import xarray as xr
import numpy as np
import subprocess
import os
import shutil
import utils

# Random number generator
rng = np.random.default_rng()






def esmda(model_prior_list, data_obs, sigma, sim_path, data_path,
          alphas=4, data_prior=None, zf_model=None, localization_matrix=None, 
          callback_post=None, return_post_data=True, return_steps=False, random=None,
          dir_save_fields=None
          ):
    
    """ES-MDA algorithm ([EmRe13]_) with optional localization.

    Consult the section :ref:`esmda` in the manual for the theory and more
    information about ES-MDA.

    Parameters
    ----------
    model_prior : ndarray
        Prior models of dimension ``(ne, ...)``, where ``ne`` is the number of
        ensembles.
    forward : callable
        Forward model that takes an ndarray of the shape of the prior models
        ``(ne, ...)``, and returns a ndarray of the shape of the prior data
        ``(ne, nd)``; ``ne`` is the number of ensembles, ``nd`` the number of
        data.
    data_obs : ndarray
        Observed data of shape ``(nd)``.
    sigma : {float, ndarray}
        Standard deviation(s) of the observation noise.
    alphas : {int, array-like}, default: 4
        Inflation factors for ES-MDA.
    data_prior : ndarray, default: None
        Prior data ensemble, of shape ``(ne, nd)``.
    callback_post : function, default: None
        Function to be executed after each ES-MDA iteration to the posterior
        model, ``callback_post(model_post)``.
    return_post_data : bool, default: True
        If true, returns also ``forward(model_post)``.
    return_steps : bool, default: False
        If true, returns model and data of all ES-MDA steps. Setting
        ``return_steps`` to True enforces ``return_post_data=True``.
    random : {None, int,  np.random.Generator}, default: None
        Seed or random generator for reproducibility; see
        :func:`rng`.
    localization_matrix : {ndarray, None}, default: None
        If provided, apply localization to the Kalman gain matrix, of shape
        ``(model-shape, nd)``.


    Returns
    -------
    model_post : ndarray
        Posterior model ensemble.
    data_post : ndarray, only returned if ``return_post_data=True``
        Posterior simulated data ensemble.

    """
    if not os.path.exists(dir_save_fields):
        print(f'Create dir to save fields: {dir_save_fields}')
        os.makedirs(dir_save_fields)


    if isinstance(model_prior_list, list):
        if (len(model_prior_list)==2) and isinstance(model_prior_list[0], xr.Dataset) and isinstance(model_prior_list[1], xr.Dataset):
            # Convert list to numpy array
            model_prior, model_prior_shape = utils.flatten_concat_wind_field(u_data=model_prior_list[0], v_data=model_prior_list[1])
        else:
            raise ValueError(f"model_prior_list must be list of xr.Dataset with order [u_data, v_data]. However, it has lenght of: {len(model_prior_list)}")
    else:
        raise TypeError(f"model_prior_list has to be of type list, and contain 2 datasets (u_data and v_data ). However type is: {type(model_prior_list)}")
    
    # Get number of ensembles and time steps
    ne = model_prior.shape[0]
    nd = data_obs.size

    

    # Expand sigma if float
    if np.asarray(sigma).size == 1:
        sigma = np.zeros(nd) + sigma

    # Get alphas
    if isinstance(alphas, int):
        alphas = np.zeros(alphas) + alphas
    else:
        alphas = np.asarray(alphas)

    # Copy prior as start of post (output)
    model_post = model_prior.copy()



    # Loop over alphas
    for i, alpha in enumerate(alphas):
        print(f"ES-MDA step {i+1: 3d}; α={alpha}")

        # == Step (a) of Emerick & Reynolds, 2013 ==
        # Run the ensemble from time zero.

        # Get data
        if i > 0 or data_prior is None:
            
            # Run model for each ensemble member
            u_data_prior = []
            v_data_prior = []
            
            for memb_idx in range(ne):
                # data_prior = forward(model_post)
                graspIn_dir  = f"{sim_path}_m{str(memb_idx+1).zfill(2)}"
                graspOut_dir = f"{data_path}_m{str(memb_idx+1).zfill(2)}"
                # Run individual ensemble member
                utils.forward_member(sims_directory=graspIn_dir, data_directory=graspOut_dir)
                # Load data, only at measurements location and times
                da_u_simdata, da_v_simdata = utils.load_SimData_member(data_directory=graspOut_dir)
                da_u_predobs, da_v_predobs = utils.measurement_operator(ds_u=da_u_simdata, ds_v=da_v_simdata, x_idx=64, y_idx=64, vert_levs=slice(10,300))
                # Append data to new np.ndarray
                u_data_prior.append(da_u_predobs)
                v_data_prior.append(da_v_predobs)
                

                ## Combine DataArrays to Dataset and add to posterior data dictionary
                da_u_predobs = da_u_predobs.reindex_like(da_v_predobs, method='nearest')
                # Concatenate along a new data variable
                ds_predobs = xr.Dataset({
                    'u_predobs': da_u_predobs,
                    'v_predobs': da_v_predobs
                    })
                
                # Store restart dataset for later analysis
                if dir_save_fields:
                    field_name = f"data_prior_step{str(i).zfill(2)}_member{str(memb_idx+1).zfill(2)}.nc" 
                    ds_predobs.to_netcdf(f"{dir_save_fields}/{field_name}")


            u_data_prior = np.asarray(u_data_prior)
            v_data_prior = np.asarray(v_data_prior)

            # Flatten and convert to numpy ndarray
            data_prior, data_prior_shape = utils.flatten_concat_wind_field(u_data=u_data_prior, v_data=v_data_prior)



        # == Step (b) of Emerick & Reynolds, 2013 ==
        # For each ensemble member, perturb the observation vector using
        # d_uc = d_obs + sqrt(α_i) * C_D^0.5 z_d; z_d ~ N(0, I_N_d)

        zd = rng.normal(size=(ne, nd))
        data_pert = data_obs + np.sqrt(alpha) * sigma * zd


        # == Step (c) of Emerick & Reynolds, 2013 ==
        # Update the ensemble using Eq. (3) with C_D replaced by α_i * C_D

        # Compute the (co-)variances
        # Note: The factor (ne-1) is part of the covariances CMD and CDD,
        # wikipedia.org/wiki/Covariance#Calculating_the_sample_covariance
        # but factored out of CMD(CDD+αCD)^-1 to be in αCD.
        cmodel = model_post - model_post.mean(axis=0)
        cdata  = data_prior - data_prior.mean(axis=0)


        CMD = np.moveaxis(cmodel, 0, -1) @ cdata
        CDD = cdata.T @ cdata
        CD = np.diag(alpha * (ne - 1) * sigma**2)


        # Compute inverse of C
        # C is a real-symmetric positive-definite matrix.
        # If issues arise in real-world problems, try using
        # - a subspace inversions with Woodbury matrix identity, or
        # - Moore-Penrose via np.linalg.pinv, sp.linalg.pinv, spp.linalg.pinvh.
        Cinv = np.linalg.inv(CDD + CD)

        # Calculate the Kalman gain
        K = CMD@Cinv
        print("Computations done for K")


        # Apply localization if provided
        if localization_matrix is not None:
            K *= localization_matrix

        # Update the ensemble parameters
        model_post += np.moveaxis(K @ (data_pert - data_prior).T, -1, 0)


        nr_nan = np.isnan(model_post).sum()
        if nr_nan>0:
            raise ValueError(f"model posterior has {nr_nan} nan-values of {model_post.size} total points")


        # Update Restart files
        ds_restart_u, ds_restart_v = utils.inflate_flattened_field(concat_membs_flat=model_post, shape_ens_3d=model_prior_shape, 
                                                                   da_u_example=model_prior_list[0]['member_1'], da_v_example=model_prior_list[1]['member_1'])
        

        

        for memb_idx in range(ne):
            graspIn_dir  = f"{sim_path}_m{str(memb_idx+1).zfill(2)}"
            print(f"Modify restart file: {graspIn_dir}")
            
            # Load xr.DataArray of graspOut.Restart.000.nc for each ensemble member (prior model)
            ds_restart_old = utils.load_ic_member(sims_directory=graspIn_dir, levels=None)
            
            # Extract correct member from new restart files
            da_u_mod = ds_restart_u[f'member_{i+1}']
            da_v_mod = ds_restart_v[f'member_{i+1}']
            
            # Replace prior with posterior and save it back to graspOut.Restart.000.nc -> Function only modifies input ds
            ds_restart_new = utils.modify_initial_condition(ds_input=ds_restart_old, da_mod=da_u_mod, var_name='u', levels=zf_model)
            ds_restart_new = utils.modify_initial_condition(ds_input=ds_restart_old, da_mod=da_v_mod, var_name='v', levels=zf_model)
            
            # Delete old and load new restart file
            utils.remove_file(sims_directory=graspIn_dir)
            ds_restart_new.to_netcdf(f"{graspIn_dir}/graspOutRestart.000.nc")
            
            # Store restart dataset for later analysis
            if dir_save_fields:
                field_name = f"model_post_step{str(i).zfill(2)}_member{str(memb_idx+1).zfill(2)}.nc" 
                ds_restart_new.to_netcdf(f"{dir_save_fields}/{field_name}")


        # Apply any provided post-checks
        if callback_post:
            callback_post(model_post)


        # # If intermediate steps are wanted, store results
        # if return_steps:
        #     # Initiate output if first iteration
        #     if i == 0:
        #         all_models = np.zeros((alphas.size+1, *model_post.shape))
        #         all_models[0, ...] = model_prior
        #         all_data = np.zeros((alphas.size+1, *data_prior.shape))
        #     all_models[i+1, ...] = model_post
        #     all_data[i, ...] = data_prior


    # # Compute posterior data if wanted
    if return_post_data or return_steps:
        for memb_idx in range(ne):
            # data_prior = forward(model_post)
            graspIn_dir  = f"{sim_path}_m{str(memb_idx+1).zfill(2)}"
            graspOut_dir = f"{data_path}_m{str(memb_idx+1).zfill(2)}"
            # Run individual ensemble member
            utils.forward_member(sims_directory=graspIn_dir, data_directory=graspOut_dir, verbose=1)
            
            # Load data, only at measurements location and times
            da_u_simdata, da_v_simdata = utils.load_SimData_member(data_directory=graspOut_dir)
            da_u_predobs, da_v_predobs = utils.measurement_operator(ds_u=da_u_simdata, ds_v=da_v_simdata, x_idx=64, y_idx=64, vert_levs=slice(10,300))

            ## Combine DataArrays to Dataset and add to posterior data dictionary
            da_u_predobs = da_u_predobs.reindex_like(da_v_predobs, method='nearest')
            # Concatenate along a new data variable
            ds_predobs = xr.Dataset({
                'u_predobs': da_u_predobs,
                'v_predobs': da_v_predobs
                })

            # Store restart dataset for later analysis
            if dir_save_fields:
                field_name = f"data_posterior_step{str(i).zfill(2)}_member{str(memb_idx+1).zfill(2)}.nc" 
                ds_predobs.to_netcdf(f"{dir_save_fields}/{field_name}")


    ## CLEAN UP 
    print("\n\n================================================================================")
    print("CLEAN-UP: Remove Restart-files from Simulation directory")
    for memb_idx in range(ne):
        graspIn_dir  = f"{sim_path}_m{str(memb_idx+1).zfill(2)}"
        utils.remove_file(sims_directory=graspIn_dir, file_type="graspOutRestart.000.nc", verbose=0)
    print("\n--------------------------------------------------------\n")
    print("Returning:\n\t-model_post)")
    print(f"\t-Posterior data is stored at {data_path}_m<ID>")
    print("\t\t\t*** END OF SCRIPT ***")
    print("================================================================================")

    # # Return posterior model and corresponding data (if wanted)
    # if return_steps:
    #     return model_post #, all_models, all_data,
    
    # elif return_post_data:
    #     return model_post # , data_post
    
    # else:
    return model_post