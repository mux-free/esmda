import xarray as xr
import numpy as np
import subprocess
import os
import shutil



####
# Functions to run simulations 
####
def forward_member(sims_directory: str, data_directory: str, verbose=0) -> None:

    # Updated command to run model within the Singularity container
    def _run_model_in_container(path_nml_file):
        singularity_exec_cmd = [
            'singularity',
            'exec',
            '--nv',
            '/home/maxf/singularity/images/aspkit-devel-cuda12.2.2-ubuntu22.04-x86_64_8.2.3.sif',
            'aspire',
            path_nml_file]
        subprocess.run(singularity_exec_cmd, check=True)


    def _move_simdata_to_datafolder(sims_directory, data_directory):
        # Ensure the destination directory exists, create it if necessary
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)
        # Define source and destination paths
        files_to_move = ["graspOutSimdata.000.nc",
                        "graspOutTFMetmast.000.nc",
                        "graspOut.000.nc"]
        # Move the files
        for file_name in files_to_move:
            src = os.path.join(sims_directory, file_name)
            dest = os.path.join(data_directory, file_name)
            # Check if the file exists before moving
            if os.path.exists(src):
                shutil.move(src, dest)

    """Run forward model for all ensemble members"""
    _run_model_in_container(path_nml_file=f"{sims_directory}/graspIn.000.nml")    
    _move_simdata_to_datafolder(sims_directory=sims_directory, data_directory=data_directory)
    if verbose>0:    
        print(f"\nMove output from {sims_directory} \nto {data_directory}/graspOut<type>.000.nc")
    





def modify_initial_condition(ds_input: xr.Dataset, da_mod: xr.DataArray, var_name: str, levels=None) -> xr.Dataset:
    """
    Function modifies the graspOutRestart.000.nc files with new data for u and v.

    Parameters:
    -----------
        ds_input (xr.Dataset): graspOutRestart.000.nc for the current ensemble member
        da_mod (xr.DataArray): updated u or v field
        var_name (str): Either u or v
    """
    #================================== ASSERTATION STATEMENTS ==================================================================================
    ## assert that dimension-order is correct --> first time, then zf
    input_dims = list(ds_input[var_name].sizes)
    assert (len(input_dims) == 3) & (input_dims[0]=='zf'), f"Dimension order is not correct, first dim must be ('zf'), but dims are {input_dims}"
    #============================================================================================================================================
    if levels is None:
        levels = sorted(ds_input.zf.values)

    ## Modification of Dataset
    ds_input[var_name].loc[{"zf": slice(levels[0],levels[-1])}] = da_mod
    return ds_input

def save_modified_restart_file(ds_restart:xr.Dataset, path_member:str, verbose=0) -> None:
    """
    Save modified restart file in correct directroy
    """
    if verbose>0:
        print(f"Save modified restart file: {path_member}")
    ds_restart.to_netcdf(f"{path_member}/graspOutRestart.000.nc")

# def update_model_posterior(ds_restart_old: xr.Dataset, da_mod: xr.Dataset, mod_var: str, sims_directory: str, NE: str) -> None:
#     print(f"Update model_posterior at: {sims_directory}_m<member{NE}>")
#     for memb_idx in range(NE):
#         graspIn_dir  = f"{sims_directory}_m{str(memb_idx+1).zfill(2)}"
#         ds_restart_new = modify_initial_condition(ds_input=ds_restart_old, da_mod=da_mod, var_name=mod_var)
#         save_modified_restart_file(ds_restart=ds_restart_new, path_member=graspIn_dir, verbose=0)




def load_SimData_member(data_directory): 
    simdata_dir = f"{data_directory}/graspOutSimdata.000.nc"    
    ds_memb = xr.open_dataset(simdata_dir)
    return ds_memb['u'], ds_memb['v'] 


def load_ic_member(sims_directory, levels=None): 
    simdata_dir = f"{sims_directory}/graspOutRestart.000.nc"    
    ds_memb = xr.open_dataset(simdata_dir)
    if levels:
        ds_memb = ds_memb.sel(zf=slice(levels[0],levels[1]))
    return ds_memb


def remove_file(sims_directory, file_type="graspOutRestart.000.nc", verbose=0):
    file_path = os.path.join(sims_directory, file_type)
    
    # Check if the file exists before trying to remove it (similar to 'rm -f')
    if os.path.exists(file_path):
        try:
            os.remove(file_path)  # Remove the file
            if verbose>0:
                print(f"File {file_path} successfully removed.")
        except Exception as e:
            print(f"Error occurred while trying to remove {file_path}: {e}")
    else:
        print(f"File {file_path} does not exist.")


def measurement_operator(ds_u, ds_v, x_idx, y_idx, vert_levs):
    """ 
    This function extracts the measurement location from the 4d simulation fields --> (time, zf)
    Return a data-array for u and v
    """
    # Load data and select measurement location
    
    da_u_pred_obs = ds_u.isel(xh=x_idx, yf=y_idx).sel(zf=vert_levs)
    da_v_pred_obs = ds_v.isel(xf=x_idx, yh=y_idx).sel(zf=vert_levs)
    
    return da_u_pred_obs, da_v_pred_obs





def flatten_concat_wind_field(u_data, v_data):


    if isinstance(u_data, xr.Dataset) and isinstance(v_data, xr.Dataset):
        # Assuming that each data variable in u_data and v_data represents an ensemble member
        list_u_memb = []
        list_v_memb = []
        nr_ensembles = len(u_data.data_vars)
        # Iterate over ensemble members
        
        for u_var, v_var in zip(u_data.data_vars, v_data.data_vars):
            # Extract the wind field values for each ensemble member
            list_u_memb.append(u_data[u_var].values)
            list_v_memb.append(v_data[v_var].values)
        

        # Convert lists to numpy arrays of shape (NE, *original_shape)
        u_membs = np.asarray(list_u_memb)  # Shape: (NE, Nz, Ny, Nx)
        v_membs = np.asarray(list_v_memb)  # Same shape as u_membs
        shape_ens_3d = u_membs.shape
        assert shape_ens_3d == v_membs.shape, "Both data-arrays must have the same shape"
        # Flatten the spatial dimensions for each ensemble member
        u_membs_flat = u_membs.reshape(nr_ensembles, -1)
        v_membs_flat = v_membs.reshape(nr_ensembles, -1)


        # Concatenate the flattened u and v arrays along the last axis
        concat_membs_flat = np.concatenate((u_membs_flat, v_membs_flat), axis=1)

    elif isinstance(u_data, xr.DataArray) and isinstance(v_data, xr.DataArray):
        print("Inputs are Data-Arrays WITHOUT members")
        u_membs = u_data.values
        v_membs = v_data.values
        shape_ens_3d = u_membs.shape
        assert shape_ens_3d == v_membs.shape, "Both data-arrays must have the same shape"
        u_membs_flat = u_membs.flatten()
        v_membs_flat = v_membs.flatten()
        
        # Concatenate the flattened u and v arrays along the last axis
        concat_membs_flat = np.concatenate((u_membs_flat, v_membs_flat))

    elif isinstance(u_data, np.ndarray) and isinstance(v_data, np.ndarray):
        nr_ensembles = u_data.shape[0]
        assert nr_ensembles == v_data.shape[0], "u and v have same nr of ensembles"
        # Flatten the spatial dimensions for each ensemble member
        shape_ens_3d = u_data.shape
        assert shape_ens_3d == v_data.shape, "Both data-arrays must have the same shape"
        u_membs_flat = u_data.reshape(nr_ensembles, -1)
        v_membs_flat = v_data.reshape(nr_ensembles, -1)
        # Concatenate the flattened u and v arrays along the last axis
        concat_membs_flat = np.concatenate((u_membs_flat, v_membs_flat), axis=1)

    else:
        raise AttributeError(f"Inputs must be of type xr.DataArray of xr.Dataset, not {type(u_data)} and {type(v_data)}")
    
    
    # Return the concatenated flattened array and the original shape
    return concat_membs_flat, shape_ens_3d



def inflate_flattened_field(concat_membs_flat, shape_ens_3d, da_u_example=None, da_v_example=None):
    nr_ensembles = shape_ens_3d[0]
    member_shape = shape_ens_3d[1:]  # Original spatial dimensions (Nz, Ny, Nx)

    # Calculate the size of one flattened member (number of elements in u or v)
    size_of_one_member = np.prod(member_shape)

    # Split the concatenated array back into u and v components
    u_membs_flat = concat_membs_flat[:, :size_of_one_member]
    v_membs_flat = concat_membs_flat[:, size_of_one_member:]

    assert u_membs_flat.shape == v_membs_flat.shape, "Both data-arrays must have the same shape"

    # Reshape the flattened arrays back to their original shapes
    u_membs = u_membs_flat.reshape((nr_ensembles,) + member_shape)
    v_membs = v_membs_flat.reshape((nr_ensembles,) + member_shape)

    if (da_u_example is not None) and (da_v_example is not None):
        ds_u, ds_v = xr.Dataset(), xr.Dataset()

        for i, (u_member, v_member) in enumerate(zip(u_membs, v_membs)): 
            ds_u[f'member_{i+1}'] = xr.DataArray(data=u_member, coords=da_u_example.coords, dims=da_u_example.dims)
            ds_v[f'member_{i+1}'] = xr.DataArray(data=v_member, coords=da_v_example.coords, dims=da_v_example.dims)

        return ds_u, ds_v
    else:
        return u_membs, v_membs




