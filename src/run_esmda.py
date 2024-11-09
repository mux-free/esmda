import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
# Random number generator
rng = np.random.default_rng()
import os
import sys

import utils
import esmda

sys.path.append('/home/maxf/projects/REFORM/src/grasp/data_handling/')
import prepare_obsdata_for_aspire as profa 


#====================================================
# !!!!!!!!!!!!       CHANGES HERE       !!!!!!!!!!!!!
#====================================================
#  Define constants:
#-------------------
NE    = 15          # Ensemble members
SIGMA = 0.5         # Standard deviation(s) of the observation noise.
ALPHA = 4
zf_model=(0,2000)    # Range where model should be changed (entire domain is to big)

OBS_TYPE = 'synthetic'
# OBS_TYPE = 'LiDAR'


if OBS_TYPE.lower() == "lidar":
    start_time = '20:00'
    end_time   = '02:00'


## Define how data should be saves
experiment_desc  = f"ESMDA_ne{NE}_a{ALPHA}_{OBS_TYPE}"
DIR_DATA_STORAGE =f"/home/maxf/projects/da4gs/esmda/data/esmda_fields/{experiment_desc}"


#===================================================
#  Define directories:
#---------------------
BASE_DIR       = "/home/maxf/projects/da4gs/esmda"
sims_directory = f"{BASE_DIR}/runs/run_esmda/2020/04/04/00"
data_directory = f"{BASE_DIR}/data/ensemble_data/2020/04/04/00"
obs_save_dir   = f"{BASE_DIR}/data/observation/synthetic_obs"


##------------------------------------------------------
## Load observtions
##-----------

if 'lidar' in OBS_TYPE.lower():

    ds_lidar = profa.load_cbw_lidar_data(date_init='20200404', date_end='20200405')
    ds_lidar = ds_lidar.sel(time=slice(f"2020-04-04T{start_time}:10.000000000", f"2020-04-05T{end_time}:00.000000000"))
    ## Interpolate dataset to cabauw levels
    levs = np.asarray([ 16.,  48.153847,  80.61688 , 113.39207, 146.4824,179.89095 , 213.62071 , 247.67484 , 282.0564  ])
    ds_obs = ds_lidar.interp(level=levs, method='linear',)
    ds_obs = ds_obs.rename({'level': 'zf', 'ucbw': 'u', 'vcbw': 'v'})

elif 'syn' in OBS_TYPE.lower():
    ds_obs = xr.open_dataset(f"{obs_save_dir}/ds_obs_synthetic.nc")

else:
    raise ValueError(f"Observation type <<{OBS_TYPE}>> not recognized")



##------------------------------------------------------
## Prepare obs and model_prior to input to ESMDA-routine
##-----------
# Convert model-prior and data_obs to flattened np-arrays
data_obs, data_obs_shape = utils.flatten_concat_wind_field(u_data=ds_obs[f'u'], v_data=ds_obs[f'v'])

# Load and prepare model-prior
ds_u_model_prior = xr.Dataset()
ds_v_model_prior = xr.Dataset()
for i in range(NE):
    # Get ensemble member index as a str of length two and filled with 0s (e.g. 1 -> "01", 10 -> "10")
    print(f"Load member {str(i+1).zfill(2)}")
    ds_restart = utils.load_ic_member(sims_directory=f"{sims_directory}_m{str(i+1).zfill(2)}", levels=zf_model)
    ds_u_model_prior[f'member_{i+1}'] = ds_restart['u']
    ds_v_model_prior[f'member_{i+1}'] = ds_restart['v']

# Convert list to numpy array
model_prior, model_prior_shape = utils.flatten_concat_wind_field(u_data=ds_u_model_prior, v_data=ds_v_model_prior)

output = esmda.esmda(model_prior_list=[ds_u_model_prior, ds_v_model_prior],
                     data_obs=data_obs,
                     sigma=SIGMA,
                     sim_path=sims_directory,
                     data_path=data_directory,
                     zf_model=zf_model,
                     alphas=ALPHA, 
                     data_prior=None,localization_matrix=None, callback_post=None, 
                     return_post_data=True,
                     dir_save_fields=DIR_DATA_STORAGE
                     )


# Inflate model output
model_post = output


ds_model_post_u, ds_model_post_v = utils.inflate_flattened_field(concat_membs_flat=model_post, shape_ens_3d=model_prior_shape,
                                                                da_u_example=ds_u_model_prior['member_1'], 
                                                                da_v_example=ds_v_model_prior['member_1'])

ds_model_post_u.to_netcdf(f"{DIR_DATA_STORAGE}/ds_u_model_post.nc")
ds_model_post_v.to_netcdf(f"{DIR_DATA_STORAGE}/ds_v_model_post.nc")



print("================================================================================")
print("================================================================================")
print("\t\tEND of it all")
print("================================================================================")
print("================================================================================")

