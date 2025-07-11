'''
Short script to create a separate netCDF file for labels and pickle objects for mean and stdev attributes from
wave and IMT files
'''

import pickle
import xarray as xr

base_dir = '/your_dir/'
#base_dir = '/your_dir/'
out_dir = base_dir + 'ml_waves/unet_test/'
imt_input_file = out_dir+'input_file_imt_westhem_africa_0.25.nc'
wave_input_file = out_dir+'input_file_0.25.nc'
imt_ds = xr.open_dataset(imt_input_file)
wave_ds = xr.open_dataset(wave_input_file)


# create pickle objects for f
wave_norms = {}
imt_norms = {}

for var in list(wave_ds.drop(['wave_labels', 'mask_frac', 'trust']).data_vars):
    wave_norms[var] = {'mean' : float(wave_ds[var].attrs['mean_pre_norm']),
                       'stdev' : float(wave_ds[var].attrs['stdev_pre_norm'])}

for var in list(imt_ds.drop(['imt_labels']).data_vars):
    imt_norms[var] = {'mean' : float(imt_ds[var].attrs['mean_pre_norm']),
                      'stdev' : float(imt_ds[var].attrs['stdev_pre_norm'])}


    
with open(out_dir+'unet_wave_norms', 'wb') as f:
    pickle.dump(wave_norms, f, protocol=pickle.HIGHEST_PROTOCOL)

with open(out_dir+'unet_imt_norms', 'wb') as f:
    pickle.dump(imt_norms, f, protocol=pickle.HIGHEST_PROTOCOL)
    


# save netcdf of just imt labels, wave labels, trusts, mask_fracs
ds = wave_ds[['wave_labels', 'mask_frac', 'trust']]
ds['imt_labels'] = imt_ds['imt_labels']

ds.astype('float32').to_netcdf(out_dir+'labels_file_combined_wave_imt_0.25.nc')



