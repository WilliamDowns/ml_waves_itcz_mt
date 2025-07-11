'''
Save predictions for wave and IMT unets since 1980. Unlike previous iterations of prediction
calculations, this reads input data from separate files rather than from one pre-formatted file
(wanted to save space)
'''

from ml_waves_itcz_mt.nn import *
import pickle
import pandas as pd

base_dir = '/your_dir/'
out_dir = base_dir + 'ml_waves/'

input_files_dir = out_dir + 'inputs/unet_wave_imt_full_pred_climo_-180_60_-17.75_32_0.25x0.25/'
input_files_middle = '_-180_60_-17.75_32_0.25x0.25_'
var_file_prefixes = ['tcw', 'U', 'V', 'q']
gridsat_file_prefix = 'gridsat_full_year_combined_wave_imt_whem_africa_0.25_'
years = [str(y) for y in range(1980, 2023)]
#years = ['2021', '2023']

checkpoint_prefix = 'checkpoint_epoch'
checkpoint_suffix = '.pth'

# Try out adding some vars from 6 hours before for waves
add_24hr_vars=True

wave_sample_height=160
wave_sample_width=640

imt_checkpoint_dir = out_dir + 'archives/20230812_imt_5tier_32init_smallepochs_2e-4lr/' #16.83
imt_checkpoint_num = 16.83
imt_sample_height=176
imt_sample_width=640

## Load in means and stdevs for normalization of input data
wave_norm_f = out_dir + 'unet_wave_norms'
imt_norm_f = out_dir + 'unet_imt_norms'

# Determined this order from logs of training runs (can also be found by dropping other variables and printing data_vars in an input_file)

#wave_var_order = ['tcw', 
wave_var_order = ['irwin_cdr', 'tcw', 
                  'u_1000', 'u_925', 'u_850', 'u_750', 'u_650', 'u_550', 
                  'v_1000', 'v_925', 'v_850', 'v_750', 'v_650', 'v_550', 
                  'q_1000', 'q_925', 'q_850', 'q_750', 'q_650', 'q_550', 
                  'lat_grid', 'lon_grid', 'day_grid']

if add_24hr_vars:
    var_names_24hr = ['u_850', 'u_750', 'u_650', 'v_850', 'v_750', 'v_650', 'q_850', 'q_750', 'q_650', 'tcw']
    wave_var_order = wave_var_order + var_names_24hr

imt_var_order = ['irwin_cdr', 'tcw', 
                  'u_1000', 'u_925', 'u_850', 'u_750', 'u_650', 'u_550', 
                  'v_1000', 'v_925', 'v_850', 'v_750', 'v_650', 'v_550', 
                  'q_1000', 'q_925', 'q_850', 'q_750', 'q_650', 'q_550']

with open(wave_norm_f, 'rb') as f:
    wave_norms = pickle.load(f)

with open(imt_norm_f, 'rb') as f:
    imt_norms = pickle.load(f)


wave_means = np.zeros([len(wave_var_order), wave_sample_height, wave_sample_width])
wave_stds = np.zeros([len(wave_var_order), wave_sample_height, wave_sample_width])
for i, var in enumerate(wave_var_order):
    wave_means[i], wave_stds[i] = wave_norms[var]['mean'], wave_norms[var]['stdev']

imt_means = np.zeros([len(imt_var_order), imt_sample_height, imt_sample_width])
imt_stds = np.zeros([len(imt_var_order), imt_sample_height, imt_sample_width])
for i, var in enumerate(imt_var_order):
    imt_means[i], imt_stds[i] = imt_norms[var]['mean'], imt_norms[var]['stdev']
    


class norm_transform:
    def __init__(self, means, stds):
        self.means = means
        self.stds = stds 
    def __call__(self, data):
        return (data-self.means)/self.stds

wave_transform = norm_transform(wave_means, wave_stds)
imt_transform = norm_transform(imt_means, imt_stds)

def pre(ds):
    # need to do this to flip file levels / latitudes for some of them
    if ds['latitude'][0] < ds['latitude'][1]:
        # reverse latitude to be decreasing            
        ds = ds.isel(latitude=slice(None,None,-1))
    if 'level' in ds.dims:
        if ds['level'][0] < ds['level'][1]:
            ds = ds.isel(level=slice(None,None,-1))
        #NOTE:using same levels for all vars here
        return ds
    else:
        return ds


### Wave prediction calculations
## Initialize saved wave UNet
wave_checkpoint_dirs = [out_dir + 'archives/' + '20240122_wave_revisedmaskfrac_skiptimes_noafrica_4trim_drop_pvsvcvw_sparselevelsimt_5tier_32_lateyears_24hrmean/']
wave_checkpoint_nums = [5.85]

do_africa=False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for wave_checkpoint_dir, wave_checkpoint_num in zip(wave_checkpoint_dirs, wave_checkpoint_nums):
    unet = UNet_3Plus_no_first_skip(len(wave_var_order), 1, initial_target_channels=32, dropout_frac=0.25, conv=5, n_tiers=5, kernel_size=5, deep_sup=True, init_full_skip=False,
                             padding=2).to(device)
    unet.to(device=device)
    state_dict = torch.load(wave_checkpoint_dir+checkpoint_prefix+str(wave_checkpoint_num)+checkpoint_suffix, map_location=device)
    unet.load_state_dict(state_dict)

    wave_start_date = '-05-01'
    wave_end_date = '-11-30T18'
    if do_africa:
        wave_min_lon = -99.75
        wave_max_lon = 60
    else:
        wave_min_lon = -160
        wave_max_lon = -0.25
    wave_min_lat = -7.75
    wave_max_lat = 32

    ## iterate over years since files are saved by year
    preds = []
    for y in years:
        ds = xr.open_mfdataset([input_files_dir + gridsat_file_prefix+y+'.nc']+[input_files_dir + v + input_files_middle + y + '.nc' for v in var_file_prefixes], 
                               chunks = {'time' : 1}, preprocess=pre)
        ds = xr.open_mfdataset([input_files_dir + v + input_files_middle + y + '.nc' for v in var_file_prefixes], 
                               chunks = {'time' : 1}, preprocess=pre)

        ds = ds.sel(time=slice(y+wave_start_date, y+wave_end_date),
                    latitude=slice(wave_max_lat,wave_min_lat),
                    longitude=slice(wave_min_lon, wave_max_lon))
        

        if add_24hr_vars:
            #time_range = range(1, len(ds.time)) #6hr
            #time_range = range(2, len(ds.time)) #12hr
            time_range = range(4, len(ds.time)) #24hr mean
        else:
            time_range = range(len(ds.time))            

        for i in time_range:
            sample = ds.isel(time=i).load().fillna(0)
            time = [ds['time'].isel(time=i).values]
            # convert variables to format where they have no level dimension, and each level is its own variable
            for v in var_file_prefixes[1:]:
                sample = sample.assign({v.lower()+'_'+str(level) : sample[v.lower()].sel(level=level) for level in [1000, 925, 850, 750, 650, 550]})
                sample = sample.drop_vars(v.lower())
            sample = sample.drop_dims('level')

            # add calendar day number, hour of day, lat, and lon as predictor vars
            days_of_year = [pd.to_datetime(t).dayofyear for t in time]
            time_of_day = [pd.to_datetime(t).hour for t in time]
            day_grid = np.full([len(sample['latitude']), len(sample['longitude'])], days_of_year[0])
            lon_grid, lat_grid = np.meshgrid(sample['longitude'], sample['latitude'])        
            sample = sample.assign({grid_name : (('latitude', 'longitude'), grid) for grid, grid_name in zip([lat_grid, lon_grid, day_grid],
                                                                                                             ['lat_grid', 'lon_grid', 'day_grid'])})

            if add_24hr_vars:
                for var_name in var_names_24hr:
                    new_var_name = var_name +'_6bef'
                    if '_' in var_name:
                        [var, level] = var_name.split('_')                    
                        #new_da = ds[var].isel(time=i-1).sel(level=int(level)).assign_coords({'time' : ds['time'][i]}).rename(new_var_name).drop('level') #6hr
                        #new_da = ds[var].isel(time=i-2).sel(level=int(level)).assign_coords({'time' : ds['time'][i]}).rename(new_var_name).drop('level') #12hr
                        new_da = ds[var].isel(time=slice(i-4, i-1)).sel(level=int(level)).mean('time').assign_coords({'time' : ds['time'][i]}).rename(new_var_name).drop('level') #24hr mean
                    else:
                        #new_da = ds[var_name].isel(time=i-1).assign_coords({'time' : ds['time'][i]}).rename(new_var_name) #6hr
                        #new_da = ds[var_name].isel(time=i-2).assign_coords({'time' : ds['time'][i]}).rename(new_var_name) #12hr
                        new_da = ds[var_name].isel(time=slice(i-4, i-1)).mean('time').assign_coords({'time' : ds['time'][i]}).rename(new_var_name) #24hr mean
                    #print(new_da)
                    sample[new_var_name] = new_da
                sample = sample.load().fillna(0)
            # run net predictions
            prediction = predict_xr(unet, sample, device, input_norm=False, transform=wave_transform).expand_dims({'time':time}).rename('pred')
            preds.append(prediction)

    preds = xr.concat(preds, 'time')
    if do_africa:
        preds.to_netcdf(wave_checkpoint_dir+'wave_pred_2021_africa.nc')
    else:
        preds.to_netcdf(wave_checkpoint_dir+'wave_pred_1980_2022.nc')
        #preds.to_netcdf(wave_checkpoint_dir+'wave_pred_2021_2023.nc')
        




### Imt prediction calculations
## Initialize saved imt UNet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
unet = UNet_3Plus_no_first_skip(len(imt_var_order), 3, initial_target_channels=32, dropout_frac=0.25, conv=5, n_tiers=5, kernel_size=5, deep_sup=True, init_full_skip=False,
                         padding=2).to(device)
unet.to(device=device)
state_dict = torch.load(imt_checkpoint_dir+checkpoint_prefix+str(imt_checkpoint_num)+checkpoint_suffix, map_location=device)
unet.load_state_dict(state_dict)

imt_start_date = '-01-01'
imt_end_date = '-12-31T18'
imt_min_lon = -160
imt_max_lon = -0.25
imt_min_lat = -15.75
imt_max_lat = 28

thresh=0.10
## iterate over years since files are saved by year
#preds = []
for y in years:
    preds = []
    ds = xr.open_mfdataset([input_files_dir + gridsat_file_prefix+y+'.nc']+[input_files_dir + v + input_files_middle + y + '.nc' for v in var_file_prefixes], preprocess=pre,
                           chunks = {'time' : 1})
    print(ds)
    ds = ds.sel(time=slice(y+imt_start_date, y+imt_end_date),
                latitude=slice(imt_max_lat,imt_min_lat),
                longitude=slice(imt_min_lon, imt_max_lon))

    for i in range(len(ds.time)):
        sample = ds.isel(time=i).load().fillna(0)
        time = [ds['time'].isel(time=i).values]
        # convert variables to format where they have no level dimension, and each level is its own variable
        for v in var_file_prefixes[1:]:
            sample = sample.assign({v.lower()+'_'+str(level) : sample[v.lower()].sel(level=level) for level in [1000, 925, 850, 750, 650, 550]})
            sample = sample.drop_vars(v.lower())
        sample = sample.drop_dims('level')
        # run net predictions
        prediction = predict_xr(unet, sample, device, input_norm=False, transform=imt_transform, out_threshold=thresh, 
                                each_class=True, class_names=['itcz', 'mt']).expand_dims({'time':time}).rename('pred')
        preds.append(prediction)
    preds = xr.concat(preds, 'time')
    preds.to_netcdf(imt_checkpoint_dir+'imt_pred_'+y+'.nc')
#preds = xr.concat(preds, 'time')
#preds.to_netcdf(imt_checkpoint_dir+'imt_pred_1980_2022.nc')
#for y, pred in zip(years, preds):
#    pred.to_netcdf(imt_checkpoint_dir+'imt_pred_'+y+'.nc')

#preds.to_netcdf(imt_checkpoint_dir+'imt_pred_2021.nc')

