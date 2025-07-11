'''
UNet for ITCZ and monsoon trough for both ATL and EPAC
'''


from ml_waves_itcz_mt.nn import *
from ml_waves_itcz_mt.util import *

import pickle
from joblib import Parallel, delayed
from torch import optim
import torch

base_dir = '/your_dir/'
out_dir = base_dir + 'ml_waves/unet_test/'

# NetCDF file to read in properly formatted input from, if it has been made already
input_file = out_dir+'input_file_imt_westhem_africa_0.25.nc_with_z'
made_input = os.path.exists(input_file)

# IMT objects
atl_imts_file = base_dir + 'ml_waves/raw_imts_atl'
with open(atl_imts_file, 'rb') as f:
    atl_imts_by_time = pickle.load(f)
epac_imts_file = base_dir +'ml_waves/raw_imts_epac'
with open(epac_imts_file, 'rb') as f:
    epac_imts_by_time = pickle.load(f)

# Get times we actually want (skip february 29th-related ones)
atl_imts_by_time = {key: val for (key, val) in atl_imts_by_time.items()  if  key[4:6] in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'] and key[4:]!='01010000'
                    and key[4:] not in ['02290000', '02290600', '02291200', '02291800', '03010000'] and key[:4] in [str(y) for y in range(2012, 2023)]}
epac_imts_by_time = {key: val for (key, val) in epac_imts_by_time.items() if  key[4:6] in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'] and key[4:]!='01010000'
                     and key[4:] not in ['02290000', '02290600', '02291200', '02291800', '03010000'] and key[:4] in [str(y) for y in range(2012, 2023)]}


# how much to smudge labels in grid cells (note: this is from a single cell center,
# unlike original unet_wave_full_res_prep which regirdded from 1 degree file)
lat_smudge=6
lon_smudge=6

# combine imts from both datasets             
for t in epac_imts_by_time.keys():
    try:
        atl_imts_by_time[t] = atl_imts_by_time[t] + epac_imts_by_time[t]
    except:
        atl_imts_by_time[t] = epac_imts_by_time[t]

imts_by_time = atl_imts_by_time


twd_years = [str(y) for y in range(2012,2023)]

max_lat = 28
min_lon = -160
min_lat = -18.75
max_lon = 80

levels = ['1000', '925', '850', '750', '650', '550']
levels = [int(level) for level in levels]
## Open the xarray datasets / variables we'll be using as our predictors,
## and reformat them to fit our requirement that the only dims are time, lat, lon
u_vars = {'dir' : base_dir + 'era5_data/unet_wave_imt_full_pred_climo_-180_60_-17.75_32_0.25x0.25/',
          'prefix' : 'U_-180_60_-17.75_32_0.25x0.25_',
          'suffix' : '.nc',
          'levels' : levels,
          'var' : 'u'}
v_vars = {'dir' : base_dir + 'era5_data/unet_wave_imt_full_pred_climo_-180_60_-17.75_32_0.25x0.25/',
          'prefix' : 'V_-180_60_-17.75_32_0.25x0.25_',
          'suffix' : '.nc',
          'levels' : levels,
          'var' : 'v'}

q_vars = {'dir' : base_dir + 'era5_data/unet_wave_imt_full_pred_climo_-180_60_-17.75_32_0.25x0.25/',
          'prefix' : 'q_-180_60_-17.75_32_0.25x0.25_',
          'suffix' : '.nc',
          'levels' : levels,
          'var' : 'q'}

tcw_vars = {'dir' : base_dir + 'era5_data/unet_wave_imt_full_pred_climo_-180_60_-17.75_32_0.25x0.25/',
          'prefix' : 'tcw_-180_60_-17.75_32_0.25x0.25_',
          'suffix' : '.nc',
          'levels' : [],
          'var' : 'tcw'}

gridsat_vars = {'dir' : base_dir + 'gridsat_data/converted_files_0.25_whem_africa/',
                'prefix' : 'gridsat_full_year_whem_africa_0.25_',
                'suffix' : '.nc',
                'levels' : [],
                'var' : 'irwin_cdr'}

# try using geopotential height too
z_vars = {'dir' : base_dir + 'era5_data/unet_wave_imt_full_pred_climo_-180_60_-17.75_32_0.25x0.25/',
          'prefix' : 'z_-180_60_-17.75_32_0.25x0.25_',
          'suffix' : '.nc',
          'levels' : levels,
          'var' : 'z'}

if not made_input:
    for year in twd_years:
        if os.path.exists(input_file+'_'+year):
            continue
        var_dicts = [u_vars, v_vars, q_vars, tcw_vars, gridsat_vars]
        files = [''.join([d['dir'],d['prefix'],year,d['suffix']]) for d in var_dicts]
        def pre(ds):
            # try to do this to save time / flip 2020 and 2021 file levels
            f = ds.encoding['source']
            year = f[f.index('.nc')-4:f.index('.nc')]
            year_slice = slice(year+'-01-01', year+'-12-31T18')
            if ds['latitude'][0] < ds['latitude'][1]:
                # reverse latitude to be decreasing                       
                ds = ds.isel(latitude=slice(None,None,-1))
            ds = ds.sel(latitude=slice(max_lat, min_lat), longitude=slice(min_lon, max_lon))
            if 'level' in ds.dims:
                if ds['level'][0] < ds['level'][1]:
                    ds = ds.isel(level=slice(None,None,-1))
                #NOTE:using same levels for all vars here                 
                return ds.sel(time=year_slice, level=levels)
            else:
                return ds.sel(time=year_slice)
        ds = xr.open_mfdataset([''.join([d['dir'],d['prefix'],year,d['suffix']]) for d in var_dicts], preprocess=pre)
        # convert variables to format where they have no level dimension, and each level is its own variable 
        for d in var_dicts:
            if len(d['levels']) > 0:
                ds = ds.assign({d['var']+'_'+str(level) : ds[d['var']].sel(level=level) for level in d['levels']})
                ds = ds.drop_vars(d['var'])
        ds = ds.drop_dims('level')
        print('opened files for', year)

        # also add calendar day number, hour of day, lat, and lon as predictor vars
        # yes, this is the correct argument order somehow                 
        days_of_year = [pd.to_datetime(t.values).dayofyear for t in ds['time']]
        time_of_day = [pd.to_datetime(t.values).hour for t in ds['time']]
        _, day_grid, _ = np.meshgrid(ds['latitude'], days_of_year, ds['longitude'])
        lat_grid, time_grid, lon_grid = np.meshgrid(ds['latitude'], time_of_day, ds['longitude'])

        ds = ds.assign({grid_name : (('time', 'latitude', 'longitude'), grid) for grid, grid_name in zip([lat_grid, time_grid, lon_grid, day_grid],
                                                                                                         ['lat_grid', 'time_grid', 'lon_grid', 'day_grid'])})
        
        input_ds = ds
        print('added time, lat, lon predictors', year)
        imts_for_year = {k: v for k,v in imts_by_time.items() if str(k)[:4] == year}
        input_ds = add_imt_mask_to_xr(ds, imts_for_year, lon_smudge=lon_smudge, lat_smudge=lat_smudge)
        input_ds.to_netcdf(input_file+'_'+year)
        print('saved', year)

    # have to norm in this way if don't have enough memory to open entire file
    input_ds = xr.open_dataset(input_file+'_'+year)
    vs = [v for v in list(input_ds.data_vars) if v not in ('imt_labels')]
    print(vs)
    # aggregate values from all files
    def get_totals(year):
        totals = {var : 0 for var in vs}
        total_squares = {var : 0 for var in vs}
        counts = {var : 0 for var in vs}
        for var in vs:
            input_ds = xr.open_dataset(input_file+'_'+year)

            #for i in range(input_ds[var].shape[0]):
            #    totals[var] += input_ds[var][i].sum()
            #    total_squares[var] += (input_ds[var][i]**2).sum()
            #    counts[var] += input_ds[var][i].shape[0]*input_ds[var][i].shape[1]

            totals[var] = float(input_ds[var].sum(skipna=True))
            total_squares[var] = float((input_ds[var]**2).sum(skipna=True))
            counts[var] = int(input_ds[var].count()-(np.isnan(input_ds[var])*1).sum())
            print(var)
            if var == 'irwin_cdr':
                print(totals[var], total_squares[var], counts[var])
        print('added totals for', year)
        return totals, total_squares, counts

    returns = Parallel(n_jobs=-1)(delayed(get_totals)(year) for year in twd_years)
    print(len(returns))
    var_totals = {var : 0 for var in vs}
    var_total_squares = {var : 0 for var in vs}
    var_counts = {var : 0 for var in vs}
    for r in returns:
        for var in vs:
            var_totals[var]+=r[0][var]
            var_total_squares[var]+=r[1][var]
            var_counts[var]+=r[2][var]
    print('added all totals together')
    # calc means and stds
    var_means = {var : float(var_totals[var]/var_counts[var]) for var in vs}
    var_stds  = {var : float(np.sqrt(var_total_squares[var]/var_counts[var] - (var_totals[var]/var_counts[var])**2)) for var in vs}
    t_start = '-01-01'
    t_end = '-12-31T18'
    for year in twd_years:
        input_ds = xr.open_dataset(input_file+'_'+year)
        for var in vs:
            # try saving as float32 to save space
            input_ds[var] = ((input_ds[var]-var_means[var])/var_stds[var]).fillna(0).astype('float32')
            # save mean and stdev values for normalizing other data
            input_ds[var].attrs['mean_pre_norm'] = var_means[var]
            input_ds[var].attrs['stdev_pre_norm'] = var_stds[var]            
        input_ds.to_netcdf(input_file+'_'+year+'_norm') # can't just overwrite old files
        print('saved norm and wave weight calcs', year)

        
    # save all years as one big file 
    input_ds=xr.open_mfdataset([input_file+'_'+year+'_norm' for year in twd_years])
    input_ds.to_netcdf(input_file)
    print('saved whole file', year)
            
input_ds = xr.open_dataset(input_file)

# print means and stds as sanity check
for v in list(input_ds.data_vars):
    input_ds = xr.open_dataset(input_file)
    print(v, float(input_ds[v].mean(dtype='float64')), float(input_ds[v].std(dtype='float64')))
