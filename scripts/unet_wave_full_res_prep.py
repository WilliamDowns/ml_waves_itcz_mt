'''
Weighted UNet for waves through 140W, with some buffer on the west side
This is the full res version, so it differs from previous versions in that the input files are split by year
'''


from ml_waves_itcz_mt.nn import *
from ml_waves_itcz_mt.util import *
from ml_waves_itcz_mt.regrid import *

import pickle
from joblib import Parallel, delayed


base_dir = '/your_dir/'
out_dir = base_dir + 'ml_waves/unet_test/'

# NetCDF file to read in properly formatted input from, if it has been made already
input_file = out_dir+'input_file_0.25.nc'
made_input = os.path.exists(input_file)

# TWD waves
atl_waves_file = base_dir + 'ml_waves/final_waves'
with open(atl_waves_file, 'rb') as f:
    atl_waves_by_time = pickle.load(f)
epac_waves_file = base_dir +'ml_waves/final_waves_epac'
with open(epac_waves_file, 'rb') as f:
    epac_waves_by_time = pickle.load(f)

# Get times we actually want    
atl_waves_by_time = {key: val for (key, val) in atl_waves_by_time.items()  if  key[4:6] in ['05', '06', '07', '08', '09', '10', '11']}
epac_waves_by_time = {key: val for (key, val) in epac_waves_by_time.items() if  key[4:6] in ['05', '06', '07', '08', '09', '10', '11']}

## Mask out waves that don't fit our ZESA criteria
# lat/lon bounds for all input fields
# note differing order of lat and lon values
extent=[-160, 0, -10, 32]
[minlon, maxlon, minlat, maxlat] = extent
lat_slice = slice(maxlat, minlat)
lon_slice = slice(minlon, maxlon)
# regridder for moving wave data from 1x1 to 0.25x0.25
template_ds = xr.open_dataset(base_dir + 'era5_data/unet_wave_0.25x0.25/U_global_0.25x0.25_2004.nc')
regrid_ds = xr.open_dataset(out_dir+'input_file_trusts_fracs_140W_smudged_norm.nc')[['wave_labels', 'mask_frac', 'trust']]
regridder = make_regridder(regrid_ds, template_ds, [minlon, maxlon, minlat, maxlat], regrid_method='nearest_s2d')

# drop waves outside of lon bounds
dropped_waves=0
total_waves=0
for waves_by_time in [atl_waves_by_time, epac_waves_by_time]:
    for t_waves in waves_by_time.values():
        for wave in t_waves:
            total_waves+=1
            wave_minlon = 0-wave.extent[0][1]
            wave_maxlon = 0-wave.extent[1][1]
            if wave_minlon < extent[0] or wave_maxlon > extent[1]:
                t_waves.remove(wave)
                dropped_waves+=1

print('dropped waves for being outside of extent:', dropped_waves, 'of', total_waves, 'waves')

# vorticity and humidity files for masking
mask_hum_folder = base_dir + 'era5_data/unet_wave_1x1/area_averaged_fields/'
mask_curv_folder = base_dir + 'era5_data/unet_wave_1x1/area_averaged_fields/'

mask_hum_prefix = 'avg_q_stdev_2d_'
mask_hum_suffix = '_8gauss.nc'
mask_curv_prefix = 'avg_cv_stdev_2d_'
mask_curv_suffix = '_8gauss.nc'

mask_hum_var = 'area_avg_anom_q'
mask_hum_level_var = 'area_avg_anom_layer_avg_q'
mask_curv_var = 'area_avg_anom_cv'
mask_curv_level_var = 'area_avg_anom_layer_avg_cv'

# levels to make pairs of humidity and vort anoms from for masks
# also doing layer averaged
mask_levels = [850, 800, 750, 700]

# parameters of ZESA masking
mask_frac = 0.65
atl_n_cell = 19
epac_n_cell = 15
atl_peak_spacing = 9
epac_peak_spacing = 7
scaling_factor = 0.3

power = 2

twd_years = set([time[:4] for time in waves_by_time.keys()])
# get times to include from files, starting on May 1st and ending November 30th
# get rid of may 1st 00z to avoid issues when converting to ERA times (6 hours back from wave times)
times_to_keep = [year+t for t in global_times if t[:2] in ['05', '06', '07', '08', '09', '10', '11'] for year in twd_years if t!='05010000']
waves_by_time = {key: val for key, val in waves_by_time.items() if key in times_to_keep}

# how much to smudge labels in longitude
lon_smudge=4#1


wave_out_file = ''.join([out_dir, 'masked_waves_weighted_140W', str(mask_frac), '_', str(atl_n_cell)])
print(wave_out_file)
# don't remake wave mask if don't have to
if not os.path.exists(wave_out_file):
    mask_hum_ds = xr.open_mfdataset([''.join([mask_hum_folder, mask_hum_prefix, year, mask_hum_suffix]) for year in twd_years]).sel(latitude=lat_slice, longitude=lon_slice)
    mask_curv_ds = xr.open_mfdataset([''.join([mask_curv_folder, mask_curv_prefix, year, mask_curv_suffix]) for year in twd_years]).sel(latitude=lat_slice, longitude=lon_slice)

    masks_hum = [mask_hum_ds[mask_hum_var].sel(level=level) for level in mask_levels] + [mask_hum_ds[mask_hum_level_var]]
    masks_curv = [mask_curv_ds[mask_curv_var].sel(level=level) for level in mask_levels] + [mask_curv_ds[mask_curv_level_var]]

    masks = [(hum, curv) for hum, curv in zip(masks_hum, masks_curv)]
    # divide waves into chunks for parallel processing cause masking function is slow as hell
    atl_waves_by_year = [{time : val for time, val in atl_waves_by_time.items() if time[:4] == year} for year in twd_years]
    epac_waves_by_year = [{time : val for time, val in epac_waves_by_time.items() if time[:4] == year} for year in twd_years]    
    def get_wave_fracs(i, waves_by_year, n_cells, peak_spacing):
        year = list(waves_by_year[i].keys())[0][:4]
        year_slice = slice(year+'-05-01',year+'-11-30T18')
        return mask_waves_by_bimodal_frac(waves_by_year[i], mask_frac, [(m1.sel(time=year_slice).copy(deep=True), m2.sel(time=year_slice).copy(deep=True)) for (m1, m2) in masks],
                                          n_cell=n_cells, peak_spacing=peak_spacing, power=power, scaling_factor=scaling_factor, save_frac=True)
    atl_waves = Parallel(n_jobs=-1)(delayed(get_wave_fracs)(i, atl_waves_by_year, atl_n_cell, atl_peak_spacing) for i in range(len(atl_waves_by_year)))
    epac_waves = Parallel(n_jobs=-1)(delayed(get_wave_fracs)(i, epac_waves_by_year, epac_n_cell, epac_peak_spacing) for i in range(len(epac_waves_by_year)))    
    atl_waves_by_time = {time : val for d in atl_waves for time, val in d.items()}
    epac_waves_by_time = {time : val for d in epac_waves for time, val in d.items()}    
    # combine waves from both datasets             
    for t in epac_waves_by_time.keys():
        try:
            atl_waves_by_time[t] = atl_waves_by_time[t] + epac_waves_by_time[t]
        except:
            atl_waves_by_time[t] = epac_waves_by_time[t]
    
    # save to disk to avoid having to rerun this if doing same parameters
    with open(wave_out_file, 'wb') as f:
        pickle.dump(atl_waves_by_time, f, protocol=pickle.HIGHEST_PROTOCOL)

        
# load in masked waves
with open(wave_out_file, 'rb') as f:
    waves = pickle.load(f)


levels = ['1000', '925', '850', '800', '750', '700', '650', '600', '550', '500']
levels = [int(level) for level in levels]
## Open the xarray datasets / variables we'll be using as our predictors,
## and reformat them to fit our requirement that the only dims are time, lat, lon
u_vars = {'dir' : base_dir + 'era5_data/unet_wave_0.25x0.25/',
          'prefix' : 'U_global_0.25x0.25_',
          'suffix' : '.nc',
          'levels' : levels,
          'var' : 'u'}
v_vars = {'dir' : base_dir + 'era5_data/unet_wave_0.25x0.25/',
          'prefix' : 'V_global_0.25x0.25_',
          'suffix' : '.nc',
          'levels' : levels,
          'var' : 'v'}
curv_vars = {'dir' : base_dir + '/era5_data/unet_wave_0.25x0.25/',
             'prefix' : 'cv_',
             'suffix' : '.nc',
             'levels' : levels,
             'var' : 'cv'}

q_vars = {'dir' : base_dir + 'era5_data/unet_wave_0.25x0.25/',
          'prefix' : 'q_global_0.25x0.25_',
          'suffix' : '.nc',
          'levels' : levels,
          'var' : 'q'}

omega_vars = {'dir' : base_dir + 'era5_data/unet_wave_0.25x0.25/',
              'prefix' : 'omega_global_0.25x0.25_',
              'suffix' : '.nc',
              'levels' : levels,
              'var' : 'w'}

pv_vars = {'dir' : base_dir + 'era5_data/unet_wave_0.25x0.25/',
           'prefix' : 'pv_global_0.25x0.25_',
           'suffix' : '.nc',
           'levels' : levels,
           'var' : 'pv'}

sv_vars = {'dir' : base_dir + 'era5_data/unet_wave_0.25x0.25/',
           'prefix' : 'sv_',
           'suffix' : '.nc',
           'levels' : levels,
           'var' : 'sv'}


tcw_vars = {'dir' : base_dir + 'era5_data/unet_wave_0.25x0.25/',
          'prefix' : 'tcw_global_0.25x0.25_',
          'suffix' : '.nc',
          'levels' : [],
          'var' : 'tcw'}

gridsat_vars = {'dir' : base_dir + 'gridsat_data/converted_files_0.25/',
                'prefix' : 'gridsat_0.25_',
                'suffix' : '.nc',
                'levels' : [],
                'var' : 'irwin_cdr'}

# Note this section is different than the 1x1 testing; don't have enough memory to make one giant file immediately here
if not made_input:
    for year in twd_years:
        if os.path.exists(input_file+'_'+year):
            continue        
        var_dicts = [u_vars, v_vars, curv_vars, q_vars, omega_vars, pv_vars, sv_vars, tcw_vars, gridsat_vars]
        files = [''.join([d['dir'],d['prefix'],year,d['suffix']]) for d in var_dicts]
        def pre(ds):
            # try to do this to save time / flip 2020 and 2021 file levels
            f = ds.encoding['source']
            year = f[f.index('.nc')-4:f.index('.nc')]#f[f.index(d['suffix'])-4:f.index(d['suffix'])]
            year_slice = slice(year+'-05-01', year+'-11-30T18')
            if ds['latitude'][0] < ds['latitude'][1]:
                # reverse latitude to be decreasing
                ds = ds.isel(latitude=slice(None,None,-1))
            if 'level' in ds.dims:
                if ds['level'][0] < ds['level'][1]:
                    ds = ds.isel(level=slice(None,None,-1))
                #NOTE:using same levels for all vars here
                return ds.sel(time=year_slice, level=levels, latitude=lat_slice, longitude=lon_slice)
            else:
                return ds.sel(time=year_slice, latitude=lat_slice, longitude=lon_slice)
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
        # add in wave labels to dataset
        for var in list(ds.data_vars):
            if var != 'time_grid':
                ds[var] = ds[var].astype('float32')
        waves_for_year = {k: v for k,v in waves.items() if str(k)[:4] == year}
        input_ds = add_wave_mask_to_xr(ds, waves_for_year, mask_frac_name='mask_frac', trust_name='trust', lon_smudge=lon_smudge)

        input_ds.to_netcdf(input_file+'_'+year)
        print('saved', year)

    # have to norm in this way if don't have enough memory to open entire file
    input_ds = xr.open_dataset(input_file+'_'+year)
    vs = [v for v in list(input_ds.data_vars) if v not in ('wave_labels', 'trust', 'mask_frac')]
    # aggregate values from all files
    def get_totals(year):
        totals = {var : 0 for var in vs}
        total_squares = {var : 0 for var in vs}
        counts = {var : 0 for var in vs}
        for var in vs:
            input_ds = xr.open_dataset(input_file+'_'+year)
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
    t_start = '-05-01'
    t_end = '-11-30T18'
    for year in twd_years:
        input_ds = xr.open_dataset(input_file+'_'+year)
        for var in vs:
            input_ds[var] = ((input_ds[var]-var_means[var])/var_stds[var]).fillna(0)
            input_ds[var].attrs['mean_pre_norm'] = var_means[var]
            input_ds[var].attrs['stdev_pre_norm'] = var_stds[var]
            
        # also do wave weight calcs while we're here (NOTE: did not do this previously; current input_file_0.25.nc was created in Python interpreter
        # using similar lines to the ones below 6/26/23)
        weight_output_var = ''
        frac_da = input_ds['mask_frac']
        trust_da = input_ds['trust']
        # basic formula: just mask_frac, no trust, no areal smoothing
        weight_da = (input_ds['wave_labels']*frac_da**0.5).rename(weight_output_var)
        # overwrite labels
        if weight_output_var == '':
            input_ds['wave_labels'] = weight_da
        else:
            input_ds = xr.merge([input_ds[[v for v in input_ds.data_vars if v not in ('mask_frac', 'trust')]] , weight_da])        

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
