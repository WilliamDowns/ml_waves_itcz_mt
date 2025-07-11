'''
Optionally: Remove variables from gridsat files, resample to requested grid, and combine to a single file
'''


import xarray as xr
from joblib import Parallel, delayed
from ml_waves_itcz_mt.regrid import *
from ml_waves_itcz_mt.constants import *
from ml_waves_itcz_mt.util import make_directory

# options
remove_vars = True
resample = True
combine_full = False
combine_annual = True
if not any([remove_vars, resample, combine_full, combine_annual]):
    # nothing is going to be done
    exit()

# resampling bounds for regridder
# these are currently setup as bounds required to undergo 3 downsampling layers to a 1 degree grid (0.125 -> 0.25 -> 0.5 -> 1.0 degrees)
#min_lon = -160.875
#max_lon = 0.875
#min_lat = -10.875
#max_lat = 32.875
#min_lon = -160-3*.125
#max_lon = 0+3*.125
#min_lat = -10-3*.125
#max_lat = 32+3*.125
## bounds for 0.125 high res extra unet input experiment
'''
min_lon = -160-(.25+.125+.125/2)
max_lon = 0+(.25+.125+.125/2)
min_lat = -10-(.25+.125+.125/2)
max_lat = 32+(.25+.125+.125/2)
res = 0.125
'''
'''
## bounds for general 0.25 degree
min_lon = -160
max_lon = 0
min_lat = -10
max_lat = 32
'''

'''
## bounds for global tropics 1 degree
min_lon = -180
max_lon = 179
min_lat = -40
max_lat = 40
#res = 0.25
res=1
'''
'''
## bounds for west hem through Africa 0.25 degree
min_lon = -160
max_lon = 80
min_lat = -18.75
max_lat = 28
res = 0.25
'''
'''
## bounds for global tropics 1 degree
min_lon = -180
max_lon = 179
min_lat = -60
max_lat = 60
#res = 0.25
res=1
'''

## bounds for combined wave + imt predictions, west hem through Africa 0.25 degree
min_lon = -180
max_lon = 60
min_lat = -17.75
max_lat = 32
res = 0.25



## get list of input files
input_dir = '/your/dir/gridsat_data/raw_files/'
prefix = 'GRIDSAT-B1.'
#output_dir = '/your/dir/gridsat_data/converted_files/' #0.125
#output_dir = '/your/dir/gridsat_data/converted_files_1x1/'
#output_dir = '/your/dir/gridsat_data/converted_files_0.25_whem_africa/'
#output_dir = '/your/dir/gridsat_data/converted_files_1x1_global_may_to_november/'
output_dir = '/your/dir/gridsat_data/converted_files_0.25_combined_wave_imt_whem_africa/'
make_directory(output_dir)
#combine_full_file = output_dir + 'gridsat_all_years.nc'
combine_full_file = output_dir + 'gridsat_2012_2022_full_year_global_tropics.nc'
#combine_annual_file_prefix = output_dir + 'gridsat_'+str(res)+'_'
#combine_annual_file_prefix = output_dir + 'gridsat_full_year_global_tropics_'+str(res)+'_'
combine_annual_file_prefix = output_dir + 'gridsat_full_year_combined_wave_imt_whem_africa_'+str(res)+'_'
#combine_annual_file_prefix = output_dir + 'gridsat_full_year_global_'+str(res)+'_'
combine_annual_file_suffix = '.nc'

# bounds inclusive 
#min_year = 2004
#min_year = 2012
#min_year = 1980
#max_year = 2022
min_year = 2023
max_year = 2023

#min_month= 5
#max_month = 11

min_month=1
max_month=12
min_day = 1
max_day = 31
#min_hour = 0
#max_hour = 23
hours = (0, 6, 12, 18)
timesteps = [t for t in global_times if
             int(t[:2]) in range(min_month, max_month+1) and
             int(t[2:4]) in range(min_day, max_day+1) and
             int(t[4:6]) in hours]

#for t in global_times:
#    print(t[:4], t[4:6], t[6:8])

times = []
for year in range(min_year, max_year+1):
    for t in timesteps:
        times.append((str(year), t[:2], t[2:4], t[4:6]))

# one opened file for getting variable and grid info
template_ds = xr.open_dataset(input_dir+prefix+'.'.join(times[0])+'.v02r01.nc')
keep_vars = list(template_ds.data_vars)

if remove_vars:    
    #keep_vars = ['ch4'] #goes
    keep_vars = ['irwin_cdr']

if resample:
    regrid_lon = np.arange(min_lon, max_lon+res, res)
    regrid_lat = np.arange(min_lat, max_lat+res, res)
    regrid_ds = xr.DataArray(data=np.zeros([len(regrid_lat), len(regrid_lon)]), dims=['lat', 'lon'], coords={'lat' : regrid_lat, 'lon' : regrid_lon}, name='_')
    regridder = make_regridder(template_ds, regrid_ds, [min_lon, max_lon, min_lat, max_lat])
    
# probably best to save intermediate files

# cannot be done in parallel because regridder cannot be pickled and there is a memory leak from calling regridder multiple times (related to underlying fortran memory allocation)
#def drop_and_resample(f):
for (y,m,d,h) in times:
    #print(f)
    print(y,m,d,h)
    # note neither dropping nor resampling needs to necessarily happen here
    in_f  = input_dir + prefix+'.'.join([y,m,d,h,'v02r01.nc'])
    out_f = output_dir + prefix + '.'.join([y,m,d,h,'v02r01.nc'])
    if not remove_vars and not resample:
        #return in_f
        continue
    ds = xr.open_dataset(in_f)
    if remove_vars:
        ds = ds[keep_vars]
    if resample:

        ds = regridder(ds)
    
    ds.to_netcdf(out_f)
    ds.close()
    
#Parallel(n_jobs=-1)(delayed(drop_and_resample)(f) for f in times)
    
    
if combine_full:
    #TODO: make this use a variable prefix and suffix for its input files in case want to combine the original files    
    ds = xr.open_mfdataset([output_dir+prefix+'.'.join([y,m,d,h,'v02r01.nc']) for (y,m,d,h) in times])
    ds.to_netcdf(combine_full_file)
    
if combine_annual:
    years = [str(y) for y in range(min_year, max_year+1)]
    for year in years:
        #TODO: make this use a variable prefix and suffix for its input files in case want to combine the original files
        ds = xr.open_mfdataset([output_dir + prefix + '.'.join([y,m,d,h,'v02r01.nc']) for (y,m,d,h) in times if y == year])
        ds = ds.rename({'lat' : 'latitude', 'lon' : 'longitude'})
        ds.to_netcdf(combine_annual_file_prefix+year+combine_annual_file_suffix)
        print('saved', year)
