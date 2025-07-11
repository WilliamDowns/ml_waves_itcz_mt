'''
Download GridSat GOES data, optionally remove variables, optionally resample it
to ERA5 grid
'''

import numpy as np
import wget
import xarray as xr
from joblib import Parallel, delayed
import os
from ml_waves_itcz_mt.constants import *

# Cull cells outside of these bounds, if requested
'''
min_lon = -70
max_lon = -20
min_lat = 0
max_lat = 23
'''
'''
min_lon = -180
max_lon = 60
min_lat = -10
max_lat = 60
'''


cull = False

# whether to do gridsat-goes (highest res, ends in 2017, only over US) or gridsat-b1 (global)
goes = False

out_dir = '/your_dir/gridsat_data/raw_files'

def get_sat_name(time):
    # GOES satellite name changes depending on date (8, 12, 13 have been
    # GOES-East)
    if time < np.datetime64('2003-04-01T18:00'):
        return '08'
    elif time < np.datetime64('2010-04-14T19:00'):
        return '12'
    else:
        return '13'


# timesteps, split into (yyyy, mm, dd, hh)
# bounds inclusive
min_year = 1980
#min_year = 2012
max_year = 2023
#min_month= 5
#max_month = 11
min_month = 1
max_month = 12

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
        

#print(timesteps)
    
# create list of file URLS to download
file_list = []
#day_count_map = {'07': 31, '08': 31, '09':30}
if goes:
    base_url = 'https://www.ncei.noaa.gov/data/gridsat-goes/access/goes/'
else:
    base_url = 'https://www.ncei.noaa.gov/data/geostationary-ir-channel-brightness-temperature-gridsat-b1/access/'
    
for (y, m, d, h) in times:
    if goes:
        #TODO: Add this from below commented out for loop
        True
    else:
        f = base_url+y+'/GRIDSAT-B1.'+'.'.join([y,m,d,h,'v02r01.nc'])
        file_list.append(f)


'''
for year in [str(y) for y in range(1995,2018)]:
    for month in ('07', '08', '09'):
        for day in [str(d) if d>9 else '0'+str(d)
                    for d in range(1, day_count_map[month]+1)]:
            for hour in ('08', '09', '10', '20', '21', '22'):                
                sat = get_sat_name(
                    np.datetime64(year+'-'+month+'-'+day+'T'+hour+':00'))
                file_list.append(base_url + year + '/' + month + '/' + \
                                 'GridSat-GOES.goes'+sat+'.'+year+'.'+month+\
                                 '.'+day+'.'+hour+'00.v01.nc')
'''

#print(file_list)                
print(len(file_list))
missing_file_list = []

#for name in file_list[:20]:
def download_file(name):
    # download (if not yet downloaded) and open (if requested)
    if not goes:
        if os.path.exists(out_dir+'/'+name.split(base_url)[1][5:]):
            #print('skipping', name)
            return
    else:
        #TODO: implement different structure for name to check and skip
        return    
    #not all files exist
    try:
        print(name, out_dir+'/'+name.split(base_url)[1])
        wget.download(name, out=out_dir)
        print('downloaded', name)
    except:
        #missing_file_list.append(name)
        print('missing', name)
        return
    if cull:
        ds = xr.open_dataset(name[name.find('GridSat'):])
        #print(name[name.find('GridSat'):])
        # eliminate unneeded variables and bounds
        ds = ds.drop([var for var in ds.data_vars if var != 'ch4'])
        ds = ds.where(ds.lon>min_lon,drop=True).where(ds.lon<max_lon,drop=True).\
            where(ds.lat>min_lat,drop=True).where(ds.lat<max_lat,drop=True)
        # resample
        #ds = ds.coarsen(lon=
        # save    
        ds.to_netcdf(name[name.find('GridSat'):][20:])
        # remove old file
        os.remove(name[name.find('GridSat'):])

Parallel(n_jobs = -1)(delayed(download_file)(name) for name in file_list)
#print(missing_file_list)
