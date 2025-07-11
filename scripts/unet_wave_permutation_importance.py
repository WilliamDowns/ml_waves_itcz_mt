'''
Calculate permutation importance scores across validation dataset
'''


from ml_waves_itcz_mt.nn import *
import pickle
import pandas as pd

base_dir = '/your_dir/'
out_dir = base_dir + 'ml_waves/'
checkpoint_dir = out_dir + 'archives/20240122_wave_revisedmaskfrac_skiptimes_noafrica_4trim_drop_pvsvcvw_sparselevelsimt_5tier_32_lateyears_24hrmean/'
checkpoint_prefix = 'checkpoint_epoch'
checkpoint_suffix = '.pth'
checkpoint_num=5.85


### Functions that I need here because I can't use util
def get_hurdat_annual_names(file_name, year):
    '''
    Retrieve list of storm names for a given year from hurdat

    filename : str, HURDAT file name
    year : str

    return : list of str
    '''
    
    storm_names = []
    
    with open(file_name, 'r') as f:
        contents = f.readlines()

    # find year section start, then start getting storm names
    start_line = 0
    while contents[start_line][:8] != 'AL01' + str(year) and contents[start_line][:8] != 'EP01' + str(year):
        start_line = start_line + 1
    cur_line = start_line
    while cur_line < len(contents) and year in contents[cur_line][:8]:
        if contents[cur_line][:2] in ('AL', 'EP'):
            storm_names.append(contents[cur_line].split()[1][:-1])
        cur_line = cur_line + 1
    return storm_names

def get_hurdat_block(file_name, storm_identifier, year, use_id=False):
    '''
    Retrieve best track data for a storm name or id in a given year from
    a local HURDAT txt file.

    filename : str, HURDAT file name
    storm_identifier : str
    year : str
    use_id: whether storm_identifier should be a storm name or id number
    
    return : pandas dataframe
    '''
    storm_identifier = storm_identifier.upper() + (',')
    check_col = 0 if use_id else 1
        
    with open(file_name, 'r') as f:
        contents = f.readlines()

    # find year section start, then storm section start
    start_line = 0
    while contents[start_line][:8] != 'AL01' + str(year) and contents[start_line][:8] != 'EP01' + str(year):
        start_line = start_line + 1
    if start_line >= len(contents):
        return []

    while contents[start_line].split()[check_col] != storm_identifier:
        start_line = start_line + 1
        
    # start actual data
    start_line = start_line + 1
    
    # get full data block
    end_line = start_line
    while end_line < len(contents) and contents[end_line][:2] not in ('AL', 'EP'):
        end_line = end_line+1

    # separate into list of lists for easier manipulation
    block = [line.replace(' ', '').split(',') for line in contents[start_line:end_line]]

    # return as pandas DataFrame
    df = pd.DataFrame(block, columns=['date', 'time', 'spec', 'state', 'lat', 
                                    'lon', 'wind', 'pres', 'ne34', 'se34', 'sw34', 
                                    'nw34', 'ne50', 'se50', 'sw50', 'nw50', 'ne64', 
                                    'se64', 'sw64', 'nw64', 'rmw'])
    # how to keep only certain columns
    #usecols=['date', 'time', 'state', 'lat', 'lon', 'wind', 'pres']
    #df=df[usecols]

    return df

###


# whether to add a wave CSI scoring exclusion in the vicinity of TCs
exclude_tcs = True
#exclude_tcs = False
if exclude_tcs:
    hurdat_f = out_dir + 'hurdat2-1851-2022-040723.txt'
    epac_hurdat_f = out_dir + 'hurdat2-nepac-1949-2022-050423.txt'
    years = ['2014', '2017']
    # Get times and locations of all TCs
    hurdat = []
    for year in years:
        for storm in get_hurdat_annual_names(hurdat_f, year):
            hurdat.append(get_hurdat_block(hurdat_f, storm, year))
        for storm in get_hurdat_annual_names(epac_hurdat_f, year):
            try:
                hurdat.append(get_hurdat_block(epac_hurdat_f, storm, year))
            except IndexError:
                print(year, storm)



# whether to exclude scoring west of 120W and east of 20W for CSI
trim_edges = True
# NetCDF file to read in properly formatted input from
input_file = out_dir +'inputs/input_file_0.25.nc'

    
# input dataset, preformatted for model
input_ds=xr.open_dataset(input_file)
input_ds = input_ds.drop(['mask_frac', 'trust'])

# filter out lats lons that aren't needed
input_ds = input_ds.sel(longitude=slice(-160,-0.25), latitude=slice(32,-7.75))  

# try skipping some variables
input_ds = input_ds.drop_vars([var for var in input_ds.data_vars if 'theta' in var or 'Z' in var])
input_ds = input_ds[[var for var in input_ds.data_vars if 'time' not in var and (var[0] != 'w' or var == 'wave_labels')]]
input_ds = input_ds[[var for var in input_ds.data_vars if 'sv' not in var and 'cv' not in var and 'pv' not in var
                     and 'time' not in var and (var[0] != 'w' or var == 'wave_labels')]]
input_ds = input_ds[[var for var in input_ds.data_vars if not '800' in var and not '700' in var and not '600' in var and not '500' in var]]

# Try out adding some means from past 24 hours
add_24hr_vars = True
if add_24hr_vars:
    # avoid extra computations
    val_starts = [np.datetime64(t) for t in ['2013-11-20', '2014-05-01','2017-05-01']]
    val_ends = [np.datetime64(t) for t in ['2013-11-30T18', '2014-11-30T18', '2017-11-30T18']]
    val_slices = [slice(s, e) for s, e in zip(val_starts, val_ends)]
    val_times = [t.data for s in val_slices for t in input_ds.time.sel(time=s)]    
    input_ds = input_ds.sel(time=val_times)
    #print('adding 6hr-before vars')                                                    
    #print('adding 12hr-before vars')                                                   
    print('adding 24hr-mean vars')                                                      
    new_times = input_ds['time']+np.timedelta64(6, 'h')                                 
    new_das=[]                
    for var_name in ['u_850', 'u_750', 'u_650', 'v_850', 'v_750', 'v_650', 'q_850', 'q_750', 'q_650', 'tcw']:                                     
        new_var_name = var_name +'_6bef'                                                
        print('adding', new_var_name)                                                   
        # note: there is a small bug right now in rolling mean vars where times in the first day of May are taking mean of variables from previous november 
        new_da = input_ds[var_name].assign_coords({'time' : new_times}).rename(new_var_name).rolling(time=4).mean().fillna(0)#.dropna('time')     
        new_times = new_da.time                                                         
        keep_times = [t for t in input_ds[var_name]['time'].values if t in new_times.values]                                                      
        new_da = new_da.sel(time=keep_times)                                            
        new_das.append(new_da)
    for var_name, da in zip(['u_850', 'u_750', 'u_650', 'v_850', 'v_750', 'v_650', 'q_850', 'q_750', 'q_650', 'tcw'], new_das):                   
        new_var_name = var_name+'_6bef'                                                 
        input_ds[new_var_name] = da                                                     
    input_ds = input_ds.sel(time=keep_times)


print(list(input_ds.data_vars))

## Initialize saved UNet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
unet = UNet_3Plus_no_first_skip(len(input_ds.data_vars)-1, 1, initial_target_channels=32, dropout_frac=0.25, conv=5, n_tiers=5, kernel_size=5, deep_sup=True, init_full_skip=False,
                         padding=2).to(device) 
unet.to(device=device)
state_dict = torch.load(checkpoint_dir+checkpoint_prefix+str(checkpoint_num)+checkpoint_suffix, map_location=device)
unet.load_state_dict(state_dict)

# choose slice in datasets to assess
val_starts = [np.datetime64(t) for t in ['2014-05-01','2017-05-01']]
val_ends = [np.datetime64(t) for t in ['2014-11-30T18', '2017-11-30T18']]
val_slices = [slice(s, e) for s, e in zip(val_starts, val_ends)]
val_times = [t.data for s in val_slices for t in input_ds.time.sel(time=s)]
print(len(val_times))
val_ds = input_ds.sel(time=val_times)


## exclude times that had no TWD available
missing_times_atl_f = base_dir+'ml_waves/missing_times_atl'
missing_times_epac_f = base_dir+'ml_waves/missing_times_epac'
missing_times = []
for f in [missing_times_atl_f, missing_times_epac_f]:
    with open(f, 'rb') as _:
        missing_times.extend(pickle.load(_))
        # skip times that would be screwed up by leap years
        missing_times.extend([str(y)+s for y in range(2004, 2023) for s in ['02290000', '02290600', '02291200', '02291800', '03010000']])
        
# redefine these here due to issue of not being able to import from util on triton
from ml_waves_itcz_mt.constants import *
def era5_time_to_np64ns(time):
    '''
    Convert one of my era5 time strings to an actual np.datetime64[ns] object as used in the NetCDFs.
    Greatly speeds up sel selections

    time: str, 'YYYYMMDDThhmm'

    return: np.datetime64
    '''
    return np.datetime64(''.join(['-'.join([time[:4],time[4:6],time[6:8]]), time[8:11], ':', time[11:]]))

def twd_time_to_era5_time(twd_time,rewind=True):
    '''
    Convert time str from a TWD to the corresponding comparison format for ERA5 timesteps, given
    TWDs are 6 hours later. Assumes you're not changing years

    twd_time: str, 'YYYYMMDDhhmm'
    rewind: bool, whether to go 6 hours back
    
    return: str, the time for use in era5 comparisons
    '''
    if rewind:
        new_time = twd_time[:4] + global_times[global_times.index(twd_time[4:])-1]
    else:
        new_time = twd_time
    return 'T'.join([new_time[:8], new_time[8:]])

for t in missing_times:
    print(t)
missing_times = [era5_time_to_np64ns(twd_time_to_era5_time(t)) for t in missing_times if t[4:] !='01010000' and t[4:] not in ['02290000', '02290600', '02291200', '02291800', '03010000']]
missing_times.extend([era5_time_to_np64ns(twd_time_to_era5_time(str(y) + '01010000', rewind=False)) for y in range(2012, 2023)])
val_ds_times = [t for t in val_ds.time.values if t not in missing_times]
val_ds = val_ds.sel(time=val_ds_times)


batch_size=36

batch_args={'sample_dims' : {'latitude' : 160, 'longitude' : 640}, 'label_var' : 'wave_labels', 'batch_dims' : {'latitude' : 160, 'longitude' : 640, 'time' : 1}
            , 'skip_zeros' : False, 'concat_input_dims' : False}    

# try out using a mask over Africa (where NHC doesn't mark waves), as well as west of 140W
mask = (input_ds['wave_labels'].isel(time=0).astype('int32'))*0+1                                                        
land_mask = mask.where((mask.longitude>=-18), other=0).values # part of west Africa and west of 140W                     
print(land_mask.max(), land_mask.min(), land_mask.sum(), land_mask.shape[0]*land_mask.shape[1])
print(land_mask.shape)         


error_metric = class_get_match_success_from_tensors(threshold=0.16, distance=28, hurdat=hurdat, hurdat_bounds=[20, 60], trim_edge_bounds=[-23, -26, -120, -123], connect_array=[20,32],
                                                    do_wave_centers=True, longitudes=val_ds['longitude'], latitudes=val_ds['latitude'])

permutation_importance = get_permutation_importance(unet, val_ds, device, batch_size=batch_size, error_metric=error_metric, batch_based_averaging=False, return_times=True, **batch_args)

# print permutation importance sorted from largest to smallest error variables, for POD, SR, and CSI
# (assuming larger error metric score is worse performance, so greater values in return = greater importance)
pod_pairs = [(key, permutation_importance[key]['pod']) for key in permutation_importance.keys()]
sr_pairs = [(key, permutation_importance[key]['sr']) for key in permutation_importance.keys()]
csi_pairs = [(key, permutation_importance[key]['csi']) for key in permutation_importance.keys()]
print('')
print('')
print('#########################################################')
print('POD scores')
indices = np.argsort(np.array([val for (_, val) in pod_pairs]))[::-1]
for i in indices:
    print(pod_pairs[i][0], pod_pairs[i][1])
print('#########################################################')    
print('SR scores')
indices = np.argsort(np.array([val for (_, val) in sr_pairs]))[::-1]
for i in indices:
    print(sr_pairs[i][0], sr_pairs[i][1])
print('#########################################################')    
print('CSI scores')
indices = np.argsort(np.array([val for (_, val) in csi_pairs]))[::-1]
for i in indices:
    print(csi_pairs[i][0], csi_pairs[i][1])
print('#########################################################')
    
# print permutation importance averaged across variable categories
print('Categorized importance for POD')
cats = set([s.split('_')[0] for (s, _) in pod_pairs])
cats = {c : [] for c in cats}
for (s, perm) in pod_pairs:
    cats[s.split('_')[0]].append(perm)    
for c in cats.keys():
    cats[c] = np.mean(cats[c])
cats = [(key, cats[key]) for key in cats.keys()]
indices = np.argsort(np.array([val for (_, val) in cats]))[::-1]
for i in indices:
    print(cats[i][0], cats[i][1])
print('#########################################################')
print('Categorized importance for SR')
cats = set([s.split('_')[0] for (s, _) in sr_pairs])
cats = {c : [] for c in cats}
for (s, perm) in sr_pairs:
    cats[s.split('_')[0]].append(perm)    
for c in cats.keys():
    cats[c] = np.mean(cats[c])
cats = [(key, cats[key]) for key in cats.keys()]
indices = np.argsort(np.array([val for (_, val) in cats]))[::-1]
for i in indices:
    print(cats[i][0], cats[i][1])
print('#########################################################')
print('Categorized importance for CSI')
cats = set([s.split('_')[0] for (s, _) in csi_pairs])
cats = {c : [] for c in cats}
for (s, perm) in csi_pairs:
    cats[s.split('_')[0]].append(perm)    
for c in cats.keys():
    cats[c] = np.mean(cats[c])
cats = [(key, cats[key]) for key in cats.keys()]
indices = np.argsort(np.array([val for (_, val) in cats]))[::-1]
for i in indices:
    print(cats[i][0], cats[i][1])

print('#########################################################')
# also print permutation importance averaged across levels
print('Level importance for POD')
cats = set([s.split('_')[1] for (s, _) in pod_pairs if '_' in s])
cats = {c : [] for c in cats}
for (s, perm) in pod_pairs:
    if '_' in s:
        cats[s.split('_')[1]].append(perm)    
for c in cats.keys():
    cats[c] = np.mean(cats[c])
cats = [(key, cats[key]) for key in cats.keys()]
indices = np.argsort(np.array([val for (_, val) in cats]))[::-1]
for i in indices:
    print(cats[i][0], cats[i][1])

print('#########################################################')
    
print('Level importance for SR')
cats = set([s.split('_')[1] for (s, _) in sr_pairs if '_' in s])
cats = {c : [] for c in cats}
for (s, perm) in sr_pairs:
    if '_' in s:
        cats[s.split('_')[1]].append(perm)    
for c in cats.keys():
    cats[c] = np.mean(cats[c])
cats = [(key, cats[key]) for key in cats.keys()]
indices = np.argsort(np.array([val for (_, val) in cats]))[::-1]
for i in indices:
    print(cats[i][0], cats[i][1])

print('#########################################################')
    
print('Level importance for CSI')
cats = set([s.split('_')[1] for (s, _) in csi_pairs if '_' in s])
cats = {c : [] for c in cats}
for (s, perm) in csi_pairs:
    if '_' in s:
        cats[s.split('_')[1]].append(perm)    
for c in cats.keys():
    cats[c] = np.mean(cats[c])
cats = [(key, cats[key]) for key in cats.keys()]
indices = np.argsort(np.array([val for (_, val) in cats]))[::-1]
for i in indices:
    print(cats[i][0], cats[i][1])
