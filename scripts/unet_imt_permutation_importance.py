'''
Plot whether matches are found, plus overall POD, SR, CSI, for the test dataset
'''


from ml_waves_itcz_mt.nn import *

import pickle

base_dir = '/your_dir/'
out_dir = base_dir + 'ml_waves/'
checkpoint_dir = out_dir+'archives/20230812_imt_5tier_32init_smallepochs_2e-4lr/'
checkpoint_prefix = 'checkpoint_epoch'
checkpoint_suffix = '.pth'
checkpoint_num=16.83
image_path = out_dir + 'plots/' + str(checkpoint_num)

# NetCDF file to read in properly formatted input from
input_file = out_dir+'inputs/input_file_imt_westhem_africa_0.25.nc_with_z'
made_input = os.path.exists(input_file)

imt_norm_f = out_dir + 'unet_imt_norms'

# input dataset, preformatted for model
input_ds=xr.open_dataset(input_file)

input_ds = input_ds.drop(['lon_grid', 'lat_grid', 'time_grid', 'day_grid'])
input_ds = input_ds.drop_vars([var for var in input_ds.data_vars if 'theta' in var or 'Z' in var])

imt_sample_height=176
imt_sample_width=640

imt_var_order = ['irwin_cdr', 'tcw', 
                  'u_1000', 'u_925', 'u_850', 'u_750', 'u_650', 'u_550', 
                  'v_1000', 'v_925', 'v_850', 'v_750', 'v_650', 'v_550', 
                  'q_1000', 'q_925', 'q_850', 'q_750', 'q_650', 'q_550']


input_ds = input_ds[imt_var_order+['imt_labels']]

# for some reason, new file has latitude sorted in wrong order
input_ds = input_ds.isel(latitude=slice(None, None, -1))

input_ds = input_ds.sel(longitude=slice(-160, -0.25), latitude=slice(28, -15.75))
labels_2014 = labels_2014.sel(longitude=slice(-160, -0.25), latitude=slice(28, -15.75))
input_ds['imt_labels'][list(input_ds.time).index(np.datetime64('2014-01-01')):list(input_ds.time).index(np.datetime64('2014-12-31T18'))+1] = labels_2014

# choose slice in datasets to assess
val_starts = [np.datetime64(t) for t in ['2014-01-01','2017-01-01']]
val_ends = [np.datetime64(t) for t in ['2014-12-31T18', '2017-12-31T18']]

val_slices = [slice(s, e) for s, e in zip(val_starts, val_ends)]
val_times = [t.data for s in val_slices for t in input_ds.time.sel(time=s)]
val_ds = input_ds.sel(time=val_times)

batch_size=36

with open(imt_norm_f, 'rb') as f:
    imt_norms = pickle.load(f)

imt_means = np.zeros([len(imt_var_order), imt_sample_height, imt_sample_width])
imt_stds = np.zeros([len(imt_var_order), imt_sample_height, imt_sample_width])
for i, var in enumerate(imt_var_order):
    imt_means[i], imt_stds[i] = imt_norms[var]['mean'], imt_norms[var]['stdev']



class norm_transform:
    def __init__(self, means, stds, wrong_means, wrong_stds):
        self.means = means#torch.as_tensor(means.astype('float32'))
        self.stds = stds #torch.as_tensor(stds.astype('float32'))
        self.wrong_means = wrong_means
        self.wrong_stds = wrong_stds
    def __call__(self, data):
        # expand along batch dimension (hopefully first dim) using broadcasting
        return (data-self.means)/self.stds
imt_transform = norm_transform(imt_means, imt_stds, wrong_means, wrong_stds)


## Initialize saved UNet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
unet = UNet_3Plus_no_first_skip(len(imt_var_order), 3, initial_target_channels=32, dropout_frac=0.25, conv=5, n_tiers=5, kernel_size=5, deep_sup=True, init_full_skip=False,
                         padding=2).to(device)
unet.to(device=device)
state_dict = torch.load(checkpoint_dir+checkpoint_prefix+str(checkpoint_num)+checkpoint_suffix, map_location=device)
unet.load_state_dict(state_dict)


## exclude times that had no TWD available                                     
missing_times_atl_f = base_dir+'ml_waves/missing_times_atl'                    
missing_times_epac_f = base_dir+'ml_waves/missing_times_epac'                  
missing_times = []                                                             
for f in [missing_times_atl_f, missing_times_epac_f]:                          
    with open(f, 'rb') as _:                                                   
        missing_times.extend(pickle.load(_))                                   
        # skip times that would be screwed up by leap years                    
        missing_times.extend([str(y)+s for y in range(2004, 2023) for s in ['02290000', '02290600', '02291200', '02291800', '03010000']])    
                 
# redefine these here due to issue of not being able to import from util 
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
                 
batch_args={'sample_dims' : {'latitude' : 176, 'longitude' : 640}, 'label_var' : 'imt_labels', 'batch_dims' : {'latitude' : 176, 'longitude' : 640, 'time' : 1}
            , 'skip_zeros' : False, 'concat_input_dims' : False, 'transform' : imt_transform} #batch_size: 128
                 
                 
# try out using a mask over south america and Africa (where NHC doesn't mark imt), as well as west of 140W             
mask = (input_ds['imt_labels'].isel(time=0).astype('int32'))*0+1               
land_mask = mask.where(((mask.longitude>=-80) & (mask.longitude<=-60) & (mask.latitude<=11)) | ((mask.longitude>=-60) & (mask.longitude<=-50) & (mask.latitude<=8)) | #western and northeast south america
                       ((mask.longitude>=-50) & (mask.longitude<=-35) & (mask.latitude<=0)) | ((mask.longitude>=-17) & (mask.latitude>=10)) | # most of brazil and northwest Africa  
                       ((mask.longitude>=-12) & (mask.latitude<=10) & (mask.latitude>=5)) | (mask.longitude<=-140), other=0).values # part of west Africa and west of 140W           
print(land_mask.max(), land_mask.min(), land_mask.sum(), land_mask.shape[0]*land_mask.shape[1])  

### Compute permutation importance ###
permutation_importances = get_permutation_importance_pixelwise_csi(unet, val_ds, device, thresholds=[0.32, 0.36, 0.40],
                                                                  match_dist=10, land_mask=land_mask, batch_size=batch_size, **batch_args)

for k, kind in enumerate(['itcz', 'mt', 'either']):
    permutation_importance = permutation_importances[k]
    print('#########################################################')
    print('#########################################################')
    print('###############BEGINNING {}##############################'.format(kind))
    print('#########################################################')
    print('#########################################################')
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
    print('#########################################################')
    print('#########################################################')
    print('###############ENDING {}##############################'.format(kind))
    print('#########################################################')
    print('#########################################################')
