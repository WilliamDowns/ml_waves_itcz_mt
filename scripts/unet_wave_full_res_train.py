'''
Weighted UNet for waves through 140W, with some buffer on the west side
This is the full res version, and does not include data prep (see the _prep version)
'''


from ml_waves_itcz_mt.nn import *
import pickle
from torch import optim
import torch


base_dir = '/your_dir/'
out_dir = base_dir + 'ml_waves/'

# NetCDF file to read in properly formatted input from, if it has been made already
input_file = out_dir+'inputs/input_file_0.25.nc'

input_ds = xr.open_dataset(input_file)
input_ds = input_ds.drop(['mask_frac', 'trust'])


# try skipping some variables
input_ds = input_ds[[var for var in input_ds.data_vars if 'sv' not in var and 'cv' not in var and 'pv' not in var 
                     and 'time' not in var and (var[0] != 'w' or var == 'wave_labels')]]

# try decreasing amount of input levels to match IMT
input_ds = input_ds[[var for var in input_ds.data_vars if not '800' in var and not '700' in var and not '600' in var and not '500' in var]]


# drop gridsat data for aidan
input_ds = input_ds[[var for var in input_ds.data_vars if 'irwin' not in var]]

# remove early years early on to make 6 hour vars take less time
input_ds = input_ds.sel(time=slice('2014-05-01', '2022-11-30T18'))

# Try out adding some means from past 24 hours
add_24hr_vars = True
if add_24hr_vars:
    #print('adding 6hr-before vars')
    #print('adding 12hr-before vars')
    print('adding 24hr-mean vars')
    new_times = input_ds['time']+np.timedelta64(6, 'h')
    #new_times = input_ds['time']+np.timedelta64(12, 'h')
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

# filter out lats lons that aren't needed
input_ds = input_ds.sel(longitude=slice(-160,-0.25), latitude=slice(32,-7.75))

## Initialize UNet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)
channels_last=False
# Final one
unet = UNet_3Plus_no_first_skip(len(input_ds.data_vars)-1, 1, initial_target_channels=32, dropout_frac=0.25, conv=5, n_tiers=5, deep_sup=True, init_full_skip=False,
                         kernel_size=5, padding=2).to(device) #batch_size 13 (sometimes up to 17)


if torch.cuda.is_available():
    # channels last (which can speed things up) is only supported for CUDA
    channels_last=True

if channels_last:
    unet = unet.to(memory_format=torch.channels_last)
    
# choose training and validation datasets

train_starts = [np.datetime64(t) for t in ['2015-05-01','2018-05-01', '2022-05-01']]
train_ends = [np.datetime64(t) for t in ['2016-11-30T18', '2020-11-30T18', '2022-11-30T18']]
train_slices = [slice(s, e) for s, e in zip(train_starts, train_ends)]
train_times = [t.data for s in train_slices for t in input_ds.time.sel(time=s)]
train_ds = input_ds.sel(time=train_times)
print('train', train_ds)

val_starts = [np.datetime64(t) for t in ['2014-05-01','2017-05-01']]
val_ends = [np.datetime64(t) for t in ['2014-11-30T18', '2017-11-30T18']]
val_slices = [slice(s, e) for s, e in zip(val_starts, val_ends)]
val_times = [t.data for s in val_slices for t in input_ds.time.sel(time=s)]
val_ds = input_ds.sel(time=val_times)
print('val', val_ds)



batch_args={'sample_dims' : {'latitude' : 160, 'longitude' : 640}, 'label_var' : 'wave_labels', 'batch_dims' : {'latitude' : 160, 'longitude' : 640, 'time' : 1}
            , 'skip_zeros' : False, 'concat_input_dims' : False}

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

train_ds_times = [t for t in train_ds.time.values if t not in missing_times]
train_ds = train_ds.sel(time=train_ds_times)
val_ds_times = [t for t in val_ds.time.values if t not in missing_times]
val_ds = val_ds.sel(time=val_ds_times)


                
# try out using a mask over Africa (where NHC doesn't mark waves), as well as west of 140W
mask = (input_ds['wave_labels'].isel(time=0).astype('int32'))*0+1
land_mask = mask.where(((mask.longitude>=-18) | (mask.longitude<=-140)), other=0).values # part of west Africa and west of 140W
print(land_mask.max(), land_mask.min(), land_mask.sum(), land_mask.shape[0]*land_mask.shape[1])

loss_func = fss_multi_tensors(9,9, drop_mask=land_mask, norm_sample_weights=True)
eval_func = fss_multi_tensors(9,9, drop_mask=land_mask, eval_mode=True, norm_sample_weights=True)

checkpoint_dir_suffix=''

weight_decay = 1e-8
optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=30)
input_norm=False
input_norm_end=0
# train the model!
# max batch size 7 for a single GPU (unstable/inconsistent), or 6 consistently for UNet_3Plus,
# or (see model assignment further up for other values)
train_model(unet, train_ds, val_ds, device, batch_size=9, checkpoint_interval=1, weight_decay=weight_decay, #sample_weight_var='wave_weights',
            loss_func=loss_func, eval_func=eval_func,optimizer=optimizer, scheduler=scheduler, input_norm=input_norm, input_norm_end=input_norm_end,
            epochs=10000, eval_interval=-50, amp=True, checkpoint_dir=out_dir+'checkpoints'+checkpoint_dir_suffix, channels_last=channels_last, #print_csi_metrics=True,
            csi_threshold=0.15, csi_distance=12, log=True, **batch_args)#eval_interval=1

