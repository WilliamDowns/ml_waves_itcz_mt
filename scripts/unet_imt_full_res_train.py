'''
UNet for ITCZ and monsoon trough for both ATL and EPAC
'''


from ml_waves_itcz_mt.nn import *
import pickle

from torch import optim
import torch

base_dir = '/your_dir/'
out_dir = base_dir + 'ml_waves/'

# NetCDF file to read in properly formatted input from, if it has been made already
input_file = out_dir+'inputs/input_file_imt_westhem_africa_0.25.nc'

input_ds = xr.open_dataset(input_file)

# filter out variables
input_ds = input_ds.drop_vars(['lon_grid', 'time_grid', 'day_grid', 'lat_grid'])

n_classes = 3

## Initialize UNet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)
channels_last=False

unet = UNet_3Plus_no_first_skip(len(input_ds.data_vars)-1, n_classes, initial_target_channels=32, dropout_frac=0.25, conv=5, n_tiers=5, deep_sup=True, init_full_skip=False, 
                         kernel_size=5, padding=2).to(device)#48 batch size


input_ds = input_ds.sel(longitude=slice(-160, -0.25), latitude=slice(28, -15.75))


if torch.cuda.is_available():
    # channels last (which can speed things up) is only supported for CUDA
    channels_last=True

if channels_last:
    unet = unet.to(memory_format=torch.channels_last)
    
# choose training and validation datasets

# Ranges I'm trying because pre-mid-2011 doesn't recognize the Monsoon Trough

train_starts = [np.datetime64(t) for t in ['2012-01-01','2015-01-01','2018-01-01', '2022-01-01']]
train_ends = [np.datetime64(t) for t in ['2013-12-31T18','2016-12-31T18', '2020-12-31T18', '2022-12-31T18']]
train_slices = [slice(s, e) for s, e in zip(train_starts, train_ends)]
train_times = [t.data for s in train_slices for t in input_ds.time.sel(time=s)]
train_ds = input_ds.sel(time=train_times)
print('train', train_ds)

val_starts = [np.datetime64(t) for t in ['2014-01-01','2017-01-01']]
val_ends = [np.datetime64(t) for t in ['2014-12-31T18', '2017-12-31T18']]
val_slices = [slice(s, e) for s, e in zip(val_starts, val_ends)]
val_times = [t.data for s in val_slices for t in input_ds.time.sel(time=s)]
val_ds = input_ds.sel(time=val_times)
print('val', val_ds)



batch_args={'sample_dims' : {'latitude' : 176, 'longitude' : 640}, 'label_var' : 'imt_labels', 'batch_dims' : {'latitude' : 176, 'longitude' : 640, 'time' : 1}
            , 'skip_zeros' : False, 'concat_input_dims' : False} #batch_size: 128


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


                
# try out using a mask over south america and Africa (where NHC doesn't mark imt), as well as west of 140W
mask = (input_ds['imt_labels'].isel(time=0).astype('int32'))*0+1
land_mask = mask.where(((mask.longitude>=-80) & (mask.longitude<=-60) & (mask.latitude<=11)) | ((mask.longitude>=-60) & (mask.longitude<=-50) & (mask.latitude<=8)) | #western and northeast south america
                       ((mask.longitude>=-50) & (mask.longitude<=-35) & (mask.latitude<=0)) | ((mask.longitude>=-17) & (mask.latitude>=10)) | # most of brazil and northwest Africa
                       ((mask.longitude>=-12) & (mask.latitude<=10) & (mask.latitude>=5)) | (mask.longitude<=-140), other=0).values # part of west Africa and west of 140W
print(land_mask.max(), land_mask.min(), land_mask.sum(), land_mask.shape[0]*land_mask.shape[1])

if n_classes == 3:
    loss_func = fss_multi_tensors_and_classes(9,9, drop_mask=land_mask)
    eval_func = fss_multi_tensors_and_classes(9,9, drop_mask=land_mask, n_classes=2, eval_mode=True) 
else:
    loss_func = fss_multi_tensors(9,9, drop_mask=land_mask)
    eval_func = fss_multi_tensors(9,9, drop_mask=land_mask, eval_mode=True) 
    # set all labels to 1
    train_ds['imt_labels'] = xr.where(train_ds['imt_labels'] > 0, 1, train_ds['imt_labels'])
    val_ds['imt_labels'] = xr.where(val_ds['imt_labels'] > 0, 1, val_ds['imt_labels'])

weight_decay = 1e-8
lr=2*1e-4
optimizer = torch.optim.Adam(unet.parameters(), lr=lr, weight_decay=weight_decay)
eval_func=loss_func
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, threshold=1e-5)#threshold=.00001)#patience=10
input_norm=False
input_norm_end=0
# train the model! #batch_size 10 for <=5 tiers, 8 for 6 tiers, 24 for 4 tiers/16init/3fss, ~10 for 9fss
train_model(unet, train_ds, val_ds, device, batch_size=6, checkpoint_interval=1, weight_decay=weight_decay, #sample_weight_var='imt_weights',
            loss_func=loss_func, eval_func=eval_func,optimizer=optimizer, scheduler=scheduler, input_norm=input_norm, input_norm_end=input_norm_end,
            epochs=10000, eval_interval=-100, amp=True, checkpoint_dir=out_dir+'checkpoints_imt', channels_last=channels_last,
            print_csi_metrics=False, csi_threshold=0.15, n_classes=n_classes, **batch_args)#eval_interval=1

