'''
Save predictions from a model for testing dataset
'''

from ml_waves_itcz_mt.nn import *
from ml_waves_itcz_mt.util import *
from ml_waves_itcz_mt.plotting import *

import pickle
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

base_dir = '/your_dir/'
out_dir = base_dir + 'ml_waves/unet_test/'
checkpoint_dir = out_dir + 'archives/20240122_wave_revisedmaskfrac_skiptimes_noafrica_4trim_drop_pvsvcvw_sparselevelsimt_5tier_32_lateyears_24hrmean/'#5.85


checkpoint_prefix = 'checkpoint_epoch'
checkpoint_suffix = '.pth'
checkpoint_num=5.85
print(checkpoint_dir)
image_path = out_dir + 'plots/' + str(checkpoint_num) #+ '_new'
make_directory(image_path)

var_names_6hr = ['u_850', 'u_750', 'u_650', 'v_850', 'v_750', 'v_650', 'q_850', 'q_750', 'q_650', 'tcw']

mask_frac=0.65
n_cell=19
wave_out_file = ''.join([out_dir, 'masked_waves_weighted_140W', str(mask_frac), '_', str(n_cell)])

# load in masked waves
with open(wave_out_file, 'rb') as f:
    waves = pickle.load(f)

# formatting changes for putting in paper
fig_mode = True
if fig_mode:
    # force error early if not in correct
    try:
        _, _ = plt.subplot_mosaic([['a', 'b']])
    except:
        print('wrong version of matplotlib for subplot mosaic')
        raise

# calculate regional scores
calc_regional_score = True
    
# only select wave centers for comparison 
do_wave_centers = True

label_var = 'wave_labels'

do_africa = False
if do_africa:
    input_file = out_dir+'input_file_0.25_2021_africa.nc'
else:
    input_file = out_dir+'input_file_0.25.nc'    
made_input = os.path.exists(input_file)


# input dataset, preformatted for model
input_ds=xr.open_dataset(input_file)
input_ds = input_ds.drop(['mask_frac', 'trust'])
input_ds = input_ds.drop_vars([var for var in input_ds.data_vars if 'theta' in var or 'Z' in var])
input_ds = input_ds[[var for var in input_ds.data_vars if 'sv' not in var and 'cv' not in var and 'pv' not in var
                     and 'time' not in var and (var[0] != 'w' or var == 'wave_labels')]]
input_ds = input_ds[[var for var in input_ds.data_vars if not '800' in var and not '700' in var and not '600' in var and not '500' in var]]

print(list(input_ds.data_vars))

## Initialize saved UNet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
unet = UNet_3Plus_no_first_skip(len(input_ds.data_vars)+len(var_names_6hr)-1, 1, initial_target_channels=32, dropout_frac=0.25, conv=5, n_tiers=5, deep_sup=True, init_full_skip=False,
                         kernel_size=5,padding=2).to(device)

unet.to(device=device)
state_dict = torch.load(checkpoint_dir+checkpoint_prefix+str(checkpoint_num)+checkpoint_suffix, map_location=device)
unet.load_state_dict(state_dict)



# choose slice in datasets to plot
test_years = ['2021']
test_slices = [slice(y+'-06-01', y+'-10-14T18') for y in test_years]
test_times = [t.data for s in test_slices for t in input_ds.time.sel(time=s)]

input_ds = input_ds.sel(latitude=slice(32, -7.75), longitude=slice(-160, -0.25))
test_ds = input_ds.sel(time=test_times)[[label_var]]


# add separate 2023 labels
test_years = ['2023']
#test_slices = [slice(y+'-06-01', y+'-10-31T18') for y in test_years]
test_slices = [slice(y+'-06-01', y+'-10-14T18') for y in test_years]
labels_2023 = xr.open_dataset(out_dir+'labels_file_combined_wave_imt_0.25_2023.nc')['wave_labels']
test_times = [t.data for s in test_slices for t in labels_2023.time.sel(time=s)]
labels_2023 = labels_2023.sel(time=test_times, latitude=slice(32, -7.75), longitude=slice(-160, -0.25))
combined_labels = xr.concat([test_ds[label_var], labels_2023], dim='time')
test_ds = xr.Dataset({label_var : combined_labels})
print(test_ds[label_var])


if fig_mode:
    min_lon = -140
    max_lon = -9
    min_lat = -1
    max_lat = 30
else:
    min_lon = -128
    max_lon = -9
    min_lat = -1
    max_lat = 30
    
# whether to exclude scoring west of 120W and east of 20W
trim_edges = True

# try out using a mask over Africa (where NHC doesn't mark waves), as well as west of 140W
mask_africa_140 = True
if mask_africa_140:
    mask = (input_ds['wave_labels'].isel(time=0).astype('int32'))*0+1               
    land_mask = mask.where((mask.longitude>=-18), other=0) #northwest Africa
    land_mask = land_mask.sel(latitude=slice(max_lat, min_lat), longitude=slice(min_lon, max_lon))


# whether to add a scoring exclusion in the vicinity of TCs
exclude_tcs = True
#exclude_tcs = False
if exclude_tcs:
    hurdat_f = base_dir + 'hurdat2-atl-1851-2023-042624.txt'
    epac_hurdat_f = base_dir + 'hurdat2-nepac-1949-2023-042624.txt'
    years = ['2021', '2023']
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

    
out_thresholds=np.arange(0.14,0.26, 0.01)


wave_colormap = {2 : 'g', 1 : 'k', False : 'r', True : 'k'}
proj = ccrs.PlateCarree()

# cmap for wave axes

cmap = plt.get_cmap('YlOrRd')

plot_daily_images = False


# make model predictions and save them before calculating the success metrics
# saves tons of time / repeated steps
pre_calc_predictions = False
pre_load_predictions = True
preds = []
if pre_calc_predictions:
    for i in range(len(test_ds.time)):    
        sample = test_ds.isel(time=i).load()
        time = [test_ds['time'].isel(time=i).values]
        print(time)
        prediction = predict_xr(unet, sample, device, label_var=label_var, input_norm=False).expand_dims({'time':time}).rename('pred')
        preds.append(prediction)
    preds = xr.concat(preds, 'time')
    if do_africa:
        preds.to_netcdf(checkpoint_dir+'pred_for_csi_africa.nc')
    else:
        preds.to_netcdf(checkpoint_dir+'pred_for_csi.nc')


if pre_load_predictions:
    preds = xr.open_dataset(checkpoint_dir+'wave_pred_2021_2023.nc')


# Load precalculated wave object locations 
load_objects = False

test_years = ['2021', '2023']
test_slices = [slice(y+'-06-01', y+'-10-14T18') for y in test_years]
test_times = [t.data for s in test_slices for t in test_ds.time.sel(time=s)]
preds = preds.sel(time=test_times)

# plot mask and real waves at each time
print_list = []


# don't check times with no TWD
## exclude times that had no TWD available                   
missing_times_atl_f = base_dir+'ml_waves/missing_times_atl'  
missing_times_epac_f = base_dir+'ml_waves/missing_times_epac'
missing_times = []                                           
for f in [missing_times_atl_f, missing_times_epac_f]:        
    with open(f, 'rb') as _:                                 
        missing_times.extend(pickle.load(_))                 
        # skip times that would be screwed up by leap years  
        missing_times.extend([str(y)+s for y in range(2004, 2023) for s in ['02290000', '02290600', '02291200', '02291800', '03010000']]) 

missing_times = [era5_time_to_np64ns(twd_time_to_era5_time(t)) for t in missing_times if t[4:] !='01010000' and t[4:] not in ['02290000', '02290600', '02291200', '02291800', '03010000']]                 
missing_times.extend([era5_time_to_np64ns(twd_time_to_era5_time(str(y) + '01010000', rewind=False)) for y in range(2012, 2023)])    
test_ds_times = [t for t in test_ds.time.values if t not in missing_times]                                                        
test_ds = test_ds.sel(time=test_ds_times)
preds = preds.sel(time=test_ds_times)


for out_threshold in out_thresholds:
    pred_match_list = []
    targ_match_list = []

    # arrays for plotting later
    cum_prob = test_ds['wave_labels'].isel(time=0).sel(longitude=slice(min_lon, max_lon), latitude=slice(max_lat,min_lat)).load()*0 # cumulative probability forecasted
    cum_true = test_ds['wave_labels'].sum(dim='time').sel(longitude=slice(min_lon, max_lon), latitude=slice(max_lat,min_lat)).load() # cumulative true labels (note: may not be only 0 and 1 if weights are involved)
    #cum_pred = test_ds['wave_labels'].isel(time=0)*0 # cumulative labels forecasted according to out_threshold. Not useful if labeling whole polygons rather than lines
    cum_pred_objs = test_ds['wave_labels'].isel(time=0).sel(longitude=slice(min_lon, max_lon), latitude=slice(max_lat,min_lat)).load()*0 # cumulative labels forecasted according to out_threshold, as retrieved from object-based detection
    cum_pred_matches = test_ds['wave_labels'].isel(time=0).sel(longitude=slice(min_lon, max_lon), latitude=slice(max_lat,min_lat)).load()*0 # cumulative matched predicted labels, as retrieved from object-based detection
    cum_true_objs = test_ds['wave_labels'].isel(time=0).sel(longitude=slice(min_lon, max_lon), latitude=slice(max_lat,min_lat)).load()*0 # cumulative true labels, as retrieved from object-based detection
    cum_true_matches = test_ds['wave_labels'].isel(time=0).sel(longitude=slice(min_lon, max_lon), latitude=slice(max_lat,min_lat)).load()*0 # cumulative matched true labels, as retrieved from object-based detection
    total_waves = 0

    # create dict of wave locations and whether they matched for all times being tested
    wave_objects = {''.join(re.split('[:-]', str(test_ds['time'][i].values))).replace('T', '')[:12] : {'pred' : [], 'real' : []} for i in range(len(test_ds.time))}
    
    
    
    def plot_compare(i):    
        if pre_load_predictions:
            prediction = preds.isel(time=i)['pred']
        else:
            sample = test_ds.isel(time=i).load()
            prediction = predict_xr(unet, sample, device, label_var=label_var, input_norm=False)

        prediction = prediction.sel(longitude=slice(min_lon, max_lon), latitude=slice(max_lat,min_lat)).load()
        longitudes, latitudes = prediction['longitude'], prediction['latitude']
        pred_max = prediction.max()
        real = test_ds[label_var].isel(time=i).sel(longitude=slice(min_lon, max_lon), latitude=slice(max_lat,min_lat)).load()

        if mask_africa_140:
            real = xr.where(land_mask, 0, real)
            prediction = xr.where(land_mask, 0, prediction)

        # calculate whether labels predicted match real labels or not
        pred_labeled = label_feature_regions(prediction.data,threshold=out_threshold)
        # realized targets should always be any labeled region presumably
        targ_labeled = label_feature_regions(real.data,threshold=0.01)#
        
        time = ''.join(re.split('[:-]', str(test_ds['time'][i].values))).replace('T', '')[:12]
        if exclude_tcs:
            # mask out in box around storms
            time_storms = []
            for storm in hurdat:
                i = 0
                while i < len(storm['date']):
                    if ''.join([storm['date'][i], storm['time'][i]]) == time and storm['state'][i] in ('SS', 'SD', 'TS', 'TD', 'HU'):
                        time_storms.append((0-abs(float(storm['lon'][i][:-1])), float(storm['lat'][i][:-1])))
                        i = len(storm['date'])
                    i = i+1    
            ## 10x20 degree (40x80 cell) box around storm center
            # 10x30 degree (40x120 cell) box around storm center
            for (storm_lon, storm_lat) in time_storms:
                # don't care about storms outside of bounds
                if storm_lon > max_lon or storm_lon < min_lon or storm_lat > max_lat or storm_lat < min_lat:
                    continue
                #print(storm_lon, storm_lat, longitudes)
                center_storm_lon_i = np.argmin(np.abs(longitudes.values-storm_lon))                                
                center_storm_lat_i = np.argmin(np.abs(latitudes.values-storm_lat))
                min_lon_i = np.max([center_storm_lon_i-20, 0])
                max_lon_i = np.min([center_storm_lon_i+21, len(longitudes)])
                min_lat_i = np.max([center_storm_lat_i-60, 0])
                max_lat_i = np.min([center_storm_lat_i+61, len(latitudes)])
                # just mask out entire meridional column of grid since waves have varying extents
                pred_labeled[min_lat_i:max_lat_i, min_lon_i:max_lon_i] = 0
                targ_labeled[min_lat_i:max_lat_i, min_lon_i:max_lon_i] = 0

        if trim_edges:
            # mask out label regions at edge of domain from pred if there is no real label nearby,
            # or vice versa            
            lon_i_23W = list(longitudes).index(-23)
            lon_i_26W = list(longitudes).index(-26)
            if not np.any(pred_labeled[:, lon_i_26W:]):
                targ_labeled[:, lon_i_23W:] = 0
            if not np.any(targ_labeled[:, lon_i_26W:]):
                pred_labeled[:, lon_i_23W:] = 0
            
            lon_i_120W = list(longitudes).index(-120)
            lon_i_123W = list(longitudes).index(-123)
            if not np.any(pred_labeled[:, :lon_i_120W]):
                targ_labeled[:, :lon_i_123W] = 0
            if not np.any(targ_labeled[:, :lon_i_120W]):
                pred_labeled[:, :lon_i_123W] = 0
        
        # get list of feature region arrays
        pred_labels = extract_full_feature_regions(pred_labeled, connect_array=[20,32])
        targ_labels = extract_full_feature_regions(targ_labeled, connect_array=[20,32])

        if do_wave_centers:
            # want to still have lines since maxima can be misaligned in latitude, so expand each array to a line axis 10 degrees tall and 0.25 degrees wide
            # doing this here rather than in label_feature_regions to avoid issues with connect_array and axes that are not close to one another
            # this does make all axes straight N/S, but that's not a huge deal
            
            for j, region in enumerate(pred_labels):
                # try out finding centroid of region
                vals = prediction.data[region[1], region[0]]
                lon_i = int(np.average(region[0], weights=vals))
                lat_i = int(np.average(region[1], weights=vals))
                min_lat_i = np.max([lat_i-20, 0])
                max_lat_i = np.min([lat_i+21, len(latitudes)])

                new_lats = np.arange(min_lat_i, max_lat_i, 1)
                new_lons = np.full(len(new_lats), lon_i)
                pred_labels[j] = (new_lons, new_lats)
                
            for j, region in enumerate(targ_labels):
                # these all have same value so just want mean
                lat_i = round(np.mean(region[1]))
                lon_i = round(np.mean(region[0]))
                min_lat_i = np.max([lat_i-20, 0])
                max_lat_i = np.min([lat_i+21, len(latitudes)])

                new_lats = np.arange(min_lat_i, max_lat_i, 1)
                new_lons = np.full(len(new_lats), lon_i)
                targ_labels[j] = (new_lons, new_lats)
                
        
        # get matches between the two lists        
        pred_matches, targ_matches = match_feature_regions_basic_distance(pred_labels, targ_labels, threshold=28) #700km / 7 degrees

        if do_wave_centers:
            # want to still have arrays for maps of CSI rather than just points, so expand each array to a line axis 3 degrees wide
            for j, [(lons, lats), match] in enumerate(pred_matches):
                lon_i = lons[0]                
                min_lon_i = np.max([lon_i-6, 0])
                max_lon_i = np.min([lon_i+7, len(longitudes)])
                height = len(lats)
                new_lons = np.full(height, min_lon_i)
                new_lats = lats.copy()
                for new_lon_i in np.arange(min_lon_i+1, max_lon_i, 1):
                    new_lons = np.append(new_lons, np.full(height, new_lon_i))
                    new_lats = np.append(new_lats, lats)
                
                pred_matches[j] = [(new_lons, new_lats), match]

            for j, [(lons, lats), match] in enumerate(targ_matches):
                lon_i = lons[0]                
                min_lon_i = np.max([lon_i-6, 0])
                max_lon_i = np.min([lon_i+7, len(longitudes)])
                height = len(lats)
                new_lons = np.full(height, min_lon_i)
                new_lats = lats.copy()
                for new_lon_i in np.arange(min_lon_i+1, max_lon_i, 1):
                    new_lons = np.append(new_lons, np.full(height, new_lon_i))
                    new_lats = np.append(new_lats, lats)
                
                targ_matches[j] = [(new_lons, new_lats), match]

        total_waves=len(pred_labels)
        if plot_daily_images:
            print(time)

            # may be an off-by-one error here
            twd_axes = [(wave, masked, frac) for (wave, masked, frac) in waves[time]]

            # plot
            fig, axes = plt.subplots(2, 1, figsize=(18, 14), subplot_kw={"projection": proj})
            plot_era_da(prediction, var='mask', title=' '.join([time[:-2], 'Wave mask (65% of coords in positive anomaly of curv vort or specific humidity)']),  
                        extent=[min_lon, max_lon, min_lat, max_lat], fig=fig, ax=axes[0], comap='binary')

            for (twd_wave, masked, frac) in twd_axes:
                shape = twd_wave.extent
                axes[0].plot([0-shape[0][1], 0-shape[1][1]], [shape[0][0], shape[1][0]], color=cmap(frac),
                             linewidth=4, transform=proj, label = 'TWD Wave ' + str(twd_wave))

            plot_era_da(prediction*0, var='mask', title=' '.join([time[:-2], 'Wave matches (blue=match, red=fail)']),  
                        extent=[min_lon, max_lon, min_lat, max_lat], fig=fig, ax=axes[1], comap='binary')

            match_cmap = {True: 'b', False : 'r'}

            for match in pred_matches+targ_matches:
                (xs, ys), matched = match
                for x, y in zip(xs[::4], ys[::4]):
                    axes[1].scatter(longitudes[x], latitudes[y], color=match_cmap[matched], linewidth=4, transform=proj)
            if exclude_tcs:
                for storm in time_storms:
                    axes[0].scatter(storm[0], storm[1], c='k', s=180, marker='*', transform=proj)#, zorder=2)
                    axes[1].scatter(storm[0], storm[1], c='k', s=180, marker='*', transform=proj)#, zorder=2) 
                
        
        if plot_daily_images:
            image_name = ''.join([image_path, '/predict_comparisons_', str(checkpoint_num), '_', time[:-2], '.png'])
            plt.savefig(image_name)
            plt.clf()
            plt.close()

        # add waves to totals                       
        return pred_matches, targ_matches, prediction, total_waves

    n_jobs = -1 if plot_daily_images else 1
    if not load_objects:
        returns = Parallel(n_jobs=n_jobs)(delayed(plot_compare)(i) for i in range(len(test_ds.time)))
    else:
        with open(checkpoint_dir+'wave_centers_out_threshold_'+str(round(out_threshold, 3)), 'rb') as f:
            loads = pickle.load(f)
        returns = []
        def coords_to_index(matches):
            new_matches = []
            for [(lons, lats), match] in matches:
                new_matches.append([(((lons+np.abs(min_lon))*4).astype('int32'), ((-lats+np.abs(max_lat))*4).astype('int32')), match])
            return new_matches
        for t in preds['time'].values:
            prediction = preds['pred'].sel(time=t)
            time = ''.join(re.split('[:-]', str(t))).replace('T', '')[:12]            
            pred_matches = coords_to_index(loads[time]['pred'])
            real_matches = coords_to_index(loads[time]['real'])
            total_waves = len(pred_matches) + len(real_matches)
            returns.append([pred_matches, real_matches, prediction, total_waves])        
        
    pred_matches = []
    targ_matches = []
    predictions = []
    total_waves = 0
    longitudes = test_ds.sel(longitude=slice(min_lon, max_lon), latitude=slice(max_lat,min_lat))['longitude']
    latitudes = test_ds.sel(longitude=slice(min_lon, max_lon), latitude=slice(max_lat,min_lat))['latitude']    
    for i, [pm, tm, p, tot] in enumerate(returns):
        pred_matches.append(pm)
        targ_matches.append(tm)
        predictions.append(p)
        total_waves+=tot
        time = ''.join(re.split('[:-]', str(test_ds['time'][i].values))).replace('T', '')[:12]
        wave_objects[time]['pred'] = [[[longitudes[lons].values, latitudes[lats].values], match] for [[lons, lats], match] in pm]
        wave_objects[time]['real'] = [[[longitudes[lons].values, latitudes[lats].values], match] for [[lons, lats], match] in tm]
        
        
    pred_matches = [p for l in pred_matches for p in l]
    targ_matches = [t for l in targ_matches for t in l]

    for p in predictions:
        cum_prob = cum_prob + p
        
    for (xs, ys), matched in pred_matches:
        pred_match_list.append(matched)
        cum_pred_objs[ys, xs] = cum_pred_objs[ys,xs]+1
        if matched:
            cum_pred_matches[ys,xs] = cum_pred_matches[ys,xs]+1
    for (xs, ys), matched in targ_matches:
        targ_match_list.append(matched)        
        cum_true_objs[ys, xs] = cum_true_objs[ys,xs]+1
        if matched:
            cum_true_matches[ys,xs] = cum_true_matches[ys,xs]+1

    
    [pod_val, sr_val, csi_val] = pod_sr_csi(pred_match_list, targ_match_list)
    print(round(out_threshold, 2), *[round(v, 4) for v in [pod_val, sr_val, csi_val]])    
    print_list.append([round(out_threshold, 2), *[round(v, 4) for v in  [pod_val, sr_val, csi_val]]])
    # Create maps of POD and SR at each grid cell across testing set.

    if calc_regional_score:
        pred_matches = [[np.mean(lons), np.mean(lats), match] for time in wave_objects.keys() for [[lons, lats], match] in wave_objects[time]['pred']]
        targ_matches = [[np.mean(lons), np.mean(lats), match] for time in wave_objects.keys() for [[lons, lats], match] in wave_objects[time]['real']]

        # calc total score first
        pred_match_list = [m for [_, _, m] in pred_matches]
        targ_match_list = [m for [_, _, m] in targ_matches]
        [pod_val, sr_val, csi_val] = pod_sr_csi(pred_match_list, targ_match_list)
        print('total', round(out_threshold, 2), *[round(v, 4) for v in [pod_val, sr_val, csi_val]])    
        
        # calc open atlantic score
        pred_match_list = [m for [lon, _, m] in pred_matches if lon > -60] 
        targ_match_list = [m for [lon, _, m] in targ_matches if lon > -60]
        [pod_val, sr_val, csi_val] = pod_sr_csi(pred_match_list, targ_match_list)
        print('atlantic', round(out_threshold, 2), *[round(v, 4) for v in [pod_val, sr_val, csi_val]])    

        # carib score
        pred_match_list = [m for [lon, _, m] in pred_matches if lon <= -60 and lon > -90] 
        targ_match_list = [m for [lon, _, m] in targ_matches if lon <= -60 and lon > -90]
        [pod_val, sr_val, csi_val] = pod_sr_csi(pred_match_list, targ_match_list)
        print('carib', round(out_threshold, 2), *[round(v, 4) for v in [pod_val, sr_val, csi_val]])    

        # epac score
        pred_match_list = [m for [lon, _, m] in pred_matches if lon <= -90] 
        targ_match_list = [m for [lon, _, m] in targ_matches if lon <= -90]
        [pod_val, sr_val, csi_val] = pod_sr_csi(pred_match_list, targ_match_list)
        print('epac', round(out_threshold, 2), *[round(v, 4) for v in [pod_val, sr_val, csi_val]])    
        
        
    vmin = 0.2
    vmax = 1
    pod = cum_true_matches/cum_true_objs
    sr = cum_pred_matches/cum_pred_objs
    csi = 1/(1/pod + 1/sr - 1)
    
    # plots    
    cmap = 'plasma_r'
    cb_params={'fraction' : 0.01, 'pad' : 0.03}

    if fig_mode:
        plot_min_lon = -140
        plot_max_lon = -10
        plot_min_lat = 5
        plot_max_lat = 20
        
        layout = [['pod', 'cb'],
                  ['sr', 'cb'],
                  ['csi', 'cb']]
        fig, axes = plt.subplot_mosaic(layout, width_ratios=[1, 0.02], per_subplot_kw = {'cb' : {}, **{k : {'projection' : proj} for l in layout for k in l if k!= 'cb'}}, figsize=(15,6))
        plot_era_da(pod, var='mask', vmin=vmin, vmax=vmax, title='', comap=cmap, gridline_labels=[False, False, False, True],
                         extent=[plot_min_lon,plot_max_lon,plot_min_lat,plot_max_lat], fig=fig, ax=axes['pod'], add_cb=False)
        plot_era_da(sr, var='mask', vmin=vmin, vmax=vmax, title='', comap=cmap, gridline_labels=[False, False, False, True],
                    extent=[plot_min_lon,plot_max_lon,plot_min_lat,plot_max_lat], fig=fig, ax=axes['sr'], add_cb=False)
        p = plot_era_da(csi, var='mask', vmin=vmin, vmax=vmax, title='', comap=cmap,
                        extent=[plot_min_lon,plot_max_lon,plot_min_lat,plot_max_lat], fig=fig, ax=axes['csi'], add_cb=False, return_plot=True)
        cb = fig.colorbar(p, cax=axes['cb'], orientation='vertical',)#, fraction=0.013, pad=0.04)
        cb.set_ticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        cb.set_label('POD / SR / CSI', rotation=90, labelpad=20, fontsize=20)
        cb.ax.tick_params(labelsize=16)
        fig.tight_layout()
        image_name = ''.join([image_path, '/fig_mode_pod_sr_csi_'+str(round(out_threshold,3))+'.png'])
        
    else:
        plot_min_lon = -120
        plot_max_lon = -10
        plot_min_lat = 5
        plot_max_lat = 20
        
        fig, axes = plt.subplots(3, 1,figsize=(17, 12), subplot_kw={"projection": proj})        
        cb = plot_era_da(pod, var='mask', vmin=vmin, vmax=vmax, title='Probability of Detection for waves in 2021 & 2023' + ', total score: ' + str(round(pod_val, 2)),
                         extent=[plot_min_lon,plot_max_lon,plot_min_lat,plot_max_lat], fig=fig, ax=axes[0], comap=cmap, cb_label='Probability of Detection', cb_params=cb_params, return_cb=True)
        cb.set_ticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])    
        cb = plot_era_da(sr, var='mask', vmin=vmin, vmax=vmax, title='Success Rate for waves in 2021 & 2023' + ', total score: ' + str(round(sr_val, 2)),
                    extent=[plot_min_lon,plot_max_lon,plot_min_lat,plot_max_lat], fig=fig, ax=axes[1], comap=cmap, cb_label='Success Rate', cb_params=cb_params, return_cb=True)
        cb.set_ticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])    
        cb = plot_era_da(csi, var='mask', vmin=vmin, vmax=vmax, title='Critical Success Index for waves in 2021 & 2023' + ', total score: ' + str(round(csi_val, 2)),
                    extent=[plot_min_lon,plot_max_lon,plot_min_lat,plot_max_lat], fig=fig, ax=axes[2], comap=cmap, cb_label='Critical Success Index', cb_params=cb_params, return_cb=True)
        cb.set_ticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])        
        image_name = ''.join([image_path, '/pod_sr_csi_'+str(round(out_threshold,3))+'.png'])
    plt.savefig(image_name)
    plt.clf()
    plt.close()
    
    # Also create map of cumulative true label counts, cumulative predicted probability,# and total number of predicted waves using sum of points above out_threshold
    fig, axes = plt.subplots(2, 1,figsize=(11, 10), subplot_kw={"projection": proj})
    vmax=max(float(cum_true.max()), float(cum_prob.max()))#, float(cum_pred.max()))
    plot_era_da(cum_true, var='mask', vmin=0, vmax=vmax, title='Total true wave probs for probability threshold' + str(round(out_threshold, 3)),
                extent=[-160,0,-10,32], fig=fig, ax=axes[0], comap='binary')
    plot_era_da(cum_prob, var='mask', vmin=0, vmax=vmax, title='Total pred wave probs  for probability threshold' + str(round(out_threshold, 3)),
                extent=[-160,0,-10,32], fig=fig, ax=axes[1], comap='binary')

    image_name = ''.join([image_path, '/cumulative_labels_thresh_'+str(round(out_threshold,3))+'.png'])
    plt.savefig(image_name)
    plt.clf()
    plt.close()

    with open(checkpoint_dir+'wave_centers_out_threshold_'+str(round(out_threshold, 3)), 'wb') as f:
        # save match lists and scores for later use too in plotting / any confidence interval calculations
        wave_objects['pod'] = pod_val
        wave_objects['sr'] = sr_val
        wave_objects['csi'] = csi_val
        wave_objects['pred_match_list'] = pred_match_list
        wave_objects['targ_match_list'] = targ_match_list
        pickle.dump(wave_objects, f, protocol=pickle.HIGHEST_PROTOCOL)

if len(out_thresholds) > 1:
    with open(checkpoint_dir+'csi_scores_2021_max', 'wb') as f:
        pickle.dump(print_list, f, protocol=pickle.HIGHEST_PROTOCOL)
    
