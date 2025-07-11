'''
Calculate and plot POD, SR, CSI for the ITCZ/MT 
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
checkpoint_dir = out_dir + 'archives/20230812_imt_5tier_32init_smallepochs_2e-4lr/' #16.83
checkpoint_num = 16.83
sample_height=176                                                                                                                                                                                       
sample_width=640

checkpoint_prefix = 'checkpoint_epoch'
checkpoint_suffix = '.pth'
image_path = out_dir + 'plots/' + str(checkpoint_num) #+ '_new'
make_directory(image_path)

label_var = 'imt_labels'

# NetCDF file to read in properly formatted labels from
input_file = out_dir + 'labels_file_combined_wave_imt_0.25_2021_2023.nc'+'_float32'
input_ds = xr.open_dataset(input_file)[[label_var]]

## Initialize saved UNet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
unet = UNet_3Plus_no_first_skip(20, 3, initial_target_channels=32, dropout_frac=0.25, conv=5, n_tiers=5, deep_sup=True, init_full_skip=False,
                         kernel_size=5,padding=2).to(device)

unet.to(device=device)
state_dict = torch.load(checkpoint_dir+checkpoint_prefix+str(checkpoint_num)+checkpoint_suffix, map_location=device)
unet.load_state_dict(state_dict)

# choose slice in datasets to plot
test_start = '2021-01-01'
test_end = '2023-12-31T18'

input_ds = input_ds.sel(longitude=slice(-160, -0.25), latitude=slice(28, -15.75))

test_ds = input_ds.sel(time=slice(test_start, test_end))
orig_test_times = test_ds.time.values

min_lon = -143
max_lon = -9
min_lat = -15
max_lat = 28

# whether to exclude scoring west of 120W and east of 20W
trim_edges = False

# try out using a mask over Africa (where NHC doesn't mark imts), as well as west of 140W
mask_africa_sa_140 = True
if mask_africa_sa_140:
    # try out using a mask over south america and Africa (where NHC doesn't mark imt), as well as west of 140W
    mask = (input_ds['imt_labels'].isel(time=0).sel(latitude=slice(max_lat, min_lat), longitude=slice(min_lon, max_lon)).astype('int32'))*0+1
    land_mask = mask.where(((mask.longitude>=-80) & (mask.longitude<=-60) & (mask.latitude<=11)) | ((mask.longitude>=-60) & (mask.longitude<=-50) & (mask.latitude<=5.5)) | #western and northeast south america 
                           ((mask.longitude>=-50) & (mask.longitude<=-35) & (mask.latitude<=-2.5)) | ((mask.longitude>=-17) & (mask.latitude>=10)) | # most of brazil and northwest Africa
                           ((mask.longitude>=-12) & (mask.latitude<=10) & (mask.latitude>=5)) | (mask.longitude<=-140), other=0).values # part of west Africa and west of 140W                   

out_thresholds=[0.32, 0.36, 0.40]

# how many grid cells away a match should be considered
match_dist=10

imt_colormap = {2 : 'g', 1 : 'k', False : 'r', True : 'k'}
proj = ccrs.PlateCarree()

plot_daily_images = False

# made model predictions and saved them before calculating the success metrics
# saves tons of time / repeated steps
pre_load_predictions = True
if pre_load_predictions:
    preds = xr.open_dataset(checkpoint_dir+'imt_pred_2021_2023.nc')
    test_times = test_ds.time.values
    preds = preds.sel(time=test_times)
    
# Load precalculated imt object locations 
load_objects = False

# plot mask and real imts at each time

print_list = []

for out_threshold in out_thresholds:
    # arrays for plotting later
    cum_prob = preds['pred'].isel(time=0).sel(longitude=slice(min_lon, max_lon), latitude=slice(max_lat,min_lat)).load().fillna(0)*0 # cumulative probability forecasted
    cum_true = preds['pred'].sum(dim='time').sel(longitude=slice(min_lon, max_lon), latitude=slice(max_lat,min_lat)).load().fillna(0) # cumulative true labels (note: may not be only 0 and 1 if weights are involved)
    #cum_pred = test_ds['pred'].isel(time=0)*0 # cumulative labels forecasted according to out_threshold. Not useful if labeling whole polygons rather than lines
    cum_pred_objs = preds['pred'].isel(time=0).sel(longitude=slice(min_lon, max_lon), latitude=slice(max_lat,min_lat)).load().fillna(0)*0 # cumulative labels forecasted according to out_threshold, as retrieved from object-based detection
    cum_pred_matches = preds['pred'].isel(time=0).sel(longitude=slice(min_lon, max_lon), latitude=slice(max_lat,min_lat)).load().fillna(0)*0 # cumulative matched predicted labels, as retrieved from object-based detection
    cum_true_objs = preds['pred'].isel(time=0).sel(longitude=slice(min_lon, max_lon), latitude=slice(max_lat,min_lat)).load().fillna(0)*0 # cumulative true labels, as retrieved from object-based detection
    cum_true_matches = preds['pred'].isel(time=0).sel(longitude=slice(min_lon, max_lon), latitude=slice(max_lat,min_lat)).load().fillna(0)*0 # cumulative matched true labels, as retrieved from object-based detection
    cum_either_matches = preds['pred'].isel(time=0).sel(longitude=slice(min_lon, max_lon), latitude=slice(max_lat,min_lat)).load().fillna(0)*0 # cumulative matched true labels, as retrieved from object-based detection

    print(out_threshold)
            
    def plot_compare(i):
        # create dict of imt locations and whether they matched for all times being tested
        
        if pre_load_predictions:
            prediction = preds.isel(time=i)['pred']
        else:
            sample = test_ds.isel(time=i).load()
            prediction = predict_xr(unet, sample, device, label_var=label_var, input_norm=False)

        prediction = prediction.sel(longitude=slice(min_lon, max_lon), latitude=slice(max_lat,min_lat)).load()
        longitudes, latitudes = prediction['longitude'], prediction['latitude']
        pred_max = prediction.max()
        real = test_ds[label_var].isel(time=i).sel(longitude=slice(min_lon, max_lon), latitude=slice(max_lat,min_lat)).load()
        if mask_africa_sa_140:
            real = xr.where(land_mask, 0, real)
            prediction = xr.where(land_mask[:,:,None], 0, prediction)
        time = ''.join(re.split('[:-]', str(test_ds['time'][i].values))).replace('T', '')[:12]            
        matches = {c : {'pred' : [], 'real' : []} for c in list(prediction['class'].values) + ['either']}
        for k, c in enumerate(prediction['class'].values):
            # separately calculate matches for each class
            class_pred = prediction.sel({'class':c})
            class_targ = xr.where(real==k+1, real, 0)
            
            # calculate whether labels predicted match real labels or not
            pred_labeled = label_feature_regions(class_pred.data,threshold=out_threshold, trim_y=True, only_center_axis=True)
            targ_labeled = label_feature_regions(class_targ.data,threshold=0.01, uniform_prob_offset=6, trim_y=True, only_center_axis=True)#

            # do pixelwise CSI for these (which are now line segments)
            pred_smudged = ((class_pred*0).fillna(0)+pred_labeled).rolling(latitude=match_dist*2+1, longitude=match_dist*2+1, center=True, min_periods=1).sum()
            targ_smudged = (class_targ*0+targ_labeled).rolling(latitude=match_dist*2+1, longitude=match_dist*2+1, center=True, min_periods=1).sum()            

            pred_matches = xr.where((pred_labeled>0) & (targ_smudged>0), 1, 0)
            pred_non_matches = xr.where((pred_labeled>0) & (~(targ_smudged>0)), 1, 0)
            targ_matches = xr.where((targ_labeled>0) & (pred_smudged>0), 1, 0)
            targ_non_matches = xr.where((targ_labeled>0) & (~(pred_smudged>0)), 1, 0)
            
            # extract indices as objects
            pred_matches = extract_full_feature_regions(label_feature_regions(pred_matches.values, threshold=0.0001))
            pred_non_matches = extract_full_feature_regions(label_feature_regions(pred_non_matches.values, threshold=0.0001))
            pred_matches = [[vals, True] for vals in pred_matches] + [[vals, False] for vals in pred_non_matches]
            
            targ_matches = extract_full_feature_regions(label_feature_regions(targ_matches.values, threshold=0.0001))
            targ_non_matches = extract_full_feature_regions(label_feature_regions(targ_non_matches.values, threshold=0.0001))
            targ_matches = [[vals, True] for vals in targ_matches] + [[vals, False] for vals in targ_non_matches]

            matches[c]['pred'] = pred_matches
            matches[c]['real'] = targ_matches
            
        # also calculate success of comparing matches between any classes

        non_class_pred=prediction.fillna(0).sum('class')
        non_class_targ=real
        pred_labeled=label_feature_regions(non_class_pred.data, threshold=out_threshold, trim_y=True, only_center_axis=True)
        targ_labeled=label_feature_regions(non_class_targ.data, threshold=out_threshold, trim_y=True, only_center_axis=True, uniform_prob_offset=6)        

        # do pixelwise CSI for these (which are now line segments)
        pred_smudged = ((non_class_pred*0)+pred_labeled).rolling(latitude=match_dist*2+1, longitude=match_dist*2+1, center=True, min_periods=1).sum()
        targ_smudged = (non_class_targ*0+targ_labeled).rolling(latitude=match_dist*2+1, longitude=match_dist*2+1, center=True, min_periods=1).sum()            

        pred_matches = xr.where((pred_labeled>0) & (targ_smudged>0), 1, 0)
        pred_non_matches = xr.where((pred_labeled>0) & (~(targ_smudged>0)), 1, 0)
        targ_matches = xr.where((targ_labeled>0) & (pred_smudged>0), 1, 0)
        targ_non_matches = xr.where((targ_labeled>0) & (~(pred_smudged>0)), 1, 0)

        # extract indices as objects
        pred_matches = extract_full_feature_regions(label_feature_regions(pred_matches.values, threshold=0.0001))
        pred_non_matches = extract_full_feature_regions(label_feature_regions(pred_non_matches.values, threshold=0.0001))
        pred_matches = [[vals, True] for vals in pred_matches] + [[vals, False] for vals in pred_non_matches]
        
        targ_matches = extract_full_feature_regions(label_feature_regions(targ_matches.values, threshold=0.0001))
        targ_non_matches = extract_full_feature_regions(label_feature_regions(targ_non_matches.values, threshold=0.0001))
        targ_matches = [[vals, True] for vals in targ_matches] + [[vals, False] for vals in targ_non_matches]

        matches['either']['pred'] = pred_matches
        matches['either']['real'] = targ_matches

        if plot_daily_images:            
            # plot prediction contours on top and success/failure lines for each class
            # plot real contours on bottom and success/failure lines for either class comparison
            
            fig, axes = plt.subplots(2, 1, figsize=(18, 15), subplot_kw={"projection": proj})
            class_colormaps={0:'Oranges', 1:'pink_r'}
            success_colormaps={False:'r', True:'b'}
            
            for c in [0, 1]:
                plot_era_da(prediction.isel({'class':c}), var='mask', title=' '.join([time[:-2], 'IMT']), vmin=out_threshold, vmax=0.95, 
                            extent=[min_lon, max_lon, min_lat, max_lat], fig=fig, ax=axes[0], comap=class_colormaps[c])
            for c in [0, 1]:
                plot_era_da(xr.where(real==c+1, real, np.nan), var='mask', title=' '.join([time[:-2], 'IMT']), vmin=out_threshold, vmax=0.95, 
                            extent=[min_lon, max_lon, min_lat, max_lat], fig=fig, ax=axes[1], comap=class_colormaps[c])
            
            for [(lons, lats), matched] in matches['itcz']['pred']+matches['itcz']['real']+matches['mt']['pred']+matches['mt']['real']:
                plot_lons = prediction['longitude'][lons].values
                plot_lats = prediction['latitude'][lats].values
                axes[0].plot(plot_lons, plot_lats, c=success_colormaps[matched], linewidth=4, transform=proj, zorder=10)
            for [(lons, lats), matched] in matches['either']['pred']+matches['either']['real']:
                plot_lons = prediction['longitude'][lons]
                plot_lats = prediction['latitude'][lats]                                
                axes[1].plot(plot_lons, plot_lats, c=success_colormaps[matched], linewidth=4, transform=proj, zorder=10)        
        
        if plot_daily_images:
            image_name = ''.join([image_path, '/predict_comparisons_', str(checkpoint_num), '_', time[:-2], '.png'])
            plt.savefig(image_name)
            plt.clf()
            plt.close()

        # add imts to totals                       
        return matches, prediction


    n_jobs = -1 #if plot_daily_images else 1
    #n_jobs = 1
    if not load_objects:
        returns = Parallel(n_jobs=n_jobs)(delayed(plot_compare)(i) for i in range(len(test_ds.time)))
    else:
        with open(checkpoint_dir+'imt_centers_out_threshold_'+str(round(out_threshold, 3)), 'rb') as f:
            imt_objects = pickle.load(f)
        returns = []
        for t in preds['time'].values:
            prediction = preds['pred'].sel(time=t)
            time = ''.join(re.split('[:-]', str(t))).replace('T', '')[:12]                        
            matches = imt_objects[time]
            returns.append([matches, prediction])    

    imt_objects = {''.join(re.split('[:-]', str(test_ds['time'][i].values))).replace('T', '')[:12] : {c : {'pred' : [], 'real' : []} for c in preds['class'].values} for i in range(len(test_ds.time))}
    
    itcz_matches = {'pred' : [], 'real' : []}
    mt_matches = {'pred' : [], 'real' : []}
    either_matches = {'pred' : [], 'real' : []}
    predictions = []
    total_imts = 0
    for i, [matches, prediction] in enumerate(returns):
        itcz_matches['pred'].append(matches['itcz']['pred'])
        itcz_matches['real'].append(matches['itcz']['real'])
        mt_matches['pred'].append(matches['mt']['pred'])
        mt_matches['real'].append(matches['mt']['real'])        
        either_matches['pred'].append(matches['either']['pred'])
        either_matches['real'].append(matches['either']['real'])        
        predictions.append(prediction)
        time = ''.join(re.split('[:-]', str(test_ds['time'][i].values))).replace('T', '')[:12]
        imt_objects[time]['itcz'] = matches['itcz']
        imt_objects[time]['mt'] = matches['mt']
        imt_objects[time]['either'] = matches['either']
            
    # add up totals from each class
    for p in predictions:
        cum_prob = cum_prob + p.fillna(0)

    for class_i, [class_name, class_dict] in enumerate(zip(['itcz', 'mt', 'either'], [itcz_matches, mt_matches, either_matches])):
        pred_match_list = []
        targ_match_list = []
        pred_matches = [p for l in class_dict['pred'] for p in l]
        targ_matches = [t for l in class_dict['real'] for t in l]

        if class_name == 'either':
            # lazy code here
            class_i = 0
            cum_pred_objs = cum_pred_objs*0
            cum_pred_matches = cum_pred_matches*0
            cum_true_objs = cum_true_objs*0
            cum_true_matches = cum_true_matches*0
        
        for (xs, ys), matched in pred_matches:
            pred_match_list.extend([matched]*len(xs))
            # have to create fake numpy arrays for adding because xarray doesn't support paired indexing
            new_pred_objs = np.zeros(np.shape(cum_pred_objs)[:-1])
            new_pred_objs[ys, xs] = 1
            cum_pred_objs[:, :, class_i] = cum_pred_objs[:, :, class_i]+new_pred_objs
            if matched:
                new_pred_matches = np.zeros(np.shape(cum_pred_matches)[:-1])
                new_pred_matches[ys, xs] = 1
                cum_pred_matches[:, :, class_i] = cum_pred_matches[:, :, class_i]+new_pred_matches
        for (xs, ys), matched in targ_matches:
            targ_match_list.extend([matched]*len(xs))
            new_true_objs = np.zeros(np.shape(cum_true_objs)[:-1])
            new_true_objs[ys, xs] = 1
            cum_true_objs[:, :, class_i] = cum_true_objs[:, :, class_i]+new_true_objs
            if matched:
                new_true_matches = np.zeros(np.shape(cum_true_matches)[:-1])
                new_true_matches[ys, xs] = 1
                cum_true_matches[:, :, class_i] = cum_true_matches[:, :, class_i]+new_true_matches


    
        #print('out threshold, pod, sr, csi: ', out_threshold, pod_sr_csi(pred_match_list, targ_match_list))
        [pod_val, sr_val, csi_val] = pod_sr_csi(pred_match_list, targ_match_list)
        print(class_name, round(out_threshold, 2), *[round(v, 4) for v in [pod_val, sr_val, csi_val]])
        print_list.append([class_name, round(out_threshold, 2), *[round(v, 4) for v in  [pod_val, sr_val, csi_val]]])
        # Create maps of POD and SR at each grid cell across testing set.
        fig, axes = plt.subplots(3, 1,figsize=(11, 10), subplot_kw={"projection": proj})
        vmin = 0.2#0
        vmax = 1

        # Note that if regions are too small matplotlib will not contour them        
        pod = cum_true_matches[:,:,class_i]/cum_true_objs[:,:,class_i]
        sr = cum_pred_matches[:,:,class_i]/cum_pred_objs[:,:,class_i]
        csi = 1/(1/pod + 1/sr - 1)
        # plots 
        plot_min_lon = -140
        plot_max_lon = -10
        plot_min_lat = -10
        plot_max_lat = 20

        cmap='plasma_r'
        
        cb = plot_era_da(pod, var='mask', vmin=vmin, vmax=vmax, title='Probability of Detection for ' +  class_name.upper() + ' in 2021 & 2023' + ', total score: ' + str(round(pod_val, 2)),
                    extent=[plot_min_lon,plot_max_lon,plot_min_lat,plot_max_lat], fig=fig, ax=axes[0], comap=cmap, cb_label='Probability of Detection', return_cb=True)
        cb.set_ticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        cb = plot_era_da(sr, var='mask', vmin=vmin, vmax=vmax, title='Success Rate for ' +  class_name.upper() + ' in 2021 & 2023' + ', total score: ' + str(round(sr_val, 2)),
                    extent=[plot_min_lon,plot_max_lon,plot_min_lat,plot_max_lat], fig=fig, ax=axes[1], comap=cmap, cb_label='Success Rate', return_cb=True)
        cb.set_ticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])        
        cb = plot_era_da(csi, var='mask', vmin=vmin, vmax=vmax, title='Critical Success Index for ' +  class_name.upper() + ' in 2021 & 2023' + ', total score: ' + str(round(csi_val, 2)),
                    extent=[plot_min_lon,plot_max_lon,plot_min_lat,plot_max_lat], fig=fig, ax=axes[2], comap=cmap, cb_label='Critical Success Index', return_cb=True)
        cb.set_ticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            
        image_name = ''.join([image_path, '/pod_sr_csi_'+str(round(out_threshold,3))+'_'+class_name+'.png'])
        plt.savefig(image_name)
        plt.clf()
        plt.close()


        # Also create map of cumulative true label counts, cumulative predicted probability,# and total number of predicted imts using sum of points above out_threshold
        fig, axes = plt.subplots(2, 1,figsize=(11, 10), subplot_kw={"projection": proj})
        vmax=max(float(cum_true.isel({'class' : class_i}).max()), float(cum_prob.isel({'class' : class_i}).max()))
        plot_era_da(cum_true.isel({'class' : class_i}), var='mask', vmin=0, vmax=vmax, title='Total true ' +  class_name.upper() + 'probs for probability threshold' + str(round(out_threshold, 3)),
                    extent=[-160,0,-10,32], fig=fig, ax=axes[0], comap='binary')
        plot_era_da(cum_prob.isel({'class' : class_i}), var='mask', vmin=0, vmax=vmax, title='Total pred ' +  class_name.upper() + 'probs for probability threshold' + str(round(out_threshold, 3)),
                    extent=[-160,0,-10,32], fig=fig, ax=axes[1], comap='binary')

        image_name = ''.join([image_path, '/cumulative_labels_thresh_'+str(round(out_threshold,3))+'_'+class_name+'.png'])
        plt.savefig(image_name)
        plt.clf()
        plt.close()

        # save metrics for future plots / confidence interval calcs
        imt_objects['pod_'+class_name]=pod_val
        imt_objects['sr_'+class_name]=sr_val
        imt_objects['csi_'+class_name]=csi_val
        imt_objects['pred_match_list_'+class_name]=pred_match_list
        imt_objects['targ_match_list_'+class_name]=targ_match_list        

        
    if not load_objects:
        with open(checkpoint_dir+'imt_centers_out_threshold_'+str(round(out_threshold, 3)), 'wb') as f:
            pickle.dump(imt_objects, f, protocol=pickle.HIGHEST_PROTOCOL)            

if len(out_thresholds) > 1:
    with open(checkpoint_dir+'csi_scores_2021_2023', 'wb') as f:
        pickle.dump(print_list, f, protocol=pickle.HIGHEST_PROTOCOL)
    

