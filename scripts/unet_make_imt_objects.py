'''
Generate IMT instantaneous location files
'''

from ml_waves_itcz_mt.nn import *
from ml_waves_itcz_mt.util import *

import pickle
from joblib import Parallel, delayed

### Various semi-global vars
base_dir = '/your_dir/'
out_dir = base_dir + 'ml_waves/unet_test/'

composite_era5_dir = base_dir + 'era5_data/unet_wave_imt_full_composite_climo_-180_60_-18_32_1x1/'
composite_era5_mid= '_full_composite_climo_'

preds_era5_dir = base_dir + 'era5_data/unet_wave_imt_full_pred_climo_-180_60_-17.75_32_0.25x0.25/'
preds_era5_mid = '_-180_60_-17.75_32_0.25x0.25_'

imt_min_lon = -160
imt_max_lon = -0.25
imt_min_lat = -17.75
imt_max_lat = 28
imt_checkpoint_dir = out_dir+'archives/20230812_imt_5tier_32init_smallepochs_2e-4lr/'#16.83
imt_checkpoint_num=16.83


## IMT net loading stuff
imt_input_file = out_dir+'input_file_imt_westhem_africa_0.25.nc_with_z'
imt_input_ds = xr.open_dataset(imt_input_file)
imt_pred_file_prefix=imt_checkpoint_dir+'imt_pred_'
imt_min_lon = -160
imt_max_lon = -0.25
imt_min_lat = -17.75
imt_max_lat = 28
imt_input_ds = imt_input_ds.sel(latitude=slice(imt_max_lat, imt_min_lat), longitude=slice(imt_min_lon, imt_max_lon))
imt_input_ds = imt_input_ds.drop(['lon_grid', 'lat_grid', 'time_grid', 'day_grid'])

imt_unet = UNet_3Plus_no_first_skip(len(imt_input_ds.data_vars)-1, 3, initial_target_channels=32, dropout_frac=0.25, conv=5, n_tiers=5, kernel_size=5, deep_sup=True, init_full_skip=False,
                             padding=2).to(device)
imt_unet.to(device=device)
state_dict = torch.load(imt_checkpoint_dir+checkpoint_prefix+str(imt_checkpoint_num)+checkpoint_suffix, map_location=device)
imt_unet.load_state_dict(state_dict)

proj = ccrs.PlateCarree()
    
def imt_create_objects():
    # Get imt objects going back to 1980 
    min_lon, max_lon = -155, -10
    min_lat, max_lat = -12, 24 
    years = [str(y) for y in range(1980,2023)]
    #years = [str(y) for y in range(1980,1981)]
    #start_date = '-07-01'
    #end_date = '-07-02T18'
    start_date = '-01-01'
    end_date = '-12-31T18'

    imt_objects = {}

    # thresholds for different class objects
    #class_threshes = {'itcz' : 0.34, 'mt' : 0.26}
    class_threshes = {'itcz' : 0.34, 'mt' : 0.34}

    # whether to save these as indices or as lats/lons
    save_indices=False

    # whether to trim predictions to only select one class per longitude based on whichever has the higher probability
    exclusive_lon=True
    
    def get_imt_by_year(y):
        imt_pred_file = imt_pred_file_prefix+y+'.nc'
        preds = xr.open_dataset(imt_pred_file).sel(time=slice(y+start_date, y+end_date), latitude=slice(max_lat, min_lat), longitude=slice(min_lon, max_lon))['pred']
        longitudes, latitudes = preds['longitude'], preds['latitude']

        year_imt_objects = {}
        
        for i in range(len(preds['time'])):
            pred = preds.isel(time=i)
            if exclusive_lon:
                # vectorized voodoo
                max_mask = pred.max(dim=['class', 'latitude'])                
                pred = xr.where(pred==max_mask.values[None, :, None], pred, 0)
            time = ''.join(re.split('[:-]', str(preds['time'][i].values))).replace('T', '')[:12]
            print(time)
            time_imt_objects = {'itcz' : [], 'mt' : []}
            for c in pred['class'].values:
                class_pred = pred.sel({'class' : c})
                thresh = class_threshes[c]                
                pred_labeled = label_feature_regions(class_pred.data,threshold=thresh, trim_y=True, only_center_axis=True)
                pred_regions = extract_full_feature_regions(pred_labeled)
                if not save_indices:
                    time_imt_objects[c] = [(longitudes.values[lons], latitudes.values[lats]) for (lons, lats) in pred_regions]
                else:
                    time_imt_objects[c] = pred_regions
            year_imt_objects[time] = time_imt_objects
        return year_imt_objects                                            

    returns = Parallel(n_jobs=-1)(delayed(get_imt_by_year)(y) for y in years)
    
    imt_objects = {}
    for year_imt_objects in returns:
        imt_objects.update(year_imt_objects)

    f_name = imt_checkpoint_dir+'imt_centers_'+years[0]+'_'+years[-1]+'_itcz_'+str(class_threshes['itcz'])+'_mt_'+str(class_threshes['mt'])
    if exclusive_lon:
        f_name = f_name + 'exclusive_lon'
    with open(f_name, 'wb') as f:
        pickle.dump(imt_objects, f, protocol=pickle.HIGHEST_PROTOCOL)


def imt_create_transition_lon_climo():
    # Get average longitudes of ITCZ to MT transition in Pacific and Atlantic,
    # and proportion of ITCZ vs. MT longitudes over each basin, and proportion of longitudes
    # uncovered by ITCZ / MT in each basin
    imt_objects_f = 'imt_centers_1980_2023_itcz_0.34_mt_0.34'
    exclusive_lon=True
    if exclusive_lon:
        imt_objects_f = imt_objects_f + 'exclusive_lon'
    with open(imt_checkpoint_dir+imt_objects_f, 'rb') as f:
        imt_objects = pickle.load(f)

    years = [str(y) for y in range(1980, 2024)]    
    start_date='-01-01'
    end_date = '-12-31T18'
    imt_objects = {k : v for k, v in imt_objects.items() if k[:4] in years and k[4:8] != '0229'}
    
    times = sorted(list(imt_objects.keys()))

    split_f = imt_checkpoint_dir + 'imt_transition_lons'
    
    natl_unique_lons = np.arange(-70, -4.75, 0.25) # this causes some incorrect impressions in the neither plots since we don't have labels over Brazil (50-60W)
    max_natl = len(natl_unique_lons)
    epac_unique_lons = np.arange(-160, -77.75, 0.25) # this causes inorrect impressions in proportions plots because it's less likely to pick up on stuff west of 140W
    max_epac = len(epac_unique_lons)
    
    if not os.path.exists(split_f):        
        times_dict = {t : {'epac_split' : np.nan, 'natl_split' : np.nan,
                           'epac_itcz_points' : np.nan, 'epac_mt_points' : np.nan, 'epac_neither_points' : np.nan,
                           'natl_itcz_points' : np.nan, 'natl_mt_points' : np.nan, 'natl_neither_points' : np.nan} for t in times}
        # Create time series of transition longitude between ITCZ and MT     
        for t in times:
            imt_coords = imt_objects[t]
            # find 18W and 30W points        
            # find closest longitude points to each of these
            for i in range(len(imt_coords['itcz'])):
                if len(imt_coords['itcz'][i][0]) < 8 and t[:4] == '1997':
                    print('Shortie at', t, [imt_coords['itcz'][j][0] for j in range(len(imt_coords['itcz']))])
            lons = np.concatenate([imt_coords['itcz'][i][0] for i in range(len(imt_coords['itcz']))] + [imt_coords['mt'][i][0] for i in range(len(imt_coords['mt']))] + [np.array([999])])
            if len(lons) == 1 and t[:4] == '1997':
                print(t, 'no imt found')

            # find transition point between MT and ITCZ in Atlantic and Pacificby searching east to west 
            itcz_lons = [('itcz', lon) for i in range(len(imt_coords['itcz'])) for lon in imt_coords['itcz'][i][0]]
            mt_lons = [('mt', lon) for i in range(len(imt_coords['mt'])) for lon in imt_coords['mt'][i][0]]
            sorted_itcz_mt = sorted(itcz_lons + mt_lons, key=lambda a:a[1])[::-1]
            
            natl_trans_lon = np.nan # otherwise get weird results when no labels exist full whole basin cause probs are too low
            for (kind, lon) in sorted_itcz_mt:                
                if lon < natl_unique_lons[-1] and not lon < natl_unique_lons[0]:
                    natl_trans_lon = lon                
                if lon < natl_unique_lons[0] or kind == 'itcz':
                    break
            if natl_trans_lon < -68:
                print(t, natl_trans_lon, sorted_itcz_mt)
            if np.isnan(natl_trans_lon):
                print(t, natl_trans_lon, sorted_itcz_mt)
            epac_trans_lon = np.nan # otherwise get weird results when no labels exist full whole basin cause probs are too low
            for (kind, lon) in sorted_itcz_mt:
                if lon >= epac_unique_lons[-1]:
                    continue
                if lon < epac_unique_lons[-1]:
                    epac_trans_lon = lon                
                if lon < epac_unique_lons[0] or kind == 'itcz':
                    break
                
            times_dict[t]['epac_split'] = epac_trans_lon
            times_dict[t]['natl_split'] = natl_trans_lon
        
            # count longitudes with itcz or mt in each basin
            itcz_epac = 0
            mt_epac = 0
            itcz_natl = 0
            mt_natl = 0

            for (kind, lon) in sorted_itcz_mt:
                if lon >= natl_unique_lons[0] and lon <= natl_unique_lons[-1]:
                    if kind == 'itcz':
                        itcz_natl+=1
                    else:
                        mt_natl +=1
                if lon >= epac_unique_lons[0] and lon <= epac_unique_lons[-1]:
                    if kind == 'itcz':
                        itcz_epac+=1
                    else:
                        mt_epac +=1
            times_dict[t]['epac_itcz_points'] = itcz_epac
            times_dict[t]['epac_mt_points'] = mt_epac
            times_dict[t]['epac_neither_points'] = max_epac-itcz_epac-mt_epac                        
            times_dict[t]['natl_itcz_points'] = itcz_natl
            times_dict[t]['natl_mt_points'] = mt_natl
            times_dict[t]['natl_neither_points'] = max_natl-itcz_natl-mt_natl                        
            
        with open(split_f, 'wb') as f:
            pickle.dump(times_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
