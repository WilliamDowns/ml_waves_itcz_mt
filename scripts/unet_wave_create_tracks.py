'''
Create wave tracks from network predictions
'''


from ml_waves_itcz_mt.nn import *
from ml_waves_itcz_mt.util import *
from skimage.feature import peak_local_max
from scipy.signal import savgol_filter

import pickle
from joblib import Parallel, delayed


### Various semi-global vars
base_dir = '/your_dir/'
out_dir = base_dir + 'ml_waves/unet_test/'

composite_era5_dir = base_dir + 'era5_data/unet_wave_imt_full_composite_climo_-180_60_-18_32_1x1/'
composite_era5_mid= '_full_composite_climo_'
wave_checkpoint_dir = out_dir + 'archives/20240122_wave_revisedmaskfrac_skiptimes_noafrica_4trim_drop_pvsvcvw_sparselevelsimt_5tier_32_lateyears_24hrmean/'#5.85, 0.16

checkpoint_prefix = 'checkpoint_epoch'
checkpoint_suffix = '.pth'
wave_checkpoint_num=5.85
wave_pred_file=wave_checkpoint_dir+'wave_pred_1980_2022.nc'
thresh=0.16

wave_obj_file =wave_checkpoint_dir+'wave_centers_out_threshold_'+str(thresh)
with open(wave_obj_file, 'rb') as f:
    wave_objs = pickle.load(f)
    
wave_input_file = out_dir+'input_file_0.25.nc'
    
# wave input dataset
wave_input_ds = xr.open_dataset(wave_input_file)
wave_input_ds = wave_input_ds.drop(['mask_frac', 'trust'])
wave_input_ds = wave_input_ds.sel(latitude=slice(32, -7.75), longitude=slice(-160, -0.25))
     
# connection distances in 0.25 degree intervals for waves near each other
lon_connect = 8
lat_connect = 32
    
## Initialize saved UNet
wave_input_ds_vars = [var for var in wave_input_ds.data_vars if 'sv' not in var and 'cv' not in var and 'pv' not in var
                      and 'time' not in var and 'theta' not in var and 'Z' not in var and
                      not '800' in var and not '700' in var and not '600' in var and not '500' in var and (var[0] != 'w' or var == 'wave_labels')]

add_6hr_vars = True
if add_6hr_vars:
    var_names_6hr = ['u_850', 'u_750', 'u_650', 'v_850', 'v_750', 'v_650', 'q_850', 'q_750', 'q_650', 'tcw']
else:
    var_names_6hr = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
wave_unet = UNet_3Plus_no_first_skip(len(wave_input_ds_vars)-1+len(var_names_6hr), 1, initial_target_channels=32, dropout_frac=0.25, conv=5, n_tiers=5, deep_sup=True, init_full_skip=False,
                              kernel_size=5,padding=2).to(device)
wave_unet.to(device=device)
state_dict = torch.load(wave_checkpoint_dir+checkpoint_prefix+str(wave_checkpoint_num)+checkpoint_suffix, map_location=device)
wave_unet.load_state_dict(state_dict)

# whether to do local maxima to find instantaneous wave centers instead of polygon connecting
local_max_method = False

# minimum timestep count for waves to be real
time_min = 8

connect_extrapolated_waves = True

exclude_sequential_westerlies = True
extra_merge_detection = True
original_merge_detection=False # Not mutually exclusive with above option


def create_instantaneous_wave_climo():
    # Get wave objects going back to 1980 and save them for making tracks
    thresh = 0.16
    years = [str(y) for y in range(1980, 2023)] #specific humidity file was unavailable for 2013 at the time of writing this function
    #years = ['2021']
    start_date = '-05-01'
    end_date = '-11-30T18'
    
    min_lat = 0
    max_lat = 32
    min_lon = -140
    max_lon = -2

    imt_objects = {}

    # whether to save these as indices or as lats/lons
    save_indices=False

    all_preds = xr.open_dataset(wave_pred_file).sel(latitude=slice(max_lat, min_lat), longitude=slice(min_lon, max_lon))

    def get_waves_by_year(year):
        preds = all_preds.sel(time=slice(year+start_date, year+end_date))
        times = preds.time
        year_wave_objects = {}
        longitudes, latitudes = preds['longitude'], preds['latitude']
        
        # flag TC locations
        # moved this code here to avoid slowdowns when not needed
        hurdat_f = base_dir + 'hurdat2-atl-1851-2023-042624.txt'
        epac_hurdat_f = base_dir + 'hurdat2-nepac-1949-2023-042624.txt'
        # Get times and locations of all TCs
        hurdat = []

        for storm in get_hurdat_annual_names(hurdat_f, year):
            hurdat.append((storm, get_hurdat_block(hurdat_f, storm, year)))
        for storm in get_hurdat_annual_names(epac_hurdat_f, year):
            try:
                hurdat.append((storm, get_hurdat_block(epac_hurdat_f, storm, year)))
            except IndexError:
                print(year, storm)

        for t_i, t in enumerate(times):#[:10]:
            # get wave centers from each dataset for every time, then find fields from their center and add those fields / a 1 count to bigger arrays
            pred = preds.sel(time=t)['pred']
            str_time = ''.join(re.split('[:-]', str(t.values))).replace('T', '')[:12]
            print(str_time)
            # look for waves in box around storms           
            time_storms = []
            for (storm_name, storm) in hurdat:
                if storm['date'][0][:4] != str_time[:4]:
                    continue
                i = 0
                while i < len(storm['date']):
                    if ''.join([storm['date'][i], storm['time'][i]]) == str_time and storm['state'][i] in ('SS', 'SD', 'TS', 'TD', 'HU'):
                        time_storms.append((0-abs(float(storm['lon'][i][:-1])), float(storm['lat'][i][:-1]), storm_name))
                        i = len(storm['date'])
                    i = i+1
            time_wave_objects = []                    
            if not local_max_method:
                # get feature regions
                region_array = label_feature_regions(pred.data, threshold=thresh)

                # get regions surrounding TCs
                tc_array = pred.data*0
                # 10x30 degree (40x120 cell) box around storm center
                storm_name_map = {}
                storm_name_ind = 1
                for (storm_lon, storm_lat, storm_name) in time_storms:
                    # don't care about storms outside of bounds                   
                    if storm_lon > max_lon or storm_lon < min_lon or storm_lat > max_lat or storm_lat < min_lat:
                        continue
                    center_storm_lon_i = np.argmin(np.abs(longitudes.values-storm_lon))
                    center_storm_lat_i = np.argmin(np.abs(latitudes.values-storm_lat))
                    min_lon_i = np.max([center_storm_lon_i-20, 0])
                    max_lon_i = np.min([center_storm_lon_i+21, len(longitudes)])
                    min_lat_i = np.max([center_storm_lat_i-60, 0])
                    max_lat_i = np.min([center_storm_lat_i+61, len(latitudes)])

                    # just mask out entire meridional column of grid since waves have varying extents                       
                    tc_array[min_lat_i:max_lat_i, min_lon_i:max_lon_i] = storm_name_ind
                    storm_name_map[storm_name_ind] = storm_name
                    storm_name_ind+=1

                region_lists = extract_full_feature_regions(region_array, connect_array=[lon_connect, lat_connect])

                for region in region_lists:
                    vals = pred.data[region[1], region[0]]
                    mean_lon_i = int(np.average(region[0], weights=vals))
                    mean_lat_i = int(np.average(region[1], weights=vals))

                    storm_ids = np.unique(tc_array[region[1], region[0]])
                    # list of nearby TCs if any are found, else empty list
                    tc = [storm_name_map[storm_i] for storm_i in storm_ids if storm_i > 0]
                    time_wave_objects.append([region, vals, mean_lon_i, mean_lat_i, tc])

                if not save_indices:
                    time_wave_objects = [[(longitudes.values[lons], latitudes.values[lats]), vals, longitudes.values[mean_lon_i], latitudes.values[mean_lat_i], tc]
                                         for [(lons, lats), vals, mean_lon_i, mean_lat_i, tc] in time_wave_objects]
            else:
                # get local maxima of wave probs
                maxes_indices = peak_local_max(pred.data, min_distance=20, threshold_abs=thresh)
                for pair in maxes_indices:
                    lat_i, lon_i = pair[0], pair[1]
                    vals = np.array([float(pred.data[lat_i, lon_i])])
                    tc = []
                    time_wave_objects.append([(np.array([longitudes.values[lon_i]]), np.array([latitudes.values[lat_i]])), vals, longitudes.values[lon_i], latitudes.values[lat_i], tc])
                
            year_wave_objects[str_time] = time_wave_objects
            
        return year_wave_objects
                
    returns = Parallel(n_jobs=-1)(delayed(get_waves_by_year)(y) for y in years)
    
    wave_objects = {}
    for year_wave_objects in returns:
        wave_objects.update(year_wave_objects)
    if not local_max_method:
        f_name = wave_checkpoint_dir+'wave_coords_instantaneous_'+years[0]+'_'+years[-1]+'_'+str(thresh)
    else:
        f_name = wave_checkpoint_dir+'wave_coords_instantaneous_local_maxima_'+years[0]+'_'+years[-1]+'_'+str(thresh)

    with open(f_name, 'wb') as f:
        pickle.dump(wave_objects, f, protocol=pickle.HIGHEST_PROTOCOL)        

        
def create_wave_tracks():
    # Generate actual connected tracks for waves

    years = [str(y) for y in range(1980, 2023)] #specific humidity file was unavailable for 2013 at the time of writing this function
    thresh = 0.16
    if local_max_method:
        input_f = wave_checkpoint_dir+'wave_coords_instantaneous_local_maxima_'+years[0]+'_'+years[-1]+'_'+str(thresh)
    else:
        input_f = wave_checkpoint_dir+'wave_coords_instantaneous_'+years[0]+'_'+years[-1]+'_'+str(thresh)
    with open(input_f, 'rb') as f:
        inst_waves = pickle.load(f)

        
    # for past 12 hours of waves [12, 6], as [westward distance bound, eastward distance bound] from current timestep wave coordinates
    # if waves were constant bodies, would only need to account for like 3 degrees / movement every 6 hours
    # but the shiftiness of the network polygons means things can shift westward (or slightly eastward) very quickly
    lon_search_windows = [[lon_connect/4-5, lon_connect/4+5], [lon_connect/4-3.5, lon_connect/4+1]] #cur_benchmark
        
    # minimum threshold of westward movement in degrees/6hr to consider something to be a wave
    min_6hr_distance = 0.4
    
    def update_track_wave(wave, time, inst_lons, inst_lats, inst_vals, inst_mean_lon, inst_mean_lat, inst_tc):
        # local function for updating track wave with new instant wave coordinates
        wave['time'].append(time)
        wave['coord_polygons'].append((inst_lons, inst_lats))
        wave['prob_polygons'].append(inst_vals)
        wave['raw_center_lon'].append(inst_mean_lon)
        wave['raw_center_lat'].append(inst_mean_lat)
        wave['TCs'].append(inst_tc)
        

    def get_waves_by_year(year):
        year_times = sorted([k for k in inst_waves.keys() if k[:4] == year])

        track_waves = []
        for t in year_times:
            print(t)
            for wave in inst_waves[t]:
                # have to convert to lists from numpy arrays because numpy array comparison is being supper annoying
                wave[0] = (list(wave[0][0]), list(wave[0][1]))
                wave[1] = list(wave[1])
            # sort waves at each time by longitude (west to east)
            inst_waves[t].sort(key=lambda wave: wave[2])
            
        time_indices = {t : i for i, t in enumerate(year_times)}        
        index_times  = {i : t for i, t in enumerate(year_times)}    
        
        past_18_waves = [[], []] #[12, 6]
        if extra_merge_detection:
            # keep waves from 18, 24 hours previous that didn't match
            dead_24_18_waves = [[], []]
        for t in year_times:
            now_inst_waves = inst_waves[t]
            now_track_waves = []
            # instantaneous wave structure: [(lons, lats), vals, mean_lon, mean_lat, tc],
            # where lons, lats are of the whole wave polygon, vals is the weights, mean_lon and mean_lat are centroids of polygon, and tc is list of nearby TCs
            # note these waves are already inherently a certain distance away, as determined by lat_connect, lon_connect in create_instantaneous_wave_climo

            # get list of matches for all existing waves first, then assign matched instant and tracked waves in order of distance, then make new track waves
            for i in range(len(past_18_waves) -1, -1, -1):
                compare_waves = past_18_waves[i]
                [west_interval, east_interval] = lon_search_windows[i]
                matched_inst_waves = [] # have to do this after to not screw up loop
                merge_parent_track_waves = []
                matched_track_waves = []
                matches = []
                for track_wave in compare_waves:
                    # list of matched pairs of waves
                    # realized should extrapolate centers to find closest match rather than matching to current time center
                    if len(track_wave['raw_center_lon']) < 2:
                        # guess a speed of 6 degrees / day (1.5 degrees / 6 hour timestep, or ~7.5m/s)
                        speed = -1.5
                    else:
                        # mean of wave's speed between periods
                        speed = np.mean([(track_wave['raw_center_lon'][j] - track_wave['raw_center_lon'][j-1])/(time_indices[track_wave['time'][j]]-time_indices[track_wave['time'][j-1]])
                                         for j in range(1,len(track_wave['raw_center_lon']))])                                
                    extrapolated_lon = track_wave['raw_center_lon'][-1]+speed*(len(past_18_waves)-i)
                    matches.extend([(track_wave, inst_wave, np.abs(inst_wave[2]-extrapolated_lon)) for inst_wave in now_inst_waves
                                    if track_wave['raw_center_lon'][-1] >= inst_wave[2]-west_interval and track_wave['raw_center_lon'][-1] <= inst_wave[2]+east_interval])
                matches.sort(key=lambda pair: pair[2]) # sort pairs by distance apart
                for (track_wave, inst_wave, _) in matches:
                    [(inst_lons, inst_lats), inst_vals, inst_mean_lon, inst_mean_lat, inst_tc] = inst_wave                        
                    if track_wave not in matched_track_waves and inst_wave not in matched_inst_waves:
                        # neither has matched yet, great                            
                        update_track_wave(track_wave, t, inst_lons, inst_lats, inst_vals, inst_mean_lon, inst_mean_lat, inst_tc)
                        matched_inst_waves.append(inst_wave)
                        matched_track_waves.append(track_wave)
                        merge_parent_track_waves.append(track_wave)
                        compare_waves.remove(track_wave)
                        now_track_waves.append(track_wave)
                    elif track_wave in matched_track_waves and inst_wave not in matched_inst_waves:
                        # instant wave might be a different wave even though there was a close existing track wave, let it live
                        continue                        
                    elif track_wave not in matched_track_waves and inst_wave in matched_inst_waves and original_merge_detection:
                        # track wave's closest match was already scooped up by another track wave, so this is likely a wave merger
                        update_track_wave(track_wave, t, inst_lons, inst_lats, inst_vals, inst_mean_lon, inst_mean_lat, inst_tc)
                        parent_wave = merge_parent_track_waves[matched_inst_waves.index(inst_wave)]
                        print('did a merge')
                        print('child ', track_wave['time'], track_wave['raw_center_lon'])
                        print('parent', parent_wave['time'], parent_wave['raw_center_lon'])
                        print()
                        track_wave['parent'] = [parent_wave, t]
                        parent_wave['children'].append([track_wave, t])
                        matched_track_waves.append(track_wave)
                        compare_waves.remove(track_wave)
                    else:
                        # both waves have matched already, carry on
                        continue
                now_inst_waves = [wave for wave in now_inst_waves if not (wave in matched_inst_waves)]

                if exclude_sequential_westerlies:
                    # stop wave if it's moved eastward for 2 consecutive timesteps
                    now_track_waves = [wave for wave in now_track_waves if len(wave['raw_center_lon']) < 3
                                       or not (wave['raw_center_lon'][-1]>wave['raw_center_lon'][-2] and wave['raw_center_lon'][-2] > wave['raw_center_lon'][-3])]
                        
                        
            # Instant waves with no matches from past 18 hours should become new waves
            # Adjust past_18_waves
            for [(inst_lons, inst_lats), inst_vals, inst_mean_lon, inst_mean_lat, inst_tc] in now_inst_waves:
                now_track_waves.append({'time' : [t], 'coord_polygons' : [(inst_lons, inst_lats)], 'prob_polygons' : [inst_vals],
                                        'raw_center_lon' : [inst_mean_lon], 'raw_center_lat' : [inst_mean_lat], 'TCs' : [inst_tc],
                                        'children' : [], 'parent' : []})
                
                track_waves.append(now_track_waves[-1])

            past_18_waves = past_18_waves[1:]
            past_18_waves.append(now_track_waves)
            
        ## Post-processing of waves
        retain_waves = []
        for wave in track_waves:
            # interpolate missing coordinates
            wave_times = wave['time']
            wave_times_indices = [time_indices[t] for t in wave_times]
            wave_coord_polygons = wave['coord_polygons']
            wave_prob_polygons = wave['prob_polygons']
            wave_raw_lons = wave['raw_center_lon']
            wave_raw_lats = wave['raw_center_lat']
            wave_tcs = wave['TCs']
            # iterate through wave_times_indices looking for gaps between timesteps
            i = 1
            while i < len(wave_times_indices):
                time_i = wave_times_indices[i]
                time_gap = time_i-wave_times_indices[i-1]
                if time_gap == 1:
                    i+=1                            
                    continue
                lon_diff = wave_raw_lons[i]-wave_raw_lons[i-1]
                lat_diff = wave_raw_lons[i]-wave_raw_lons[i-1]            
                for time_j in range(1, time_gap):
                    # keep inserting new stuff at same index until gap is filled
                    new_time_i = time_i-time_j
                    new_time = index_times[new_time_i]
                    # interpolate lats and lons
                    diff_factor = time_j/time_gap
                    wave_times.insert(i, new_time)
                    wave_times_indices.insert(i, new_time_i)
                    wave_raw_lons.insert(i, wave_raw_lons[i]-(lon_diff*diff_factor))
                    wave_raw_lats.insert(i, wave_raw_lats[i]-(lat_diff*diff_factor))
                    # just gonna assume TC matches are the same as the starting ones
                    wave_tcs.insert(i, wave['TCs'][i])
                    wave_prob_polygons.insert(i, wave_prob_polygons[i])
                    wave_coord_polygons.insert(i, (wave_coord_polygons[i][0]-(lon_diff*diff_factor), wave_coord_polygons[i][1]-(lat_diff*diff_factor)))
                i+=1
            # trim off consecutive eastward points at end, unless it's a merge point
            i = len(wave_times_indices)-1
            while i > 0 and wave_raw_lons[i] > wave_raw_lons[i-1] and len(wave['parent']) == 0:
                i-=1

            if i < len(wave_times_indices)-1:
                wave['time'] = wave['time'][:i+1]
                wave['coord_polygons'] = wave['coord_polygons'][:i+1]
                wave['prob_polygons'] = wave['prob_polygons'][:i+1]
                wave['raw_center_lon'] = wave['raw_center_lon'][:i+1]
                wave['raw_center_lat'] = wave['raw_center_lat'][:i+1]
                wave['TCs'] = wave['TCs'][:i+1]
                wave['children'] = [[child, time] for [child, time] in wave['children'][:i+1] if time_indices[time] <= time_indices[wave['time'][-1]]]

                
        # connect waves whose extrapolated motion based on mean of their initial and final 24 hours overlap with one another within a range
        connect_distance = 3
        track_waves.sort(reverse=True, key=lambda wave:wave['time'][0])
        n_extrap_points = 6
        if connect_extrapolated_waves:
            for wave in track_waves:
                if len(wave['time']) < 4: #==1
                    wave['end_connection'] = None
                    wave['beginning_connection'] = None                
                    continue
                # add 48 hr extrapolated motion at each end starting 24hrs from each end using the mean motion of that 24hrs            
                lon_len = len(wave['raw_center_lon'])
                mean_first_24_hour_motion = np.mean([wave['raw_center_lon'][i] - wave['raw_center_lon'][i-1] for i in range(1,
                                                                                                                            min(5, lon_len))])            
                mean_final_24_hour_motion = np.mean([wave['raw_center_lon'][i] - wave['raw_center_lon'][i-1] for i in range(lon_len-1,
                                                                                                                            max(0, lon_len-5), -1)])
                n_extrap_beginning_points = min(lon_len+int(n_extrap_points/2), n_extrap_points) # at most 48 hours extrap at beginning, at minimum 30 hours
                extrap_beginning_index = min(int(n_extrap_points/2), lon_len-1) # first point should be at 24 hours into wave life, so extrapolate from 30 hours in
                extrap_beginning_time_index = time_indices[wave['time'][extrap_beginning_index]]
                extrap_beginning_lon = wave['raw_center_lon'][extrap_beginning_index]
                extrap_beginning_times = []
                extrap_beginning_lons = []
                for i in range(n_extrap_beginning_points):
                    # create series of lons and times going back in time
                    extrap_beginning_time_index = extrap_beginning_time_index-1
                    extrap_beginning_lon = extrap_beginning_lon-mean_first_24_hour_motion
                    try:
                        extrap_beginning_times.append(index_times[extrap_beginning_time_index])
                    except KeyError:
                        # beginning of year
                        break
                    extrap_beginning_lons.append(extrap_beginning_lon)

                wave['extrap_beginning_times'] = extrap_beginning_times
                wave['extrap_beginning_lons'] = extrap_beginning_lons            

                n_extrap_end_points = min(lon_len+int(n_extrap_points/2), n_extrap_points)
                extrap_end_index = max(lon_len-int(n_extrap_points/2), 0)
                extrap_end_time_index = time_indices[wave['time'][extrap_end_index]]
                extrap_end_lon = wave['raw_center_lon'][extrap_end_index]
                extrap_end_times = []
                extrap_end_lons = []
                for i in range(n_extrap_end_points):
                    # create series of lons and times going forward in time
                    extrap_end_time_index+=1
                    extrap_end_lon += mean_final_24_hour_motion
                    try:
                        extrap_end_times.append(index_times[extrap_end_time_index])
                    except KeyError:
                        # end of year
                        break
                    extrap_end_lons.append(extrap_end_lon)

                wave['extrap_end_times'] = extrap_end_times            
                wave['extrap_end_lons'] = extrap_end_lons
                # only allow one connection at either end 
                wave['end_connection'] = None
                wave['beginning_connection'] = None

            # match up waves that are within 3 degrees of each other at some point in their respective extrap ends and beginnings
            for wave1 in track_waves:
                # wave1 is end extrapolation, wave2 is beginning extrapolation
                for wave2 in track_waves:                
                    if len(wave1['time']) < 4 or len(wave2['time']) < 4 or wave1 is wave2 or not wave2['beginning_connection'] is None:
                        continue
                    _, wave1_inds, wave2_inds = np.intersect1d(wave1['extrap_end_times'], wave2['extrap_beginning_times'], return_indices=True)
                    if len(wave1_inds) == 0:
                        continue
                    wave1_lons = list(np.array(wave1['extrap_end_lons'])[wave1_inds])
                    wave2_lons = list(np.array(wave2['extrap_beginning_lons'])[wave2_inds])
                    overlap_times = []
                    overlap_lons = []
                    for lon1, lon2, ind1, ind2, in zip(wave1_lons, wave2_lons, wave1_inds, wave2_inds):
                        if np.abs(lon1-lon2) <= connect_distance:
                            overlap_times.append(wave1['extrap_end_times'][ind1])
                            overlap_lons.append(float(np.mean([lon1, lon2])))
                    # of overlap times, want maximum possible coordinates that are real wave coordinates and not extrapolations

                    # elif an overlap time exists for extrapolated coordinates in both waves (it has to), take their mean
                    new_times = []
                    new_lats = []
                    new_lons = []                
                    new_coord_polygons = []
                    new_prob_polygons = []
                    new_TCs = []

                    for i, t in enumerate(overlap_times):
                        if t in wave1['time'] and t in wave2['time']:
                            # if an overlap time exists for real wave coordinates in both waves, take their mean                        
                            new_lons.append(float(np.mean([wave1['raw_center_lon'][wave1['time'].index(t)], wave2['raw_center_lon'][wave2['time'].index(t)]])))
                            new_lats.append(float(np.mean([wave1['raw_center_lat'][wave1['time'].index(t)], wave2['raw_center_lat'][wave2['time'].index(t)]])))
                            new_coord_polygons.append([np.nan]) 
                            new_prob_polygons.append([np.nan])
                            new_TCs.append([wave1['TCs'][wave1['time'].index(t)], wave2['TCs'][wave2['time'].index(t)]])
                            new_times.append(t)
                        elif t in wave1['time']:
                            # elif an overlap time exists in one real wave, use that coordinate                
                            new_lons.append(wave1['raw_center_lon'][wave1['time'].index(t)])
                            new_lats.append(wave1['raw_center_lat'][wave1['time'].index(t)])
                            new_coord_polygons.append(wave1['coord_polygons'][wave1['time'].index(t)])
                            new_prob_polygons.append(wave1['prob_polygons'][wave1['time'].index(t)])
                            new_TCs.append(wave1['TCs'][wave1['time'].index(t)])                        
                            new_times.append(t)                        
                        elif t in wave2['time']:
                            new_lons.append(wave2['raw_center_lon'][wave2['time'].index(t)])
                            new_lats.append(wave2['raw_center_lat'][wave2['time'].index(t)])
                            new_coord_polygons.append(wave2['coord_polygons'][wave2['time'].index(t)])
                            new_prob_polygons.append(wave2['prob_polygons'][wave2['time'].index(t)])
                            new_TCs.append(wave2['TCs'][wave2['time'].index(t)])                        

                            new_times.append(t)                        
                        else:
                            new_lons.append(overlap_lons[i])
                            # use mean of end and beginning lat of two waves, except for categories where it doesn't make sense to take the mean
                            new_lats.append(float(np.mean([wave1['raw_center_lat'][-1], wave2['raw_center_lat'][0]])))
                            new_TCs.append([wave1['TCs'][-1], wave2['TCs'][0]])
                            new_coord_polygons.append([np.nan]) 
                            new_prob_polygons.append([np.nan])                        
                            new_times.append(t)

                    if len(overlap_times) > 0:
                        # save info for joining waves together next
                        wave1['end_lons'] = new_lons
                        wave1['end_lats'] = new_lats
                        wave1['end_times'] = new_times
                        wave1['end_coord_polygons'] =  new_coord_polygons
                        wave1['end_prob_polygons'] =  new_prob_polygons
                        wave1['end_TCs'] = new_TCs
                        wave1['end_connection'] = wave2
                        wave2['beginning_connection'] = wave1
                        
            # create new waves from joined waves
            post_connected_waves = []
            drop_waves = []
            # sort by latest termination time to join these up
            track_waves.sort(reverse=True, key=lambda wave:wave['time'][0])
            for wave in track_waves:
                if wave['end_connection'] is None and wave['beginning_connection'] is None: post_connected_waves.append(wave)
                elif not wave['end_connection'] is None:
                    times_lons_lats = [[t, lon, lat, coords, probs, tcs] for t, lon, lat, coords, probs, tcs in zip(wave['end_times'], wave['end_lons'], wave['end_lats'],
                                                                                                                  wave['end_coord_polygons'], wave['end_prob_polygons'], wave['end_TCs'])]

                    times_lons_lats.sort(key=lambda triple: triple[0])
                    # add times, lons, lats from extrapolation check
                    for [time, lon, lat, end_coord_polygon, end_prob_polygon, TCs] in times_lons_lats:
                        try:
                            time_ind = wave['time'].index(time)
                            wave['raw_center_lon'][time_ind] = lon
                            wave['raw_center_lat'][time_ind] = lat
                            wave['coord_polygons'][time_ind] = end_coord_polygon
                            wave['prob_polygons'][time_ind] = end_prob_polygon
                            wave['TCs'][time_ind] = TCs                        
                        except ValueError:
                            wave['time'].append(time)
                            wave['raw_center_lon'].append(lon)
                            wave['raw_center_lat'].append(lat)

                    # add other times, lons, lats from wave2
                    wave2 = wave['end_connection']

                    wave2_start_ind = 0
                    for time in [l[0] for l in times_lons_lats]:                
                        # last time in list that is in wave2's times should be the one we start including wave2's data after
                        try:
                            wave2_start_ind = wave2['time'].index(time)
                            wave2_start_ind+=1 
                        except ValueError:
                            continue

                    for var in ['time', 'raw_center_lon', 'raw_center_lat', 'coord_polygons', 'prob_polygons', 'TCs', 'children']:
                        wave[var].extend(wave2[var][wave2_start_ind:])
                    # might have to do something with parent here
                    post_connected_waves.append(wave)
                    drop_waves.append(wave2)
                    if len(set(wave['time'])) != len(wave['time']):
                        print('repeated times', wave['time'], wave2_start_ind, wave2['time'], [l[0] for l in times_lons_lats])
                        print()

            track_waves = [wave for wave in post_connected_waves[::-1] if wave not in drop_waves]

        # eliminate waves that last less than time_min    
        for wave in track_waves:
            # memory issues right now with saving coords
            wave['coord_polygons'] = []
            wave['prob_polygons'] = []

            if len(wave['time']) < time_min:
                if len(wave['parent']) > 0:
                    wave_merge_time, wave_parent = wave['parent'][1], wave['parent'][0]                    
                    wave_parent['children'] = [[child, time] for [child, time] in wave_parent['children'] if child != wave]
                if len(wave['children']) > 0:
                    for [child_wave, time] in wave['children']:
                        child_wave['parent'] = []
            elif (wave['raw_center_lon'][0] - wave['raw_center_lon'][-1]) / len(wave['time']) < min_6hr_distance:
                if len(wave['parent']) > 0:
                    print('losing merger', wave['time'], wave['raw_center_lon'], (wave['raw_center_lon'][0] - wave['raw_center_lon'][-3]) / (len(wave['time']) - 2))
                    wave_merge_time, wave_parent = wave['parent'][1], wave['parent'][0]
                    wave_parent['children'] = [[child, time] for [child, time] in wave_parent['children'] if child != wave]
                else:
                    retain_waves.append(wave)
                if len(wave['children']) > 0:                    
                    for [child_wave, time] in wave['children']:
                        child_wave['parent'] = []
            else:
                retain_waves.append(wave)

        # final pass of merge checks, and also smoothing
        merge_extraps = 4
        merge_detection_dist = 5
        for wave in retain_waves:
            if extra_merge_detection:
                found_parent=False
                wave_beginning_time_ind = time_indices[wave['time'][0]]        
                wave_final_time_ind = time_indices[wave['time'][-1]]
                wave_final_lon = wave['raw_center_lon'][-1]
                wave_speed = np.mean([(wave['raw_center_lon'][j] - wave['raw_center_lon'][j-1])/(time_indices[wave['time'][j]]-time_indices[wave['time'][j-1]])
                                                     for j in range(-min(len(wave['raw_center_lon'])-1, 8), 0, 1)])
                try:
                    wave_final_times_extrap = [index_times[i] for i in range(wave_final_time_ind+1, wave_final_time_ind+merge_extraps+1)]
                except KeyError:
                    # have to smooth anyway
                    wave['smooth_center_lon'] = savgol_filter(wave['raw_center_lon'], 7, 2)
                    wave['smooth_center_lat'] = savgol_filter(wave['raw_center_lat'], 7, 2)                
                    continue
                wave_final_lons_extrap = [wave_final_lon+wave_speed*(i+1) for i in range(len(wave_final_times_extrap))]
                #NOTE: Will have to fill some time gaps still            
                # check if any track waves from 24 or 18 hours ago with no matches were within 5 degrees of another (not new) track wave for any of their extrapolation points
                # if they were, merge into that surviving wave
                for wave2 in retain_waves:
                    if found_parent:
                        break
                    wave2_beginning_time_ind = time_indices[wave2['time'][0]]        
                    wave2_final_time_ind = time_indices[wave2['time'][-1]]            
                    if wave2_beginning_time_ind >= wave_final_time_ind or wave2_final_time_ind <= wave_final_time_ind or wave2_beginning_time_ind >= wave_final_time_ind-4:
                        # want waves with some time overlap that finish after our wave and also aren't too new
                        continue
                    for t, wave_lon in zip(wave_final_times_extrap, wave_final_lons_extrap):
                        if found_parent:
                            break
                        try:
                            wave2_time_index = wave2['time'].index(t)
                            wave2_lon = wave2['raw_center_lon'][wave2_time_index]
                        except ValueError:
                            continue                    
                        if np.abs(wave_lon - wave2_lon) < 5 or (np.abs(wave_lon - wave2_lon) < 5 and wave_lon > wave2_lon):                        
                            found_parent=True
                            try:
                                update_track_wave(wave, t, [], [],
                                                  [], wave2['raw_center_lon'][wave2_time_index], wave2['raw_center_lat'][wave2_time_index], wave2['TCs'][wave2_time_index])
                            except IndexError:
                                # TC appending issues sometimes
                                update_track_wave(wave, t, [], [],
                                                  [], wave2['raw_center_lon'][wave2_time_index], wave2['raw_center_lat'][wave2_time_index], [])
                                
                            print('did a new method merge at', t, wave2_lon)
                            print('child ', wave['time'], wave['raw_center_lon'], wave_lon)
                            print('parent', wave2['time'], wave2['raw_center_lon'], wave2_lon)
                            print()
                            wave['parent'] = [wave2, t]
                            wave2['children'].append([wave, t])                                                        
                            # run postprocessing again
                            # interpolate missing coordinates
                            wave_times = wave['time']
                            wave_times_indices = [time_indices[t] for t in wave_times]
                            wave_coord_polygons = wave['coord_polygons']
                            wave_prob_polygons = wave['prob_polygons']
                            wave_raw_lons = wave['raw_center_lon']
                            wave_raw_lats = wave['raw_center_lat']
                            wave_tcs = wave['TCs']
                            # iterate through wave_times_indices looking for gaps between timesteps
                            i = 1
                            while i < len(wave_times_indices):
                                time_i = wave_times_indices[i]
                                time_gap = time_i-wave_times_indices[i-1]
                                if time_gap == 1:
                                    i+=1                            
                                    continue
                                lon_diff = wave_raw_lons[i]-wave_raw_lons[i-1]
                                lat_diff = wave_raw_lons[i]-wave_raw_lons[i-1]            
                                for time_j in range(1, time_gap):
                                    # keep inserting new stuff at same index until gap is filled
                                    new_time_i = time_i-time_j
                                    new_time = index_times[new_time_i]
                                    # interpolate lats and lons
                                    diff_factor = time_j/time_gap
                                    wave_times.insert(i, new_time)
                                    wave_times_indices.insert(i, new_time_i)
                                    wave_raw_lons.insert(i, wave_raw_lons[i]-(lon_diff*diff_factor))
                                    wave_raw_lats.insert(i, wave_raw_lats[i]-(lat_diff*diff_factor))
                                    # just gonna assume TC matches are the same as the starting ones                                
                                    try:
                                        wave_tcs.insert(i, wave['TCs'][i])
                                    except IndexError:
                                        print('failed to insert TCs', len(wave_tcs), len(wave_times), len(wave_raw_lons))
                                i+=1
            wave['smooth_center_lon'] = savgol_filter(wave['raw_center_lon'], 7, 2)
            wave['smooth_center_lat'] = savgol_filter(wave['raw_center_lat'], 7, 2)
            
        return retain_waves
        
    returns = Parallel(n_jobs=-1)(delayed(get_waves_by_year)(y) for y in years)

    retain_waves = []
    for r in returns:
        retain_waves = retain_waves + r
    
    if local_max_method:
        f_name = wave_checkpoint_dir+'wave_tracks_local_maxima_'+years[0]+'_'+years[-1]+'_'+str(thresh)
    else:
        f_name = wave_checkpoint_dir+'wave_tracks_'+years[0]+'_'+years[-1]+'_'+str(thresh)        

    print(len(retain_waves))
    
    with open(f_name, 'wb') as f:
        pickle.dump(retain_waves, f, protocol=pickle.HIGHEST_PROTOCOL)        
    
        
    ### Outline of tracking methodology ###
    '''
    In each year, at each timestep, get list of instant waves

        Compare wave locations to waves from past 18 hours
            - Stop looking at waves as go further back in time if have already found a point from them 
            - Search eastward further as go further back in time
            - Also search slightly westward for 6 hours before
            - If match found, append coordinates
                  - If multiple matches found, merge and keep oldest wave
            - Else, initialize new wave object

        Merge waves within 6 degrees of one another, keeping oldest wave
            - 
    
    
    Interpolate missing coordinates
    Smooth coordinates
    Get unique storm names at each timestep
    Filter out 24-hr periods in waves that don't move westward fast enough (
    Filter out waves that don't move westward fast enough
    Filter out waves that don't last at least 24 hours (maybe 48)
    Make sure references to these waves are deleted from children / parents

    
    Wave object components: dict of: 'time':               list of times,
                                     'coord_polygons' :    list of tuples of (lons, lats) of polygons,
                                     'prob_polygons'  :    list of polygons of probabilities,
                                     'raw_center_lon' :    list of mean lons,
                                     'raw_center_lat' :    list of mean lats,
                                     'smooth_center_lon' : list of smoothed mean lons,
                                     'smooth_center_lat' : list of smoothed mean lats,                            
                                     'TCs' :               list of list of colocated TCs at each time,
                                     'children' :          list of times waves merged into this wave and pointers to those waves,    
                                     'parent' :            list of time this wave merged with another wave and pointer to that wave
    '''
    
    
            
create_instantaneous_wave_climo()
lon_connect = 20 # this is here to keep wave window tracking consistent when tweaking lon_connect
create_wave_tracks()
