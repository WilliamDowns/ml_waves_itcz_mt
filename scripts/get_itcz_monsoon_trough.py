'''
Get all ITCZ/Monsoon Troughs from Tropical Weather Discussion files
'''

from ml_waves_itcz_mt.util import *
from ml_waves_itcz_mt.plotting import *
from ml_waves_itcz_mt.constants import *
from joblib import Parallel, delayed
import re
import pickle
import csv

topdir = '/your_dir/ml_waves/'

# set year bounds
years = [str(year) for year in range(2004,2024)]

# choose basin
basin = 'TWDAT'
#basin = 'TWDEP'

out_suffix = 'atl' if basin == 'TWDAT' else 'epac'

# whether to save list of missing times and exit
save_missing_times = False

# get filenames in date and year bounds, rounding to nearest 6 hour interval
filenames = []
dates_by_filename = {}

# uncomment section to rerun TWD reader and imt location grabber

skip_files = global_skip_TWD_files + list(global_manual_assignments.keys())

for year in years:
    # 4. For non-removed ones that meet these criteria, assign them to the advisory time corresponding to their issuance time
    # 5. Check if timesteps have multiple TWDs associated with them
    # 6. For timesteps with multiple TWDs, choose the latest issued one

    path = os.path.join(topdir,'TWDs/archive/text/'+basin+'/', year)    
    yearly_filenames = sorted(os.listdir(path))

    # 1. Remove TWDs that aren't needed and have bad timesteps (manually checked based on following 2 conditions)    
    for f in skip_files:
        if f[6:10] == year:
            try:
                yearly_filenames.remove(f)
            except:
                pass
        
    yearly_dates_by_filename = {f : closest_time_str(f[6:])
                                for f in yearly_filenames}
    

    for f in yearly_filenames:
        # 2. Check which ones don't have analysis times matching the typical format based on time in filename        
        closest_time = yearly_dates_by_filename[f][-4:-2]
        check_str = str((int(closest_time)-6)%24) + '00'
        if len(check_str) == 3:
            check_str = '0' + check_str
        found_corresponding_time = False
        time_line = ''
        contents = ''
        
        with open(os.path.join(path,f), 'r') as text:
            contents = text.readlines()
            time_line_index = -1
            for line in contents:
                if len(time_line) == 0:
                    time_line_index = time_line_index + 1
                if ' utc surface analysis' in line.lower():
                    time_line = line
                    if check_str + ' UTC' in line:                        
                        found_corresponding_time = True
                        break

        if not found_corresponding_time:
            # 3. Check which ones also don't have issuance times matching the filename time
            # match AM / PM / UTC timestamps to their respective appropriate file times
            
            second_check_str = yearly_dates_by_filename[f][-4:]
            first_UTC_index = time_line.find('UTC')
            first_time = time_line[first_UTC_index-5:first_UTC_index-1]
            second_UTC_index = time_line[first_UTC_index+3:].find('UTC')
            second_time = time_line[first_UTC_index+3:][second_UTC_index-5:second_UTC_index-1]
            
            # fix a few typos
            def fix_time_typos(t):
                if len(t) > 0:
                    if t[0] == ' ':
                        t = t[1:] + '0'
                    if t[0] == 'O':
                        t = '0' + t[1:]
                    if t[0] in ['0','1','2','3','4','5','6','7','8','9'] and int(t[0]) > 2:
                        t = '0' + t[:-1]
                    #if t[-1] == ' ':
                    #    if t[0]
                return t
            first_time, second_time = fix_time_typos(first_time), fix_time_typos(second_time)
            
            if second_UTC_index == -1:
                # there's an extra line in the TWD for the satellite imagery time
                try:
                    second_UTC_index = contents[time_line_index+1].find('UTC')
                except:
                    # skip trying to figure out problem cases that don't have times
                    print('This file has a problematic satellite analysis time string ' , f,
                          ' but has likely been checked already')
                    continue
                # avoid occasional issue of 3 digit times, assuming the leading 0 is missing (may cause a few mishaps)
                if second_UTC_index == 4:
                    second_time = '0' + contents[time_line_index+1][second_UTC_index-4:second_UTC_index-1]
                else:
                    second_time = contents[time_line_index+1][second_UTC_index-5:second_UTC_index-1]
                
            #print(first_time, second_time)
            # timestamp here refers to the time in the header
            timestamp_index = 7
            timestamp_line = contents[timestamp_index]
            while 'UTC' not in timestamp_line and 'AM' not in timestamp_line and 'PM' not in timestamp_line:
                #20220216 case (possibly elsewhere) of an extra line before timestamp
                timestamp_index = timestamp_index + 1
                timestamp_line = contents[timestamp_index]
            if timestamp_line.find('AM') != -1:
                part_of_day = 'AM'
                timestamp_hour_str = timestamp_line[:timestamp_line.find('AM')-1]
            elif timestamp_line.find('PM') != -1:
                part_of_day = 'PM'
                timestamp_hour_str = timestamp_line[:timestamp_line.find('PM')-1]
            elif timestamp_line.find('UTC') != -1:
                part_of_day = 'UTC'
                timestamp_hour_str = timestamp_line[:timestamp_line.find('UTC')-1]
            if len(timestamp_hour_str) == 3:
                # single digit hours
                timestamp_hour_str = '0' + timestamp_hour_str[0]
            elif len(timestamp_hour_str) == 4:
                timestamp_hour_str = timestamp_hour_str[:2]

            # the timestamp does not match what we expect
            # assign correct time to filename based on timestamp
            correct_timestamps_dict = {('PM', '06') : '00', ('PM', '07') : '00', ('PM', '08') : '00',
                                       ('AM', '12') : '06', ('AM', '01') : '06', ('AM', '02') : '06',
                                       ('AM', '06') : '12', ('AM', '07') : '12', ('AM', '08') : '12',
                                       ('PM', '12') : '18', ('PM', '01') : '18', ('PM', '02') : '18',
                                       ('UTC', '23') : '00', ('UTC', '00') : '00',
                                       ('UTC', '05') : '06', ('UTC', '06') : '06',
                                       ('UTC', '11') : '12', ('UTC', '12') : '12',
                                       ('UTC', '17') : '18', ('UTC', '18') : '18'}



            # can't be bothered to comb through these manually 
            # so will probably move timestamp forward according to most recent of regular timestamp, sfc analysis timestamp and satellite timestamp
            sat_timestamps_dict = {'19' : '00', '20' : '00', '21' : '00', '22' : '00', '23' : '00', '00' : '00',
                                   '01' : '06', '02' : '06', '03' : '06', '04' : '06', '05' : '06', '06' : '06',
                                   '07' : '12', '08' : '12', '09' : '12', '10' : '12', '11' : '12', '12' : '12',
                                   '13' : '18', '14' : '18', '15' : '18', '16' : '18', '17' : '18', '18' : '18'}
            sfc_timestamps_dict = {'12' : '18', '13' : '18', '14' : '18', '15' : '18', '16' : '18', '17' : '18',
                                   '18' : '00', '19' : '00', '20' : '00', '21' : '00', '22' : '00', '23' : '00',
                                   '00' : '06', '01' : '06', '02' : '06', '03' : '06', '04' : '06', '05' : '06',
                                   '06' : '12', '07' : '12', '08' : '12', '09' : '12', '10' : '12', '11' : '12'}
            main_timestamps_dict = {'00' : '00', '01' : '00', '02' : '06', '03' : '06', '04' : '06',
                                    '05' : '06', '06' : '06', '07' : '06', '08' : '12', '09' : '12',
                                    '10' : '12', '11' : '12', '12' : '12', '13' : '12', '14' : '18',
                                    '15' : '18', '16' : '18', '17' : '18', '18' : '18', '19' : '18',
                                    '20' : '00', '21' : '00', '22' : '00', '23' : '00'}
            try:
                # assuming first time is sfc analysis time, second time is satellite imagery time
                # Have commented out sfc analysis contribution for now because it is pretty ambiguous
                sfc_time_match = sfc_timestamps_dict[first_time[:2]]
                sat_time_match = sat_timestamps_dict[second_time[:2]]
                main_time_match = main_timestamps_dict[timestamp_hour_str]
                # want latest time (so 0>18>12>6>0)
                latest_time_dict = {('18', '12') : '18', ('12', '06') : '12', ('06', '00') : '06', ('00', '18') : '00',
                                    ('06', '06') : '06', ('12', '12') : '12', ('18', '18') : '18', ('00', '00') : '00'}
                latest_time_dict.update({(k[1], k[0]) : v for k, v in latest_time_dict.items()})
                timestamp_UTC_match = latest_time_dict[(sat_time_match, main_time_match)]

            except:
                # catch any files with ambiguous timestamps that haven't manually been weeded out
                print('ISSUE FILE REMAINS:', f, closest_time, first_time, second_time)
                continue
            new_time = yearly_dates_by_filename[f]
            # make sure we account for when the closest time and the timestamp don't match to the same date
            if (closest_time == '00' and timestamp_UTC_match == '18'):
                # change day backward, making sure we account for possibly changing month and year
                month = new_time[4:6]
                year = new_time[:4]
                new_day = str(int(new_time[6:8])-1)
                new_end = '1800'                    
                if len(new_day) == 1: new_day = '0' + new_day
                if int(new_day) < 1:
                    # going back a month
                    if int(month) == 1:
                        # going back a year
                        new_year = str(int(year)-1)
                        new_month = '12'
                        new_day = '31'
                        yearly_dates_by_filename[f] = new_year+new_month+new_day+new_end
                    else:
                        new_month = str(int(month)-1)
                        if len(new_month) == 1: new_month = '0' + new_month
                        new_day = global_month_days[new_month][-1]
                        yearly_dates_by_filename[f] = year+new_month+new_day+new_end
                else:
                    yearly_dates_by_filename[f] = year+month+new_day+new_end

            elif (closest_time == '18' and timestamp_UTC_match == '00'):
                # change day forward, making sure we account for possibly changing month and year
                month = new_time[4:6]
                year = new_time[:4]
                new_day = str(int(new_time[6:8])+1)
                new_end = '0000'
                if len(new_day) == 1: new_day = '0' + new_day
                if int(new_day) > int(global_month_days[month][-1]):
                    # going forward a month
                    if int(month) == 12:
                        # going forward a year
                        new_year = str(int(year)+1)
                        new_month = '01'
                        new_day = '01'
                        yearly_dates_by_filename[f] = new_year+new_month+new_day+new_end
                    else:
                        new_month = str(int(month)+1)
                        if len(new_month) == 1: new_month = '0' + new_month
                        new_day = '01'
                        yearly_dates_by_filename[f] = year+new_month+new_day+new_end
                else:
                    yearly_dates_by_filename[f] = year+month+new_day+new_end
            else:
                yearly_dates_by_filename[f] = new_time[:-4] + timestamp_UTC_match + '00'

                
    filenames.extend(yearly_filenames)
    dates_by_filename.update(yearly_dates_by_filename)

    
# add back in our manually assigned times    
dates_by_filename.update({k: v for k,v in global_manual_assignments.items() if basin[-2:] in k})
filenames.extend([k for k in global_manual_assignments.keys() if basin[-2:] in k])
filenames.sort()


# debugging to check for duplicates and missed times
expected_dates = []
expected_times = ['0000', '0600', '1200', '1800']
for year in years:    
    for month in global_months:
        if year == '2022' and month == '09':
            #break
            True
        for day in global_month_days[month]:
            for time in expected_times:
                expected_dates.append(year+month+day+time)

# check for missing times
missing_times = []
for date in expected_dates:
    if date not in dates_by_filename.values():
        print('missing ', date)
        missing_times.append(date)
    
if save_missing_times:
    # save list of missing times so I can make sure they don't get used in various analyses
    with open(os.path.join(topdir, 'missing_times_'+out_suffix), 'wb') as f:
        pickle.dump(missing_times, f, protocol=pickle.HIGHEST_PROTOCOL)
    exit()
    
# check for times with duplicate files
filenames_by_date = {dates_by_filename[f] : [] for f in filenames}
for f in filenames:
    filenames_by_date[dates_by_filename[f]].append(f)

# assign latest TWD at any given timestep as the actual corresponding file
# for each date
for date in filenames_by_date.keys():
    filenames_by_date[date] = filenames_by_date[date][-1]

# get sections to check for imt in each file
imt_blocks_by_time = {time : [] for time in filenames_by_date.keys()}

imt_headers = ["tropical waves/itcz...", "..itcz", "...itcz", "...the itcz", "...the monsoon trough", "...monsoon trough", "..monsoon trough", "monsoon trough...",
               "...intertropical"]

no_imt_headers = ['Tropical Weather Discussion for ']

for time in filenames_by_date.keys():
    f = filenames_by_date[time]
    path = os.path.join(topdir,'TWDs/archive/text/'+basin+'/', f[6:10])
    imt_block = get_twd_block(os.path.join(path,f), *imt_headers)
    if len(imt_block) == 0:        
        no_imt_block = get_twd_block(os.path.join(path,f), *no_imt_headers)
    else:
        no_imt_block = []

    imt_blocks_by_time[time].extend([block for block in [imt_block, no_imt_block] if len(block) > 0])


####### Find imt in each corresponding block in each TWD
imts_by_time = {time : [] for time in filenames_by_date.keys()}

word_counts={}


### various string searches that get checked for each wave paragraph
exclusion_pairs = [('short[ -]*', 'b'), ('[ ]*field', 'a'), ('[ ]*event', 'a'),
                   ('[ ]*height', 'a'), ('s', 'a'), ('long[ -]*', 'b'), ('[ ]*runup', 'a'),
                   ('micro', 'b'), ('trak', 'a'), ('[ ]*of[ ]*african[ ]*dust[ ]*', 'a'),
                   ('[ ]*conditions', 'a'), ('[ ]*model', 'a')]

# Don't include the enormous possible spread of locations given by mexico / brazil (also most of these have an accompanying coordinate anyway)
location_re = '(' + '|'.join([s.lower().replace(' ', '[ ]*') for s in global_loc_keyword_map.keys()-['Brazil', 'brazil', 'gulf of mexico', 'mexico', 'Mexico']]) + ')'
direction_re = '(([, -]|^)(' + '|'.join([s.lower().replace(' ', '[ ]*') for s in global_direction_map.keys() - [None]]) + '))'
c_re = '[0-9]+[\.]?[0-9]*'
missing_decimal_re = '[2-9][0-9][0-9]'
missing_decimal_re_2 = '[0-9][0-9][0-9][0-9]'
missing_lat_decimal_re = '[0-9][0-9][0-9][ns]'
start_re = 'from[ ]*'
end_re = '(to|near)[ ]*'
# lat/lon pairs
coord_pair_re = '{}(?:n|s)[/ ]*{}w'.format(c_re, c_re)
#coord_pair_re = '(?:({}(?:n|s))|eq)[/ ]*{}w'.format(c_re, c_re)
coord_pair_start_re = '{}\D*{}'.format(start_re, coord_pair_re)
coord_pair_end_re = '{}\D*{}'.format(end_re, coord_pair_re)
coord_pair_typo_re = '{}n/?{}[a-z ,]'.format(c_re, c_re)
# exclude "st." since that is used in island names,
# as well as 't.s', 't.d', 'u.s'
sentence_re = '(?<! st|[s|t|u|n]\.[s|d])\.[\D ]+'
# also don't want "from XXn to XXw" or "from XXw to XXn", NEED TO ADD THIS ONE IN
# Might be hard to confirm whether it's supposed to be lats or lons; maybe have a max value for
# it to be considered a mistyped lat?
lon_coord_re = '(?<![ns0-9]){}/?{}w'.format(c_re, c_re)
lat_coord_re = '{}?/?{}n'.format(c_re,c_re)
equator_re = '(?:the)?[ ]*equator(?!ial|ward)'
equator_coord_re = 'eq(?:{})'.format(lon_coord_re)
# an issue in like 5 EPAC files where there are multiple decimal places in a lon coord
lon_coord_typo_re = '[\d]+\.[\d]+\.[\d]+w'
# pick up if there's a location descriptor after the word to
to_re = '(?<!according)[, ]+(to|across|over|from) '
not_kt_re = '{}(?![ ]*[\d\-]*[ ]*(kt|knot|be |the[ ]+past|the[ ]+low))'.format(to_re)
of_re = ' of[ ]*(the)?{}'.format(location_re)
# do not want to capture locations like 'west of cabo verde' as they're too vague
# had to add extra code in loop to handle this because not including 'of the' but
# including just 'the' was very arduous
not_of_re = '(?<! of)[ ]*'
geographic_re = '{}({}[ ]*)*[ ]*(near)?[ ]*(of)?[ ]*(the)?[ ]*{}?[, /-]+{}'.format(not_of_re, to_re[:-1], direction_re, location_re)
# not currently specifically looking at borders
border_re = '(border[ ]*of[ ]*{}[ ]*and[ ]*{}|{}[ -/]*{}'.format(location_re, location_re, location_re, location_re)
# don't like when these words are in sentences
convection_strs = ['scattered', 'moderate', 'convection', 'precipitation', 'isolated',
                   'shower', 'thunderstorm', 'rainfall', 'clouds', 'fresh',
                   'moisture', 'cumulus',  'upper level',  'precipitable', 'pw',
                   'convective', 'cloud']
reposition_strs = ['reposition', 'previously', 'moved back', 'dissipated', 'expected',
                   'was absorbed', 'has been asbored', 'has moved', 'dropped', 'will be', 'dispersed',
                   'has developed', 'replaced', 'was analyzed']


min_lat = 0
# loop through discussion blocks

for time in filenames_by_date.keys():    
    blocks = imt_blocks_by_time[time]

    paragraphs = []
    
    imt_list = [] #imt objects in it
    for block in blocks:
        # find each paragraph in a block (though we will probably only ever have 1 here)
        breaks = []
        cur_line_i = 0
        while cur_line_i < len(block):
            cur_line = block[cur_line_i]
            if len(cur_line) == cur_line.count(' ') + cur_line.count('\n') and cur_line_i != len(block) - 1:                
                # blank line, next line starts a new paragraph
                breaks.append(cur_line_i+1)
            elif cur_line_i == 0:
                # first line of block isn't blank, must be start of first paragraph
                breaks.append(cur_line_i)
            cur_line_i = cur_line_i + 1

        for i in range(len(breaks)-1):
            # one paragraph here is the set of lines before next break
            paragraphs.append([line for line in block[breaks[i]:breaks[i+1]]])

        # add paragraph for last line break
        paragraphs.append([line for line in block[breaks[-1]:len(block)]])

    # convert paragraphs into actual paragraphs rather than lists of strings
    # get rid of usesless / confusing characters for upcoming sentence search
    for i in range(len(paragraphs)):
        paragraphs[i] = ' '.join(paragraphs[i]).replace('\n', ' ').replace('...', ',').replace('..', ',').lower()

    # Only keep paragraphs where ITCZ or monsoon trough are actually mentioned
    keep_strs = ['itcz', 'monsoon', 'intertropical']
    paragraphs = [p for p in paragraphs if any([s in p for s in keep_strs])]
    # Now deduce location of imt from these paragraphs    
    for p in paragraphs:
        coord_pairs = []        
        lat_coords = []
        lon_coords = []
        
        # All sentences could have information / their own separate objects in them,
        # but may also need to know what type the previous sentence was
        sentence_ends, _ = get_locs_and_matches(sentence_re, p)
        sentence_indices = [0, *[s[0] for s in sentence_ends], len(p)]
        sentences = [p[sentence_indices[i]:sentence_indices[i+1]]
                     for i in range(len(sentence_indices)-1)]

        # get rid of sentences that are just periods
        sentences = [s for s in sentences if set(s) != set(['.', ' '])]
        
        # get rid of equator in favor of 00n, and also try to fix typos of 'XXnXXn' to be 'XXnXXw'
        # (and other incorrect coordinate pair endings)
        # also fix issues with south of XXw-type strings and when there's a missing decimal
        # in coords
        def replace_patterns(s, *patterns):
            s_new = s
            for pat in patterns:
                _, typos = get_locs_and_matches(pat, s_new)
                for typo in typos:
                    if pat == coord_pair_typo_re:
                        s_new = s_new.replace(typo, typo[:-1]+'w')
                    elif pat == missing_decimal_re or pat==missing_decimal_re_2:
                        s_new = s_new.replace(typo, typo[:-1] + '.' + typo[-1])
                    elif pat == missing_lat_decimal_re:
                        s_new = s_new.replace(typo, typo[:-2] + '.' + typo[-2:])
                    elif pat == equator_re or pat == equator_coord_re:
                        s_new = s_new.replace(typo, '00n')
                    elif pat == lon_coord_typo_re:
                        # this one we don't have a good idea what to do with, just get rid of the coord
                        s_new = s_new.replace(typo, 'filler')
            return s_new

        typo_patterns = [coord_pair_typo_re, missing_decimal_re, missing_decimal_re_2, equator_re, equator_coord_re, lon_coord_typo_re,
                         missing_lat_decimal_re]        
        sentences = [replace_patterns(s, *typo_patterns) for s in sentences]

        # check to see if terms are present in imt_sentence that describe locations of
        # other things (like convection or past / future wave locations)
        issue_strs = [*convection_strs, *reposition_strs]
        

        # go through each sentence, adding new imt objects as needed
        #found_enough = False
        cat = ''
        
        for sentence in sentences:
            lat_coords = []
            lon_coords = []
            coord_pairs = []
            verbal_locations = []
            coord_pair_locs = []
            lat_coord_locs = []
            lon_coord_locs = []
            
            problem, problem_str = check_for_problem_strs(sentence, issue_strs)
            # skip problem sentences and don't bother reading more if we have enough info
            if problem:
                continue
            # first look for pairs of coordinates, which may be the ideal location denoter
            coord_pair_locs_temp, coord_pair_temp = get_locs_and_matches(coord_pair_re, sentence)
            coord_pairs = coord_pairs + coord_pair_temp
            coord_pair_locs.extend(coord_pair_locs_temp)

            # now begin looking for all coordinates listed, making sure they're not duplicates of coord pairs (only issue for lats
            # based on current regex setup)
            lon_coord_locs_temp, lon_coord_temp = get_locs_and_matches(lon_coord_re, sentence)            
            lon_coords = lon_coords + lon_coord_temp
            lon_coord_locs.extend(lon_coord_locs_temp)
            
            raw_lat_locs, raw_lats = get_locs_and_matches(lat_coord_re, sentence)
            for loc, lat in zip(raw_lat_locs, raw_lats):
                # Make sure lat coords aren't repeats
                min_loc = loc[0]
                max_loc = loc[1]
                repeat = False
                for check_loc in lat_coord_locs+coord_pair_locs:
                    if min_loc >= check_loc[0] and max_loc <= check_loc[1]:
                        repeat = True
                        break
                if not repeat:
                    lat_coords.append(lat), lat_coord_locs.append(loc)

            '''
            # code bit for identifying location names we might have to check
            words = sentence.replace(',', '').replace('.', '').split(" ")
            for s in words:
                if s in word_counts:
                    word_counts[s] = word_counts[s]+1
                else:
                    # checking if string is just letters
                    if s.isalpha():
                        word_counts[s] = 1
            '''
            # Now we get into the real grit: identifying imt locations using landmarks        
            verbal_location_locs, verbal_locations_temp, verbal_locations_groups = get_locs_and_matches(geographic_re, sentence, [8,9])
            # this is that annoying 'of the' code I mentioned before
            # actually need to exclude when 'of' precedes a location without a
            # directional modifier before it as well, since the regex is not working correctly
            for i in range(len(verbal_locations_temp)):
                # make search radius beforehand large enough to include 'northwest of the'
                start_of_search = max(0, verbal_location_locs[i][0]-18)
                search = sentence[start_of_search:min(verbal_location_locs[i][0]+6, len(sentence))]
                if len(re.findall(direction_re+'[ ]*of[ ]*', search)) == 0 and \
                   len(re.findall(direction_re+'[ ]*of[ ]*the[ ]*', search)) == 0:
                    # appending the groups, which have both directions and places in them
                    # added a manual dominican republic check here
                    # also don't want both maracaibo and venezuela, or guajira and colombia or venezuela
                    if verbal_locations_groups[i][1] == 'dominica' and 'dominican' in sentence:
                        verbal_locations.append([verbal_locations_groups[i][0], 'dominican republic'])
                        verbal_location_locs.append(verbal_location_locs[i])
                    elif not ((verbal_locations_groups[i][1] == 'venezuela' and 'maracaibo' in sentence)
                               or (verbal_locations_groups[i][1] == 'venezuela' and 'guajira' in sentence)
                               or (verbal_locations_groups[i][1] == 'colombia' and 'guajira' in sentence)):
                        verbal_locations.append(verbal_locations_groups[i])
                        verbal_location_locs.append(verbal_location_locs[i])

            # Now decide whether this is an ITCZ or Monsoon Trough sentence
            itcz_strs = ['itcz', 'itc', 'itz', 'icz', 'tcz', 'intertropical', 'inter-tropical']
            mt_strs = ['monsoon', 'monson']
            itcz = any([s in sentence for s in itcz_strs])
            mt =  any([s in sentence for s in mt_strs])
            if itcz and not mt:
                cat = 'itcz'
            elif not itcz and mt:
                cat = 'mt'
            elif not itcz and not mt:
                # assume type is same as one in previous sentence if some hinting strings are present
                continue_strs = ['continues', 'resumes', 'segment', 'section', 'portion', 'extends']
                start_strs = ['.d', '.s'] # sentence is actually just a break following a t.s. or t.d.
                if any([sentence.startswith(s) for s in start_strs]) or any([s in sentence for s in continue_strs]):
                    cat = cat
                else:
                    # not a useful sentence
                    cat = ''
            else:
                # both are in sentence
                # find which of itcz and mt are listed first and second, then create two imt objects corresponding to each                
                itcz_loc = min([sentence.index(s) for s in itcz_strs if s in sentence])
                mt_loc = min([sentence.index(s) for s in mt_strs if s in sentence])
                if itcz_loc < mt_loc:
                    first_cat = 'itcz'
                    first_loc = itcz_loc
                    second_cat = 'mt'
                    second_loc = mt_loc
                else:
                    first_cat = 'mt'
                    first_loc = mt_loc
                    second_cat = 'itcz'
                    second_loc = itcz_loc

                def split_coords_and_locs(coords, locs, index, geq=False):
                    if not geq:
                        coords = [c for c, l in zip(coords, locs) if l[1]<index]
                        locs   = [l for l in locs if l[1]<index]
                    else:
                        coords = [c for c, l in zip(coords, locs) if l[0]>=index]
                        locs = [l for l in locs if l[0]>=index]
                    return coords, locs
                first_coord_pairs, first_coord_pair_locs = split_coords_and_locs(coord_pairs, coord_pair_locs, second_loc, geq=False)
                first_lat_coords, first_lat_coord_locs = split_coords_and_locs(lat_coords, lat_coord_locs, second_loc, geq=False)
                first_lon_coords, first_lon_coord_locs = split_coords_and_locs(lon_coords, lon_coord_locs, second_loc, geq=False)
                first_verbal_locations, first_verbal_location_locs = split_coords_and_locs(verbal_locations, verbal_location_locs, second_loc, geq=False)
                second_coord_pairs, second_coord_pair_locs = split_coords_and_locs(coord_pairs, coord_pair_locs, second_loc, geq=True)
                second_lat_coords, second_lat_coord_locs = split_coords_and_locs(lat_coords, lat_coord_locs, second_loc, geq=True)
                second_lon_coords, second_lon_coord_locs = split_coords_and_locs(lon_coords, lon_coord_locs, second_loc, geq=True)
                second_verbal_locations, second_verbal_location_locs = split_coords_and_locs(verbal_locations, verbal_location_locs, second_loc, geq=True)
                

                first_imt = TWDIMT(time, first_cat, first_coord_pairs, first_lat_coords, first_lon_coords, first_verbal_locations,
                                  first_coord_pair_locs, first_lat_coord_locs, first_lon_coord_locs, first_verbal_location_locs, full_text=sentence)
                
                if len(first_imt.coords) > 1:
                    imt_list.append(first_imt)
                    #first_imt.full_print()
                    # in the majority of these sentences, the westernmost coordinate of the first one is also the easternmost coordinate of the second one
                    western_lon_str, western_lat = str(first_imt.coords[0][1])+'w', first_imt.coords[0][0]
                    western_lat_str = str(western_lat)
                    western_lat_str = western_lat_str[1:] + 's' if western_lat < 0 else western_lat_str+'n'
                    second_coord_pairs = [western_lat_str+western_lon_str] + second_coord_pairs
                    second_coord_pair_locs = [(0,1)] + second_coord_pair_locs

                second_imt = TWDIMT(time, second_cat, second_coord_pairs, second_lat_coords, second_lon_coords, second_verbal_locations,
                                 second_coord_pair_locs, second_lat_coord_locs, second_lon_coord_locs, second_verbal_location_locs, full_text=sentence)
                    
                if len(second_imt.coords) >1:                    
                    imt_list.append(second_imt)
                    min_lat = min(min([lat for (lat, lon) in second_imt.coords]), min_lat)                    
                continue
            
            if cat != '':
                # create new imt object (if only have one to make from this sentence, otherwise we did above)
                try:
                    new_imt = TWDIMT(time, cat, coord_pairs, lat_coords, lon_coords, verbal_locations,
                                  coord_pair_locs, lat_coord_locs, lon_coord_locs, verbal_location_locs, full_text=sentence)
                except E:
                    print(time, cat, coord_pairs, lat_coords, lon_coords, verbal_locations,
                          coord_pair_locs, lat_coord_locs, lon_coord_locs, verbal_location_locs, sentence)
                    raise
                # sometimes things get mentioned as not existing at this time / being inland over africa at only one location
                if len(new_imt.coords) > 1:
                    imt_list.append(new_imt)
                    min_lat = min(min([lat for (lat, lon) in new_imt.coords]), min_lat)
                                    
    imts_by_time[time] = imt_list + imts_by_time[time]

# Save imt objects with raw vertex info
with open(os.path.join(topdir, 'raw_imts_'+out_suffix), 'wb') as f:
    pickle.dump(imts_by_time, f, protocol=pickle.HIGHEST_PROTOCOL)


with open(os.path.join(topdir, 'raw_imts_'+out_suffix), 'rb') as f:
    imts_by_time = pickle.load(f)


