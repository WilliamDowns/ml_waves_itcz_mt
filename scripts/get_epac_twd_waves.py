'''
Get all EPAC tropical waves from Tropical Weather Discussion files
'''

from ml_waves_itcz_mt.util import *
from ml_waves_itcz_mt.plotting import *
from ml_waves_itcz_mt.constants import *
from joblib import Parallel, delayed
import re
import pickle
import csv

#topdir = '/your_dir/ml_waves/TWDs/archive/text/TWDAT'
#topdir = '/your_dir/ml_waves/TWDs/archive/text/TWDAT'
topdir = '/your_dir/ml_waves/'
#topdir = '/your_dir/ml_waves/'

# set year bounds
years = [str(year) for year in range(2004,2024)]

# get filenames in date and year bounds, rounding to nearest 6 hour interval
filenames = []
dates_by_filename = {}

# uncomment section to rerun TWD reader and wave location grabber

skip_files = global_skip_TWD_files + list(global_manual_assignments.keys())

for year in years:
    # 4. For non-removed ones that meet these criteria, assign them to the advisory time corresponding to their issuance time
    # 5. Check if timesteps have multiple TWDs associated with them
    # 6. For timesteps with multiple TWDs, choose the latest issued one

    path = os.path.join(topdir,'TWDs/archive/text/TWDEP/', year)    
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
            

            # assign correct time to filename based on timestamp
            correct_timestamps_dict = {('PM', '06') : '00', ('PM', '07') : '00', ('PM', '08') : '00',
                                       ('AM', '12') : '06', ('AM', '01') : '06', ('AM', '02') : '06',
                                       ('AM', '06') : '12', ('AM', '07') : '12', ('AM', '08') : '12',
                                       ('PM', '12') : '18', ('PM', '01') : '18', ('PM', '02') : '18',
                                       ('UTC', '23') : '00', ('UTC', '00') : '00',
                                       ('UTC', '05') : '06', ('UTC', '06') : '06',
                                       ('UTC', '11') : '12', ('UTC', '12') : '12',
                                       ('UTC', '17') : '18', ('UTC', '18') : '18'}

            # can't be bothered to comb through these manually like the Atlantic because there's so dang many of them,
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
                #timestamp_UTC_match = [sfc_time_match, sat_time_match, main_time_match][np.argmax([(int(sfc_time_match)-6)%24,
                #timestamp_UTC_match = [sat_time_match, main_time_match][np.argmax([(int(sat_time_match)-6)%24,
                #                                                                   (int(main_time_match)-6)%24])]
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
dates_by_filename.update({k: v for k,v in global_manual_assignments.items() if 'EP' in k})
filenames.extend([k for k in global_manual_assignments.keys() if 'EP' in k])
filenames.sort()


# debugging to check for duplicates and missed times
expected_dates = []
expected_times = ['0000', '0600', '1200', '1800']
for year in years:    
    for month in global_months:
        if year == '2022' and month == '09':
            break
        for day in global_month_days[month]:
            for time in expected_times:
                expected_dates.append(year+month+day+time)

# check for missing times
for date in expected_dates:
    if date not in dates_by_filename.values():
        print('missing ', date)
        continue
    
    
# check for times with duplicate files
filenames_by_date = {dates_by_filename[f] : [] for f in filenames}
for f in filenames:
    filenames_by_date[dates_by_filename[f]].append(f)

# assign latest TWD at any given timestep as the actual corresponding file
# for each date
for date in filenames_by_date.keys():
    filenames_by_date[date] = filenames_by_date[date][-1]

# get sections to check for waves in each file
wave_blocks_by_time = {time : [] for time in filenames_by_date.keys()}

waves_headers = ["...tropical", "tropical waves/itcz...", "..tropical", "tropical waves..."]
special_features_headers = ["...SPECIAL FEATURE", "SPECIAL FEATURE..."]
no_waves_headers = ['Tropical Weather Discussion for ']


for time in filenames_by_date.keys():
    f = filenames_by_date[time]
    path = os.path.join(topdir,'TWDs/archive/text/TWDEP/', f[6:10])
    wave_block = get_twd_block(os.path.join(path,f), *waves_headers)
    special_block = get_twd_block(os.path.join(path,f), *special_features_headers)
    if len(wave_block) == 0 and len(special_block) == 0:        
        no_waves_block = get_twd_block(os.path.join(path,f), *no_waves_headers)
    else:
        no_waves_block = []

    wave_blocks_by_time[time].extend([block for block in [wave_block, special_block,
                                                          no_waves_block] if len(block) > 0])


# Find waves in each corresponding block in each TWD
waves_by_time = {time : [] for time in filenames_by_date.keys()}

# global_manual_waves is currently just the Atlantic
#for wave in global_manual_waves:
#    waves_by_time[wave[0]].append(TwdWave(*wave))

metrics = {'sorted_waves' : 0, 'in_progress_waves' : 0, 'problem_waves' : 0, 'pairs_range_count' : 0, 'pairs_sufficient_count' : 0,
           'lat_range_count' : 0, 'lat_vector_count' : 0, 'three_locations_count' : 0, 'just_lon_count' : 0, 'two_lat_one_lon_count' : 0,
           'one_lat_vec_one_lon' : 0, 'one_lat_one_lon_count' : 0}
word_counts={}


### various string searches that get checked for each wave paragraph
exclusion_pairs = [('short[ -]*', 'b'), ('[ ]*field', 'a'), ('[ ]*event', 'a'),
                   ('[ ]*height', 'a'), ('s', 'a'), ('long[ -]*', 'b'), ('[ ]*runup', 'a'),
                   ('micro', 'b'), ('trak', 'a'), ('[ ]*of[ ]*african[ ]*dust[ ]*', 'a'),
                   ('[ ]*conditions', 'a'), ('[ ]*model', 'a')]

# Trying out including Mexico as an option for the EPAC specifically (still will mix up gulf of mexico and mexico)
location_re = '(' + '|'.join([s.lower().replace(' ', '[ ]*') for s in global_loc_keyword_map.keys()]) + ')'
direction_re = '(([, -]|^)(' + '|'.join([s.lower().replace(' ', '[ ]*') for s in global_direction_map.keys() - [None]]) + '))'
c_re = '[0-9]+[\.]?[0-9]*'
missing_decimal_re = '[2-9][0-9][0-9]'
start_re = 'from[ ]*'
end_re = '(to|near)[ ]*'
# lat/lon pairs
coord_pair_re = '{}(?:n|s)[/ ]*{}w'.format(c_re, c_re)
coord_pair_start_re = '{}\D*{}'.format(start_re, coord_pair_re)
coord_pair_end_re = '{}\D*{}'.format(end_re, coord_pair_re)
coord_pair_typo_re = '{}n/?{}[a-z ,]'.format(c_re, c_re)
# exclude "st." since that is used in island names,
# as well as 't.s', 't.d', 'u.s'
sentence_re = '(?<! st|[s|t|u|n]\.[s|d])\.[\D ]+'
# lat to lat ranges
lat_range_re = '(?<!southward)[ ]*(?:from|between)\D*{}(?:n|s)?\D*(?:-|to|and)[ ]*\D*{}(?:n|s)'.format(c_re, c_re)
# don't want "XXw from YYn to ZZw" or vice versa
lat_range_typo_re_1 = '{}[ ]*w[ ](?<!southward)[ ]*(?:from|between)\D*{}w'.format(c_re,c_re)
lat_range_typo_re_2 = '{}[ ]*w[ ](?<!southward)[ ]*(?:from|between)\D*{}(?:n|s)\D*(?:-|to|and)[ ]*\D*{}w'.format(c_re,c_re,c_re)
# want "s of", "south of", " southward", 'n of', 'north of', 'northward'
lat_vector_re = '(?: s|south|southward|equatorward|n|north|northward)[ ]*(?:of|from)[ ]*{}(?:n|s)|[, ]+{}(?:n|s)[ ]*(?:southward|south|equatorward|n|north|northward)'.format(c_re, c_re)
# don't want "s/n of XXw"
lat_vector_typo_re = '(?: s|south|n|north)[ ]*(?:of|from)[ ]*{}w'.format(c_re)
# also don't want "from XXn to XXw" or "from XXw to XXn", NEED TO ADD THIS ONE IN
# Might be hard to confirm whether it's supposed to be lats or lons; maybe have a max value for
# it to be considered a mistyped lat?
lon_coord_re = '(?<![n0-9]){}/?{}w'.format(c_re, c_re)
lat_coord_re = '{}?/?{}n'.format(c_re,c_re)
equator_re = '(?:the)?[ ]*equator(?!ial|ward)'
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
                   'shower', 'thunderstorm', 'rainfall', 'trough', 'clouds', 'fresh',
                   'moisture', 'cumulus',  'upper level', 'trof', 'precipitable', 'pw',
                   'convective', 'cloud']
reposition_strs = ['reposition', 'previously', 'moved back', 'dissipated', 'expected',
                   'was absorbed', 'has been asbored', 'has moved', 'dropped', 'will be', 'dispersed',
                   'has developed', 'replaced', 'was analyzed']


# save mentions of merge
merge_fields = ['date', 'paragraphs']
merge_rows = []


for time in filenames_by_date.keys():    
    blocks = wave_blocks_by_time[time]

    paragraphs = []
    waves = []
    time_metrics = {key : 0 for key in metrics.keys()}
    for block in blocks:
        # find each paragraph in a block
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

    # Only keep paragraphs where "wave" is mentioned and is not a component of various irrelevant strings
    # Also have some conditionals to eliminate other irrelevant paragraphs that have wave in them
    paragraphs = [p for p in paragraphs if exclude_by_string(p, 'wave', exclusion_pairs)]
    
    
    # Now deduce location of waves from these paragraphs    
    for p in paragraphs:
        if ('merge' in p and not 'emerge' in p) or 'coalesce' in p:
            merge_rows.append([time, p])
        # skip a few specific paragraphs that are problematic and have been accounted for in global_manual_waves
        # as needed
        skip = False
        for pair in global_skips:
            if time == pair[0] and pair[1] in p:
                skip = True
        if skip:
            continue
        
        # wave locations will be given different weights depending on how reliable we think they are
        # 1 = good, 8 = bad. 1's determine climo N/S extent, 2's receive climo N/S extent if they are
        # less than 50% of climo N/S, 3's receive climo N/S extent with their one lat coord as their northward end,
        # 4's receive climo N/S extent while ensuring their one lat coord is in those bounds,
        # 5's receive climo N/S extent always as they have no latitude data, 6's are potentially from
        # problematic (convection-related for instance) location descriptors and will follow
        # the same latitude assignment process as 4's but for multiple latitudes potentially,
        # 7's are the same idea but for 5's, 8 is a special designation for likely repeated waves
        # that is discussed further along in this script. Last element in wave list will be trust level.
        coord_pairs = []        
        lat_coords = []
        lon_coords = []
        
        # find sentence with 'wave' in it and prioritize searching there
        # also need other sentence bounds because we may need to search
        # sentences before and after        
        sentence_ends, _ = get_locs_and_matches(sentence_re, p)
        sentence_indices = [0, *[s[0] for s in sentence_ends], len(p)]
        sentences = [p[sentence_indices[i]:sentence_indices[i+1]]
                     for i in range(len(sentence_indices)-1)]
        wave_sentence = ''
        pre_wave_sentence = ''
        post_wave_sentence = ''
        wave_sentence_i = 0
        while wave_sentence == '' and wave_sentence_i < len(sentences):
            if 'wave' in sentences[wave_sentence_i]:
                wave_sentence = sentences[wave_sentence_i]
                pre_wave_sentence = sentences[max(0, wave_sentence_i-1)] if wave_sentence_i != 0 else ''
                post_wave_sentence = sentences[min(len(sentences)-1, wave_sentence_i+1)] if wave_sentence_i+1 != len(sentences) else ''
            wave_sentence_i = wave_sentence_i+1

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
                    elif pat == missing_decimal_re:
                        s_new = s_new.replace(typo, typo[:-1] + '.' + typo[-1])
                    elif pat in (lat_vector_typo_re, lat_range_typo_re_1, lat_range_typo_re_2):
                        s_new = s_new.replace(typo, typo[:-1]+'n')
                    elif pat == equator_re:
                        s_new = s_new.replace(typo, '00n')
            return s_new
        
        typo_patterns = [coord_pair_typo_re, missing_decimal_re, lat_vector_typo_re, equator_re,
                         lat_range_typo_re_1, lat_range_typo_re_2]

        wave_sentence = replace_patterns(wave_sentence, *typo_patterns)
        pre_wave_sentence = replace_patterns(pre_wave_sentence, *typo_patterns)
        post_wave_sentence = replace_patterns(post_wave_sentence, *typo_patterns)        
        sentences = [wave_sentence, pre_wave_sentence, post_wave_sentence]
        
        # check to see if terms are present in wave_sentence that describe locations of
        # other things (like convection or past / future wave locations)
        issue_strs = [*convection_strs, *reposition_strs]
        

        # go through the wave sentence, then the sentences before and after it,
        # searching for enough lat/lon info to use
        found_enough = False
        lat_coords = []
        lon_coords = []
        coord_pairs = []
        verbal_locations = []
        lat_coord_locs = {}
        coord_pair_locs = {}
        lon_coord_locs = {}
        
        for sentence in sentences:
            problem, problem_str = check_for_problem_strs(sentence, issue_strs)
            # skip problem sentences and don't bother reading more if we have enough info
            if problem:
                continue
            if found_enough:
                break
            # first look for pairs of coordinates, which may be the ideal location denoter
            coord_pair_locs_temp, coord_pair_temp = get_locs_and_matches(coord_pair_re, sentence)
            coord_pairs = coord_pairs + coord_pair_temp
            coord_pair_locs[sentence] = coord_pair_locs_temp
            if len(coord_pairs) > 1 and not problem:
                # found multiple coordinates, most likely these each refer to the wave axis,
                # unlikely to find any better location descriptors
                range_starts = re.findall(coord_pair_start_re, sentence)
                range_ends = re.findall(coord_pair_end_re, sentence)
                if len(range_starts) > 0 and len(range_ends) > 0:
                    # definitive wave locations should read 'from ... to ...', and we will
                    # take these and run
                    waves.append(TwdWave(time, coord_pairs, lat_coords, lon_coords, verbal_locations, 1, p))
                    time_metrics['sorted_waves'] = time_metrics['sorted_waves'] + 1
                    time_metrics['pairs_range_count'] = time_metrics['pairs_range_count'] + 1
                    found_enough=True
                    continue
                elif len(coord_pair_locs_temp) > 0:
                    # check if there's any more specific location data to the right of the
                    # last coord pair in this sentence, otherwise assume we've got enough to go on
                    max_end = max(pair[1] for pair in coord_pair_locs_temp)
                    if len(re.findall(not_kt_re, sentence[max_end:])) == 0:
                        waves.append(TwdWave(time, coord_pairs, lat_coords, lon_coords, verbal_locations, 2, p))
                        time_metrics['sorted_waves'] = time_metrics['sorted_waves'] + 1
                        time_metrics['pairs_sufficient_count'] = time_metrics['pairs_sufficient_count'] + 1
                        found_enough=True
                        continue
            # now begin looking for all coordinates listed, which we can use as long as there
            # is not a 'to ...', 'through ...' after the last coordinate
            # ranges are two coords separated by to or -
            lat_range_locs_temp, lat_ranges = get_locs_and_matches(lat_range_re, sentence)
            # the vectors are the ones where it mentions south of a certain latitude
            lat_vector_locs_temp, lat_vectors = get_locs_and_matches(lat_vector_re, sentence)
            lat_coords = lat_coords + lat_ranges + lat_vectors
            lat_coord_locs[sentence] = lat_range_locs_temp + lat_vector_locs_temp
            
            lon_coord_locs_temp, lon_coord_temp = get_locs_and_matches(lon_coord_re, sentence)
            lon_coords = lon_coords + lon_coord_temp
            lon_coord_locs[sentence] = lon_coord_locs_temp
            
            if ((len(lat_coords) > 0 and len(lon_coords) > 0) or (len(lat_coords) > 0 and len(coord_pairs) > 0)) and \
               not problem and (len(lat_range_locs_temp) > 0 or len(lat_vector_locs_temp) > 0 or len(lon_coord_locs_temp) > 0):            
                # we have some location data and are happy to stop if there's no more specific location
                # data to the right (as currently denoted by the letters 'to' ending a word and not being followed
                # by a kt value)
                max_end = max(pair[1] for pair in [*lat_range_locs_temp, *lat_vector_locs_temp, *lon_coord_locs_temp])
                if len(re.findall(not_kt_re, sentence[max_end:])) == 0:
                    # trust lat ranges more than lat vectors
                    if len(lat_ranges) > 0:
                        waves.append(TwdWave(time, coord_pairs, lat_coords, lon_coords, verbal_locations, 1, p))
                        time_metrics['lat_range_count'] = time_metrics['lat_range_count'] + 1
                    elif len(lat_vectors) > 0:
                        waves.append(TwdWave(time, coord_pairs, lat_coords, lon_coords, verbal_locations, 3, p))
                        time_metrics['lat_vector_count'] = time_metrics['lat_vector_count'] + 1
                    time_metrics['sorted_waves'] = time_metrics['sorted_waves'] + 1
                    found_enough=True
                    continue
        wave_problem, wave_problem_str = check_for_problem_strs(wave_sentence, issue_strs)

        # stop searching if we have enough
        if found_enough:
            continue

        # Add extra lat coords that didn't get picked up in the original search
        found_new_lat=False
        for sentence in sentences:            
            problem, problem_str = check_for_problem_strs(sentence, issue_strs)
            if problem:
                continue
            raw_lat_locs, raw_lats = get_locs_and_matches(lat_coord_re, sentence)
            for loc, lat in zip(raw_lat_locs, raw_lats):
                min_loc = loc[0]
                max_loc = loc[1]
                repeat = False
                for check_loc in lat_coord_locs[sentence]+coord_pair_locs[sentence]:
                    if min_loc >= check_loc[0] and max_loc <= check_loc[1]:
                        repeat = True
                        break
                if not repeat:
                    found_new_lat=True
                    # if this is actually a southward coordinate, assign
                    # only do this if no other lat coords have been found yet (since any already-found vectors
                    # should have this instance of 'southward' in them already). Mostly this is to avoid
                    # some typo instances of mixing up N and W
                    if 'southward' in sentence and len(lat_coords) == 0:
                        lat_coords.append(lat + ' southward'), lat_coord_locs[sentence].append(loc)
                        # check if we're done (modified from above lat vector check)
                        if ((len(lat_coords) > 0 and len(lon_coords) > 0) or (len(lat_coords) > 0 and len(coord_pairs) > 0)) and not problem:            
                            # we have some location data and are happy to stop if there's no more specific location
                            # data to the right (as currently denoted by the letters 'to' ending a word and not being followed
                            # by a kt value)
                            if len(re.findall(not_kt_re, sentence[max_loc:])) == 0:
                                # trust lat ranges more than lat vectors
                                waves.append(TwdWave(time, coord_pairs, lat_coords, lon_coords, verbal_locations, 3, p))
                                time_metrics['lat_vector_count'] = time_metrics['lat_vector_count'] + 1
                                time_metrics['sorted_waves'] = time_metrics['sorted_waves'] + 1
                                found_enough=True
                                continue

                        
                    else:
                        lat_coords.append(lat), lat_coord_locs[sentence].append(loc)
                        
        # code bit for identifying location names we might have to check
        for sentence in sentences:
            words = sentence.replace(',', '').replace('.', '').split(" ")
            for s in words:
                if s in word_counts:
                    word_counts[s] = word_counts[s]+1
                else:
                    # checking if string is just letters
                    if s.isalpha():
                        word_counts[s] = 1
        if found_enough:
            continue
        # Now we get into the real grit: identifying wave locations using landmarks        
        for sentence in sentences:
            problem, problem_str = check_for_problem_strs(sentence, issue_strs)
            if problem:
                continue
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
                    elif not ((verbal_locations_groups[i][1] == 'venezuela' and 'maracaibo' in sentence)
                               or (verbal_locations_groups[i][1] == 'venezuela' and 'guajira' in sentence)
                               or (verbal_locations_groups[i][1] == 'colombia' and 'guajira' in sentence)):
                        verbal_locations.append(verbal_locations_groups[i])
            
        # use 3 total locations (including at least one non-vector of lat and lon each) as a good cutoff point for enough location info
        if len(verbal_locations) + len(coord_pairs) + len(lat_coords) + len(lon_coords) > 2 and\
           len(verbal_locations) + len(coord_pairs) + len(lon_coords) > 0 and\
           len(verbal_locations) + len(coord_pairs) + len(lat_coords) > 0 and\
           not all([len(re.findall('of|from|south', lat)) > 0 for lat in lat_coords]):
            time_metrics['sorted_waves'] = time_metrics['sorted_waves'] + 1
            time_metrics['three_locations_count'] = time_metrics['three_locations_count'] + 1
            waves.append(TwdWave(time, coord_pairs, lat_coords, lon_coords, verbal_locations, 2, p))
            continue
        # if a wave only has longitude values, will end up assigning it climatological n/s extent
        if len(verbal_locations) + len(coord_pairs) + len(lat_coords) == 0 and len(lon_coords) > 0:
            time_metrics['sorted_waves'] = time_metrics['sorted_waves'] + 1
            time_metrics['just_lon_count'] = time_metrics['just_lon_count'] + 1
            waves.append(TwdWave(time, coord_pairs, lat_coords, lon_coords, verbal_locations, 5, p))
            continue
        # if a wave has two or more lat points and at least 1 lon point including verbal locations,
        # assign it medium trust. Slight redundancy with the 3 total locations step above
        if len(verbal_locations) + len(coord_pairs) + len(lat_coords) > 1 and \
           len(verbal_locations) + len(coord_pairs) + len(lon_coords) > 0:
            time_metrics['sorted_waves'] = time_metrics['sorted_waves'] + 1
            time_metrics['two_lat_one_lon_count'] = time_metrics['two_lat_one_lon_count'] + 1
            waves.append(TwdWave(time, coord_pairs, lat_coords, lon_coords, verbal_locations, 2, p))
            continue
        # if a wave has one lat point with one of the southward descriptions in it and a lon value,
        # assume climatological length starting with coord as the northernmost point
        if len(verbal_locations) + len(coord_pairs) + len(lon_coords) > 0 and \
           len(lat_coords) > 0 and len(re.findall('of|from|south', lat_coords[0])) > 0:
            time_metrics['sorted_waves'] = time_metrics['sorted_waves'] + 1
            time_metrics['one_lat_vec_one_lon'] = time_metrics['one_lat_vec_one_lon'] + 1
            waves.append(TwdWave(time, coord_pairs, lat_coords, lon_coords, verbal_locations, 3, p))
            continue
        # if a wave has one lat point and a lon value, assume climatological N/S extent while ensuring
        # the listed lat value is within those bounds
        # NOTE: This may have some issues since sometimes the verbal locations include southward terms
        # that are missed by this
        if len(verbal_locations) + len(coord_pairs) + len(lat_coords) > 0 and \
           len(verbal_locations) + len(coord_pairs) + len(lon_coords) > 0:
            time_metrics['sorted_waves'] = time_metrics['sorted_waves'] + 1
            time_metrics['one_lat_one_lon_count'] = time_metrics['one_lat_one_lon_count'] + 1
            waves.append(TwdWave(time, coord_pairs, lat_coords, lon_coords, verbal_locations, 4, p))
            continue

        # now to deal with waves that only have a north or have no locations at all
        if len(verbal_locations) + len(coord_pairs) + len(lon_coords) == 0:
            # get rid of paragraphs about dissipation or irrelevant stuff
            if check_for_problem_strs(pre_wave_sentence + wave_sentence + post_wave_sentence,
                                      ['discernible', 'dissipated', 'special', 'dispersed', 'absorbed', 'dropped',
                                       'no longer'])[0]:
                continue

            # resort to using longitude and latitude averages of all locations listed in the wave description
            # if still searching (which means this wave either doesn't have any location data or it's
            # tripping the problematic string flag). These should be the least accurate wave locations of them all
            # Note that all the searches here are a condensed version of the process above
            for sentence in sentences:
                coord_pair_locs_temp, coord_pair_temp = get_locs_and_matches(coord_pair_re, sentence)
                coord_pairs = coord_pairs + coord_pair_temp
                coord_pair_locs[sentence] = coord_pair_locs_temp
                
                lat_range_locs_temp, lat_ranges = get_locs_and_matches(lat_range_re, sentence)
                lat_vector_locs_temp, lat_vectors = get_locs_and_matches(lat_vector_re, sentence)
                lat_coords = lat_coords + lat_ranges + lat_vectors
                lat_coord_locs[sentence] = lat_range_locs_temp + lat_vector_locs_temp
                raw_lat_locs, raw_lats = get_locs_and_matches(lat_coord_re, sentence)                
                for loc, lat in zip(raw_lat_locs, raw_lats):
                    min_loc = loc[0]
                    max_loc = loc[1]
                    repeat = False
                    for check_loc in lat_coord_locs[sentence]+coord_pair_locs[sentence]:
                        if min_loc >= check_loc[0] and max_loc <= check_loc[1]:
                            repeat = True
                            break
                    if not repeat:
                        lat_coords.append(lat), lat_coord_locs[sentence].append(loc)
                
                lon_coord_locs_temp, lon_coord_temp = get_locs_and_matches(lon_coord_re, sentence)
                lon_coords = lon_coords + lon_coord_temp
                lon_coord_locs[sentence] = lon_coord_locs_temp
                
                verbal_location_locs, verbal_locations_temp, verbal_locations_groups = get_locs_and_matches(geographic_re, sentence, [8,9])
                for i in range(len(verbal_locations_temp)):
                    if len(re.findall('of[ ]* the', verbal_locations_temp[i])) == 0:
                        verbal_locations.append(verbal_locations_groups[i])

            # now check for 1 or more lat/lon or at least 1 lon
            if len(verbal_locations) + len(coord_pairs) + len(lat_coords) > 0 and \
               len(verbal_locations) + len(coord_pairs) + len(lon_coords) > 0:
                waves.append(TwdWave(time, coord_pairs, lat_coords, lon_coords, verbal_locations, 6, p))
                time_metrics['sorted_waves'] = time_metrics['sorted_waves'] + 1
                continue
            
            # check for at least 1 lon
            if len(verbal_locations) + len(coord_pairs) + len(lon_coords) > 0:
                waves.append(TwdWave(time, coord_pairs, lat_coords, lon_coords, verbal_locations, 7, p))
                time_metrics['sorted_waves'] = time_metrics['sorted_waves'] + 1
                continue

            # The remaining wave paragraphs must either be irrelevant or have typos, so need to manually
            # look for their locations. If I actually get locations from the graphics,
            # these will be some of the most accurate
            if time not in [w[0] for w in global_manual_waves]:
                continue
            continue

    # eliminate duplicate waves (which happens when a wave is mentioned as already having been
    # covered in the special features section)
    for wave in waves:
        wave_lon = wave.get_center()[1]        
        for wave2 in waves:
            wave2_lon = wave2.get_center()[1]
            if np.abs(wave_lon-wave2_lon) < 1 and wave2.trust > wave.trust:
                waves.remove(wave2)
    waves_by_time[time] = waves + waves_by_time[time]
    for key in metrics:
        metrics[key] = metrics[key] + time_metrics[key]
    
# save merge waves
with open(os.path.join(topdir, 'merge_waves.csv'), 'w') as f:
    write = csv.writer(f)
    write.writerow(merge_fields)
    write.writerows(merge_rows)
        

print(metrics)

# Save raw waves (no lat assignments done yet for non trust==1 waves)
with open(os.path.join(topdir, 'raw_waves_epac'), 'wb') as f:
    pickle.dump(waves_by_time, f, protocol=pickle.HIGHEST_PROTOCOL)
#END COMMENT BLOCK FOR RAW WAVE CALCULATIONS

with open(os.path.join(topdir, 'raw_waves_epac'), 'rb') as f:
    waves_by_time = pickle.load(f)


# sort waves into days of the hurricane season
calendar_days_climo_waves = {}
for time in waves_by_time.keys():
    if int(time[4:6]) in (5,6,7,8,9,10,11):
        if time[4:8] not in calendar_days_climo_waves:
            calendar_days_climo_waves[time[4:8]] = []
        for wave in waves_by_time[time]:
            if wave.trust == 1:
                calendar_days_climo_waves[time[4:8]].append(wave)

'''
# uncomment this block to rerun lat climatology calculations


# get latitude climatology
lons = np.array(range(0, 122, 1))
days = sorted(calendar_days_climo_waves.keys())

climo_lat_sums = xr.DataArray(np.zeros([len(days), len(lons), 3]),
                          coords=dict(day=days, lon=lons, range_end=['bottom', 'top', 'count']))
climo_lats_raw = xr.DataArray(np.full([len(days), len(lons), 2], np.nan),
                          coords=dict(day=days, lon=lons, range_end=['bottom', 'top']))
climo_lats = xr.DataArray(np.full([len(days), len(lons), 2], np.nan),
                          coords=dict(day=days, lon=lons, range_end=['bottom', 'top']))

# add sums
#for day_i in range(len(days)):
def add_sums(day_i):
    day = days[day_i]
    temp_array = climo_lat_sums[day_i].copy(deep=True)
    temp_return_array = climo_lats_raw[day_i].copy(deep=True)
    
    for wave in calendar_days_climo_waves[day]:
        lon = np.nanmean([wave.extent[0][1], wave.extent[1][1]])
        min_lat, max_lat = wave.extent[0][0], wave.extent[1][0]
        lon_i = np.abs(np.array(lons) - lon).argmin()
        #climo_lat_sums[day_i, lon_i, 0] = climo_lat_sums[day_i, lon_i, 0] + min_lat
        #climo_lat_sums[day_i, lon_i, 1] = climo_lat_sums[day_i, lon_i, 1] + max_lat
        #climo_lat_sums[day_i, lon_i, 2] = climo_lat_sums[day_i, lon_i, 2] + 1
        temp_array[lon_i, 0] = climo_lat_sums[day_i, lon_i, 0] + min_lat
        temp_array[lon_i, 1] = climo_lat_sums[day_i, lon_i, 1] + max_lat
        temp_array[lon_i, 2] = climo_lat_sums[day_i, lon_i, 2] + 1

    # calculate mean for a longitude for a day
    for lon_i in range(len(lons)):
        if temp_array[lon_i, 2] != 0:
            temp_return_array[lon_i, 0] = temp_array[lon_i, 0]/temp_array[lon_i, 2]
            temp_return_array[lon_i, 1] = temp_array[lon_i, 1]/temp_array[lon_i, 2]
            
    return temp_return_array

climo_lats_temp = Parallel(n_jobs=-1)(delayed(add_sums)(day_i)
                                      for day_i in range(len(days)))

for day_i in range(len(days)):
    climo_lats_raw[day_i] = climo_lats_temp[day_i]

# calculate weighted average latitude climatology for each longitude and time
# weights here are the distance from one index to another in time, longitude index space,
# but with greater weighting towards time
np.seterr(divide="ignore")
for day_i in range(len(days)):
    for lon_i in range(len(lons)):
        weights = np.full([len(days), len(lons)], 1)
        rows, cols = np.indices(weights.shape, sparse=True)
        weights = 1/np.sqrt((abs(rows-day_i))**3 + (abs(cols-lon_i))**3.5)
        # get rid of infinity from 1/0
        weights[day_i, lon_i] = 1.5
        # ignore nans in calculations
        weights[np.isnan(climo_lats_raw[:,:,0])] = 0
        #print(weights, day_i, lon_i)        
        climo_lats_temp = climo_lats_raw.copy(deep=True).fillna(0)
        #print(climo_lats_temp)
        climo_lats[day_i, lon_i, 0] = np.average(climo_lats_temp[:, :, 0], weights=weights)        
        climo_lats[day_i, lon_i, 1] = np.average(climo_lats_temp[:, :, 1], weights=weights)


with open(os.path.join(topdir, 'wave_lat_climo_epac'), 'wb') as f:
    pickle.dump(climo_lats, f, protocol=pickle.HIGHEST_PROTOCOL)
#END COMMENT BLOCK FOR CLIMO LAT CALCULATIONS

'''
with open(os.path.join(topdir, 'wave_lat_climo_epac'), 'rb') as f:
    climo_lats = pickle.load(f)

# assign lat values to waves that need it
# comment block below for latitude reassignment and final wave creation

for time in waves_by_time.keys():
    new_waves = []
    #print(time)
    for wave in waves_by_time[time]:
        if int(time[4:6]) in (5,6,7,8,9,10,11):
            wave.assign_lat(climo_lats)
        new_waves.append(wave)
    
    waves_by_time[time] = new_waves
    
# catch if waves have same description as previous timestep, indicating
# a likely failure to relocate them by the TWD author
# assign them a trust of 8 if so (note that this assignment is past the stage of all other wave
# tinkering and thus lat climatology is minorly affected)

for time_i in range(1, len(waves_by_time.keys())):
    cur_time = list(waves_by_time.keys())[time_i]
    pre_time = list(waves_by_time.keys())[time_i-1]
    new_waves = []
    cur_waves = waves_by_time[cur_time]
    pre_waves = waves_by_time[pre_time]    
    for cur_wave in cur_waves:
        #print(str(cur_wave)[16:])
        for pre_wave in pre_waves:
            if str(cur_wave)[16:] == str(pre_wave)[16:]:
                #print(cur_wave, pre_wave)
                cur_wave.trust = 8
        new_waves.append(cur_wave)
    waves_by_time[time] = new_waves

    
with open(os.path.join(topdir, 'final_waves_epac'), 'wb') as f:
    pickle.dump(waves_by_time, f, protocol=pickle.HIGHEST_PROTOCOL)
#END COMMENT BLOCK FOR FINAL WAVES


# Done making wave files
with open(os.path.join(topdir, 'final_waves_epac'), 'rb') as f:
    waves_by_time = pickle.load(f)

# print any waves which have a single remaining latitude coordinate (which is not desired)    
for time in waves_by_time.keys():
    for wave in waves_by_time[time]:
        if wave.extent[0][0] == wave.extent[1][0]:
            wave.full_print()
    


