
'''
Assorted utility functions and constants
'''

import pandas as pd
import xarray as xr
import numpy as np
import os
from netCDF4 import Dataset
import re
import requests

from joblib import Parallel, delayed

from ml_waves_itcz_mt.constants import *



def get_coords_via_descriptor(loc, descriptor='central'):
    '''
    Get coordinates from a location name and a verbal descriptor (like central or north).

    loc : str, location name
    descriptor : str, from direction_map dict

    return: tuple of (float, float) of lat lon
    '''
    (box, center) = global_place_coords[global_loc_keyword_map[loc]]
    short = direction_map[descriptor]
    if short == 'c':
        return tuple(center)
    if short == 'n':
        return (box[1], center[1])
    if short == 'ne':
        return (box[1], box[3])
    if short == 'e':
        return (center[0], box[3])
    if short == 'se':
        return (box[0], box[3])
    if short == 's':
        return (box[0], center[1])
    if short == 'sw':
        return (box[0], box[2])
    if short == 'w':
        return (center[0], box[2])
    if short == 'nw':
        return (box[1], box[2])

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
    while contents[start_line][:8] != 'AL01' + str(year) and contents[start_line][:8] != 'EP01' + str(year) and contents[start_line][:8] != 'CP01' + str(year):
        start_line = start_line + 1
    if start_line >= len(contents):
        return []

    while contents[start_line].split()[check_col] != storm_identifier:
        start_line = start_line + 1
        
    # start actual data
    start_line = start_line + 1
    
    # get full data block
    end_line = start_line
    while end_line < len(contents) and contents[end_line][:2] not in ('AL', 'EP', 'CP'):
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
    while contents[start_line][:8] != 'AL01' + str(year) and contents[start_line][:8] != 'EP01' + str(year) and contents[start_line][:8] != 'CP01' + str(year):
        start_line = start_line + 1
    cur_line = start_line
    while cur_line < len(contents) and year in contents[cur_line][:8]:
        if contents[cur_line][:2] in ('AL', 'EP', 'CP'):
            storm_names.append(contents[cur_line].split()[1][:-1])
        cur_line = cur_line + 1
    return storm_names

def get_hurdat_annual_ids(file_name, year):
    '''
    Retrieve list of IDs names for a given year from hurdat (some storms share names)

    filename : str, HURDAT file name
    year : str

    return : list of str
    '''
    
    storm_ids = []
    
    with open(file_name, 'r') as f:
        contents = f.readlines()

    # find year section start, then start getting storm names
    start_line = 0
    while contents[start_line][:8] != 'AL01' + str(year) and contents[start_line][:8] != 'EP01' + str(year) and contents[start_line][:8] != 'CP01' + str(year):
        start_line = start_line + 1
    cur_line = start_line
    while cur_line < len(contents) and year in contents[cur_line][:8]:
        if contents[cur_line][:2] in ('AL', 'EP', 'CP'):
            storm_ids.append(contents[cur_line].split()[0][:-1])
        cur_line = cur_line + 1
    return storm_ids



def get_hurdat_formation_info(df):
    '''
    Get formation date and time from HURDAT block (as a pandas DataFrame) for a specific storm
    '''
 
    # find first line once genesis has actually occurred (TD rather than PTC)
    states = [state for state in df['state']]
    i = 0
    while states[i] not in ('TS', 'TD', 'SS', 'SD', 'HU'):
        i = i+1

    # returned in format ['YYYYMMDD', 'HHMM', '-XX.X'(W), 'XX.X'N]
    return df['date'][i], df['time'][i], '-' + df['lon'][i][:-1], df['lat'][i][:-1]


def get_hurdat_lat_lon(block, i):
    '''
    Get lat / lon info at index i in HURDAT block as floats
    '''
    lon, lat = block['lon'][i], block['lat'][i]
    if type(lon) == str and type(lat) == str:
        if 'W' in lon: lon = 0-float(lon[:-1])
        elif 'E' in lon: lon = float(lon[:-1])
        else: lon = float(lon)
        lat = block['lat'][i]
        if 'N' in lat: lat = float(lat[:-1])
        elif 'S' in lat: lat = 0-float(lat[:-1])
        else: lat = float(lat)
    else:
        lon = float(lon)
        lat = float(lat)
    return lat, lon
    
def get_file_times_for_era_fetch(date, time, hours_before, hours_after, interval):
    '''
    Get set of dates and times to fetch ERA files for within a range around
    a specified date and time.

    date : 'YYYYMMDD'
    time : 'HHMM'
    hours_before : int
    hours_after : int
    interval : int, hours spacing
    '''

    year = date[:4]
    month = date[4:6]
    day = date[6:]
    
    #month_day_counts = {'05' : 31, '06' : 30, '07' : 31, '08' : 31, '09' : 30, '10' : 31, '11' : 30, '12' : 31}

    # build list of all timesteps for the requested year    
    day_strs = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
                '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
                '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31']
    month_days = {'05' : day_strs, '06' : day_strs[:-1], '07' : day_strs, '08' : day_strs,
                  '09' : day_strs[:-1], '10' : day_strs, '11' : day_strs[:-1], '12' : day_strs}
    months = ['05', '06', '07', '08', '09', '10', '11', '12']
    
    times = []
    hour = 0
    while hour < 10:
        times.append('0' + str(hour) + ':00')
        hour = hour + interval
    while hour < 24:
        times.append(str(hour) + ':00')
        hour = hour + interval

    # where one timestep is [yyyy, mm, dd, hh:00]
    timesteps = [[year, m, d, t] for m in months for d in month_days[m] for t in times]
    
    # find the date and time requested
    i = 0
    while timesteps[i][0] != year or timesteps[i][1] != month or timesteps[i][2] != day or \
          timesteps[i][3][:2] + timesteps[i][3][3:] != time:
        i = i + 1
        
    mid = i
    end = i
    start = i
    
    # iterate forward and backward to get start and end indices
    for i in range(int(hours_after/interval)+1):
        end = mid + i
    
    end = end + 1
    for i in range(int(hours_before/interval)+1):
        start = mid - i

    return timesteps[start:end]


def make_directory(*folders):
    # create any directories that don't exist in the passed, ordered hierarchy
    base = ''
    for folder in folders:
        base = os.path.join(base, folder)
        if not os.path.isdir(base):
            try:
                os.mkdir(base)
            except FileExistsError:
                pass
    
    return base


def get_twd_block(file_name, *header_strings):
    '''
    Retrieve block of text with specific header options from Tropical Weather
    Discussion file

    filename : str, TWD file name
    *header_strings: str, possible header starting strings
    
    returns : list of str
    '''
        
    with open(file_name, 'r') as f:
        contents = f.readlines()

    # find Tropical wave section using ... (or .. if there's a typo),
    # which also gets used to replace commas in older files it seems.
    # To work around this, make sure preceding line is blank (only spaces and new line character)
    # Also, until July 06 18z 2004 they started the section with TROPICAL WAVES/ITCZ...
    start_line = 1

    '''
    # this block is problematic because some files have "...tropical" in them and a blank line
    # before due to inconsistent line breaks
    while start_line < len(contents) and (all([h.lower() not in contents[start_line].lower()
                                               for h in header_strings])
                                          or len(contents[start_line-1].lower()) >
                                          contents[start_line-1].lower().count(' ') +
                                          contents[start_line-1].lower().count('\n')):

        start_line = start_line + 1
    '''
    while start_line < len(contents) and (all([not contents[start_line].lower().startswith(h.lower())
                                               for h in header_strings])
                                          or any([contents[start_line].lower().startswith(h)
                                                  for h in ['...tropical s', '...tropical d']])
                                          or len(contents[start_line-1].lower()) >
                                          contents[start_line-1].lower().count(' ') +
                                          contents[start_line-1].lower().count('\n')):

        start_line = start_line + 1

    # skip actual header line
    start_line = start_line + 1

    '''
    # start actual data
    while start_line < len(contents) and \
          len(contents[start_line]) > contents[start_line].lower().count(' ') + \
          contents[start_line].lower().count('\n'):
        start_line = start_line + 1
    '''
    
    # get full data block until next section, which should start with '...' and have a blank line before
    # EDIT: does not always start with '...', so now also checking to see if line starts with
    # "gulf of mexico..." or "monsoon trough/itcz..." or a couple other strings used in hurricane season 2004
    # (Note that any waves that may be listed pre-June 1st 2004 are going to be excluded so not adding in all
    # the weird starting strings there)
    starting_strings = ["...", " ...", "  ...", "gulf of mexico...", "monsoon trough/itcz...", "subtropical atlantic...",
                        "central atlantic...", "e united states and the w atlantic...",
                        "northern gulf of mexico...", "northern gulf of mexico and far western atlantic...",
                        "central and east atlantic...", "gulf of mexico and nw caribbean...",
                        "the rest of the caribbean sea and the subtropical atlantic...",
                        "rest of the caribbean sea and subtropical atlantic...",
                        "itcz...", "the itcz...", "the caribbean sea...", 
                        "the caribbean sea and parts of the southwest north atlantic ocean...",
                        "the atlantic ocean...", "tropical atlantic...", "...intertropical",
                        "intertropical convergence zone/monsoon trough...", "...discussion...", #EPAC
                        "offshore waters within", "remainder of the area..."] #EPAC
    end_line = start_line
    while  end_line < len(contents) and (all(contents[end_line].lower()[:len(s)] != s
                                             for s in starting_strings) or \
                                         len(contents[end_line-1].lower()) >
                                         contents[end_line-1].lower().count(' ') +
                                         contents[end_line-1].lower().count('\n')):
        end_line = end_line+1
        
    end_line = end_line - 1

    '''
    if "...tropical" in header_strings and file_name in ('/home/will/research/ml_waves/TWDs/archive/text/TWDAT/2010/TWDAT.201008040005.txt',
                                                         '/home/will/research/ml_waves/TWDs/archive/text/TWDAT/2010/TWDAT.201008031804.txt'):# and not start_line < len(contents):
        print(file_name, start_line, end_line, contents[start_line:end_line])
    '''
    if start_line < len(contents):
        return contents[start_line:end_line]
    else:
        return []


def closest_time_str(in_str, interval='000000000600'):
    '''
    Return input date/time string rounded to nearest interval,
    where interval is one of years, months, days, hours, minutes.
    Note leap years aren't accounted for right now. Also skipping
    everything that isn't years or hours for now.

    in_str : str, 'YYYYMMDDHHMM'
    interval : str, 'YYYYMMDDHHMM'

    return : str, 'YYYYMMDDHHMM'
    '''

    # lop off .txt if it's present at the end of the in_str
    # (this is the case for all TWD files beginning in March 2006)
    txt_index = in_str.find('.txt')
    if txt_index != -1:
        in_str = in_str[:txt_index]
    
    year_indices = (0, 4)
    month_indices = (4, 6)
    day_indices = (6, 8)
    hour_indices = (8, 10)
    minute_indices = (10, 12)

    # for figuring out which part of the string we're in and efficiently
    # accessing the associated index range
    index_dict = {0 : year_indices, 1 : year_indices, 2 : year_indices, 3 : year_indices,
                  4 : month_indices, 5 : month_indices, 6 : day_indices, 7 : day_indices,
                  8 : hour_indices, 9 : hour_indices, 10 : minute_indices, 11 : minute_indices}
    

    # go left to right for checking rounding    
    current_digit = 0
    while current_digit < len(in_str):
        if len(in_str) > 12:
            print(in_str)
        current_indices = index_dict[current_digit]
        interval_var = interval[current_indices[0] : current_indices[1]]
        in_str_var =  in_str[current_indices[0] : current_indices[1]]

        if interval_var == '0' * len(interval_var):
            # nothing to round here (at least not yet)
            current_digit = current_indices[1]
        else:
            # Rounding required. Account for:
            # 1. Rounding for each date / time type (years, months, days, hours, minutes)
            #    must obey calendar rules and use data from further right in the string
            #    to decide what closest value is
            # 2. After rounding, reset values to the right to 0s (for hours or minutes),
            #    '01' (for days or months)
            current_digit = current_indices[0]
            if current_digit == 0:
                # year rounding
                in_year_int = int(in_str_var)
                interval_year = int(interval_var)
                min_possible_year = in_year_int - in_year_int%interval_year
                max_possible_year = in_year_int + interval_year-in_year_int%interval_year
                datestr_end = '01010000'
                min_datestr = str(min_possible_year) + datestr_end
                max_datestr = str(max_possible_year) + datestr_end
                
                #if interval_year%2 != 0 and in_year_int == int((max_possible_year+min_possible_year)/2):
                if interval_year%2 != 0 and in_year_int == min_possible_year+int(interval_year/2):
                    # odd intervals where we're in the center year make things more interesting
                    in_str_month_int = int(in_str[month_indices[0]:month_indices[1]])
                    if in_str_month_int == 7:
                        in_str_day_int = int(in_str[day_indices[0]:day_indices[1]])
                        # leap years not accounted for here (in non leap years center day is July 2nd)
                        if in_str_day_int == 2:
                            in_str_hour_int = int(in_str[hour_indices[0]:hour_indices[1]])
                            if in_str_hour_int < 12:
                                out_str = min_datestr
                            else:
                                out_str = max_datestr
                        else:
                            if in_str_day_int < 2:
                                out_str = min_datestr
                            else:
                                out_str = max_datestr
                    else:
                        if in_str_month_int < 7:
                            out_str = min_datestr
                        else:
                            out_str = max_datestr
                        
                else:
                    # closest year interval will be one with least difference in year digit
                    # round up if tied (equal distance of year digits)
                    if in_year_int - min_possible_year < max_possible_year-in_year_int:
                        out_str = min_datestr
                    else:
                        out_str = max_datestr

            elif current_digit == 4:
                # month rounding
                # note this is only written to support periods contained within a single year
                # (i.e. 3 month periods work but not 5 months)
                # also need to use 0-11 as months
                # skipping for now
                '''
                in_month_int = int(in_str_var)-1
                interval_month = int(interval_var)
                min_possible_month = in_month_int - in_month_int%interval_month
                max_possible_month = in_month_int + interval_month - in_month_int%interval_month
                datestr_end = '01010000'
                min_datestr = str(min_possible_month) + datestr_end
                max_datestr = str(max_possible_month) + datestr_end
                '''
                True

            elif current_digit == 6:
                # day rounding
                # skip for now
                True

            elif current_digit == 8:
                # hour rounding
                # only support factors of 24
                in_hour_int   = int(in_str_var)
                interval_hour = int(interval_var)
                # make sure to go to 00 on the next day if needed
                '''
                if in_hour_int >= 24-interval_hour:
                    min_possible_hour = 24-interval_hour
                    max_possible_hour = 0
                else:
                    min_possible_hour = in_hour_int - in_hour_int%interval_hour
                    max_possible_hour = in_hour_int + interval_hour-in_hour_int%interval_hour
                '''
                datestr_end = '00'
                min_possible_hour = in_hour_int - in_hour_int%interval_hour
                max_possible_hour = in_hour_int + interval_hour-in_hour_int%interval_hour
                #if in_hour_int%interval_hour >= 2 and in_hour_int%interval_hour < 5:
                    #print(in_str, in_hour_int%interval_hour)
                    
                # only possible date change issues should be if within
                # one half interval hours of the end of the day,
                # and extra check beyond just the hours digit is only needed
                # if hour interval is odd (like the year interval check above)
                in_str_minute_int = int(in_str[minute_indices[0]:minute_indices[1]])
                if in_hour_int >= min_possible_hour + int(interval_hour/2) and \
                   max_possible_hour==24 and (interval_hour%2 == 0 or in_str_minute_int >= 30):
                    # go to next day, which could need a new month and year as well
                    out_str = '00' + datestr_end
                    new_day = int(in_str[day_indices[0]:day_indices[1]])+1
                    if new_day > int(global_month_days[in_str[month_indices[0]:month_indices[1]]][-1]):
                        # have to change month
                        out_str = '01' + out_str
                        new_month = int(in_str[month_indices[0]:month_indices[1]]) + 1
                        if new_month > 12:
                            out_str = '01' + out_str
                            new_year = int(in_str[year_indices[0]:year_indices[1]]) + 1
                            out_str = str(new_year) + out_str
                        else:
                            new_month_str = str(new_month) if new_month > 9 else '0' + str(new_month)
                            out_str = in_str[:year_indices[1]] + new_month_str + out_str
                        
                    else:
                        new_day_str = str(new_day) if new_day > 9 else '0' + str(new_day)
                        out_str = in_str[:month_indices[1]] + new_day_str + out_str
                else:
                    min_possible_end_str = str(min_possible_hour) + datestr_end
                    max_possible_end_str = str(max_possible_hour) + datestr_end \
                        if max_possible_hour != 24 else '00' + datestr_end
                    if min_possible_hour < 10: min_possible_end_str = '0' + min_possible_end_str
                    if max_possible_hour < 10: max_possible_end_str = '0' + max_possible_end_str
                    if in_hour_int - min_possible_hour < max_possible_hour - in_hour_int:
                        out_str = in_str[:hour_indices[0]] + min_possible_end_str
                    else:
                        out_str = in_str[:hour_indices[0]] + max_possible_end_str
            current_digit = 12
    
                        
    return out_str



def exclude_by_string(p, match_str, exclusion_pairs):
    '''
    Takes an input string p and checks if an instance of match_str in p
    is not also an instance of any of the strings from exclude_strs.
    This could probably just be rewritten entirely as regular expressions.

    p : str, the string to search in
    match_str : str, the string we want to verify exists
    exclusion_pairs : list of tuples of (str, 'a' or 'b'), strings we don't want
                   and whether they're after or before our match_str

    return : bool, False if the only instances of match_str are associated with
             strings from exclude_strs or no instance of match_str is found.
             Otherwise return True
    '''
    if match_str not in p:
        return False
    else:
        match_str_locs = [_.start() for _ in re.finditer(match_str, p)]
        for loc in match_str_locs:
            found_good_match = True
            for (exclude, pos) in exclusion_pairs:
                if pos == 'a':
                    # check if match string is immediately followed by a bad string
                    try:
                        exclude_loc = list(re.finditer(exclude, p[loc+len(match_str):]))[0].start()
                    except IndexError:
                        # didn't find this exclude string
                        continue
                    if exclude_loc == 0:
                        found_good_match = False
                        break
                else:
                    # check if match string is immediately preceded by a bad string
                    try:
                        exclude_loc = list(re.finditer(exclude, p[:loc]))[-1].end()
                    except IndexError:
                        # didn't find this exclude string                        
                        continue
                    if exclude_loc == len(p[:loc]):
                        found_good_match = False
                        break
                    
            if found_good_match:
                return True
    return False
    

def get_locs_and_matches(pat, string, groups=[]):
    '''
    Uses regex to get instances of pattern out of string.

    pat : str, pattern to search for
    string : str, sentence to search in
    groups : list of int, groups to return also

    return : list of tuples of locations, list of tuples of strings, (optional) list of list of strings
    '''
    iter = re.finditer(pat, string)
    spans = [m.span() for m in iter]
    matches = [string[span[0]:span[1]] for span in spans]
    if len(groups) > 0:        
        return spans, matches, [[m.group(g) for g in groups] for m in re.finditer(pat, string)]
    return spans, matches
    
    
def check_for_problem_strs(string, problem_strs):
    '''
    Checks if anything we don't want is in string, and returns True if so.

    string : str, to search in
    problem_strs : list of str, which we don't want

    return : bool
    '''
    for s in problem_strs:
        if s in string:
            return True, s
    return False, ''

def get_location_coords(loc, t='country'):
    '''
    Retrieve location coordinates in box form and center form online.
    Have an ongoing dictionary in the global variables so don't need
    to fetch these everytime

    loc : str
    t : str, 'country' or 'q' for non countries

    return : list of int of min/max lat/lon, list of int of center lat/lon
    '''

    url = '{}{}={}{}'.format('http://nominatim.openstreetmap.org/search?', t, loc,
                          '&format=json&polygon=0')
    #print(url)    
    response = requests.get(url).json()[0]

    box = [float(coord) for coord in response['boundingbox']]
    center = [float(response.get(key)) for key in ['lat', 'lon']]
    return box, center



def curv_vort(u, v, dx, dy, x_index=1, y_index=0):
    '''
    Added from Quinton's code.
    Calculates curvature vorticity from wind and distance arrays.

    u : 2D np.array, u component of wind
    v : 2D np.array, v component of wind
    dx : 2D np.array, x distance across point in meters
    dy : 2D np.array, y distance across point in meters
    x_index : int, index of the x (longitude) axis in the data
    y_index : int, index of the y (latitude) axis in the data

    return : 2D np.array, 
    '''
    V_2 = (u**2+v**2)
    curv_vort_raw = (1/V_2)*(u*u*np.gradient(v, axis=x_index)/dx - v*v*np.gradient(u, axis=y_index)/dy
                             - v*u*np.gradient(u, axis=x_index)/dx + u*v*np.gradient(v, axis=y_index)/dy)

    return curv_vort_raw


def advection_2d(u, v, var, dx, dy, x_index=1, y_index=0):
    '''
    Calculates advection of a variable in 2D using horizontal wind and grid distance fields

    u : 2D np.array, u component of wind
    v : 2D np.array, v component of wind
    var : 2D np.array, variable to be advected
    dx : 2D np.array, x distance across point in meters
    dy : 2D np.array, y distance across point in meters
    x_index : int, index of the x (longitude) axis in the data
    y_index : int, index of the y (latitude) axis in the data

    return : 2D np.array of x direction advection, 2D np.array of y direction advection, 2D np.array of advection magnitude
    '''
    
    u_adv = -u*np.gradient(var, axis=x_index)/dx
    v_adv = -v*np.gradient(var, axis=y_index)/dy
    #total_adv = (u_adv**2+v_adv**2)**0.5
    total_adv = u_adv + v_adv
    
    return u_adv, v_adv, total_adv
    
def interval_round(x, prec=2, base=0.25):
    # Round x to nearest instance of base with precision prec
    return round(base * round(float(x)/base),prec)

def convert_360_to_lon(f, target_f, lon_coord='longitude'):
    '''
    Converts degrees east longitudes to regular longitudes

    f : netCDF file path to open
    target_f : path to save new file
    lon_coord : str, default 'longitude' (ERA5 default)
    '''

    ds = xr.open_dataset(f)
    # Check if values are already converted, in which case do nothing to avoid freaky files
    if float(ds[lon_coord].max()) <= 180:
        return
    # formula from https://confluence.ecmwf.int/display/CUSF/Longitude+conversion+0~360+to+-180~180
    ds[lon_coord] = ((ds[lon_coord]+180)%360)-180
    ds = ds.sortby(ds[lon_coord])
    ds.to_netcdf(target_f)


def convert_360_to_lon_ds(ds, lon_coord='longitude'):
    '''
    Converts degrees east longitudes to regular longitudes, but with an already open dataset

    f : netCDF file path to open
    target_f : path to save new file
    lon_coord : str, default 'longitude' (ERA5 default)
    '''

    # Check if values are already converted, in which case do nothing to avoid freaky files
    if float(ds[lon_coord].max()) <= 180:
        return
    # formula from https://confluence.ecmwf.int/display/CUSF/Longitude+conversion+0~360+to+-180~180
    ds[lon_coord] = ((ds[lon_coord]+180)%360)-180
    ds = ds.sortby(ds[lon_coord])
    return ds


def reorder_ds_at_lon(ds, new_end_lon=-180, lon_coord='longitude'):
    '''
    Converts ds with regular longitudes with ends at 180 to ds with regular longitudes with ends at chosen longitude

    ds: xarray Dataset
    lon_coord: str, default 'longitude' (ERA5 default)
    '''

    # Check if values are already converted, in which case do nothing to avoid freaky files
    #if ds[lon_coord][0] > ds[lon_coord][-1]:
    #    return
    # formula from https://confluence.ecmwf.int/display/CUSF/Longitude+conversion+0~360+to+-180~180
    ds[lon_coord] = (ds[lon_coord]-new_end_lon)%360    
    ds = ds.sortby(ds[lon_coord])
    ds[lon_coord] = ((ds[lon_coord]+new_end_lon+180)%360)-180
    
    return ds


    
def mask_local_advect_minima(u, v, var, dx, dy):
    '''
    Masks out local minima in advection of var from the original var array. Follows Berry 2007 Mask A2

    u : 2D np.array, u component of wind
    v : 2D np.array, v component of wind
    var : 2D np.array, variable to be advected
    dx : 2D np.array, x distance across point in meters
    dy : 2D np.array, y distance across point in meters
    x_index : int, index of the x (longitude) axis in the data
    y_index : int, index of the y (latitude) axis in the data

    #return : 2D np.array of advection magnitude
    return : 2D np.array of advection magnitude without masking done yet
    '''
    
    new_u_adv, new_v_adv, _ = advection_2d(u, v, var, dx, dy)
    new_u_adv, new_v_adv = 0-new_u_adv, 0-new_v_adv
    new_advect_da = new_u_adv + new_v_adv
    #u_adv = u*np.gradient(var, axis=1)/dx
    #v_adv = v*np.gradient(var, axis=0)/dy
    #new_advect_da = u_adv + v_adv
    return new_advect_da > 0


def berry_2007_masks(u, v, adv, dx, dy, var=[], extra_minima=False, var_cutoff=0):
    '''
    Applies the masks from the Berry 2007 tracker. Created this function because
    was experimenting with many different variables / levels.

    u : 2D array, u component of wind
    v : 2D array, v component of wind    
    adv : 2D array, advected variable
    dx : 2D np.array, x distance across point in meters
    dy : 2D np.array, y distance across point in meters    
    var : 2D array, variable to be used for masking out negative values (vorticity)
    extra_minima : bool, whether to take the minimum of local variables again (may be needed for vars that don't have the < 0 check)
    var_cutoff : float, minima for cutoff of var value

    return : 2D array of fully masked adv, a complete mask array
    '''

    # easterly flow mask (A3)
    u_mask = u < 0
    #u_mask = u > -99999

    # first local minima mask (A2)
    min_mask = mask_local_advect_minima(u, v, adv, dx, dy)
    #min_mask = u_mask
    
    # should always do var=something or extra_minima mask, but create a placeholder mask just in case
    second_min_mask = np.abs(adv) >= 0
    if len(var) > 0:
        # get rid of points <= 0
        second_min_mask = var > var_cutoff
    elif extra_minima:
        # get rid of local minima
        second_min_mask = np.gradient(adv,axis=1)/dx + np.gradient(adv,axis=0)/dy < 0


    # return fully masked adv array, and the masks combined
    return adv.where(u_mask).where(min_mask).where(second_min_mask), u_mask & min_mask & second_min_mask



def create_bimodal(n_cell=19, peak_spacing=5, power=2, scaling_factor=0.5):
    '''
    Creates symmetric bimodal distribution. Default values are chosen based on 1 degree tropical wave analysis
    Note that distribution will be a parabola if peak_spacing>=floor(n_cell/2)

    n_cell: window width (must be odd)                                                                                                                                                            
    peak_spacing: distance from midpoint in cells                                                                                                                                                 
    power: exponent to increment as base                                                                                                                                                          
    scaling_factor: increase in power scaling per distance from middle cell, then decrease beyond peak indices

    return: xarray DataArray of weights
    '''
    bimodal = xr.DataArray(np.full((n_cell), power), dims=['window'])
    midpoint = int(n_cell/2)
    peak_1 = midpoint-peak_spacing
    peak_2 = midpoint+peak_spacing
    peak_scaling = peak_spacing*scaling_factor
    for i in range(len(bimodal)):
        if i <= midpoint:
            bimodal[i] = float(bimodal[i])**(peak_scaling-scaling_factor*abs((peak_1-i)))
        else:
            bimodal[i] = float(bimodal[i])**(peak_scaling-scaling_factor*abs((peak_2-i)))
    return bimodal



def calc_zonal_eddy_mean(data, n_cell=19, peak_spacing=5, power=2, scaling_factor=0.5):
    '''
    Calculate zonal eddy mean from xarray data ('longitude' is hardcoded as x right now)

    data : xr.DataArray, to take rolling mean of
    n_cell: window width (must be odd)                                                                                                                                                            
    peak_spacing: distance from midpoint in cells                                                                                                                                                 
    power: exponent to increment as base                                                                                                                                                          
    scaling_factor: increase in power scaling per distance from middle cell, then decrease beyond peak indices

    return: xarray DataArray of calculated zonal mean
    '''
    weight = create_bimodal(n_cell, peak_spacing, power, scaling_factor)
    return data - data.rolling(longitude=n_cell, min_periods=1, center=True).construct('window').dot(weight)/weight.sum()

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

    twd_time: str, 'YYYYMMDDmmhh'
    rewind: bool, whether to go 6 hours back
    
    return: str, the time for use in era5 comparisons
    '''
    if rewind:
        new_time = twd_time[:4] + global_times[global_times.index(twd_time[4:])-1]
    else:
        new_time = twd_time
    return 'T'.join([new_time[:8], new_time[8:]])


def wave_times_for_era5_comparison(waves):
    '''
    Replace all timesteps in TWD waves with suitable timesteps for matching with ERA5 timesteps in xarray.
    Operates inplace

    waves: dict, of TwdWave objects
    '''
    # Two different formats depending on whether mask bool inclusion has happened
    #for wave_list in waves.values():
    sorted_waves = sorted(waves.items())
    for i, (time, wave_list) in enumerate(sorted_waves):
        new_list = []
        try:            
            for wave, masked in wave_list:                
                wave.time = twd_time_to_era5_time(wave.time)
                #new_list.append(wave) # commented out on 4-16-24
                new_list.append((wave, masked))
        except:
            for wave in wave_list:
                wave.time = twd_time_to_era5_time(wave.time)
                new_list.append(wave)
        # adjust dict
        waves[time[:4] + global_times[global_times.index(time[4:])-1]]=new_list
        

def add_wave_mask_to_xr(ds, waves_orig, lat_name='latitude', lon_name='longitude', time_name='time',
                        label_name='wave_labels', lon_smudge=0, weight_name='', trust_name='', mask_frac_name='', n_jobs=1):
    '''
    Add a tropical wave data variable to an xarray Dataset that takes the form of a mask
    indicating the presence of a tropical wave (1), no tropical wave (0), or a masked tropical wave due to input constraints (2).
    Note that whether a wave is masked or not is determined in mask_waves_by_bimodal_frac

    ds: xr.Dataset, where all vars have same dimensions time, lat_name, lon_name
    waves_orig: dict, time : list of (TwdWave object, masked_or_not)
    lat_name: str, name of latitude dimension of input ds
    lon_name: str, name of longitude dimension of input ds
    time_name: str, name of time dimension of input ds
    label_name: str, name of wave mask data var to add
    lon_smudge: int, gridpoints on each side to expand label
    weight_name: str, name of weight dimension to add. If empty, none added
    trust_name: str, name of trust dimension to add. If empty, none added
    mask_frac_name: str, name of max mask frac dimension to add. If empty, none added

    return: xr.Dataset, the original ds merged with the new wave mask
    '''
    # grab dim info from existing DataArray
    copy_da = ds[list(ds.data_vars)[0]]
    wave_da = xr.DataArray(np.zeros(copy_da.shape), dims=copy_da.dims, coords=copy_da.coords).rename(label_name)
    weight_da = xr.DataArray(np.zeros(copy_da.shape), dims=copy_da.dims, coords=copy_da.coords).rename(weight_name)
    trust_da = xr.DataArray(np.zeros(copy_da.shape), dims=copy_da.dims, coords=copy_da.coords).rename(trust_name)
    frac_da = xr.DataArray(np.zeros(copy_da.shape), dims=copy_da.dims, coords=copy_da.coords).rename(mask_frac_name)            

    waves = waves_orig.copy()
    # change wave times to match ERA5 format if needed
    for time in waves.keys():
        if len(waves[time]) == 0:
            continue
        if 'T' not in list(waves[time])[0][0].time:
            wave_times_for_era5_comparison(waves)
            break

    
    # insert wave data at appropriate axis location in wave_da
    #for time in waves.keys():
    time_list = list(wave_da['time'].values)
    lats = wave_da[lat_name].values.copy()
    lons = wave_da[lon_name].values.copy()
    def get_wave_data(time):
        #if time[:4] != '2018':
        #     return []
        # note parallelizing it actually managed to make it slower seemingly
        # get time index in the da for this time
        era5_time = twd_time_to_era5_time(time, rewind=False)
        #print(wave_da)
        #itime = list(wave_da['time'].values).index(wave_da['time'].sel(time=era5_time))
        #itime = list(wave_da['time'].values).index(era5_time_to_np64ns(era5_time))
        itime = time_list.index(era5_time_to_np64ns(era5_time))
        #print(era5_time)
        wave_mask_coords = []
        #for wave, masked in waves[time]:
        for wave_info in waves[time]:
            wave = wave_info[0]
            masked = wave_info[1]
            wave_lon_coords = []
            wave_lat_coords = []
            
            #print(era5_time, wave)
            if weight_name == '' and trust_name == '' and mask_frac_name == '':
                masked_val  = 2 if not masked else 1
            else:
                # co-opting this variable name when doing weights to make things easier (since won't have drop threshold anyway)
                # tuple of (mask frac, trust)
                masked_val = (wave_info[2], wave.trust)
            wave_coords = [masked_val, itime]
            wave_minlon = 0-wave.extent[0][1]
            wave_maxlon = 0-wave.extent[1][1]
            wave_minlat = wave.extent[0][0]
            wave_maxlat = wave.extent[1][0]
            # will create interpolated coordinates of wave, using 3*number of grid cells from minlat to maxlat as number of vertices
            spacing = abs(float(lats[1]-lats[0]))
            num_coords = int(3*((wave_maxlat-wave_minlat)/spacing))
            wave_lons = np.linspace(wave_minlon, wave_maxlon, num=num_coords)
            wave_lats = np.linspace(wave_minlat, wave_maxlat, num=num_coords)

            for lon, lat in zip(wave_lons, wave_lats):
                ilon = np.argmin(np.abs(lons-lon))
                ilat = np.argmin(np.abs(lats-lat))
                #print(lat, lon, ilat, ilon, lats, lons)
                #wave_da[itime, ilat, ilon] = masked_val
                # try getting smudge stuff here as opposed to how it was originally done to speed things up
                extra_lons = [ilon-i for i in range(1, lon_smudge+1) if ilon-i > -1] + [ilon+i for i in range(1, lon_smudge+1) if ilon+i < len(wave_da[lon_name])]
                ilats = [ilat] + [ilat for i in range(len(extra_lons))]
                ilons = [ilon] + extra_lons 
                wave_lon_coords.extend(ilons)
                wave_lat_coords.extend(ilats)
                
            wave_coords.extend([wave_lat_coords, wave_lon_coords])
            wave_mask_coords.append(wave_coords)
            #print(wave_mask_coords)
        return wave_mask_coords
    coords = Parallel(n_jobs=1)(delayed(get_wave_data)(time) for time in waves.keys() if len(waves[time]) > 0)
    for time_list in coords:
        for wave in time_list:
            #print(wave.time)
            #print(wave)
            masked_val = wave[0]
            itime = wave[1]
            if weight_name != '' or mask_frac_name != '' or trust_name != '':
                # weight based on trust and mask fraction will get its own dimension too
                # also, masked_val will stay as 1 for all waves (1 = Good)
                frac = masked_val[0]
                trust = masked_val[1]
                masked_val = 1                
                if weight_name != '':
                    weight = frac*(1/(trust**(1/4)))
                    #print(frac, trust, weight)
            
            #print(wave_da['time'][itime])
            #print(wave[2:])
            wave_lats = wave[2]
            wave_lons = wave[3]
            # for some reason xarray indexing does not function in the same way as numpy indexing when sending two lists of indices,
            # so need to do a numpy intermediate step            
            intermed = wave_da[itime].values
            intermed[wave_lats, wave_lons] = masked_val
            wave_da[itime] = intermed
            if weight_name != '':
                intermed = weight_da[itime].values
                intermed[wave_lats, wave_lons] = weight
                weight_da[itime] = intermed
            if trust_name != '':
                intermed = trust_da[itime].values
                intermed[wave_lats, wave_lons] = trust
                trust_da[itime] = intermed
            if mask_frac_name != '':
                intermed = frac_da[itime].values
                intermed[wave_lats, wave_lons] = frac
                frac_da[itime] = intermed
            
    # merge wave_da and ds and any other added layers
    ret = xr.merge([ds, wave_da])
    if weight_name != '':
        ret = xr.merge([ret, weight_da])
    if trust_name != '':
        ret = xr.merge([ret, trust_da])
    if mask_frac_name != '':
        ret = xr.merge([ret, frac_da])
    return ret


def add_imt_mask_to_xr(ds, imts_orig, lat_name='latitude', lon_name='longitude', time_name='time',
                        label_name='imt_labels', lon_smudge=0, lat_smudge=0, n_jobs=1):
    '''
    Add imt data variable to an xarray Dataset that takes the form of a mask
    indicating the presence of monsoon trough (2), itcz (1), or neither (0)

    ds: xr.Dataset, where all vars have same dimensions time, lat_name, lon_name
    imts_orig: dict, time : list of imt objects
    lat_name: str, name of latitude dimension of input ds
    lon_name: str, name of longitude dimension of input ds
    time_name: str, name of time dimension of input ds
    label_name: str, name of imt mask data var to add
    lon_smudge: int, gridpoints left and right  to expand label
    lat_smudge: int, gridpoints above and below to expand label

    return: xr.Dataset, the original ds merged with the new imt mask
    '''
    # grab dim info from existing DataArray
    copy_da = ds[list(ds.data_vars)[0]]
    imt_da = xr.DataArray(np.zeros(copy_da.shape), dims=copy_da.dims, coords=copy_da.coords).rename(label_name)

    imts = imts_orig.copy()
    # change imt times to match ERA5 format if needed
    for time in imts.keys():
        if len(imts[time]) == 0:
            continue
        if 'T' not in list(imts[time])[0].time:
            wave_times_for_era5_comparison(imts)
            break

        
    # insert imt data at appropriate axis location in imt_da
    #for time in imts.keys():
    time_list = list(imt_da[time_name].values)
    lats = imt_da[lat_name].values.copy()
    lons = imt_da[lon_name].values.copy()
    spacing = abs(float(lats[1]-lats[0])) # assumes equal spacing in zonal and meridional
    def get_imt_data(time):
        # note parallelizing it actually managed to make it slower seemingly
        # get time index in the da for this time
        era5_time = twd_time_to_era5_time(time, rewind=False)
        itime = time_list.index(era5_time_to_np64ns(era5_time))
        imt_mask_coords = []
        #print(time)
        for imt in imts[time]:            
            imt_coords = [imt.cat, itime]
            imt_lon_coords = []
            imt_lat_coords = []
            # will create interpolated coordinates of imt, using 4*number of grid cells between longitudes of each point as number of vertices
            for (lat1, lon1), (lat2, lon2) in zip(imt.coords[:len(imt.coords)-1], imt.coords[1:]):
                lon1=0-lon1
                lon2=0-lon2
                num_coords = int(3*((lon2-lon1)/spacing))
                imt_lons = np.linspace(lon1, lon2, num=num_coords)
                imt_lats = np.linspace(lat1, lat2, num=num_coords)
                
                for lon, lat in zip(imt_lons, imt_lats):
                    ilon = int(np.argmin(np.abs(lons-lon)))
                    ilat = int(np.argmin(np.abs(lats-lat)))
                    #print(lat, lon, ilat, ilon, lats, lons)
                    #imt_coords.append((ilat, ilon))
                    # try getting smudge stuff here as opposed to how it's done with waves, may help speed things up                    
                    extra_lons = [ilon-i for i in range(1, lon_smudge+1) if ilon-i > -1] + [ilon+i for i in range(1, lon_smudge+1) if ilon+i < len(imt_da[lon_name])]
                    extra_lats = [ilat-i for i in range(1, lat_smudge+1) if ilat-i > -1] + [ilat+i for i in range(1, lat_smudge+1) if ilat+i < len(imt_da[lat_name])]
                    ilats = [ilat] + [ilat for i in range(len(extra_lons))] + extra_lats
                    ilons = [ilon] + extra_lons + [ilon for i in range(len(extra_lats))]
                    imt_lon_coords.extend(ilons)
                    imt_lat_coords.extend(ilats)
            if len(imt_lat_coords) > 1000 or len(imt_lon_coords) > 1000:
                # ISSUE WITH MISSING DECIMAL REGULAR EXPRESSION, may need a specific one for EPAC                
                print(time, imt.coords, len(imt_lat_coords), len(imt_lon_coords))

            imt_coords.extend([imt_lat_coords, imt_lon_coords])
            imt_mask_coords.append(imt_coords)
            #print(imt_mask_coords)

        return imt_mask_coords
    coords = Parallel(n_jobs=n_jobs)(delayed(get_imt_data)(time) for time in imts.keys() if len(imts[time]) > 0)# and time[:4] == '2021')
    for time_list in coords:
        for imt in time_list:
            cat = 2 if imt[0] == 'mt' else 1
            itime = imt[1]
            imt_lats = imt[2]
            imt_lons = imt[3]
            #if len(imt_lats) > 1000:
            #    print(itime, cat, len(imt_lats), len(imt_lons))
            #print(len(imt_lats), len(imt_lons))
            
            # for some reason xarray indexing does not function in the same way as numpy indexing when sending two lists of indices,
            # so need to do a numpy intermediate step            
            intermed = imt_da[itime].values
            intermed[imt_lats, imt_lons] = cat
            imt_da[itime] = intermed
            '''
            #troubleshooting
            import matplotlib.pyplot as plt
            import cartopy.crs as ccrs
            proj = ccrs.PlateCarree()
            from ml_waves_itcz_mt.plotting import plot_era_da
            imt_colormap = {'itcz' : 'g', 'mt' : 'b'}
            fig, ax = plt.subplots(1, 1, figsize=(18, 15), subplot_kw={"projection": proj})
            plot_era_da(imt_da[itime], var='mask', title=' '.join([str(imt_da['time'][itime].values), 'true itcz/mt grid']),  
                        extent=[-160, 1, -7, 24], fig=fig, ax=ax, comap='binary')            
            plt.show()
            '''
    # merge imt_da and ds and any other added layers
    ret = xr.merge([ds, imt_da])
    return ret


def mask_waves_by_bimodal_frac(waves_orig, mask_frac, das_for_masks=[], n_cell=19, peak_spacing=5, power=2, scaling_factor=0.5,
                               time_dim='time', save_frac=False):
    '''
    Calculate whether to call waves possibly wrong based on their overlap with zonal eddies

    waves_orig: dict, time : list of TwdWave objects
    mask_frac: float, fraction of any mask a wave's axis must overlap with to be considered valid
    das_for_masks: list of pairs of xr.DataArray, for masking by level (will calculate zonal eddy info from these)
    n_cell: int, window width (must be odd)
    peak_spacing: int, distance from midpoint in cells
    power: int or float, exponent to increment as base
    scaling_factor: float, increase in power scaling per distance from middle cell, then decrease beyond peak indices
    time_dim: str, name of time dimension for input arrays
    save_frac: bool, whether to also save the highest fraction an axis overlaps with a mask
    
    return: dict, time : list of (TwdWave object, masked or not)
    '''
    
    
    waves = waves_orig.copy()
    # change wave times to match ERA5 format if needed
    for time in waves.keys():
        if len(waves[time]) == 0:
            continue
        if 'T' not in list(waves[time])[0].time:
            wave_times_for_era5_comparison(waves)
            break

    # for saving vals later
    waves_new = {}

    waves_masked = 0
    # will create interpolated coordinates of wave, using 3*number of grid cells from minlat to maxlat as number of vertices
    spacing = abs(float(das_for_masks[0][0]['latitude'][1]-das_for_masks[0][0]['latitude'][0]))

    # Trying out a speedup method
    # NOTE THIS ONLY ACHIEVES PRECISION UP TO .1 DEGREES
    speedup=True
    if speedup:
        # create dicts of lon/lat values from spacings mapped to lon/lat indices
        lon_dict = {}
        lat_dict = {}
        all_waves = [wave for time in waves.keys() for wave in waves[time]]
        for wave in all_waves:
            wave_minlon = 0-wave.extent[0][1]            
            wave_maxlon = 0-wave.extent[1][1]
            wave_minlat = wave.extent[0][0]
            wave_maxlat = wave.extent[1][0]
            
            num_coords = int(3*((wave_maxlat-wave_minlat)/spacing))            
            wave_lons = np.linspace(wave_minlon, wave_maxlon, num=num_coords)
            wave_lats = np.linspace(wave_minlat, wave_maxlat, num=num_coords)
            for lon in wave_lons:
                rlon = round(lon, 1)
                try:
                    lon_dict[rlon]
                except KeyError:
                    lon_dict[rlon] = np.argmin(np.abs(das_for_masks[0][0]['longitude'].values-rlon))
            for lat in wave_lats:
                rlat = round(lat, 1)
                try:
                    lat_dict[rlat]
                except KeyError:
                    lat_dict[rlat] = np.argmin(np.abs(das_for_masks[0][0]['latitude'].values-rlat))
                    
                
    
    
    for time in waves.keys():
        if time[:4] == '2018':
            print(time)
        waves_new[time] = []
        if len(waves[time]) == 0:
            continue
        # create bimodal zonal eddy arrays for masking
        np_time=era5_time_to_np64ns(twd_time_to_era5_time(time, rewind=False))
        # NOTE: Made this np_time edit after already having run wave calculations for my unet_test
        bimodal_zes = [(calc_zonal_eddy_mean(da1.sel({time_dim:np_time}), n_cell=n_cell, peak_spacing=peak_spacing, power=power, scaling_factor=scaling_factor),
                        calc_zonal_eddy_mean(da2.sel({time_dim:np_time}), n_cell=n_cell, peak_spacing=peak_spacing, power=power, scaling_factor=scaling_factor))
                       for (da1, da2) in das_for_masks]
        # create masking arrays
        masks = [(da1 > 0) | (da2 > 0) for (da1, da2) in bimodal_zes]
        #print(time, float(masks[2].mean()))
        
        for wave in waves[time]:            
            wave_minlon = 0-wave.extent[0][1]
            wave_maxlon = 0-wave.extent[1][1]
            # skip waves outside of our mask bounds
            if wave_minlon < float(min(das_for_masks[0][0]['longitude'])) or wave_maxlon > float(max(das_for_masks[0][0]['longitude'])):
                continue
            
            wave_minlat = wave.extent[0][0]
            wave_maxlat = wave.extent[1][0]
            num_coords = int(3*((wave_maxlat-wave_minlat)/spacing))
            #num_coords = 50
            #print(num_coords)

            # BUG CURRENTLY THAT WOULD ONLY IMPACT NOTABLY TILTED WAVES: ALWAYS ASSUMES WAVES TILTED LIKE /
            wave_lons = np.linspace(wave_minlon, wave_maxlon, num=num_coords)
            wave_lats = np.linspace(wave_minlat, wave_maxlat, num=num_coords)

            
            mask_coords = [[] for mask in masks]
            for lat, lon in zip(wave_lats, wave_lons):
                if not speedup:
                    ilon = np.argmin(np.abs(masks[0]['longitude'].values-lon))
                    ilat = np.argmin(np.abs(masks[0]['latitude'].values-lat))
                else:
                    ilon = lon_dict[round(lon, 1)]
                    ilat = lat_dict[round(lat, 1)]                    
                for mask, coords in zip(masks, mask_coords):
                    coords.append(float(mask[ilat, ilon]*1))
                    
            masked = any([np.sum(coords) >= mask_frac*num_coords for coords in mask_coords]) \
                and not( abs(wave_maxlat-wave_minlat) < abs(wave_maxlon-wave_minlon)) # also mask out waves that are too tilted to be real
            #print(time, wave.extent, masked)
            if not save_frac:
                waves_new[time].append((wave, masked))
            else:
                # Also want to still make sure that super tilted waves have low weighting
                frac = max([np.sum(coords)/num_coords for coords in mask_coords]) if not (abs(wave_maxlat-wave_minlat) < abs(wave_maxlon-wave_minlon)) and num_coords > 0 else 0
                #print(frac, masked)
                waves_new[time].append((wave, masked, frac))
            
            if not masked:
                waves_masked = waves_masked + 1
                
    print('masked out', waves_masked, 'waves in', time[:4])
    return waves_new
        

class TWDIMT:
    '''
    Class for ITCZ / monsoon trough axes from Tropical Weather Discussions
    '''
    def __init__(self, time, cat, coord_pairs, lat_coords, lon_coords, verbal_locations,
                 coord_pair_locs, lat_coord_locs, lon_coord_locs, verbal_location_locs, full_text='Manual'):
        self.time = time
        self.cat = cat #'itcz' or 'mt'
        self.coord_pairs = coord_pairs
        self.lat_coords = lat_coords
        self.lon_coords = lon_coords
        self.verbal_locations = [[d, loc.replace('   ', ' ').replace('  ', ' ')] for [d, loc] in verbal_locations]
        self.coord_pair_locs = coord_pair_locs
        self.lat_coord_locs = lat_coord_locs
        self.lon_coord_locs = lon_coord_locs
        self.verbal_location_locs = verbal_location_locs
        self.full_text = full_text
        self._give_coords()
        
    def __str__(self):
        return str([self.time, self.cat, self.coords, self.coord_pairs, self.lat_coords, self.lon_coords,
                    self.verbal_locations])

    def full_print(self):
        print(str(self), self.full_text)
                
    def _get_info_from_coord_pairs(self):
        lats, lons = [], []
        for pair in self.coord_pairs:
            w_loc = pair.find('w')
            n_loc = pair.find('n')
            north = True
            if n_loc == -1:
                # coordinate south of equator
                north = False
                n_loc = pair.find('s')
            if '/' not in pair:
                lat, lon = float(pair[:n_loc]), float(pair[n_loc+1:-1])
            else:
                lat, lon = float(pair[:n_loc]), float(pair[n_loc+2:-1])
            if not north:
                lat = 0-lat
            lats.append(lat), lons.append(abs(lon))
        return lats, lons

    def _get_info_from_lat_coords(self):
        lats = []
        lat_pattern = '([0-9]+)[ns]'
        for s in self.lat_coords:
            north = True
            pat_lats = [float(lat) for lat in re.findall(lat_pattern, s)]
            if s.find('n') == -1:
                # south
                north = False
            if not north: pat_lats = [0-lat for lat in pat_lats]
            lats.extend(pat_lats)
        return lats

    
    def _get_info_from_lon_coords(self):
        lons = []
        #lon_pattern = '([0-9]+)[w]'
        lon_pattern = '([0-9\.]+)[/w]'
        for s in self.lon_coords:
            lons.extend(abs(float(lon)) for lon in re.findall(lon_pattern, s))
        return lons

    def _get_coords_from_verbal_locations(self):
        '''
        Convert word-based locations and direction descriptors to lat/lon pairs
        '''
        lats, lons = [], []
        for pair in self.verbal_locations:
            direction = global_direction_map[pair[0]]
            loc = pair[1]
            # mexico specifically needs to be southern bound for coast rather than northern bound
            if pair[0] and 'coast' in pair[0] and 'mexico' in loc:
                direction = 's'
            (bounds, center) = global_place_coords[global_loc_keyword_map[pair[1]]]            
            if 'n' in direction:
                lat = bounds[1]
            elif 's' in direction:
                lat = bounds[0]
            else:
                lat = center[0]
            if 'w' in direction:
                lon = bounds[2]
            elif 'e' in direction:
                lon = bounds[3]
            else:
                lon = center[1]
            lats.append(round(lat, 2)), lons.append(round(abs(lon), 2))
        return lats, lons
                
    
    def _give_coords(self):
        '''
        Interprets provided location information into set of lat/lon pairs (note lons are in degrees west).
        '''

        ## get numerical coords from all info
        pair_lats, pair_lons = self._get_info_from_coord_pairs()
        ind_lats = self._get_info_from_lat_coords()
        ind_lons = self._get_info_from_lon_coords()
        verbal_lats, verbal_lons = self._get_coords_from_verbal_locations()
        verbal_names = [v for (_,v) in self.verbal_locations]
        
        # sort all coords by location in sentence
        #pairs_by_loc =  [((lat, lon, False), loc) for lat, lon, loc in zip(pair_lats, pair_lons, self.coord_pair_locs)]
        #lons_by_loc =   [((-1000, lon, False), loc) for lon, loc in zip(ind_lons, self.lon_coord_locs)]
        #lats_by_loc =   [((lat, -1000, False), loc) for lat, loc in zip(ind_lats, self.lat_coord_locs)]        
        #verbal_by_loc = [((lat, lon, True), loc) for lat, lon, loc in zip(verbal_lats, verbal_lons, self.verbal_location_locs)]
        pairs_by_loc =  [((lat, lon, ''), loc) for lat, lon, loc in zip(pair_lats, pair_lons, self.coord_pair_locs)]
        lons_by_loc =   [((-1000, lon, ''), loc) for lon, loc in zip(ind_lons, self.lon_coord_locs)]
        lats_by_loc =   [((lat, -1000, ''), loc) for lat, loc in zip(ind_lats, self.lat_coord_locs)]        
        verbal_by_loc = [((lat, lon, name), loc) for lat, lon, loc, name in zip(verbal_lats, verbal_lons, self.verbal_location_locs, verbal_names)]

        ordered_by_loc = [(lat,lon,verbal_name) for ((lat,lon,verbal_name), loc) in sorted(pairs_by_loc+lons_by_loc+lats_by_loc+verbal_by_loc, key=lambda a:a[1][0])]
        #print(ordered_by_loc)
        
        ## match up loose lon coords with preceding lat info (from coord pair / lat coord / verbal loc)
        previous_lat = -1000
        for i, (lat, lon, verbal_name) in enumerate(ordered_by_loc):
            if lat != -1000:
                previous_lat = lat
            elif previous_lat != -1000:
                ordered_by_loc[i] = (previous_lat, lon, verbal_name)
                

        ## dispose of verbal locs that are not the final coordinates in the sentence, and also loose lat coords
        #trimmed_verbals = [(lat, lon) for i, (lat, lon, verbal) in enumerate(ordered_by_loc) if lon!=-1000 and lat!=-1000 and not (verbal and i < len(ordered_by_loc)-1)]
        ## dispose of verbal locs that are the second to last coordinates in the sentence if there is a non-verbal location after, unless full length of coords is 2
        #trimmed_verbals = [(lat, lon) for i, (lat, lon, verbal) in enumerate(ordered_by_loc) if lon!=-1000 and lat!=-1000
        #                   and not (verbal and i == len(ordered_by_loc)-2 and not ordered_by_loc[-1][2] and len(ordered_by_loc) != 2)]
        # dispose of verbal locs which are within the diameter of that location's distance, or 2 degrees (max of those two), of any other coords, and are not last location in sentence
        verbals = [(lat,lon, verbal_name) for (lat,lon, verbal_name) in ordered_by_loc if verbal_name and lon!=-1000 and lat!=-1000]
        trimmed = [(lat, lon) for (lat,lon,verbal_name) in ordered_by_loc if not verbal_name and lon!=-1000 and lat!=-1000]
        kept_verbals = []
        for (lat, lon, verbal_name) in verbals:
            #round(lat, 2)), lons.append(round(abs(lon), 2))            
            [min_lat, max_lat, min_lon, max_lon] = global_place_coords[global_loc_keyword_map[verbal_name]][0]
            min_lon, max_lon = 0-min_lon, 0-max_lon
            diameter = np.sqrt((max_lat-min_lat)**2 + (max_lon-min_lon)**2)
            found_close = False
            for (coord_lat, coord_lon) in trimmed:
                dist = np.sqrt((coord_lat-lat)**2 + (coord_lon-lon)**2)
                if dist < diameter or dist < 2:
                    found_close = True
                    break
            if not found_close or ordered_by_loc.index((lat,lon,verbal_name)) == len(ordered_by_loc)-1: #or len(trimmed) == 0:
                kept_verbals.append((lat,lon))
            #elif self.time == '202107291200':
            #    print(trimmed, lat, lon, verbal_name, min_lat, max_lat, min_lon, max_lon, diameter)
        trimmed.extend(kept_verbals)
        ## sort coordinate pairs from west to east
        self.coords = sorted(trimmed, key=lambda a:a[1])[::-1]
        #print(self.coords)



class TwdWave:
    '''
    Class for tropical waves from Tropical Weather Discussions
    '''
    def __init__(self, time, coord_pairs, lat_coords, lon_coords, verbal_locations, trust, full_text='Manual'):
        self.time = time
        self.coord_pairs = coord_pairs
        self.lat_coords = lat_coords
        self.lon_coords = lon_coords
        self.verbal_locations = [[d, loc.replace('   ', ' ').replace('  ', ' ')] for [d, loc] in verbal_locations]
        self.trust = trust
        self.full_text = full_text
        self._give_coords()
        # for determining whether wave slope is unrealistically large (something screwed up)
        self.lon_range = self._get_lon_range()
        
    def __str__(self):
        return str([self.time, self.coord_pairs, self.lat_coords, self.lon_coords,
                    self.verbal_locations, self.trust, self.extent, self.lon_range])

    def full_print(self):
        print(str(self), self.full_text)
    
    def get_center(self):
        return (np.nanmean([self.extent[0][0], self.extent[1][0]]), np.nanmean([self.extent[0][1], self.extent[1][1]]))
        
    def _get_lon_range(self):
        lons = [np.nan]
        _, pair_lons = self._get_info_from_coord_pairs()
        lons.extend(pair_lons)
        lons.extend(self._get_info_from_lon_coords())
        _, verbal_lons = self._get_coords_from_verbal_locations()
        lons.extend(verbal_lons)
        # converting to all positive so switching max and min 
        min_lon, max_lon = np.nanmax(lons), np.nanmin(lons)
        return [min_lon, max_lon]
    
    def _get_info_from_coord_pairs(self):
        lats, lons = [], []
        for pair in self.coord_pairs:
            w_loc = pair.find('w')
            n_loc = pair.find('n')
            north = True
            if n_loc == -1:
                # coordinate south of equator
                north = False
                n_loc = pair.find('s')
            if '/' not in pair:
                lat, lon = float(pair[:n_loc]), float(pair[n_loc+1:-1])
            else:
                lat, lon = float(pair[:n_loc]), float(pair[n_loc+2:-1])
            if not north:
                lat = 0-lat
            lats.append(lat), lons.append(abs(lon))
        return lats, lons

    def _get_info_from_lat_coords(self):
        lats = []
        lat_pattern = '([0-9]+)[ns]'
        for s in self.lat_coords:
            lats.extend(float(lat) for lat in re.findall(lat_pattern, s))
        return lats

    def _get_vector_max_from_lat_coords(self):
        lats = []
        lat_pattern = '([0-9]+)[ns]'
        for s in self.lat_coords:
            if s.startswith(' s ') or s.startswith('south') or s.endswith('southward'):
                return float(re.findall(lat_pattern, s)[0])
        return -99

    def _get_vector_min_from_lat_coords(self):
        lats = []
        lat_pattern = '([0-9]+)[ns]'
        for s in self.lat_coords:
            if s.startswith(' n ') or s.startswith('north') or s.endswith('northward'):
                return float(re.findall(lat_pattern, s)[0])
        return -99
    
    
    def _get_explicit_lat_max(self, lat):
        # take any 'south of', 'southward', lat coordinates as the actual max lat
        vector_max = self._get_vector_max_from_lat_coords()
        if vector_max > -99:
            return vector_max
        return lat

    def _get_explicit_lat_min(self, lat):
        # take any 'north of', 'north', lat coordinates as the actual min lat
        vector_min = self._get_vector_min_from_lat_coords()
        if vector_min > -99:
            return vector_min
        return lat

    
    def _get_info_from_lon_coords(self):
        lons = []
        #lon_pattern = '([0-9]+)[w]'
        lon_pattern = '([0-9\.]+)[/w]'
        for s in self.lon_coords:
            lons.extend(abs(float(lon)) for lon in re.findall(lon_pattern, s))
        return lons

    def _get_coords_from_verbal_locations(self):
        '''
        Convert word-based locations and direction descriptors to lat/lon pairs
        '''
        lats, lons = [], []
        for pair in self.verbal_locations:
            direction = global_direction_map[pair[0]]
            loc = pair[1]
            # mexico specifically needs to be southern bound for coast rather than northern bound
            if pair[0] and 'coast' in pair[0] and 'mexico' in loc:
                direction = 's'
            (bounds, center) = global_place_coords[global_loc_keyword_map[pair[1]]]            
            if 'n' in direction:
                lat = bounds[1]
            elif 's' in direction:
                lat = bounds[0]
            else:
                lat = center[0]
            if 'w' in direction:
                lon = bounds[2]
            elif 'e' in direction:
                lon = bounds[3]
            else:
                lon = center[1]
            lats.append(round(lat, 2)), lons.append(round(abs(lon), 2))
        return lats, lons
                
    
    def _give_coords(self):
        '''
        Interprets provided location information into two lat/lon end pairs (note lons are in degrees west).
        '''
        lats = [np.nan]
        lons = [np.nan]
        #[min_lat, min_lon, max_lat, max_lon] = [-99, 0, -99, 0]
        [min_lat, min_lon, max_lat, max_lon] = [np.nan, np.nan, np.nan, np.nan]
        if self.trust == 1 and len(self.coord_pairs) > 1:
            # when trust is 1, we already have coordinate pairs that we will use
            # for our extent, just need to convert them to floats
            pair_lats, pair_lons = self._get_info_from_coord_pairs()
            lats.extend(pair_lats)
            lons.extend(pair_lons)
            # make sure lons remain paired to the correct lats since waves can have tilt
            # so "min_lon" really means the longitude paired with the minimum latitude
            min_lat, max_lat = np.nanmin(lats), np.nanmax(lats)
            min_lon, max_lon = lons[lats.index(min_lat)], lons[lats.index(max_lat)]
        elif self.trust == 1:
            # have lat ranges and longitude data from somewhere, but not in clean pair form
            pair_lats, pair_lons = self._get_info_from_coord_pairs()
            lats.extend(pair_lats)
            lons.extend(pair_lons)
            lats.extend(self._get_info_from_lat_coords())
            lons.extend(self._get_info_from_lon_coords())
            avg_lon = np.nanmean(lons)
            min_lat, max_lat = np.nanmin(lats), np.nanmax(lats)
            min_lon, max_lon = avg_lon, avg_lon
        elif self.trust == 2:        
            # may actually include a low location rather than full extent of a wave,
            # so don't get used for climo
            # need to experiment with a threshold below-which waves should get extended
            # to full climo length - perhaps 50% of climo?
            lons = self._get_info_from_lon_coords()
            # first choice: when we have a standalone longitude coordinate, use that as our
            # longitude for all points
            if len(lons) > 0:
                pair_lats, _ = self._get_info_from_coord_pairs()
                lats.extend(pair_lats)
                lats.extend(self._get_info_from_lat_coords())
                verbal_lats, _ = self._get_coords_from_verbal_locations()
                lats.extend(verbal_lats)
                min_lat, max_lat = np.nanmin(lats), np.nanmax(lats)
                max_lat = self._get_explicit_lat_max(max_lat)
                min_lat = self._get_explicit_lat_min(min_lat)
                min_lon, max_lon = np.nanmean(lons), np.nanmean(lons)
            else:
                # otherwise, take northernmost lat/lon pair and southernmost lat/lon pair
                # unless another lat is more extreme than the ones from the pairs, in which
                # case use that and the lon from the closest pair
                pair_lats, pair_lons = self._get_info_from_coord_pairs()
                verbal_lats, verbal_lons = self._get_coords_from_verbal_locations()
                pair_lats.extend(verbal_lats)
                pair_lons.extend(verbal_lons)
                lats.extend(self._get_info_from_lat_coords())
                min_lat, max_lat = np.nanmin(pair_lats), np.nanmax(pair_lats)
                min_lon, max_lon = pair_lons[pair_lats.index(min_lat)], pair_lons[pair_lats.index(max_lat)]
                max_lat = np.nanmax([max_lat, *lats])
                max_lat = self._get_explicit_lat_max(max_lat)
                min_lat = np.nanmin([min_lat, *lats])
                min_lat = self._get_explicit_lat_min(min_lat)
                
        else:
            lons = self._get_info_from_lon_coords()
            # first choice: when we have a standalone longitude coordinate, use that as our
            # longitude for all points
            if len(lons) > 0:
                pair_lats, _ = self._get_info_from_coord_pairs()
                verbal_lats, _ = self._get_coords_from_verbal_locations()                
            else:
                pair_lats, pair_lons = self._get_info_from_coord_pairs()
                lons.extend(pair_lons)
                verbal_lats, verbal_lons = self._get_coords_from_verbal_locations()
                lons.extend(verbal_lons)
            lats.extend(pair_lats)
            lats.extend(self._get_info_from_lat_coords())
            lats.extend(verbal_lats)
            min_lat, max_lat = np.nanmin(lats), np.nanmax(lats)
            max_lat = self._get_explicit_lat_max(max_lat)
            min_lat = self._get_explicit_lat_min(min_lat)
            min_lon, max_lon = np.nanmean(lons), np.nanmean(lons)
                
        if self.trust == 1 and min_lat == max_lat:
            # 4 or 5 of these are screwed up and actually do have identical latitudes
            self.trust = 4

        self.set_extent((min_lat, min_lon), (max_lat, max_lon))
        #return ((round(min_lat, 2), round(min_lon, 2)), (round(max_lat, 2), round(max_lon, 2)))
    
    def set_extent(self, *pairs):
        (min_lat, min_lon), (max_lat, max_lon) = pairs[0], pairs[1]        
        self.extent = ((round(min_lat, 2), round(min_lon, 2)),
                            (round(max_lat, 2), round(max_lon, 2)))
        #self.extent = tuple(pairs)
                            

    def assign_lat(self, climatology):
        '''
        Changes lat values of this wave to match climatology if needed

        climatology : xarray DataArray of shape (calendar_days, lons, (min_lat, max_lat))
        '''
        if self.trust == 1:
            # these are our climo waves, no change needed
            return
        cur_min_lat, cur_max_lat = self.extent[0][0], self.extent[1][0]
        cur_length = round(np.abs(cur_max_lat - cur_min_lat), 2)
        min_lon, max_lon = self.extent[0][1], self.extent[1][1]
        lon_i = np.abs(np.array(climatology.lon)-np.mean([min_lon, max_lon])).argmin()
        date = self.time[4:8]
        date_i = list(climatology.day).index(date)
        climo_min_lat = round(float(climatology[date_i, lon_i, 0]), 2)
        climo_max_lat = round(float(climatology[date_i, lon_i, 1]), 2)
        climo_length = round(np.abs(climo_max_lat-climo_min_lat), 2)
        vector_max_lat = self._get_vector_max_from_lat_coords()
        vector_min_lat = self._get_vector_min_from_lat_coords()
        #if self.trust == 2 and cur_length < climo_length * 0.5 and not (cur_min_lat == cur_max_lat and vector_max_lat == -99):
        if self.trust == 2 and cur_length < climo_length * 0.5 and not (cur_min_lat == cur_max_lat and vector_max_lat == -99 and vector_min_lat == -99):
            # Assign climo lat extent if extent is less than 50% of climo
            # Add half of the missing extent to each end of the wave's extent
            # Note that we don't do waves with one latitude point here unless it's in a vector
            missing_length = climo_length - cur_length
            if vector_max_lat != -99:
                # use explicit northern end as max_lat
                new_max_lat = vector_max_lat
                new_min_lat = cur_min_lat-missing_length
            elif vector_min_lat != -99:
                # use explicit southern end as min_lat
                new_min_lat = vector_min_lat
                new_max_lat = cur_max_lat+missing_length
            else:
                new_max_lat = cur_max_lat + missing_length/2
                new_min_lat = cur_min_lat - missing_length/2
            
            self.set_extent((new_min_lat, min_lon), (new_max_lat, max_lon))
        elif self.trust == 3:
            # Northern end is the given max_lat, southern end is from climo_length, or vice versa
            #self.set_extent((cur_max_lat - climo_length, min_lon), (cur_max_lat, max_lon))
            if vector_max_lat != -99:
                self.set_extent((cur_max_lat - climo_length, min_lon), (cur_max_lat, max_lon))
            else:
                self.set_extent((cur_min_lat, min_lon), (cur_min_lat + climo_length, max_lon))                
        elif self.trust == 4 or (self.trust == 6 and cur_min_lat == cur_max_lat) or (self.trust==2 and cur_min_lat == cur_max_lat):
            # Ensure the one given lat coord is in bounds, otherwise shift north or south to have
            # given lat coord be at 20% or 80% of the N/S climo extent
            #print(climo_min_lat, climo_max_lat)
            if cur_min_lat != cur_max_lat:
                print('These lat values should be equal', self)
            if cur_min_lat > climo_min_lat and cur_max_lat < climo_max_lat:
                self.set_extent((climo_min_lat, min_lon), (climo_max_lat, max_lon))
            elif cur_max_lat >= climo_max_lat:
                self.set_extent((climo_min_lat + 0.2 * climo_length, min_lon), (cur_max_lat + 0.2 * climo_length, max_lon))
            elif cur_min_lat <= climo_min_lat:
                self.set_extent((cur_min_lat - 0.2 * climo_length, min_lon), (climo_max_lat - 0.2 * climo_length, max_lon))
            else:
                print("This spot, should never have been reached", self)
        elif self.trust == 5 or self.trust == 7:
            # No latitude data given, so assign climo values
            self.set_extent((climo_min_lat, min_lon), (climo_max_lat, max_lon))
        elif self.trust == 6:
            # Don't trust latitude values very much, so don't go beyond climo length
            # If climo length is exceeded, shrink by equal amount on north and south to reach climo length again
            if cur_length > climo_length:
                diff = cur_length - climo_length
                self.set_extent((cur_min_lat + diff/2, min_lon), (cur_max_lat - diff/2, max_lon))
                
        
