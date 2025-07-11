# Download ERA5 data for 1980-2023 for creating full climatology of predictions of waves and ITCZ.
# Lots of redundancy with data downloaded in original training scripts, so that data will be
# deleted when this is done.

import cdsapi
from joblib import Parallel, delayed
import pandas as pd

from ml_waves_itcz_mt.util import *
from ml_waves_itcz_mt.plotting import *

if __name__ == "__main__":

    years = [str(year) for year in range(1980, 2024)]
    # added vorticity back because am lazy and that's the easiest way to calculate shear vorticity
    variables = ['u_component_of_wind', 'v_component_of_wind', 'specific_humidity',
                 'total_column_water']
    
    variable_name_map = {'u_component_of_wind' : 'U', 'v_component_of_wind' : 'V', 'temperature' : 'T', 'relative_humidity' : 'r',
                         'vorticity' : 'vo', 'potential_vorticity' : 'pv', 'divergence' : 'div', 'geopotential' : 'z', 'vertical_velocity' : 'omega',
                         'specific_humidity' : 'q', 'total_column_water' : 'tcw'}

    levels = ['1000', '925', '850', '750', '650', '550']#[::-1]
    area = [32, -180, -17.75, 60]
    days =  ['01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
            '13', '14', '15',
            '16', '17', '18',
            '19', '20', '21',
            '22', '23', '24',
            '25', '26', '27',
            '28', '29', '30', '31']
    months = ['01', '02', '03',
              '04', '05', '06',
              '07', '08', '09',
              '10', '11', '12']
    times = ['00:00', '06:00', '12:00',
            '18:00']
    target_dir = '/your_dir/era5_data/unet_wave_imt_full_pred_climo_-180_60_-17.75_32_0.25x0.25/'

    
    #def data_fetch(year):
    def data_fetch(var):
        #for var in variables:
        for year in years:        
            f = target_dir + variable_name_map[var] + '_-180_60_-17.75_32_0.25x0.25_' + year +  '.nc'
            if os.path.exists(f):
                continue
            c = cdsapi.Client()
            if var != 'total_column_water':
                c.retrieve(
                    'reanalysis-era5-pressure-levels',
                    {
                        'product_type': 'reanalysis',
                        'variable': var,
                        'pressure_level': levels,
                        'year': year,
                        'month': months,
                        'day': days,
                        'time': times,                    
                        'format': 'netcdf',
                        'area' : area,
                    },
                    f)
            else:
                c.retrieve(
                    'reanalysis-era5-single-levels',
                    {
                        'product_type': 'reanalysis',
                        'variable': var,
                        'year': year,
                        'month': months,
                        'day': days,
                        'time': times,
                        'area': area,
                        'format': 'netcdf',
                    },
                    f)
                
            
    #Parallel(n_jobs=-1)(delayed(data_fetch)(year) for year in years)
    Parallel(n_jobs=-1)(delayed(data_fetch)(var) for var in variables)


