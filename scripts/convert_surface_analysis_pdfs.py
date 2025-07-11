'''
Convert PDFs of surface analysis maps into images optionally cropped 
'''

from joblib import Parallel, delayed
from pdf2image import convert_from_path
import os

base_dir = '/your_dir/'
#base_dir = '/your_dir/'

crop = False

# trim image to 40n, 0n, 145w, 0e
# used pixspy.com for pixel ID 
# pixel bounds (origin at upper left): 1168 upper, 2870 lower, 951 left, 6595 right
crop_upper = 1168
crop_lower = 2870
crop_left  = 951
crop_right = 6595

#years = [str(y) for y in range(2004, 2023)]
years = [str(2021)]
day_strs = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
            '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
            '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31']
month_days = {'01' : day_strs, '02' : day_strs[:-3], '03' : day_strs, '04' : day_strs[:-1], '05': day_strs,
              '06' : day_strs[:-1], '07' : day_strs, '08' : day_strs, '09' : day_strs[:-1], '10' : day_strs, '11' : day_strs[:-1], '12' : day_strs}
months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
timesteps = ['0000', '0600', '1200', '1800']
#times = [''.join([m, d, t]) for m in months[1:-1]
#         for d in month_days[m] for t in timesteps]
times = [''.join([m, d, t]) for m in months
         for d in month_days[m] for t in timesteps]

surface_analysis_path = '/your_dir/ml_waves/surface_analysis_maps/tafb/surface_analysis/'
target_path = '/your_dir/ml_waves/surface_analysis_maps/converted/'

for year in years:
    year_times = [''.join([year, time]) for time in times]
    def convert_image(time):
        pdf_path = ''.join([surface_analysis_path, time[:4], '/', time[4:6], '/', 'tsfc_', time[:-2], '.pdf'])
        image_path = ''.join([target_path, 'tsfc_atl_epac', time[:-2], '.png'])
        if os.path.exists(image_path):
            return
        # not all surface analyses exist
        try:
            im = convert_from_path(pdf_path)
        except:
            return
        im = convert_from_path(pdf_path)
        # optionally crop image
        if crop:
            im = [im[0].crop((crop_left, crop_upper, crop_right, crop_lower))]
        im[0].save(image_path)
        print(time)

    Parallel(n_jobs=-1)(delayed(convert_image)(time) for time in year_times)
#Parallel(n_jobs=-1)(delayed(convert_image)(year) for year in [str(y) for y in range(2020, 2023)])

