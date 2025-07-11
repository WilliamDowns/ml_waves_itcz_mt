'''
A collection of regridding routines mostly taken from GCPy
'''


import os
import xesmf as xe
import numpy as np
import xarray as xr
import pandas as pd
import scipy.sparse
import warnings



def regrid_datasets(ref, dev, weightsdir='~/regridding_weights', direction='high'):
    '''
    Regrids two input datasets to the largest overlapping extent,
    as well as a common coordinate format.

    ref : xarray Dataset or DataArray
    dev : xarray Dataset or DataArray
    weightsdir: str, where regridding weights files are stored / should be stored
    direction : str, whether to regrid to higher or lower res data. possible values: high or low
    
    returns
    -------
    new_ref : regridded ref dataset
    new_dev : regridded dev dataset
    cmp_extent : overlapping extent of grids
    '''


    # get maximum extent of each dataset and their overlap
    ref_extent = get_grid_extents(ref)
    dev_extent = get_grid_extents(dev)
    cmp_extent = get_maximum_extent(ref_extent, dev_extent)
    
    # check to make sure data is not screwed up (0-length extent dimension)
    # if only one is, return the extent of the other and a full array of 0s
    # for the bad one, otherwise if both are bad return the whole globe
    bad_ref = ref_extent[0] == ref_extent[1] or ref_extent[2] == ref_extent[3]
    bad_dev = dev_extent[0] == dev_extent[1] or dev_extent[2] == dev_extent[3]
    
    if bad_ref and bad_dev:
        print('WARNING: ref_extent and dev_extent both have 0-length dimensions: ',
              ref_extent, dev_extent, ' using [-180, 180, -90, 90] and not regridding data')
        return ref, dev, [-180, 180, -90, 90]
    
    if bad_ref:
        new_ref = dev.copy(deep=True)
        new_ref = new_ref.where(new_ref==0, other=0)
        new_ref = ds_to_lat_lon(new_ref)
        new_dev = ds_to_lat_lon(dev)        
        print('WARNING: ref_extent has 0-length dimension: ', ref_extent,
              ' using ', dev_extent, ' and not regridding data')
        return new_ref, new_dev, dev_extent

    if bad_dev:
        new_dev = ref.copy(deep=True)
        new_dev = new_dev.where(new_dev==0, other=0)
        new_ref = ds_to_lat_lon(ref)
        new_dev = ds_to_lat_lon(new_dev)        
        print('WARNING: dev_extent has 0-length dimension: ', dev_extent,
              ' using ', ref_extent, ' and not regridding data')
        return new_ref, new_dev, ref_extent        
    
    # limit each dataset to overlap between them
    new_ref = filter_by_extent(ref, cmp_extent)
    new_dev = filter_by_extent(dev, cmp_extent)

    ref_res, ref_gridtype = get_input_res(new_ref)
    dev_res, dev_gridtype = get_input_res(new_dev)

    # change coord names to 'lat' and 'lon' for consistency / regridding
    new_ref = ds_to_lat_lon(new_ref)
    new_dev = ds_to_lat_lon(new_dev)

    # Right now going to assume that xESMF can handle regridding rectilinear
    # to and from curvilinear innately rather than worrying about differentiating
    
    # Regrid if necessary. Note smaller res value here = higher resolution
    float_ref_res = float(ref_res.split('x')[0])*float(ref_res.split('x')[1])
    float_dev_res = float(dev_res.split('x')[0])*float(dev_res.split('x')[1])    
    if (float_ref_res < float_dev_res and direction=='high') or (float_ref_res > float_dev_res and direction=='low'):
        # make regridder from dev to ref, regrid dev to ref
        regridder = make_regridder(new_dev, new_ref, cmp_extent, weightsdir, dev_res, ref_res)
        new_dev = regridder(new_dev)
    elif (float_dev_res < float_ref_res and direction=='high') or (float_dev_res > float_ref_res and direction=='low'):
        # make regridder from ref to dev, regrid ref to dev
        regridder = make_regridder(new_ref, new_dev, cmp_extent, weightsdir, ref_res, dev_res)
        new_ref = regridder(new_ref)
        
    return new_ref, new_dev, cmp_extent


def make_regridder(init_ds, targ_ds, cmp_extent, weightsdir='~/regridding_weights',
                   init_res = '', targ_res = '', regrid_method='conservative'):
    '''
    Creates (if necessary) an xESMF regridder between the grid of
    init_ds and that of targ_ds and performs regridding of init_ds

    init_ds : xr.Dataset or xr.DataArray
    targ_ds : xr.Dataset or xr.DataArray
    cmp_extent : [minlon, maxlon, minlat, maxlat], for file naming
    weightsdir : str, target for weight files
    init_res : str, resolution of init_ds
    targ_res : str, resolution of targ_ds
    
    Return : xr.Dataset, the regridded version of init_ds
    '''
    if init_res == '':
        init_res, _ = get_input_res(init_ds)
    if targ_res == '':
        targ_res, _ = get_input_res(targ_ds)

    cmp_extent_str = str(cmp_extent).replace(
        '[', '').replace(
            ']', '').replace(
                ', ', 'x')

    #regrid_method = 'bilinear'
    #regrid_method = 'conservative'
    weightsfile = os.path.expanduser(os.path.join(weightsdir, regrid_method + '_{}_{}_{}.nc'.format(
    init_res, targ_res, cmp_extent_str)))

    reuse_weights = True
    if not os.path.isfile(weightsfile):
        #prevent error with more recent versions of xesmf
        reuse_weights=False

    # regridder needs xarray datasets (or dicts, but I don't want to deal with those)
    if type(init_ds) is xr.DataArray:
        init_ds = init_ds.to_dataset()
    if type(targ_ds) is xr.DataArray:
        targ_ds = targ_ds.to_dataset()

    
    try:
        regridder = xe.Regridder(
            init_ds,
            targ_ds,
            method=regrid_method,
            filename=weightsfile,
            reuse_weights=reuse_weights)
    except BaseException:
        #joblib issue
        regridder = xe.Regridder(
            init_ds,
            targ_ds,
            method=regrid_method,
            filename=weightsfile,
            reuse_weights=reuse_weights)

    return regridder
            
    
def ds_to_lat_lon(ds):
    '''
    Takes an xarray Dataset or DataArray and renames
    in a format with lat and lon coordinates. Assumes single timestep

    Supported input formats: ERA5, WRF
    '''
    lat_name, lon_name = get_lat_lon_names(ds)
    return ds.rename({lat_name : 'lat', lon_name : 'lon'})


def get_lat_lon_names(data):
    '''
    Retrieve the names of the lat and lon coords of the input dataset

    data : xr.Dataset or xr.DataArray

    Return : lat_name, lon_name
    '''

    if "XLONG" in data.coords:        
        #WRF
        lat_name ="XLAT"
        lon_name ="XLONG"

    elif "latitude" in data.coords:        
        #ERA5
        lat_name = "latitude"
        lon_name = "longitude"        
    elif "lat" in data.coords:
        # assume lat/lon format
        lat_name = "lat"
        lon_name = "lon"
    else:
        # can't find the coords
        lat_name = 'NO_LAT_NAME_FOUND'
        lon_name = 'NO_LON_NAME_FOUND'

    return lat_name, lon_name
    
    
def get_grid_extents(data, edges=False):
    """
    Get min and max lat and lon from an input dataset

    Args:
        data: xarray Dataset or DataArray            
        edges (optional): bool, whether grid extents should use cell edges instead of centers

    Returns:
        minlon: float
            Minimum longitude of data grid
        maxlon: float
            Maximum longitude of data grid
        minlat: float
            Minimum latitude of data grid
        maxlat: float
            Maximum latitude of data grid
    """
    lat_name, lon_name = get_lat_lon_names(data)
    lat = data[lat_name].values
    lon = data[lon_name].values

    minlat = float(np.min(lat))
    maxlat = float(np.max(lat))
    minlon = float(np.min(lon))
    maxlon = float(np.max(lon))
    # add longitude res to max longitude if needed
    if edges == True:
        maxlon = float(np.max(lon) + abs(abs(lon[-1]) - abs(lon[-2])))
    return [minlon, maxlon, minlat, maxlat]
    
    
def get_maximum_extent(ref_extent, dev_extent):
    '''
    Return largest overlapping area between two extent lists
    ref_extent : [minlon, maxlon, minlat, maxlat]
    dev_extent : [minlon, maxlon, minlat, maxlat]

    Return : [minlon, maxlon, minlat, maxlat]
    '''

    [refminlon, refmaxlon, refminlat, refmaxlat] = ref_extent
    [devminlon, devmaxlon, devminlat, devmaxlat] = dev_extent

    return [np.max([refminlon, devminlon]), np.min([refmaxlon, devmaxlon]),
            np.max([refminlat, devminlat]), np.min([refmaxlat, devmaxlat])]


def filter_by_extent(data, extent):
    '''
    Trim data down to match extent bounds.

    data : xr.Dataset or xr.DataArray
    extent : [minlon, maxlon, minlat, maxlat]

    Return : new_data filtered down
    '''

    new_data = data.copy(deep=True)

    [minlon, maxlon, minlat, maxlat] = extent

    lat_name, lon_name = get_lat_lon_names(new_data)

    # do differently for rectilinear vs. curvilinear grids
    if len(np.shape(new_data[lon_name])) < 2:
        new_data = new_data.where(new_data[lon_name] >= minlon, drop=True).\
            where(new_data[lon_name] <= maxlon, drop=True).\
            where(new_data[lat_name] >= minlat, drop=True).\
            where(new_data[lat_name] <= maxlat, drop=True)
    else:
        # assume format [south_north, west_east] for coordinates
        mask = np.full(np.shape(new_data[lon_name]), True)
        for i in range(len(mask)):
            # want to drop rows where minimum value in all cols is < minlat
            # want to drop rows where maximum value in all cols is > maxlat        
            if np.all(new_data[lat_name][i] < minlat):
                mask[i] = np.full(np.shape(mask[i]), False)
            if np.all(new_data[lat_name][i] > maxlat):
                mask[i] = np.full(np.shape(mask[i]), False)
        for i in range(len(mask[0])):
            # want to drop cols where minimum value in all rows is < minlon
            # want to drop cols where maximum value in all rows is > maxlon
            if np.all(new_data[lon_name][:, i] < minlon):
                mask[:, i] = np.full(np.shape(mask[:, i]), False)
            if np.all(new_data[lon_name][:, i] > maxlon):
                mask[:, i] = np.full(np.shape(mask[:, i]), False)
        mask = xr.DataArray(mask, dims=["south_north", "west_east"])
        new_data = new_data.where(mask, drop=True)

    return new_data


def get_input_res(data):
    """
    Returns resolution of data and whether it's rectilinear or curvilinear

    data : xarray Dataset or xarray DataArray
    
    Return : 
        res : str or int
            Lat/lon res of the form 'latresxlonres'
        gridtype : str
            'rect' for rectilinear or 'curv' for curvilinear

    """
    lat_name, lon_name = get_lat_lon_names(data)
    lat = data[lat_name].values
    lon = data[lon_name].values
    if 'XLONG' == lon_name:
        #WRFs
        lat_diffs = [lat[i][j] - lat[i-1][j]
                     for i in range(1, len(lat))
                     for j in range(len(lat[i]))]

        lon_diffs = [lon[j][i] - lon[j][i-1]
                     for j in range(len(lon))
                     for i in range(1, len(lon[j]))]
        
        lat_res = round(float(np.mean(lat_diffs)), 4)
        lon_res = round(float(np.mean(lon_diffs)), 4)           
    else:
        #ERA5 or some other lat/lon grid
        lat_res = float(np.abs(lat[2] - lat[1]))
        lon_res = float(np.abs(lon[2] - lon[1]))

    #curvilinear should have (x,y) format for lat and lon
    if len(np.shape(lat)) == 1:        
        gridtype = 'rect'
    else:
        gridtype = 'curv'

    return str(lat_res) + "x" + str(lon_res), gridtype

