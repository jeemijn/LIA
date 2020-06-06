#!/usr/bin/env python
# coding: utf-8

"""This file contains helper functions that are read in and used in LIA.ipynb. 
Functions to compute are in the first half; functions to plot in the second half (in random order)."""


def load_data_multiple_runs(folder, runs, spinup_yr=1765, full=True, full_inst=False):
    """Input: 
    - folder 
    - runs is string array of runnames 
    - spinup_yr [optional; if other than 1765] is an int if all simulations have equal spinup; otherwise int array
      N.B. needed since file_path = folder + runname + spinup_yr 
    - full [optional] if you want (no) full_ave.nc file (e.g. not generated for runs with output every time step)
    - full_inst [optional] if you want full_inst.nc file as well (for special runs diagnosing convection or seasonal cycle)
    
    Output:
    - [datas, data_fulls (optional; default), data_full_inst(optional)] 
       contains 1 to 3 dictionaries with runs; depending on chosen parameters

    Explanation of output:
    1) data = data from timeseries_ave.nc output file
    2) data_full = data from full_ave.nc output file 
    3) data_full_inst = data from full_inst.nc output file

    For all 3: the year axis is changed from simulation years to years they represent in C.E.
    
    Author: Jeemijn Scheen, jeemijn.scheen@climate.unibe.ch"""
    
    from xarray import open_dataset
    from numpy import ndarray
    
    datas = {}           # empty dict for timeseries output DataSets
    if full:
        data_fulls = {}  # empty dict for full output DataSets
    if full_inst:
        data_fulls_inst = {} 
        
    # change years to simulation years
    subtract_yrs = spinup_yr 
    
    for nr, runname in enumerate(runs):
        # set up path
        if spinup_yr == 0:  # ignoring all problems with cases of spinup 0 C.E. to 999 C.E.
            spinup_yr_str = '0000'
        else:
            if isinstance(spinup_yr, (list, tuple, ndarray)):  # test if spinup_yr is a list/tuple/array
                spinup_yr = spinup_yr[nr]
            spinup_yr_str = str(spinup_yr)
        path = folder + runname + '.000' + spinup_yr_str
        # create datas
        datas[runname] = open_dataset(path + '_timeseries_ave.nc', decode_times=False) 
        datas[runname]['time'] -= subtract_yrs
        # create data_fulls
        if full:
            data_fulls[runname] = open_dataset(path + '_full_ave.nc', decode_times=False)
            data_fulls[runname]['time'] -= subtract_yrs
        # create data_fulls_inst using dask arrays (chunks) because of large file sizes
        if full_inst:
            data_fulls_inst[runname] = open_dataset(path + '_full_inst.nc', decode_times=False, chunks={'yearstep_oc': 20})
            data_fulls_inst[runname]['time'] -= subtract_yrs
            
    # generate resulting array in correct order
    res = [datas]
    if full:
        res.append(data_fulls)
    if full_inst:
        res.append(data_fulls_inst)
    return res
        
def area_mean(obj, obj_with_data_var, keep_lat=False, keep_lon=False, basin=""):
    '''Takes horizontal area-weighted average of a certain data_var. 
    Note: another averaging method is implemented in vol_mean(): more intuitive; same result
    although this function area_mean rounds differently in the last digits (* data_var_no0 / data_var_no0 below)
    SO IT IS RECOMMENDED TO USE VOL_MEAN INSTEAD
    
    - obj must be a DataSet with data variable 'area' and coordinates 'lat_t', 'lon_t'
    - obj_with_data_var must contain the data_var wanted e.g. data_full.TEMP
    - basin can be set; otherwise the result will be too small by a fixed factor.   
    options: 'pac' and 'atl' (mask 2 and 1, resp.) and 'pacso' and 'atlso' (masks 2 and 1, resp.)
    - if keep_lat is True then latitude is kept as a variable and the area_weight is only done over longitude.       
    - if keep_lon is True then area_weight is only done over latitude.
    
    Author: Jeemijn Scheen, jeemijn.scheen@climate.unibe.ch'''       
        
    if keep_lat and keep_lon:
        raise Exception("not possible to average when both keep_lat and keep_lon.")
        
    weighted_data = obj_with_data_var * obj.area       # element-wise mult 
    
    if 'z_t' in obj_with_data_var.dims: 
        mask = obj.mask
        masks = obj.masks
    else: # alternative if no z_t in obj_with_data_var, otherwise a z_t dimension would be added to obj_with_data_var 
        mask = obj.mask.isel(z_t=0)
        masks = obj.masks.isel(z_t=0)
    
    ## find area weights:
    # first we construct a data_var_no0, only used for correct shape and mask of required area object
    # replace 0 by 1 since we don't want to replace 0 with nan or divide by 0 in next step: 
    if basin == 'pac':
        data_var_no0 = obj_with_data_var.where(mask==2).where(obj_with_data_var != 0.00000000, 1.0)         
    elif basin == 'atl':
        data_var_no0 = obj_with_data_var.where(mask==1).where(obj_with_data_var != 0.00000000, 1.0)
    elif basin == 'so':
        data_var_no0 = obj_with_data_var.where(mask==4).where(obj_with_data_var != 0.00000000, 1.0)
    elif basin == 'pacso':
        data_var_no0 = obj_with_data_var.where(masks==2).where(obj_with_data_var != 0.00000000, 1.0) 
    elif basin == 'atlso':
        data_var_no0 = obj_with_data_var.where(masks==1).where(obj_with_data_var != 0.00000000, 1.0)
    elif basin == "":
        data_var_no0 = obj_with_data_var.where(obj_with_data_var != 0.00000000, 1.0) 
    else:
        raise Exception("basin should be empty '' or one out of: 'pac', 'atl', 'pacso', 'atlso', 'so'.")
    area = obj.area * data_var_no0 / data_var_no0    # expand area to dimensions & MASK of data_var
    # important; otherwise sum of weights still includes land (global avg SST will be 4 degrees too low)

    if keep_lat:
        weights = area.sum(dim='lon_t')            # one value for each z_t,time
        return weighted_data.sum(dim='lon_t') / weights.where(weights != 0) # weighted average
    elif keep_lon:
        weights = area.sum(dim='lat_t')            # one value for each z_t,time
        return weighted_data.sum(dim='lat_t') / weights.where(weights != 0) # weighted average
    else:
        weights = area.sum(dim='lat_t').sum(dim='lon_t')            # one value for each z_t,time
        return weighted_data.sum(dim='lon_t').sum(dim='lat_t') / weights.where(weights != 0) # weighted average

def vol_mean(data_obj, vol, keep_z=False, keep_latlon=False):
    '''Takes volume-weighted average of a certain data_var in horizontal and/or vertical direction. 
    If the data_var has a time coord, then this time coord is always kept (output is array).
    If the data_var has a z coord, then this is kept only if keep_z is True (otherwise averaged over z as well).
    Input:
    - data_obj must be the data_var wanted (e.g. data_full.TEMP) with coords: 
            lat_t & lon_t & optionally time or z_t
    - vol must contain the grid-cell volumes i.e. data_full.boxvol
    - keep_z indicates whether to keep the z_t dimension [default False]
    - keep_latlon indicates whether to keep the lat and lon dimension [default False]
    NB keep_z and keep_latlon cannot both be true.
    Output:
    - average over lat and lon (if keep_latlon is False) and over z (if keep_z is False).
         Default output:               scalar
         If obj has a time coord:      1D array in time
         If keep_z is True:            1D array in z 
         If keep_latlon is True:       2D array in lat,lon
         If both keep_z and time:      2D array in time and z
         If both keep_latlon and time: 3D array in time, lat and lon
         
    Author: Jeemijn Scheen, jeemijn.scheen@climate.unibe.ch'''

    from numpy import tile, isnan, average, sort
    from xarray import DataArray
    
    obj = data_obj.copy(deep=True) # this line is important because:
    # Replacing nans with zeros in obj (done below) otherwise affects the original input data 
    # in certain cases. Quite unpredictable because it occured:
    # when first calling vol_mean, then area_mean and then vol_mean and THEN vol_mean for the 2nd time.
    # This gave a wrong global average SST of 13.1 instead of 17.9 C.
    
    coords = obj.dims
    
    if keep_z and keep_latlon:
        raise Exception("with keep_z and keep_latlon both True, there is no average to compute.")
    
    if 'z_t' not in coords and 'z_t' in vol.dims: 
            weights = vol.isel(z_t=0).values#.copy()
    else: 
        # weighting by surface layer works for any fixed depth layer because ocean bottom values are nan in the data
        weights = vol.values   
    if 'time' in coords:
        if 'z_t' in coords:
            weights = tile(weights, (len(obj.time),1,1,1)) # extend weights in time direction
        else:
            weights = tile(weights, (len(obj.time),1,1)) # extend weights in time direction

    # handle NaN values (where land or masked out)
    try:
        weights[isnan(obj.values)] = 0             # needed such that total weights are correct
        obj.values[isnan(obj.values)] = 0          # needed, otherwise still nan * 0 = nan
    except:
        raise Exception("the shape of weights " + str(weights.shape) +
              " is not equal to that of data_var " + str(obj.shape))

    axes = [] # to save axes over which to average
    if keep_latlon is False:
        axes.append(coords.index('lat_t'))
        axes.append(coords.index('lon_t'))
    if 'z_t' in coords and keep_z is False:
        axes.append(coords.index('z_t'))
    axes = tuple(sort(axes))
    
    res = average(obj, axis = axes, weights = weights)    
    # using np.average because it has weigths as parameter (xr.mean() does not) 
    # but now we have to convert result back to xarray dataset
    
    if len(coords) == len(axes):  # res is scalar 
        return res
    elif len(res.shape) == 1:     # res is 1D
        if 'time' in coords:      # res has 1D time axis
            return DataArray(res, coords=[obj.time], dims=['time'])
        elif keep_z:              # res has 1D z_t axis
            return DataArray(res, coords=[obj.z_t], dims=['z_t'])
    elif len(res.shape) == 2:     # res is 2D
        if 'time' in coords:      # res has 2D time, z_t axes
            return DataArray(res, coords=[obj.time, obj.z_t], dims=['time', 'z_t'])
        elif keep_latlon:         # res has 2D lat_t, lon_t axes
            return DataArray(res, coords=[obj.lat_t, obj.lon_t], dims=['lat_t', 'lon_t'])    
    elif len(res.shape) == 3:     # res is 3D
        return DataArray(res, coords=[obj.time, obj.lat_t, obj.lon_t], dims=['time', 'lat_t', 'lon_t'])
    else:
        raise Exception("something went wrong")
# TESTED this vol_mean function is perfectly identical to the previous area_mean():
# obj_var = data_full_ref.TEMP.where(data_full_ref.mask == 2)
# temp_start_pac = f.area_mean(data_full_ref, obj_var, basin = 'pac') 
# temp_start_pac2 = f.vol_mean(obj_var, data_full_ref.boxvol, keep_z = True)
# abs(temp_start_pac2 - temp_start_pac2) < 1e-14  # all True


def area_mean_dye_regions(obj, boxvol, region=''):
    """Takes area averaged over a certain dye region, regarding mask with 8 dye tracers as defined below.
    Input:
    - obj is object to average with [time, lat, lon] coords e.g. sst
    - boxvol is .boxvol object (not sliced)
    - dye is a string out of 'NADW' 'NAIW' 'SAIW' 'NPIW' 'SPIW' 'SO' 'Arctic' 'Tropics'
    Note that this function is confusing because input is the dye name of a water mass (NADW) but what is actually used
    is a certain (surface) area corresponding to that like North Atlantic. 
    Output:
    - area average over requested dye region, keeping the time coordinate
    Author: Jeemijn Scheen, jeemijn.scheen@climate.unibe.ch"""
    
    from xarray import where
    from numpy import nan
    
    vol = boxvol.isel(z_t=0)
    
    if region == '':
        raise Exception("Enter a dye region")
    elif region == 'NADW':
        return vol_mean(obj[:,33:38,20:32], vol[33:38,20:32])
    elif region == 'SO':
        return vol_mean(obj[:,0:9,:], vol[0:9,:])
    elif region == 'NAIW':
        return vol_mean(obj[:,29:33,19:35], vol[29:33,19:35])
    elif region == 'SAIW':
        return vol_mean(obj[:,9:13,21:33], vol[9:13,21:33])
    elif region == 'NPIW':
        return vol_mean(obj[:,29:37,2:14], vol[29:37,2:14])
    elif region == 'SPIW':
        return vol_mean(obj[:,9:13,5:20], vol[9:13,5:20])
    elif region == 'Arctic' or region == 'arctic':
        # first subtract a few grid cells that belong to NADW, not Nordic Seas...
        mask = (obj.lat_t > 70) & (obj.lat_t < 75) & (obj.lon_t > 290) & (obj.lon_t < 375) 
        obj_nordic = where(mask, nan, obj)  # this changes order of obj[time,lat,lon] to [lat,lon,time]
        obj_nordic = obj_nordic.transpose('time', 'lat_t', 'lon_t') # change back
        vol_nordic = where(mask, nan, vol)  # vol order unchanged [lat,lon]
        obj_nordic = obj_nordic[:,37:40,:] # ... then slice to Nordic Seas
        vol_nordic = vol_nordic[37:40,:] 
        return vol_mean(obj_nordic, vol_nordic)
    elif region == 'Tropics' or region == 'tropics':
        # first subtract a few grid cells that belong to SIW, not rest/tropics...
        mask = (obj.lat_t > -48) & (obj.lat_t < -30) & (obj.lon_t > 150) & (obj.lon_t < 380) 
        obj_trop = where(mask, nan, obj) # this changes order of obj[time,lat,lon] to [lat,lon,time]
        obj_trop = obj_trop.transpose('time', 'lat_t', 'lon_t')
        vol_trop = where(mask, nan, vol) # vol order unchanged [lat,lon]
        obj_trop = obj_trop[:,9:29,:] # ... then slice to rest/tropics
        vol_trop = vol_trop[9:29,:] 
        return vol_mean(obj_trop, vol_trop)    
    else:
        raise Exception("Enter a valid dye region")

def temp_basin(run_t, run_f, anoms=True):
    """Prepares temperature anomaly data per basin. This can be used for Hoevmiller plot or leads and lags plot.
    Input:
    - run_t is data_full of transient run
    - run_f is data_full of fixed run
    - anoms determines whether returned values are anomalies (and the unit of output)
    
    Output:
    - if anoms [default]: 
    temperature anomaly in centi-Kelvin w.r.t. year 0 per basin and per simulation (transient or fixed) 
    in this order:
    [pac_t, pac_f, atl_t, atl_f, so_t, so_f]
    - if not anoms:
    temperature in centi-Celsius per basin and per simulation (transient or fixed) in the same order
    
    NB the pac and atl mask exclude the southern ocean.
    
    Author: Jeemijn Scheen, jeemijn.scheen@climate.unibe.ch"""
    
    vol = run_t.boxvol
    
    # temp anomaly vs depth in Kelvin:
    pac_t = vol_mean(run_t.TEMP.where(run_t.mask == 2), vol, keep_z=True)
    pac_f = vol_mean(run_f.TEMP.where(run_t.mask == 2), vol, keep_z=True) 
    atl_t = vol_mean(run_t.TEMP.where(run_t.mask == 1), vol, keep_z=True)
    atl_f = vol_mean(run_f.TEMP.where(run_t.mask == 1), vol, keep_z=True)
    so_t =  vol_mean(run_t.TEMP.where(run_t.mask == 4), vol, keep_z=True)
    so_f =  vol_mean(run_f.TEMP.where(run_t.mask == 4), vol, keep_z=True)
    
    if anoms == False:
        return [100 * x for x in [pac_t, pac_f, atl_t, atl_f, so_t, so_f]]  # temperature in centi-Celsius
    else:
        # save temperatures in year 0 in order to subtract for anomaly
        temp0pac = pac_t[0] # shape is time, z
        temp0atl = atl_t[0]
        temp0so  = so_t[0]
        # subtract anomaly and convert to cK (centi-Kelvin)
        pac_t = (pac_t - temp0pac) * 100
        pac_f = (pac_f - temp0pac) * 100
        atl_t = (atl_t - temp0atl) * 100
        atl_f = (atl_f - temp0atl) * 100
        so_t  = (so_t  - temp0so) * 100
        so_f  = (so_f  - temp0so) * 100

        return [pac_t, pac_f, atl_t, atl_f, so_t, so_f]  # temp anomalies in cK

def find_min(obj, in_obj=None):
    '''Gives minimum of a dataarray in a convenient way. 
    Input: an xarray dataarray called obj with a coordinate named time
    Output: [t,y] where y is the value of the minimum of the array and this occurs at time t
    Optional: in_obj = something.time is a time coordinate that has a larger stepsize (i.e. from data_full)
    Then the output is given as: [t, y, t_rounded] where t_rounded is the time closest to the minimum in coarser grid
    Author: Jeemijn Scheen, jeemijn.scheen@climate.unibe.ch'''

    minm = obj.where(obj == obj.min(), drop = True)
    y = minm.item()
    t = minm.time.item()
    
    if in_obj is not None:
        # compare time t with the coarser time dataset by minimizing the difference.
        t_rounded = in_obj.where(abs(in_obj - t) == abs(in_obj - t).min(), drop=True)
        if len(t_rounded) > 1: # in case t is exactly in between 2 times in courser grid
            t_rounded = t_rounded[1] # pick the 2nd one
        return [t, y, t_rounded.item()]
    return [t,y]

# find ridges of minimal and maximal temperature i.e. where cooling changes to warming or vice versa 
def find_ridges(obj, only_min=True, max_guess=600.0, min_guess=1750.0, fast=True):
    """Finds the ridges of minimal and maximal values within e.g. temperature time series. 
    That is, where the warming changes to a cooling or vice versa. The ridges can then be plot in a contour plot.
    
    If only_min, then a global minimum is searched for (e.g. for LIA-Industrial_warming simulations);
    else both 1 minimum and 1 maximum are searched for (e.g. for MCA-LIA-Industrial_warming simulations).
    
    Input: 
    - obj must be an xarray DataArray with z_t and time coords, here: temp_diff_per_depth (values from TEMP)
    - only_min [default True] see above
      NB in this case the search is highly simplified since we can take the global minimum
    - max_guess [optional] is year C.E. of first maximum in forcing; used as a guess of first maximum at surface 
      NB max_guess is not used if only_min
    - min_guess [optional] is year C.E. of first minimum in forcing; used as a guess of first minimum at surface
    - fast can be set if only_min. In this case the ridges are found faster, but less precise; namely, rounded to the 
    frequency of the output (e.g. 5 years) instead of interpolating with a parabola in between.
    For contour plots with an output frequency of 5 years there is no visual difference (so use fast), 
    but when using the integer number of delays it is better without rounding (without fast).

    Output:
    - if only_min: ridge_min
    - else: [ridge_min, ridge_max],
    where each ridge is an array over depth steps containing the year of min/max temp value at this depth step.
    
    NB IF YOU GET AN ERROR "too many values to unpack" then somewhere in calling this function or its subfunctions you did 
    [a,b] = call... instead of a = call...
    
    Author: Jeemijn Scheen, jeemijn.scheen@climate.unibe.ch"""
    
    from numpy import zeros, sort, array, where
   
    delta_t = (obj.time[2] - obj.time[1]).item() # assuming delta_t is constant
    ridge_min = zeros(len(obj.z_t))
    ridge_max = zeros(len(obj.z_t))
    
    if only_min: # ONLY FIND 1 MINIMUM RIDGE
        if fast: # METHOD: just take global minimum (fast; can be good enough if high resolution output)
            for n,z in enumerate(obj.z_t):
                minm_rough = find_min(obj.sel(z_t=z)) # finds global minm [time, val] rounded to output freq eg 5 yr
                ridge_min[n] = minm_rough[0]
        else:    # METHOD: take global minm (has output resolution) & find nearest true minm with method similar to below
            for n,z in enumerate(obj.z_t):
                # find timestamps of the interesting min:
                minm_rough = find_min(obj.sel(z_t=z)) # finds global minm [time, val] rounded to output freq eg 5 yr
                # trick: global minm now given in as min_guess for each depth layer separately:
                t_min = find_local_min(obj.sel(z_t=z), min_guess=minm_rough[0]) 
                ridge_min[n] = t_min
    else:  # METHOD TO FIND BOTH 1 MIN AND 1 MAX RIDGE: 
        # fitting parabola with polifit() using 5 datapoints in running window & selecting by 
        # A) monotonically and B) starting from min_guess, max_guess and C) following ridge downwards
        # THIS IS VERY SIMILAR TO find_local_min_max() BUT FOR ALL DEPTHS IN 1 GO & BETTER DOCUMENTED:
        max_init = max_guess
        min_init = min_guess
        
        # 1.) find all local maxima and minima at the surface
        surf = obj.sel(z_t=obj.z_t[0]) # depth slice at the surface
        [t_max_arr, t_min_arr] = find_local_min_max(surf) # finds 2 subarrays
        
        # 2.) pick the first min and max found directly AFTER min_init and max_init
        off = 2.0 # N.B. assuming misfit will not be more than off*delta_t (depends on sep in find_local_extr..())
        min_guess = sort([t - min_guess for t in t_min_arr if t > (min_init - off*delta_t)])[0] + min_guess
        max_guess = sort([t - max_guess for t in t_max_arr if t > (max_init - off*delta_t)])[0] + max_guess
        ridge_min[0] = min_guess # also save result in output array
        ridge_max[0] = max_guess 

        # 3.) find all local max and min for all deeper layers
        for z in range(1, len(obj.z_t)):
            d_slice = obj.sel(z_t=obj.z_t[z]) # depth slice
            [t_max_arr, t_min_arr] = find_local_min_max(d_slice) 
            
            # 4.) pick min,max CLOSEST to min,max of previous depth(=max_guess,min_guess) and AFTER min_init and max_init
            t_min_arr = array(t_min_arr)  # to be sure that np.where() works
            t_max_arr = array(t_max_arr) 
            t_min_arr = t_min_arr[t_min_arr > min_init - off*delta_t] 
            t_max_arr = t_max_arr[t_max_arr > max_init - off*delta_t] 
            min_guess = t_min_arr[where(abs(t_min_arr-min_guess) == abs(t_min_arr-min_guess).min())].item() 
            max_guess = t_max_arr[where(abs(t_max_arr-max_guess) == abs(t_max_arr-max_guess).min())].item() 
            ridge_min[z] = min_guess # also save result in output array        
            ridge_max[z] = max_guess 
            
    if only_min:
        return ridge_min
    else: 
        return [ridge_min, ridge_max]

def find_local_min(obj, min_guess=1750.0):
    """This function is a wrapper around find_local_min_max(),
    combined with the magic of guess_min in find_ridges. It is only used for searching 1 min (no max).
    
    It is just a copy of part of find_ridges, but now it is also available if you are only interested 
    in one depth slice (since find_ridges takes only variables with z_t coordinate).
    
    Usage:
    - trick to always find your global minimum: use global minm 'f.find_min(obj)[0]' 
    (is rounded to output frequency) as value for min_guess.
    
    Author: Jeemijn Scheen, jeemijn.scheen@climate.unibe.ch"""
    
    from numpy import sort
    
    t_min_arr = find_local_min_max(obj=obj, only_min=True)
    delta_t = (obj.time[2] - obj.time[1]).item() # assuming delta_t is constant
    off = 2.0  # N.B. assuming misfit will not be more than off*delta_t (depends on sep in find_local_extr..())
    min_init = min_guess
    min_guess = sort([t - min_guess for t in t_min_arr if t > (min_init - off*delta_t)])[0] + min_guess  
    return min_guess 
    
def find_local_min_max(obj, only_min=False):
    '''Finds 2 local extrema (1 min and 1 max) in obj by fitting a parabola through each 5 consecutive data points.
    Input:
    - obj is e.g. a temperature time series at a fixed depth (a depth slice)
    - only_min [optional] if not looking for a max, only for 1 min   
    Output: [t_min, t_max] as time indices
    Author: Jeemijn Scheen, jeemijn.scheen@climate.unibe.ch'''
    
    from numpy import asarray, round, polyfit, argmax, split, diff, average, insert, abs, sign, zeros, nan, isnan, unique

    if 'z_t' in obj.dims:
        raise Exception("obj must be independent of z_t; only dependent on time coordinate")
    if 'time' not in obj.dims:
        raise Exception("obj must be dependent on time coordinate")

    delta_t = (obj.time[2] - obj.time[1]).item() # assuming delta_t is constant
    signs = sign(diff(obj)) # sign of gradient vs time at this fixed depth 
    
    ## find possible extrema
    extr = [] # empty list to save result i.e. the found extr_time 
    a_coefs = [] # empty list to save value of a in ax^2+bx+c. That is, the extremum is a min if a>0; max if a<0.
    for n,this_temp in enumerate(obj): 
        if n==0 or n==1 or n==len(obj)-2 or n==len(obj)-1: continue # not over 2 start & 2 end points of obj
        
        ## METHOD: discard this extremum/timestep if all datapoints in this interval increase/decrease monotonically
        x_arr_steps = [n+i for i in range(-2,2)] # 4 of the 5 time coords in time steps (exclude the last one)
        # recall that signs contains only -1.0 and +1.0, i.e. this data point to next is de- resp. increasing
        discard = len(unique(signs[x_arr_steps]))==1 
        # discard is true if only either -1 or +1 occurs in our interval i.e. if monotonically de-/increasing
        if discard: continue
        else: # end of the above METHOD; go on with looking for extrema 
            # find 5 neighbouring points
            x_arr = [obj.time[n+i].item() for i in range(-2,3)] # the 5 time coords
            y_arr = [obj.sel(time = t) for t in x_arr] # the 5 temp coords 
            this_time = x_arr[2] # timestamp of the current time at step n   

            # fit parabola y=ax^2 + bx + c through 5 datapoints
            coefs = polyfit(x_arr, y_arr, deg = 2) # fits parabola = degree 2 polynomial
            extr_time = - coefs[1] / (2.0*coefs[0]) # extremum is located at x = -b/2a

            if extr_time>x_arr[-1] or extr_time<x_arr[0]: # if the found extr is NOT within interval of those 5 datapoints
                continue
            else: # save found extremum
                extr.append(extr_time)
                a_coefs.append(coefs[0])   
                
    ## average extrema that are found double (each extremum is found 1 to 5 times):
    # split extr[] into array with minima and array with maxima
    extr_min = asarray([val for n,val in enumerate(extr) if a_coefs[n] > 0.0]) 
    extr_max = asarray([val for n,val in enumerate(extr) if a_coefs[n] <= 0.0])
    
    # grab indices of extr[] where next extr is more than 5 datapoints away
    sep = 1.5 # can play with this separation value; should be somewhere between 0.5 and 4
    split_arr_min = [i+1 for i,val in enumerate(diff(extr_min)) if val>sep*delta_t] 
    split_arr_max = [i+1 for i,val in enumerate(diff(extr_max)) if val>sep*delta_t] 
    extr_min = split(extr_min, split_arr_min) # split into subarrays that represent the same min
    extr_max = split(extr_max, split_arr_max) 
    for i,subarr in enumerate(extr_min):
        extr_min[i] = average(subarr) # now change subarray to average of its values i.e. average the found extremum
    if not only_min: # only compute if searching for maxima
        # NB this construction is because in the above maxima are always computed but above it didn't give problems
        for i,subarr in enumerate(extr_max):
            extr_max[i] = average(subarr) 

    if only_min:
        return extr_min
    else:
        return [extr_max, extr_min]
    
def calc_leads_lags(obj_t, obj_f, d=26):
    """Calculate leads and lags between transient and fixed simulation at a fixed depth of 3 km. 
    This function is for the case of a global temperature minm (no maxm) e.g. LIA and industrial warming. 
    Input:
    - obj_t is transient simulation (a variable e.g. TEMP depending on z_t and time)
    - obj_f idem for fixed simulation
    - d can be set to another depth than the default of d=26 (3km). Useful: 14, 21, 29 are 1, 2, 4 km, respectively.
    Output:
    In this order [time_min_t, time_min_f, val_min_t, val_min_f, lead] for transient (_t) resp. fixed (_f) :
    - time of global minimum (time_min) 
    - value of global minimum (val_min)  [this is the amplitude, but with a minus sign] 
    - lead of transient w.r.t. fixed in yr (lead)
    Example usage:
    [time_min_t, time_min_f, val_min_t, val_min_f, lead] = calc_leads_lags(obj_t, obj_f) 
    Author: Jeemijn Scheen, jeemijn.scheen@climate.unibe.ch"""
    
    if 'z_t' in obj_t.dims:  # possibly already sliced to 1 depth before fnc call
        obj_t = obj_t.isel(z_t=d)
    if 'z_t' in obj_f.dims: 
        obj_f = obj_f.isel(z_t=d)
    
    # First find global minimum [time, val] in data:
    # This time of minimum is rough since rounded to output frequency e.g. 5 yrs
    # Value is not called 'rough' since it can't be made more precise below
    [time_min_t_rough, val_min_t] = find_min(obj_t)  
    [time_min_f_rough, val_min_f] = find_min(obj_f)

    # Find more precise time of minimum:
    time_min_t = find_local_min(obj_t, min_guess=time_min_t_rough)
    time_min_f = find_local_min(obj_f, min_guess=time_min_f_rough)

    # Quantify lead of transient w.r.t. fixed [yr]
    # transient is leading (lead>0) if it adjusts faster than fixed i.e. if time_min_t < time_min_f
    lead = int(round(time_min_f - time_min_t))  # rounded to entire years
    
    # amplitude of transient minimum equals abs(val_min_t) 
    # but we output val to preserve sign information
    return [time_min_t, time_min_f, val_min_t, val_min_f, lead]


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

# ABOVE: FUNCTIONS TO COMPUTE
# BELOW: FUNCTIONS TO PLOT

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


def plot_leads_lags(obj_t, obj_f, ax, color='blue', labels=['transient', 'fixed'], align='lower', indic=True, d=26):
    """Plots leads and lags between transient and fixed simulation at a fixed depth of 3 km.
    Input:
    - obj_t is transient simulation (a variable e.g. TEMP depending on z_t and time)
    - obj_f idem for fixed simulation
    - ax object of the subplot
    - color of graph line can be set
    - labels array can be given in. Legend is not plotted automatically but must be called
    - align is 'lower' (text and arrow appear below graph) or 'upper' (above graph)
    - if indic then indications and arrows of lead or lag are plotted along [default]
    - d can be set to another depth than the default of d=26 (3.1 km). Useful: 14, 21, 29 are ca. 1, 2, 4 km, resp.
    Output:
    - a plot is made on this axis, including an arrow and indication of the lead or lag in yr
    - axis object is returned
    Example usage:
    ax[1,1] = plot_leads_lags(obj_t, obj_f, ax=ax[1,1]) 
    Author: Jeemijn Scheen, jeemijn.scheen@climate.unibe.ch"""
    
    if color == 'blue':
        col_pos = 'forestgreen' # color of arrow and annotations if positive (transient leading w.r.t fixed)
        col_neg = 'deeppink'    # color of arrow and annotations if negative (transient lagging w.r.t fixed)
    else:  # combines nicely with C0:
        col_pos = 'g'      
        col_neg = 'purple' 
    
    pad = 5            # padding between minima and arrow [cK]
    
    if align not in ('lower', 'upper'):
        raise Exception("Align should be 'upper' or 'lower'.")
    if len(labels) is not 2:
        raise Exception("Labels should have length 2.")
        return
    
    obj_t = obj_t.isel(z_t=d)
    obj_f = obj_f.isel(z_t=d)
    
    ax.plot(obj_t.time, obj_t, color, linestyle='solid', label=labels[0])
    ax.plot(obj_f.time, obj_f, color, linestyle='dashed', label=labels[1])

    if indic:
        [time_min_t, time_min_f, val_min_t, val_min_f, lead] = calc_leads_lags(obj_t, obj_f, d=d)    

        if lead >= 0:
            lead_str = '+' + str(lead)
            lag_str = str(-lead) # in ax.text() below either lead_str or lag_str can be used
            col_ar = col_pos 
        else:
            lead_str = str(lead)
            lag_str = '+' + str(-lead)
            col_ar = col_neg 
        # plot marker on top of minima    
        ax.scatter([time_min_t], [val_min_t], color=col_ar, marker='x')  # 'o' or 'D' diamond or '+' or 'x'
        ax.scatter([time_min_f], [val_min_f], color=col_ar, marker='x')

        # plot arrow (color depends on lead vs lag) and text
        if align == 'lower':
            arrow_y = val_min_t - pad      # y coord of arrow
            text_y = arrow_y - 1.5 * pad   # y coord of text
        elif align == 'upper':
            arrow_y = 0 + pad
            text_y = arrow_y + pad
        ax.annotate('', xytext=(time_min_f, arrow_y), xy=(time_min_t, arrow_y), 
                     arrowprops=dict(arrowstyle="->", color=col_ar, linewidth=2)) # arrow from xytext to xy 
        # indicate the lag in text [yr]:
        ax.text(min(time_min_t, time_min_f), text_y, lag_str + ' yr', fontsize=16, color=col_ar, ha='left')
 
    return ax

def plot_surface(fig, ax, x, y, z, title="", grid='T', cbar=True, cbar_label='', cmap=None, 
                 vmin=None, vmax=None, ticklabels=False):
    """Makes a (lat,lon) plot using pcolor. 
    Input:
    - fig and ax must be given
    - x and y are either obj.lon_u and obj.lat_u (if var on T grid) or obj.lon_t and obj.lat_t (if var on U grid)
    - z must be a lat x lon array 
    Optional input:
    - title
    - grid can be 'T' [default] (if z values on lon_t x lat_t) or 'U' (if on lon_u x lat_u)
    - cbar determines whether a cbar is plotted [default True]
    - cbar_label can be set
    - cmap gives colormap to use
    - vmin, vmax give min and max of colorbar
    - ticklabels prints the tick labels of lat/lon
    Output:
    - [ax, cbar] axis and cbar object are given back.
    Example usage:
    ax[0] = plot_surface(fig, ax[0], obj.lon_u, obj.lat_u, obj.TEMP.isel(z_t=0, time=0).values) """
    
    if grid == 'T':
        if (len(x) != 42) or (len(y) != 41) or (z.shape != (40,41)):
            raise Exception("x,y must be on u-grid and z on T-grid (if var is not on T-grid, then set: grid = 'U')")
        if cmap is None:
            cpf = ax.pcolor(x, y, extend(z), vmin=vmin, vmax=vmax) # pcolor takes default for vmin=None
        else:
            cpf = ax.pcolor(x, y, extend(z), cmap=cmap, vmin=vmin, vmax=vmax)    
    elif grid == 'U':
        # U grid implemented analogously but not tested yet
        if (len(x) != 41) or (len(y) != 40) or (z.shape != (41,42)):
            raise Exception("x,y must be on T-grid and z on U-grid (if var is not on U-grid, then set: grid = 'T')")
            return
        if cmap is None:
            cpf = ax.pcolor(x, y, z, vmin=vmin, vmax=vmax)        
        else:
            cpf = ax.pcolor(x, y, extend(z), cmap=cmap, vmin=vmin, vmax=vmax)            
    else:
        raise Exception("Set grid to 'T' or 'U'.")
    
    ax.set_title(title)
    if not ticklabels:
        ax.tick_params(labelbottom=False, labelleft=False)  # remove tick labels with lat/lon values
    if cbar:
        cbar = fig.colorbar(cpf, ax=ax, label=cbar_label)
        return [ax, cbar]
    else:
        return [ax, cpf]
    
def make_colormap(seq):
    """Return a LinearSegmentedColormap
    Input:
    - seq is a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1). 
    Explanation:
    - For discrete boundaries, located at the floats: mention every colour once.
    - For fluent gradient boundaries: define the same colour on both sides of the float.
    Source: 
    https://stackoverflow.com/questions/16834861/create-own-colormap-using-matplotlib-and-plot-color-scale
    """
    
    from matplotlib import colors as colors
    
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i-1]
            r2, g2, b2 = seq[i+1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return colors.LinearSegmentedColormap('CustomMap', cdict)

def extend(var):
    """
    Adds one element of rank-2 matrices to be plotted by pcolor
    Author: Raphael Roth, roth@climate.unibe.ch
    """
    
    from numpy import ma as ma
    from numpy import ones, nan
    
    [a,b] = var.shape
    field = ma.masked_invalid(ones((a+1,b+1))*nan)
    field[0:a,0:b] = var
    return field

def create_land_mask(obj, data_full):
    """Creates land mask suitable for a (lat,z)-plot or (lon,lat)-plot.
    Input:
    - obj from whose nan values to create land mask e.g. any var masked to atlantic basin 
    obj should either have lat_t and z_t coord OR lat_u and z_w coord OR lon_t and lat_t coord
    - data_full: xarray data set that contains respective coords on u,v,w grid
    Output: 
    - [mask, cmap_land]
    NB cmap_land is independent of the land mask itself but just needed for plotting
    Usage example for (lat,z)-plot:
    - plot land by:
    X, Y = np.meshgrid(data_full.lat_u.values, data_full.z_w.values)
    if obj has z_t, lat_t coord:
        ax[i].pcolormesh(X,Y,extend(mask), cmap = cmap_land, vmin = -0.5, vmax = 0.5)
    if obj has z_w, lat_u coord: 
        identical but without extend()"""

    from numpy import isnan, ma, unique
    from matplotlib import pyplot as plt
    
    if 'time' in obj.dims:
        obj = obj.isel(time = 0)
    if not( ('lat_t' in obj.dims and 'z_t' in obj.dims) or ('lat_u' in obj.dims and 'z_w' in obj.dims) 
          or ('lon_t' in obj.dims and 'lat_t' in obj.dims)):
        raise Exception("obj should have either (z_t,lat_t) or (z_w,lat_u) or (lon_t,lat_t) coord")
    if unique(isnan(obj)).any() == False:
        raise Exception("obj has no nan values. Change 0 values to nan values first (if appropriate).")
    
    mask = isnan(obj) 
    mask = ma.masked_where(mask == False, mask) # change False values to nan such that cmap.set_bad picks them
    cmap_land = plt.cm.Greys 
    cmap_land.set_bad(color='white') # where there is still ocean but no contour interpolation

    return [mask, cmap_land]

def plot_hovmoeller(obj, fig, ax, zoom=None, levels=None, hi=None, lo=None, levelarr=-1, 
                    ridges=True, cbar=True, title=''):
    '''Makes a Hovmoeller diagram:
    a contour plot with x=time, y=depth and colors=temperature (or another variable on T-grid).
    
    Input:
    - obj needs to be set to an xarray DataArray containing coords time and z_t (values eg TEMP)
    - fig needs to be given (in order to be able to plot colorbar);   from e.g. fig,ax=plt.subplots(1,2)
    - ax needs to be set to an axis object;   use e.g. ax or ax[2] if multiple subplots
    Optional inputs:
    - zoom can be set to a fixed nr of years (eg if 1200, only the first 1200 years are plotted)
    - hi and lo are min and max of contours, respectively.
    - either levels or levelarr is used to define the number of contours
       - if none of them, then automatic contour levels are used and the colorbar will not be nicely centered
       - levelarr (if supplied) overwrites levels
       - levels is the number of uniformly spaced levels
       - levelarr=[a,b,c] with 3x explicit levels (stepsize may vary; 0 should be in the middle)
    - ridges plots ridge where cooling/warming starts. Can behave badly if changes are small.
    - cbar to be plotted along
    - title of the plot can be set

    Output:
    - makes the plot on this axis object 
    - returns:
        - if not cbar: cpf (=cbar object)
        - if ridges: ridge values
        - if both of the above: [cpf, ridge values]
    
    Author: Jeemijn Scheen, jeemijn.scheen@climate.unibe.ch'''
    
    from numpy import meshgrid, arange, concatenate
    
    # avoid the somewhat cryptic TypeError: "Input z must be at least a 2x2 array."
    if obj.shape[0] < 2:
        raise Exception("Object needs to have at least 2 timesteps.")
    elif obj.shape[1] < 2:
        raise Exception("Object needs to have at least 2 depth steps.")
    
    # make x,y,z arrays for contour plot
    if zoom != None:
        obj = obj.sel(time=slice(obj.time[0], zoom))
    xlist = obj.time.values
    ylist = obj.z_t.values
    Z = obj.values.transpose()
    X, Y = meshgrid(xlist, ylist)
    
    if levelarr != -1: # use levelarr (overwrites levels,hi,lo)
        [a,b,c] = levelarr
        level_arr = concatenate((a,b,c)).tolist()
        # contour fill
        cpf = ax.contourf(X, Y, Z, level_arr, cmap='coolwarm', extend='both') # choose coolwarm, RdBu or seismic
        # only plot labels for subset 1 & 3 of level_arr
        cp1 = ax.contour(X, Y, Z, a, colors='k', linestyles='-', linewidths=0.5) # contour lines
        cp2 = ax.contour(X, Y, Z, b, colors='k', linestyles='-', linewidths=0.5) 
        cp3 = ax.contour(X, Y, Z, c, colors='k', linestyles='-', linewidths=0.5) 
        cp0 = ax.contour(X, Y, Z, [0.0], colors='k', linestyles='-', linewidths=0.5) # 0 another time manually
        ax.clabel(cp1, inline=True, fontsize=12, colors='k', use_clabeltext=True, fmt='%1.0f') # labels    
        ax.clabel(cp3, inline=True, fontsize=12, colors='k', use_clabeltext=True, fmt='%1.0f')     
        ax.clabel(cp0, inline=True, fontsize=12, colors='k', use_clabeltext=True, fmt='%1.0f') # add 0 label    
    elif levels != None: # use levels,hi,lo
        # define contour levels
        step = (hi-lo) / levels
        level_arr = arange(lo, hi+step, step) 
        # contour fill
        cpf = ax.contourf(X, Y, Z, level_arr, cmap='coolwarm', extend='both') # choose coolwarm, RdBu or seismic
        cp = ax.contour(X, Y, Z, level_arr, colors='k', linestyles='-', linewidths=0.5) # contour lines
        ax.clabel(cp, inline=True, fontsize=12, colors='k', use_clabeltext=True, fmt='%1.0f') # labels    
    else: # choose levels automatically (colorbar ugly/not centered)
        cpf = ax.contourf(X, Y, Z, cmap='coolwarm') # contour fill
        cp = ax.contour(X, Y, Z, colors='k', linestyles='-', linewidths=0.5) # contour lines
        ax.clabel(cp, inline=True, fontsize=12, colors='k', use_clabeltext=True, fmt='%1.0f') # labels 

    ax.invert_yaxis()    # depth goes down on y axis   
    ax.set_xlabel('simulation year')
    ax.set_ylabel('Depth [km]')
    ax.set_title(title)

    if ridges == True: # plot nice ridges at points where cooling or warming starts
        ridge_min = find_ridges(obj, only_min=True, min_guess=1750.0, fast=True)
        ax.plot(ridge_min, obj.z_t.values, 'r')  # or 'k' for black 
    if cbar:
        fig.colorbar(cpf, ax=ax, label='temp. anomaly [cK]') 
        if ridges:
            return ridge_min
    else:
        if ridges:
            return [cpf, ridge_min]
        else:
            return cpf # in order to be able to plot 1 colorbar separately next to subplots
    
def plot_contour(obj, fig, ax, var='OPSI', levels=None, hi=None, lo=None, cbar=True, title='', 
                 cmap=None, add_perc=False, extend=None):
    """Makes a contour plot with x=lat, y=depth and for colors 3 variables are possible:
    1) var = 'OPSI' then colors = OPSI (overturning psi, stream function)
    2) var = 'TEMP' then a lat-lon plot is made e.g. for a temperature, which must be in cK.
    3) var = 'CONC' idem as OPSI but plots a concentration so no dotted streamline contours etc

    Input (required):
    - obj needs to be set to an xarray DataArray containing coords lat and z_t (values eg OPSI+GM_OPSI)
    - fig needs to be given (in order to be able to plot colorbar);  from e.g. fig,ax=plt.subplots(1,2)
    - ax needs to be set to an axis object;   use e.g. ax or ax[2] if multiple subplots
    
    Input (optional):
    - var: see 3 options above
    - levels, hi and lo can be set to nr of contours, highest and lowest value, respectively.
      NB if levels = None, automatic contour levels are used and the colorbar will not be nicely centered.
      NB if var='CONC', colorbar ticks are hardcoded to maximal 6 ticks
    - cbar can be plotted or not
    - title of the plot can be set
    - cmap sets colormap  (inspiration: Oranges, Blues, Purples, PuOr_r, viridis) [default: coolwarm]
    - add_perc adds a '%' after each colorbar tick label (=unit for dye concentrations)
    - extend tells the colorbar to extend: 'both', 'neither', 'upper' or 'lower' (default: automatic)
    
    Output:
    - plot is made on axis object
    - if cbar: cbar object is returned (allows you to change the cbar ticks)
    - else: cpf object is returned (allows you to make a cbar)
    
    Author: Jeemijn Scheen, jeemijn.scheen@climate.unibe.ch"""  
    
    from numpy import meshgrid, arange, ceil, concatenate, floor, asarray, unique, sort
    from matplotlib import ticker, colors
    
    # avoid the somewhat cryptic TypeError: "Input z must be at least a 2x2 array."
    if obj.shape[0] < 2:
        raise Exception("object needs to have at least 2 depth steps.")
    elif obj.shape[1] < 2:
        raise Exception("object needs to have at least 2 steps on the x axis of contour plot.")

    if cmap is None:
        cmap = 'coolwarm'
            
    # make x,y,z arrays for contour plot
    if var == 'OPSI':
        xlist = obj.lat_u.values
        ylist = obj.z_w.values
        unit = 'Sv'      # unit for colorbar label
    elif var == 'TEMP':
        xlist = obj.lon_t.values
        ylist = obj.lat_t.values
        unit = '[cK]'    # assuming obj is in centi-Kelvin
    elif var == 'CONC':  # only difference wrt OPSI: need T-grid
        xlist = obj.lat_t.values 
        ylist = obj.z_t.values
        unit = ''
    else:
        raise Exception('var must be equal to OPSI or TEMP or CONC')
    Z = obj.values 
    X, Y = meshgrid(xlist, ylist)
    
    # prepare array for contour levels:
    if levels != None: # use levels, hi, lo
        # define contour levels
        step = (hi-lo) / float(levels)         # go to float to avoid integer rounding (can => division by zero)  
        level_arr = arange(lo, hi+step, step) 
        level_arr[abs(level_arr) < 1e-4] = 0.  # to avoid contour level labels called '-0.0' 

        # set nr of decimals for labels of contour lines
        if asarray(level_arr).max() >= 10.0:
            fmt = '%1.0f' 
        else:
            fmt = '%1.1f'
        
        # PLOT CONTOUR LINES ACCORDING TO levels, hi, lo PARAMETERS
        if var == 'OPSI':
            # contour lines
            cp_neg = ax.contour(X,Y,Z, level_arr[level_arr<0], colors='k', linestyles='dashed', linewidths=0.5) 
            cp_pos = ax.contour(X,Y,Z, level_arr[level_arr>0], colors='k', linestyles='-', linewidths=0.5) 
            cp0 = ax.contour(X,Y,Z, level_arr[abs(level_arr)<1e-4], colors='k', linestyles='-', linewidths=1.5)
            # contour labels
            for cp in [cp_neg, cp_pos, cp0]:    # add level values to contour lines
                ax.clabel(cp, inline=True, fontsize=12, colors='k', use_clabeltext=True, fmt=fmt) 
                # NB fmt formats text (standard %1.3f); use_clabeltext sets labels parallel to contour line
        elif var == 'TEMP':
            # hardcoded in order to resemble Gebbie & Huybers 2019, Fig 2
            # => contourf on the far bottom will plot for every 2.5 cK
            # but for contour lines:
            # keep small steps within [-10,10] but go to steps of 10 outside that interval
            this_level_arr = concatenate((level_arr[abs(level_arr) <= 10.0], level_arr[level_arr%10==0]))
            this_level_arr = unique(sort(this_level_arr))

            cp = ax.contour(X, Y, Z, this_level_arr, colors='k', linestyles='-', linewidths=0.5) 
            # contour labels via dict comprehension 
            # round to integer if levels are not like -7.5,-2.5,2.5,7.5 s.t. no '.0' appears
            fmt_dict = {x : str(x) if x-floor(x) == 0.5 else str(int(round(x))) for x in this_level_arr}
            ax.clabel(cp, inline=True, fontsize=12, colors='k', use_clabeltext=True, fmt=fmt_dict) 
        elif var == 'CONC':             
            # contour lines
            cp_non0 = ax.contour(X, Y, Z, level_arr[abs(level_arr)>1e-4], colors='k', linestyles='-', linewidths=0.5) 
            cp0 = ax.contour(X,Y,Z, level_arr[abs(level_arr)<1e-4], colors='k', linestyles='-', linewidths=1.5) 
            # contour labels
            for cp in [cp_non0, cp0]:    # add level values to contour lines
                ax.clabel(cp, inline=True, fontsize=12, colors='k', use_clabeltext=True, fmt=fmt) 
                
    else: # PLOT CONTOUR LINES WITH AUTOMATIC LEVELS (colorbar ugly/not centered)
        cpf = ax.contourf(X, Y, Z, cmap=cmap) # contour fill
        cp = ax.contour(X, Y, Z, colors='k', linestyles='-', linewidths=0.5) # contour lines
        # contour labels
        ax.clabel(cp, inline=True, fontsize=12, colors='k', use_clabeltext=True) # labels
        
    # PLOT CONTOUR FILL
    if var == 'TEMP':
        zorder=0 # makes things on top possible, here: grid lines
    else:
        zorder=1
    if extend is None:
        cpf = ax.contourf(X, Y, Z, level_arr, cmap=cmap, zorder=zorder) 
    else:
        cpf = ax.contourf(X, Y, Z, level_arr, cmap=cmap, extend=extend, zorder=zorder)
        
    # COLOUR BAR
    if cbar:
        if var == 'TEMP':
            # hardcoded in order to resemble Gebbie & Huybers 2019 science, Fig 2
            cbar_obj = fig.colorbar(cpf, ax=ax, label=unit, orientation='horizontal', pad=0.15)
        else:
            cbar_obj = fig.colorbar(cpf, ax=ax, label=unit)  
        # set cbar ticks and labels
        if var == 'CONC':
            tick_locator = ticker.MaxNLocator(nbins=6) # to force 0-100% cbar to reasonable ticks without manually
            cbar_obj.locator = tick_locator
            cbar_obj.update_ticks()
        if add_perc:
            perc = "%"
        else:
            perc = ""
        cticks = cbar_obj.get_ticks()
        ctick_labels = ['0'+perc if abs(x) < 1e-4
                        else str(int(round(x)))+perc if hi >= 10.0 
                        else str(round(x,1))+perc if hi > 5.0 
                        else str(round(x,2))+perc for x in cticks]
        cbar_obj.ax.set_yticklabels(ctick_labels)  
        
    # LABELS AND TICKS
    ax.set_title(title)
    if var != 'TEMP':
        # for TEMP do nothing; the axes (lon, lat) are obvious from world map
        ax.set_xlabel('Latitude')
        ax.set_ylabel('Depth [km]')
        ax.set_ylim(0,5)
        ax.invert_yaxis()    # depth goes down on y axis    
        ax.set_yticks(range(0,6,1))
    
    if cbar:
        return cbar_obj  # return already plotted cbar object s.t. it can be adjusted
    else:
        return cpf       # return object with which plotting cbar is possible
    
def plot_overturning(data, data_full, times, time_avg=False, atl=True, pac=False, sozoom=False, 
                     levels=None, lo=None, hi=None, land=True, all_anoms=False):
    """Plots figure of overturning stream function panels at certain time steps and basins.
    Columns:
    - a column for every t in times array
    Rows: 
    - if atl: overturning as measured only in Atlantic basin
    - if pac: overturning as measured only in Pacific basin
    - if sozoom: Southern Ocean sector of global overturning
    - global overturning (always plotted)
    
    Input:
    - data and data_full xarray datasets with depth in kilometers
    - times array with time indices, e.g., 50 stands for data_full.time[50]
    - time_avg [default False] plots a 30 year average around the selected time steps instead of the 1 annual value 
      NB for t=0 a 15 year average on the future side is taken
    - atl, pac and/or sozoom basins (rows; see above)
    - levels, lo and hi set the number of colour levels and min resp. max boundaries
    - land [optional] prints black land on top
    - all_anoms [optional] plots all values as anomalies w.r.t. t1 except the first column (t1)
      NB anomaly plots have a hardcoded colorbar between -2 and 2 Sv
    
    Output:
    - returns [fig, ax]: a figure and axis handle
        
    Author: Jeemijn Scheen, jeemijn.scheen@climate.unibe.ch"""

    # SETTINGS:
    so_bnd = -80  # S.O. southern boundary e.g. -90 or -80
    # color of land: black when (vmin,vmax) = (-0.5,0.5) and grey when (0.5,1.5) and light grey when (0.8,1.5)
    vmin = 0.8
    vmax = 1.5  
    
    from matplotlib.pyplot import subplots, suptitle, tight_layout
    from numpy import zeros, ceil, sum, nan, meshgrid
    
    row_nr = 1 + sum([atl, pac, sozoom]) # np.sum() gives nr of True values 
    col_nr = len(times)
        
    if all_anoms:
        # anomaly plots have a hardcoded colorbar between -2 and 2 Sv
        hi_anom = 2.0
        lo_anom = -2.0
        levels_anom = 10
           
    opsi_all_t = data_full.OPSI + data_full.GMOPSI      # total global overturning; still for all times
    opsi_a_all_t = data_full.OPSIA + data_full.GMOPSIA  # atlantic overturning
    opsi_p_all_t = data_full.OPSIP + data_full.GMOPSIP  # pacific overturning
       
    if land:     
        # in this case we want to replace 0 values (land) by nan values
        # such that the opsi variables are not plotted on land & the nan values are needed to plot land
        opsi_all_t = opsi_all_t.where(opsi_all_t != 0.0, nan)
        opsi_a_all_t = opsi_a_all_t.where(opsi_a_all_t != 0.0, nan)
        opsi_p_all_t = opsi_p_all_t.where(opsi_p_all_t != 0.0, nan)
        [mask_gl, cmap_land_gl]   = create_land_mask(opsi_all_t, data_full) 
        [mask_atl, cmap_land_atl] = create_land_mask(opsi_a_all_t, data_full) 
        [mask_pac, cmap_land_pac] = create_land_mask(opsi_p_all_t, data_full) 
        X, Y = meshgrid(data_full.lat_u.values, data_full.z_w.values) # same for all subplots    
    
    fig, ax = subplots(nrows=row_nr, ncols=col_nr, figsize=(14, 3*row_nr)) 
        
    for i in range(0,row_nr):
        for j in range(0,col_nr):
            ax[i,j].set_xticks(range(-75,80,25))
            
    ## now we go through the rows/basins one by one (if they exist); 
    ## within each we have a for loop over cols/times 
    ## where we save plotted objects for all times in opsi{} s.t. we can take the anomaly between them
            
    this_row = 0
    if atl:    
        opsi = {}  # keys=cols, vals=opsi plot object for that col/timestep
        for n,t in enumerate(times):
            # pick correct ax object
            if row_nr == 1: 
                this_ax = ax[n]
            else: 
                this_ax = ax[this_row,n] 

            # select correct data for time step and row in t_slice:
            if time_avg:
                # over 30 yrs e.g. 1750 now becomes average over [1735, 1765]
                opsi[n] = opsi_a_all_t.sel(time=slice(t-16,t+16)).mean(dim='time')
            else:
                opsi[n] = opsi_a_all_t.sel(time=t) 
            this_title = "Atlantic overturning"
            this_ax.set_xlim([-50,90]) # exclude S.O. 
            
            # plot
            if land:  
                this_ax.pcolormesh(X, Y, mask_atl, cmap=cmap_land_atl, vmin=vmin, vmax=vmax) 
            if all_anoms and n != 0:  # first row is never anomaly
                opsi_diff = opsi[n] - opsi[0]
                plot_contour(opsi_diff, fig, ax=this_ax, levels=levels_anom, lo=lo_anom, hi=hi_anom, 
                             var='OPSI', extend='both', title=this_title + ' anomaly')             
            else:
                plot_contour(opsi[n], fig, ax=this_ax, levels=levels, lo=lo, hi=hi, 
                             var='OPSI', extend='both', title=this_title)
        this_row += 1

    if pac:
        opsi = {}  # keys=cols, vals=opsi plot object for that col/timestep
        for n,t in enumerate(times):
            # pick correct ax object
            if row_nr == 1: 
                this_ax = ax[n]
            else: 
                this_ax = ax[this_row,n] 

            # select correct data for time step and row in t_slice:
            if time_avg:
                opsi[n] = opsi_p_all_t.sel(time=slice(t-16,t+16)).mean(dim='time')
            else:
                opsi[n] = opsi_p_all_t.sel(time=t) 
            this_title = "Indo-Pacific overturning"
            this_ax.set_xlim([-50,90]) # exclude S.O.
            
            # plot
            if land:  
                this_ax.pcolormesh(X, Y, mask_pac, cmap=cmap_land_pac, vmin=vmin, vmax=vmax) 
            if all_anoms and n != 0:  # first row is never anomaly
                opsi_diff = opsi[n] - opsi[0]
                plot_contour(opsi_diff, fig, ax=this_ax, levels=levels_anom, lo=lo_anom, hi=hi_anom, 
                             var='OPSI', extend='both', title=this_title + ' anomaly')             
            else:
                plot_contour(opsi[n], fig, ax=this_ax, levels=levels, lo=lo, hi=hi, 
                             var='OPSI', extend='both', title=this_title)                
        this_row += 1

    if sozoom:
        opsi = {}  # keys=cols, vals=opsi plot object for that col/timestep
        for n,t in enumerate(times):
            # pick correct ax object
            if row_nr == 1: 
                this_ax = ax[n]
            else: 
                this_ax = ax[this_row,n] 

            # select correct data for time step and row in t_slice:
            if time_avg:
                opsi[n] = opsi_all_t.sel(lat_u=slice(so_bnd, -50), time=slice(t-16,t+16)).mean(dim='time')
            else:
                opsi[n] = opsi_all_t.sel(lat_u=slice(so_bnd, -50), time=t)
            this_title = "Southern Ocean overturning"
            # make exception for SO (otherwise only 1 tick at -75)
            this_ax.set_xticks(range(-90,-45,10)) 
            
            # plot
            if land:  
                this_ax.pcolormesh(X, Y, mask_gl, cmap=cmap_land_gl, vmin=vmin, vmax=vmax) 
                this_ax.set_xlim([so_bnd,-50]) # ticks change to automatic but that seems fine
            if all_anoms and n != 0:  # first row is never anomaly
                opsi_diff = opsi[n] - opsi[0]
                plot_contour(opsi_diff, fig, ax=this_ax, levels=levels_anom, lo=lo_anom, hi=hi_anom, 
                             var='OPSI', extend='both', title=this_title + ' anomaly')             
            else:
                plot_contour(opsi[n], fig, ax=this_ax, levels=levels, lo=lo, hi=hi, 
                             var='OPSI', extend='both', title=this_title)
        this_row += 1

    ## always plot global overturning in last row
    opsi = {}  # keys=cols, vals=opsi plot object for that col/timestep
    for n,t in enumerate(times):
        # pick correct ax object
        if row_nr == 1: 
            this_ax = ax[n]
        else: 
            this_ax = ax[this_row,n] 

        # select correct data for time step and row in t_slice:
        if time_avg:
            opsi[n] = opsi_all_t.sel(time=slice(t-16,t+16)).mean(dim='time')
        else:
            opsi[n] = opsi_all_t.sel(time=t)
        this_title = "Global overturning"
        
        # plot
        if land:  
            this_ax.pcolormesh(X, Y, mask_gl, cmap=cmap_land_gl, vmin=vmin, vmax=vmax) 
        if all_anoms and n != 0:  # first row is never anomaly
            opsi_diff = opsi[n] - opsi[0]
            plot_contour(opsi_diff, fig, ax=this_ax, levels=levels_anom, lo=lo_anom, hi=hi_anom, 
                         var='OPSI', extend='both', title=this_title + ' anomaly')             
        else:
            plot_contour(opsi[n], fig, ax=this_ax, levels=levels, lo=lo, hi=hi, 
                         var='OPSI', extend='both', title=this_title)
        # print info on minimum and maximum
        if time_avg:
            print('@%1.0f CE: global MOC min=%1.2f Sv, AMOC max=%1.2f Sv' 
                  %(ceil(times[n]), data.OPSI_min.sel(time=slice(t-16,t+16)).mean(dim='time'),
                    data.OPSIA_max.sel(time=slice(t-16,t+16)).mean(dim='time')))
        else:
            print('@%1.0f CE: global MOC min=%1.2f Sv, AMOC max=%1.2f Sv' 
                  %(ceil(times[n]), data.OPSI_min.sel(time=t).item(), data.OPSIA_max.sel(time=t).item()))

    tight_layout()    
    
    return fig, ax
