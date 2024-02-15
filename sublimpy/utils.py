"""
Utilities for downloading, cleaning, and otherwise working with SoS data.
"""

import pandas as pd
import xarray as xr
import os
import urllib
from urllib.error import URLError
import numpy as np
from metpy.units import units
import datetime as dt
from dateutil.relativedelta import relativedelta

def download_sos_highrate_data_hour(date = '20221031', hour = '00', local_download_dir = 'hr_noqc_geo', cache=False):
    """Download a netcdf file from the ftp url provided by the Earth Observing Laboratory at NCAR.
    Data is the high-rate, 20hz data.

    Args:
        date (str, optional): String date. Defaults to '20221101'.
        hour (str, optional): String hour, two digits. Defaults to '00'.
        local_download_dir (str, optional): _description_. Defaults to 'hr_noqc_geo'.
        cache (bool, optional): Whether or not to check the local_download_dir for the requested dataset. 
                If the file is already there, does not download it. Defaults to False.

    Returns:
        str: file path to the downloaded file
    """
    base_url = 'ftp.eol.ucar.edu'
    path = 'pub/archive/isfs/projects/SOS/netcdf/hr_noqc_geo'
    file_example = f'isfs_geo_hr_{date}_{hour}.nc'

    os.makedirs(local_download_dir, exist_ok=True)

    full_file_path = os.path.join('ftp://', base_url, path, file_example)
    download_file_path = os.path.join(local_download_dir, file_example)

    if cache and os.path.isfile(download_file_path):
        print(f"Caching...skipping download for {date}, {hour}")
    else:
        urllib.request.urlretrieve(
            full_file_path,
            download_file_path   
        )
        
    return download_file_path

def download_sos_data_day(date = '20221101', local_download_dir = 'sosnoqc', cache=False,  planar_fit = False):
    """Download a netcdf file from the ftp url provided by the Earth Observing Laboratory at NCAR.
    Data is the daily data reynolds averaged to 5 minutes.

    Args:
        date (str, optional): Date to download data. in format '%Y%m%d', i.e. 20230101 for Jan 1, 2023. Defaults 
                to '20221101'.
        local_download_dir (str, optional): Directory to which files will be downloaded. Defaults to 'sosnoqc'; 
                this directory will be created if it does not already exist.
        cache (bool, optional): Whether or not to check the local_download_dir for the requested dataset. 
                If the file is already there, does not download it. Defaults to False.
        planar_fit (bool, optional): Whether or not to download data that has been planar fit by NCAR. These 
                datasets are not available for all dates. Defaults to False.

    Returns:
        str: file path to the downloaded file
    """
    base_url = 'ftp.eol.ucar.edu'
    if planar_fit:
        path = 'pub/archive/isfs/projects/SOS/netcdf/noqc_geo_tiltcor/'
    else:
        path = 'pub/archive/isfs/projects/SOS/netcdf/noqc_geo'
    
    if planar_fit:
        file_example =  f'isfs_sos_tiltcor_{date}.nc'

    else:
        file_example = f'isfs_{date}.nc'

    os.makedirs(local_download_dir, exist_ok=True)

    full_file_path = os.path.join('ftp://', base_url, path, file_example)
    if planar_fit:
        download_file_path = os.path.join(local_download_dir, 'planar_fit', file_example)
    else:
        download_file_path = os.path.join(local_download_dir, file_example)
    

    if cache and os.path.isfile(download_file_path):
        print(f"Caching...skipping download for {date}")
    else:
        urllib.request.urlretrieve(
            full_file_path,
            download_file_path   
        )

    return download_file_path

def download_sos_data(
    start_date,
    end_date,
    variable_names,
    local_download_dir = 'sosnoqc',
    cache = False,
    planar_fit = False
):
    """Download SoS datasets and perform a few preprocessing steps to clean up the data. 
    SoS datasets are NetCDF files from the ftp url provided by the Earth Observing Laboratory at NCAR.
    Data is the daily data reynolds averaged to 5 minutes. This function requires the caller to specify 
    the variables to be included in the output dataset because memory requirements are extensive if all 
    variables are included when merging datasets from many dates. 
    
    Specifically, this function:
    1. Downloads multiple netcdf files form the NCAR-EOL FTP server,
    2. Catches the URLERror thrown if the netcdf file for a specific date does not exist, and prints a 
        note that a failure occured,
    3. Merges the datasets into a single dataset, dealing with conflicts that arrise if some variables 
        are available in some datasets but not in others.
    4. Fills in missing timestamps so align with the 5 minute index that the datasets come in. 
        Timestamps may be missing in a single day's dataset if data-loss occured at the beginning 
        or end of a day.

    Args:
        start_date (str): first date to download data. in format '%Y%m%d', i.e. 20230101 for Jan 1, 2023/
        end_date (str): last date to download data. in format '%Y%m%d'.
        variable_names (list(str)): List of strings that represent NetCDF variable names to include in the
                combined dataset.
        local_download_dir (str, optional): Directory to which files will be downloaded. Defaults to 'sosnoqc'; 
                this directory will be created if it does not already exist.
        cache (bool, optional): Whether or not to check the local_download_dir for the requested dataset. 
                If the file is already there, does not download it. Defaults to False.
        planar_fit (bool, optional): Whether or not to download data that has been planar fit by NCAR. These 
                datasets are not available for all dates. Defaults to False.
    Returns:
        xr.Dataset: Merged and cleaned dataset with specified data variables between specified dates.
    """
    datelist = pd.date_range(
        dt.datetime.strptime(start_date, '%Y%m%d'),
        dt.datetime.strptime(end_date, '%Y%m%d'),
        freq='d'
    ).strftime('%Y%m%d').tolist()

    # We make sure that we aren't accessing variables that don't exist in the datasets
    # This is necessary because some daily NetCDF files don't have all the expected variables
    # (for example because an instrument was down). In that case, we want to add that variable
    # to the dataset, filled with nans, which sosutils.merge_datasets_with_different_variables
    # handles for us
    datasets = []
    for date in datelist:
        try:
            ds = xr.open_dataset(download_sos_data_day(date, local_download_dir, cache=cache, planar_fit=planar_fit))
        # Some dates are missing
        except URLError:
            print(f"failed on {date}, skipping")
        ds_new = ds[set(ds.data_vars).intersection(variable_names)]
        datasets.append(ds_new)
        
    sos_ds = xr.concat(datasets, dim='time')        
    sos_ds = fill_missing_timestamps(sos_ds)
    return sos_ds

def modify_df_timezone(df, source_tz, target_tz, time_col='time'):
    """Modify the timezone of a dataframe. The time data should NOT be an index of the provided 
        dataframe, there must be a column (time_col) with time data.

    Args:
        df (pd.DataFrame): _description_
        source_tz (_type_): A pytz timezone object specifying the timezone the data is already in. 
                For example, `pytz.UTC`.
        target_tz (_type_): A pytz timezone object specifying the timezone the data is to be 
                converted to. For example, `pytz.timezone('US/Mountain')`.
        time_col (str, optional): The dataframe column name that contains the time data. Defaults 
                to 'time'.

    Returns:
        pd.DataFrame: Dataframe with the time_col object overwritten with the modified timestamps.
    """
    df = df.copy()
    df[time_col] = df[time_col].dt.tz_localize(source_tz).dt.tz_convert(target_tz).dt.tz_localize(None)
    return df

def modify_xarray_timezone(ds, source_tz, target_tz):
    """Modify the timezone of an xr.Dataset. The dataset should have a coordinate and dimension 'time'.
    The returned xr.Dataset object will have the original 'time' coordinate/dimension overwritten.

    Args:
        ds (xr.Dataset): xarray Dataset object to have its time coordinate/dimension converted.
        source_tz (_type_): A pytz timezone object specifying the timezone the data is already in. 
                For example, `pytz.UTC`.
        target_tz (_type_): A pytz timezone object specifying the timezone the data is to be 
                converted to. For example, `pytz.timezone('US/Mountain')`.

    Returns:
        xr.Dataset: xarray Dataset with the time coordinate/dimension overwritten with the modified 
                timestamps.
    """
    ds = ds.copy()
    time_utc = ds['time'].to_index().tz_localize(source_tz)
    tz_corrected = time_utc.tz_convert(target_tz).tz_localize(None)
    local_da=xr.DataArray.from_series(tz_corrected)
    ds.coords.update({f'time ({target_tz})': tz_corrected})
    ds.coords.update({f'time ({source_tz})': ds['time'].to_index()})
    ds = ds.assign_coords({
        'time': ds[f'time ({target_tz})'].values
    })
    return ds

def fill_missing_timestamps(ds):
    """Fills in missing timestamps in an xr.Dataset for all data variables with NaN values. This is
    particularly useful when multiple daily NetCDF files have been merged together. SoS NetCDF files 
    generally have data every 5 minutes. If data is missing from the beginning or end of the day,
    there may be missing timestamps (e.g. if the power was out at the study site between 12am and 2am
    on a day, the first timestamp in the dataset will be 02:02:30). This can be confusing when we want 
    to combine datasets from different days. This function fills in all missing timestamps between the 
    first timestamp and the last timestamp in the provided xr.Dataset. It makes sure there is one 
    timestamp every 5 minutes. 

    Args:
        ds (xr.Dataset): Dataset to be filled. 
    """
    def date_range(start_date, end_date, increment, period):
        result = []
        nxt = start_date
        delta = relativedelta(**{period:increment})
        while nxt <= end_date:
            result.append(nxt)
            nxt += delta

        return result
    dt_list = date_range(pd.to_datetime(ds.time.values[0]), pd.to_datetime(ds.time.values[-1]), 5, 'minutes')
    ds = ds.drop_duplicates(dim='time').reindex(time=dt_list)

    return ds
