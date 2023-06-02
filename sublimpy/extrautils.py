import pandas as pd
import datetime
import xarray as xr
import os
import urllib
import geopy
import geopy.distance
import shapely.geometry
import py3dep
import pytz
import numpy as np
import rasterio
import geopandas as gpd
from sklearn import linear_model

from astral import LocationInfo
from astral.sun import sun


from metpy.units import units

import turbpy


ROTATION_SUPPORTED_MEASUREMENTS = [
     'u',     'v',     
     'u_w_',     'v_w_', 
     'u_tc_',     'v_tc_',    
    'u_h2o_',     'v_h2o_',    
    ]

def open_datasets_as_dataframe(file_list, variables = None):
    """Create a dataframe from a list of files. Files have extension ".nc" and are from the EOL lab's ftp address

    ftp://ftp.eol.ucar.edu/pub/archive/isfs/projects/SOS/netcdf/noqc_geo

    Args:
        file_list (_type_): _description_
    """
    if variables:
        return pd.concat([
            xr.open_dataset(f)[variables].to_dataframe().reset_index() for f in file_list
        ])
    else:
        return pd.concat([
            xr.open_dataset(f).to_dataframe().reset_index() for f in file_list
        ])
        
def time_from_day_and_hhmm(
    day,
    hhmm,
    base_day = datetime.datetime(2022, 1, 1)
):
    """ Get a datetime object from an integer day and a an integer hhmm

    This was created for ingesting data from the Stretch (Tilden's) tower data.

    Args:
        day (int): a number indicating an ordinal day
        hhmm (int): an integer indicating the hour and minute during a single day. E.G., 12:01am is 1, 1:01am is 101, 1:32pm is 1332
        base_day (_type_, optional): _description_. Defaults to datetime.datetime(2022, 1, 1).

    Returns:
        _type_: _description_
    """
    hours = float(str(int(hhmm)).zfill(4)[:2])
    minutes = float(str(int(hhmm)).zfill(4)[2:])
    return base_day \
        + datetime.timedelta(days = day - 1) \
        + datetime.timedelta(hours = hours) \
        + datetime.timedelta(minutes = minutes)

# if already downloaded

def merge_datasets_with_different_variables(ds_list, dim):
    """ Take a list of datasets and merge them using xr.merge. First check that the two datasets
    have the same data vars. If they do not, missing data vars in each dataset are added with nan values
    so that the two datasets have the same set of data vars. NOTE: This gets slow with lots of datasets

    Args:
        ds_list (_type_): _description_
        dim (_type_): _description_
    """
    def _merge_datasets_with_different_variables(ds1, ds2, dim):
        vars1 = set(ds1.data_vars)
        vars2 = set(ds2.data_vars)
        in1_notin2 = vars1.difference(vars2)
        in2_notin1 = vars2.difference(vars1)
        # add vars with NaN values to ds1
        for v in in2_notin1:
            ds1[v] = xr.DataArray(coords=ds1.coords, dims=ds1.dims)
        # add vars with NaN values to ds2
        for v in in1_notin2:
            ds2[v] = xr.DataArray(coords=ds2.coords, dims=ds2.dims)
        return xr.concat([ds1, ds2], dim=dim)

    new_ds = ds_list.pop(0)
    while ds_list:
        new_ds = _merge_datasets_with_different_variables(
            new_ds,
            ds_list.pop(0),
            dim=dim
        )
    return new_ds

def get_linestring(lon, lat, bearing, radius):
    radar_location = geopy.Point(lat, lon)
    radar_elevation = py3dep.elevation_bycoords(
        [(radar_location.longitude, radar_location.latitude)]
    )[0]

    positive_distance = geopy.distance.distance(
        kilometers=radius
    ).destination(
        point=radar_location, 
        bearing=bearing
    )
    negitive_distance = geopy.distance.distance(
        kilometers=radius
    ).destination(
        point=radar_location, 
        bearing=bearing-180
    )
    return shapely.geometry.LineString([
        shapely.geometry.Point(negitive_distance.longitude, negitive_distance.latitude),
        shapely.geometry.Point(radar_location.longitude, radar_location.latitude),
        shapely.geometry.Point(positive_distance.longitude, positive_distance.latitude),
        
    ])


def get_radar_scan_ground_profile(lon, lat, bearing, radius, spacing = 10):
    """Returns the ground profile relative to the provided point, i.e. the elevation at the
    provided lon/lat is equal to 0 in the returned dataset.

    Args:
        lon (_type_): _description_
        lat (_type_): _description_
        bearing (_type_): _description_
        radius (_type_): _description_
        spacing (int, optional): _description_. Defaults to 10.
    """
    radar_location = geopy.Point(lat, lon)
    radar_elevation = py3dep.elevation_bycoords(
        [(radar_location.longitude, radar_location.latitude)]
    )[0]
    line = get_linestring(lon, lat, bearing, radius)

    elevation_profile = py3dep.elevation_profile(line, spacing=spacing, crs='EPSG:4326')
    elevation_profile.values = elevation_profile.values - radar_elevation
    elevation_profile_df = elevation_profile.to_dataframe().reset_index()
    elevation_profile_df['distance'] = elevation_profile_df['distance'] - radius*1000
    elevation_profile_df['zero'] = 0

    return elevation_profile_df

def get_radar_scan_ground_profile_from_raster(dem_file, lon, lat, bearing, radius, n_points):
    """Returns the ground profile relative to the provided point, i.e. the elevation at the
    provided lon/lat is equal to 0 in the returned dataset.

    Args:
        lon (_type_): _description_
        lat (_type_): _description_
        bearing (_type_): _description_
        radius (float): radius in kilometers
        spacing (int, optional): _description_. Defaults to 10.
    """
    radar_location = geopy.Point(lat, lon)
    radar_elevation = py3dep.elevation_bycoords(
        [(radar_location.longitude, radar_location.latitude)]
    )[0]
    line = get_linestring(lon, lat, bearing, radius)

    line = gpd.GeoDataFrame(geometry=pd.Series([line])).set_crs('EPSG:4326').to_crs('EPSG:32613').geometry.iloc[0]
    distances = np.linspace(0, line.length, n_points)
    points = [line.interpolate(distance) for distance in distances]
    line = gpd.GeoDataFrame(geometry=[shapely.geometry.LineString(points)]).set_crs('EPSG:32613').to_crs('EPSG:4326').geometry.iloc[0]
    points = list(line.coords)
    dem = rasterio.open(dem_file)
    elevation_profile = rasterio.sample.sample_gen(dem, points)
    elevation_profile = [f[0]-radar_elevation for f in elevation_profile]

    return pd.DataFrame({
        'elevation': elevation_profile,
        'distance': pd.Series(distances) - radius*1000
    })


def get_nightime_df(timezone, lat, lon, dates):
    nighttime_df = pd.DataFrame()
    loc = LocationInfo(timezone=timezone, latitude=lat, longitude=lon)
    for date in dates:
        s = sun(loc.observer, date=date, tzinfo=loc.timezone)
        nighttime_df = pd.concat([
            nighttime_df,
            pd.DataFrame({
                'datetime': [date],
                'sunset': [s['sunset']],
                'sunrise': [s['sunrise']],
            } )
        ], ignore_index=True)
    # remove timezone info
    nighttime_df['sunset'] = nighttime_df['sunset'].dt.tz_localize(None)
    nighttime_df['sunrise'] = nighttime_df['sunrise'].dt.tz_localize(None)
    # add partial first night:
    nighttime_df = pd.concat([
            pd.DataFrame({
                'datetime': [np.nan],
                'sunset': [np.nan],
                'sunrise': [np.nan],
            }),
            nighttime_df,
            pd.DataFrame({
                'datetime': [np.nan],
                'sunset': [np.nan],
                'sunrise': [np.nan],
            })
        ], ignore_index=True)
    nighttime_df['sunrise'] = nighttime_df['sunrise'].shift(-1)
    nighttime_df.loc[0, 'sunset'] = pd.Series(sun(loc.observer, date=dates[0] - datetime.timedelta(days=1), tzinfo=loc.timezone)['sunset']).dt.tz_localize(None).iloc[0]
    nighttime_df.loc[len(dates), 'sunrise'] = dates[-1] + datetime.timedelta(days=1) 
    return nighttime_df.drop(nighttime_df.tail(1).index)


def calculate_planar_fit(u,v,w):
    # remove nans from u,v,w
    temp_df = pd.DataFrame({'u': u, 'v': v, 'w': w}).dropna()
    u = temp_df['u']
    v = temp_df['v']
    w = temp_df['w']
    
    # Fit, using least-squares, the averaged wind components to the equation of the plane,
    # $ a = -bu - cv + w$
    X_data = np.array([u, v]).reshape((-1,2))
    Y_data = np.array(w)
    reg = linear_model.LinearRegression().fit(X_data, Y_data)
    a = reg.intercept_
    b,c = reg.coef_

    # Now define a new set of coordinates, streamwise coordinates, 
    # <$U_f$, $V_f$, $W_f$>
    # The normal vector to the plane defined by U_f and V_f is Wf, and is calculated
    W_f = np.array([-b, -c, 1]) / np.sqrt( b**2 + c**2 + 1**2 )
    tilt = np.arctan(np.sqrt(b**2 + c**2))
    tiltaz = np.arctan2(-c, -b)
    # Check that these two angles correctly define $W_f$ (floating point errors are ok)
    assert (
        all(W_f - ( np.sin(tilt)*np.cos(tiltaz), np.sin(tilt)*np.sin(tiltaz), np.cos(tilt)) < 10e-7)
    )

    return (a,b,c), (tilt, tiltaz), W_f

def apply_planar_fit(u, v, w, a, W_f):
    # # Define the sonic coordinate system unit vectors
    U_s = np.array([1,0,0])
    V_s = np.array([0,1,0])
    W_s = np.array([0,0,1])

    # Calculate the other axes of the mean flowwise coordinate system
    U_f_normal_vector = np.cross(np.cross(W_f, U_s), W_f)
    U_f = U_f_normal_vector / np.sqrt((U_f_normal_vector**2).sum())
    V_f = np.cross(W_f, U_f)
    
    # Transform velocity measurements in sonic coords to mean flowwise coords
    u_streamwise = np.dot(U_f, np.array([u, v, w - a]))
    v_streamwise = np.dot(V_f, np.array([u, v, w - a]))
    w_streamwise = np.dot(W_f, np.array([u, v, w - a]))

    return (u_streamwise, v_streamwise, w_streamwise)

def calculate_and_apply_planar_fit(u, v, w):
    """Provides planar fit adjusted velocity components for a given dataset of u,v,w.

    Args:
        u (_type_): _description_
        v (_type_): _description_
        w (_type_): _description_

    Returns:
       (float,float,float), (float, float), (np.array, np.array, np.array):
                 (a,b,c), (tilt (radians), tiltaz (rad)), (u_streamwise, v_streamwise, w_streamwise)
    """
    (a,b,c), (tilt, tiltaz), W_f = calculate_planar_fit(u, v, w)
    (u_streamwise, v_streamwise, w_streamwise) = apply_planar_fit(u, v, w, a, W_f)
    return (u_streamwise, v_streamwise, w_streamwise)

    


def streamwise_coordinates_xarray(ds):
    assert type(ds) == xr.DataSet

    """ToDo: Implement for the dataset as it comes in the SOSNOQC format"""

def streamwise_coordinates_single_rotation_tidy_df(original_tidy_df):
    """Modify multiple data variables of a tidy_df created from a SoS netcdf dataset (see 
    `get_tidy_dataset` abozve). Namely, the u and v velocity variables and all first and second 
    covariances. A single rotation is used (no adjustment to the  vertical axis) based on the mean
    u and v velocity over the provided time period. The returned dataset will have all relevant data 
    variables replaced with rotated values.
    Args:
        tidy_df (pd.DataFrame): In the same format as the result of the `get_tidy_dataset` function 
        above.
    """

    assert type(original_tidy_df) == pd.DataFrame

    tidy_df = original_tidy_df.copy()

    u_avg = tidy_df[tidy_df['measurement'] == 'u'].groupby(['tower', 'height'])['value'].mean()
    v_avg = tidy_df[tidy_df['measurement'] == 'v'].groupby(['tower', 'height'])['value'].mean()
    angles = np.arctan2(v_avg, u_avg)
    D = angles.to_dict()
    D

    variable_pairs = [
        ('u', 'v'), 
        ('u_w_', 'v_w_'), 
        ('u_tc_', 'v_tc_'), 
        ('u_h2o_', 'v_h2o_'), 
    ]
    for u_var, v_var in variable_pairs:
        for (tower, height), angle_rad in D.items():
            us = tidy_df.loc[
                    (tidy_df['tower'] == tower) & (tidy_df['height'] == height) & (tidy_df['measurement'] == u_var)
                ]['value'].reset_index(drop=True)
            vs = tidy_df.loc[
                    (tidy_df['tower'] == tower) & (tidy_df['height'] == height) & (tidy_df['measurement'] == v_var)
                ]['value'].reset_index(drop=True)
            new_u = np.array(us*np.cos(angle_rad) + vs*np.sin(angle_rad))
            new_v = np.array(-us*np.sin(angle_rad) + vs*np.cos(angle_rad))
            tidy_df.loc[
                    (tidy_df['tower'] == tower) & (tidy_df['height'] == height) & (tidy_df['measurement'] == u_var), 'value'] = new_u
            tidy_df.loc[
                    (tidy_df['tower'] == tower) & (tidy_df['height'] == height) & (tidy_df['measurement'] == v_var), 'value'] = new_v
    return tidy_df


def fast_xarray_resample_mean(ds, resampling_time):
    df_h = ds.to_dataframe().resample(resampling_time).mean()  # what we want (quickly), but in Pandas form
    vals = [xr.DataArray(data=df_h[c], dims=['time'], coords={'time':df_h.index}, attrs=ds[c].attrs) for c in df_h.columns]
    return  xr.Dataset(dict(zip(df_h.columns,vals)), attrs=ds.attrs)

def fast_xarray_resample_median(ds, resampling_time):
    df_h = ds.to_dataframe().resample(resampling_time).median()  # what we want (quickly), but in Pandas form
    vals = [xr.DataArray(data=df_h[c], dims=['time'], coords={'time':df_h.index}, attrs=ds[c].attrs) for c in df_h.columns]
    return  xr.Dataset(dict(zip(df_h.columns,vals)), attrs=ds.attrs)

    # u_avg = tidy_df[tidy_df['measurement'] == 'u'].groupby(['tower', 'height'])['value'].mean()
    # v_avg = tidy_df[tidy_df['measurement'] == 'v'].groupby(['tower', 'height'])['value'].mean()
    # angles = np.arctan(v_avg/u_avg)
    # print(np.rad2deg(angles))
    # D = angles.to_dict()
    
        
    # df_wide = tidy_df.pivot_table(index=['time','tower','height'], values='value', columns='measurement').reset_index()
    
    # # convert u velocity
    # df_wide['u'] = df_wide.apply(
    #     lambda row: row['u']*np.cos(
    #         D.get((row['tower'], row['height']), np.nan)
    #     ) + row['v']*np.sin(
    #             D.get((row['tower'], row['height']), np.nan)
    #     ),
    #     axis=1
    # )

    # # # convert v velocity
    # df_wide['v'] = df_wide.apply(
    #     lambda row: - row['u']*np.sin(
    #         D.get((row['tower'], row['height']), np.nan)) + row['v']*np.cos(
    #         D.get((row['tower'], row['height']), np.nan)),
    #     axis=1
    # )

    # df_wide['u_w_'] = df_wide.apply(
    #     lambda row: row['u_w_']*np.cos(
    #         D.get((row['tower'], row['height']), np.nan)) 
    #     + row['v_w_']*np.sin(
    #         D.get((row['tower'], row['height']), np.nan)
    #     ),
    #     axis=1
    # )

    # df_wide['v_w_'] = df_wide.apply(
    #     lambda row: - row['u_w_']*np.sin(
    #         D.get((row['tower'], row['height']), np.nan)
    #     ) + row['v_w_']*np.cos(
    #         D.get((row['tower'], row['height']), np.nan)
    #     ),
    #     axis=1
    # )

    # df_wide['u_tc_'] = df_wide.apply(
    #     lambda row: row['u_tc_']*np.cos(
    #         D.get((row['tower'], row['height']), np.nan)) + row['v_tc_']*np.sin(
    #             D.get((row['tower'], row['height']), np.nan)),
    #     axis=1
    # )

    # df_wide['v_tc_'] = df_wide.apply(
    #     lambda row: - row['u_tc_']*np.sin(
    #         D.get((row['tower'], row['height']), np.nan)) + row['v_tc_']*np.cos(
    #         D.get((row['tower'], row['height']), np.nan)),
    #     axis=1
    # )

    # df_wide['u_h2o_'] = df_wide.apply(
    #     lambda row: row['u_h2o_']*np.cos(
    #         D.get((row['tower'], row['height']), np.nan)) + row['v_h2o_']*np.sin(
    #             D.get((row['tower'], row['height']), np.nan)),
    #     axis=1
    # )

    # df_wide['v_h2o_'] = df_wide.apply(
    #     lambda row: - row['u_h2o_']*np.sin(
    #         D.get((row['tower'], row['height']), np.nan)) + row['v_h2o_']*np.cos(
    #         D.get((row['tower'], row['height']), np.nan)),
    #     axis=1
    # )

    # return df_wide.melt(
    #     id_vars=['time', 'tower', 'height'], 
    #     value_vars=tidy_df['measurement'].unique()
    # )

def get_turbpy_schemes():
    # stab_titles are the names given to each stability scheme when plotting. In this example they correspond 
    # to the dictionaries that contain the parameter values for each run. T
    stab_titles = ('Standard',
                #    'Louis (b = 4.7)',
                #    'Louis (b = 12)',
                #    'Louis (Ri capped, MJ98)',
                'MO_HdB', #Holtslag/de Bruin',
                #    'MO (Holtslag/de Bruin - capped)',
                # 'MO (Beljaars/Holtslag)',
                #    'MO (Webb - NoahMP)',
                # 'MO (Cheng/Brutsaert)',
                )

    # A mapping between the titles and the stability methods used in each test.
    stab_methods = {'Standard': 'standard',
                    # 'Louis (b = 4.7)': 'louis',
                    # 'Louis (b = 12)': 'louis',
                    # 'Louis (Ri capped, MJ98)': 'louis',
                    'MO_HdB': 'monin_obukhov',
                    # 'MO (Holtslag/de Bruin - capped)': 'monin_obukhov',
                    # 'MO (Beljaars/Holtslag)': 'monin_obukhov',
                    # 'MO (Webb - NoahMP)': 'monin_obukhov',
                    # 'MO (Cheng/Brutsaert)': 'monin_obukhov',
                }

    # Thes gradient functions for the Monin-Obukhov methods
    gradient_funcs = {'MO_HdB': 'holtslag_debruin',
                    #   'MO (Holtslag/de Bruin - capped)': 'holtslag_debruin',
                    # 'MO (Beljaars/Holtslag)': 'beljaar_holtslag',
                    #   'MO (Beljaars/Holtslag - capped)': 'beljaar_holtslag',
                    # 'MO (Cheng/Brutsaert)': 'cheng_brutsaert',
                    #   'MO (Webb - NoahMP)': 'webb_noahmp',
                    }

    # Parameters for the Louis scheme. Any method without a parameter value provided 
    # is filled in with the default value
    params = {'Louis (b = 4.7)': 9.4,
            'Louis (Ri capped, MJ98)': 9.4,
            'Louis (b = 12)': 24.}

    # Indicates which methods have capping of the conductance. Any method without capping 
    # indicated is assumed to have no capping.
    capping = {
        'Louis (Ri capped, MJ98)': 'louis_Ri_capping',
        'MO (Holtslag/de Bruin - capped)': 'windless_exchange',
        }

    # Initialize the multi-level parameter dictionary
    stab_dict = {}
    # stab_dict['stability_params'] = {}

    for st in stab_methods:
        stab_dict[st] = {}
        
        # Assigning the stability method
        stab_dict[st]['stability_method'] = stab_methods[st]
        
        # Assigning the gradient method
        if 'monin_obukhov' in stab_methods[st]:
            stab_dict[st]['monin_obukhov'] = {}
            stab_dict[st]['monin_obukhov']['gradient_function'] = gradient_funcs[st]
            
        # Assiging the capping behavior
        if st in capping.keys():
            stab_dict[st]['capping'] = capping[st]
        
        # Determine stability params
        if st in params.keys():
            stab_dict[st]['stability_params'] = {stab_methods[st]: params[st]}    
        
    return stab_titles, stab_methods, stab_dict


def tidy_df_calculate_richardson_number_with_turbpy(
    tidy_df_original, 
    tower, 
    height, 
    snowDepth,
    pressure_height, 
    fillna_method='ffill',
    surface_temp_col_substitute = None
):

    if surface_temp_col_substitute:
        sfcTemp = (tidy_df_original.query(f"variable == '{surface_temp_col_substitute}'")['value']+273.15).fillna(method=fillna_method)
    else:
        sfcTemp = (tidy_df_original.query(f"variable == 'Tsurf_{tower}'")['value']+273.15).fillna(method=fillna_method)
    # all temperature sensors are on tower c
    airTemp = (tidy_df_original.query(f"variable == 'T_{height}m_c'")['value']+273.15).fillna(method=fillna_method)
    # in the case where you try to grab 1m Tower C data, it will not be available for much of the season - get 2m temp data to replace
    if (not all(airTemp) and height == 1) or (len(airTemp) == 0 and height == 1):
        airTemp = (tidy_df_original.query(f"variable == 'T_2m_c'")['value']+273.15).fillna(method=fillna_method)

    windspd = (tidy_df_original.query(f"variable == 'spd_{height}m_{tower}'")['value']).fillna(method=fillna_method)

    return turbpy.bulkRichardson(
        airTemp.values,
        sfcTemp.values,
        windspd.values,
        height
    )

def tidy_df_model_heat_fluxes_with_turbpy(
    tidy_df_original,
    stab_titles, 
    stab_methods,
    stab_dict,
    tower, 
    height, 
    snowDepth,
    pressure_height, 
    fillna_method='ffill',
    surface_temp_col_substitute = None
):
    # collect inputs
    if surface_temp_col_substitute:
        sfcTemp = (tidy_df_original.query(f"variable == '{surface_temp_col_substitute}'")['value']+273.15).fillna(method=fillna_method)
    else:
        sfcTemp = (tidy_df_original.query(f"variable == 'Tsurf_{tower}'")['value']+273.15).fillna(method=fillna_method)
    # all temperature sensors are on tower c
    airTemp = (tidy_df_original.query(f"variable == 'T_{height}m_c'")['value']+273.15).fillna(method=fillna_method)
    # in the case where you try to grab 1m Tower C data, it will not be available for much of the season - get 2m temp data to replace
    if (not all(airTemp) and height == 1) or (len(airTemp) == 0 and height == 1):
        airTemp = (tidy_df_original.query(f"variable == 'T_2m_c'")['value']+273.15).fillna(method=fillna_method)
    windspd = (tidy_df_original.query(f"variable == 'spd_{height}m_{tower}'")['value']).fillna(method=fillna_method)
    
    windspd
    airPressure = (
        tidy_df_original.query(
            f"variable == 'P_{pressure_height}m_{tower}'"
        )['value'].fillna(
            method=fillna_method
        ).values * units.millibar
    ).to(units.pascal).magnitude

    (airVaporPress, _) = turbpy.satVapPress(airTemp - 273.15)
    (sfcVaporPress, _) = turbpy.satVapPress(sfcTemp - 273.15)

    ## Calculate stability

    # Initialzie dictionaries for containing output
    stability_correction = {}
    conductance_sensible = {}
    conductance_latent = {}
    sensible_heat = {}
    latent_heat = {}
    zeta = {}

    for stab in stab_titles:
        stability_correction[stab] = np.zeros_like(sfcTemp)
        conductance_sensible[stab] = np.zeros_like(sfcTemp)
        conductance_latent[stab] = np.zeros_like(sfcTemp)
        sensible_heat[stab] = np.zeros_like(sfcTemp)
        latent_heat[stab] = np.zeros_like(sfcTemp)
        zeta[stab] = np.zeros_like(sfcTemp)

    ## Calculate stability
    for stab in stab_titles:
        for n, (tair, vpair, tsfc, vpsfc, u, airP) in enumerate(zip(airTemp, airVaporPress, sfcTemp, sfcVaporPress, windspd, airPressure)):
            try:
                # Offline Turbulence Package
                (conductance_sensible[stab][n], 
                conductance_latent[stab][n], 
                sensible_heat[stab][n],
                latent_heat[stab][n],
                stab_output, p_test) = turbpy.turbFluxes(tair, airP,
                                                        vpair, u, tsfc,
                                                        vpsfc, snowDepth,
                                                        height, param_dict=stab_dict[stab],
                                                        z0Ground=.005, groundSnowFraction=1)
                # Unpack stability parameters dictionary
                if not 'monin_obukhov' in stab_methods[stab]:
                    stability_correction[stab][n] = stab_output['stabilityCorrection']
                else:
                    stability_correction[stab][n] = np.nan
                    zeta[stab][n] = stab_output['zeta']
            except:
                conductance_sensible[stab][n] = None
                conductance_latent[stab][n] = None
                sensible_heat[stab][n] = None
                latent_heat[stab][n] = None
                stab_output = None
                p_test = None
                # Unpack stability parameters dictionary
                if not 'monin_obukhov' in stab_methods[stab]:
                    stability_correction[stab][n] = np.nan
                else:
                    stability_correction[stab][n] = np.nan
                    zeta[stab][n] = np.nan

    return (stability_correction, conductance_sensible, conductance_latent, sensible_heat, latent_heat, zeta)