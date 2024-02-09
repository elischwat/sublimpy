"""
Convenience functions for creating a tidy dataframe dataset from SoS NetCDF datasets.
"""
import pandas as pd 
import xarray as xr
    
def _height_from_variable_name(name):
    """
    Parse instrument/sensor height from EOL variable names. 

    Args:
        name (str): variable name from SoS noqc netcdf datasets

    Returns:
        float: height of measurement in units of meters
    """
    # handle the soil moisture depths
    if '_0_6cm' in name:
        return -0.006
    elif '_1_9cm' in name:
        return -0.019
    elif '_3_1cm' in name:
        return -0.031
    elif '_4_4cm' in name:
        return -0.044
    elif '_8_1cm' in name:
        return -0.081
    elif '_9_4cm' in name:
        return -0.094
    elif '_10_6cm' in name:
        return -.106
    elif '_11_9cm' in name:
        return -.119
    elif '_18_1cm' in name:
        return -.181
    elif '_19_4cm' in name:
        return -.194
    elif '_20_6cm' in name:
        return -.206
    elif '_21_9cm' in name:
        return -.219
    elif '_28_1cm' in name:
        return -.281
    elif '_29_4cm' in name:
        return -.294
    elif '_30_6cm' in name:
        return -.306
    elif '_31_9cm' in name:
        return -.319
    # snow temperature depths - these must be handled before tower depths because,
    # for example, '_4m_' is in '_0_4m_'
    elif '_0_4m_' in name:
        return 0.4
    elif '_0_5m_' in name:
        return 0.5
    elif '_0_6m_' in name:
        return 0.6
    elif '_0_7m_' in name:
        return 0.7
    elif '_0_8m_' in name:
        return 0.8
    elif '_0_9m_' in name:
        return 0.9
    elif '_1_0m_' in name:
        return 1.0
    elif '_1_1m_' in name:
        return 1.1
    elif '_1_2m_' in name:
        return 1.2
    elif '_1_3m_' in name:
        return 1.3
    elif '_1_4m_' in name:
        return 1.4
    elif '_1_5m_' in name:
        return 1.5
    # tower depths
    elif '_1m_' in name:
        return 1.0
    elif '_2m_' in name:
        return 2.0
    elif '_3m_' in name:
        return 3.0
    elif '_4m_' in name:
        return 4.0
    elif '_5m_' in name:
        return 5.0
    elif '_6m_' in name:
        return 6.0
    elif '_7m_' in name:
        return 7.0
    elif '_8m_' in name:
        return 8.0
    elif '_9m_' in name:
        return 9.0
    elif '_10m_' in name:
        return 10.0
    elif '_11m_' in name:
        return 11.0
    elif '_12m_' in name:
        return 12.0
    elif '_13m_' in name:
        return 13.0
    elif '_14m_' in name:
        return 14.0
    elif '_15m_' in name:
        return 15.0
    elif '_16m_' in name:
        return 16.0
    elif '_17m_' in name:
        return 17.0
    elif '_18m_' in name:
        return 18.0
    elif '_19m_' in name:
        return 19.0
    elif '_20m_' in name:
        return 20.0
    # surface measurements
    elif 'surf' in name:
        return 0.0
    else:
        return 0.0


def _tower_from_variable_name(name):
    """Parse instrument/sensor tower from EOL variable names.

    Args:
        name (str): variable name from SoS noqc netcdf datasets

    Returns:
        str: tower of measurement
    """
    if name.startswith('SWE_'):
        return {
            'SWE_p1_c': 'uw',
            'SWE_p2_c': 'c',
            'SWE_p3_c': 'ue',
            'SWE_p4_c': 'd',
        }.get(name)
    if name.endswith('_d') or name.endswith('_d_1') or name.endswith('_d_2') or name.endswith('_d_3'):
        return 'd'
    if name.endswith('_c') or name.endswith('_c_1') or name.endswith('_c_2') or name.endswith('_c_3'):
        return 'c'
    if name.endswith('_ue') or name.endswith('_ue_1') or name.endswith('_ue_2') or name.endswith('_ue_3'):
        return 'ue'
    if name.endswith('_uw') or name.endswith('_uw_1') or name.endswith('_uw_2') or name.endswith('_uw_3'):
        return 'uw'


def _measurement_from_variable_name(name):
    """Provide plain text measurement name from EOL variable names.

    Args:
        name (_type_): _description_

    Returns:
        _type_: _description_
    """
    # VARIABLE NAMES THAT COME FROM THE SOSNOQC DATASETS
    if name.startswith('counts_'):
        if name.endswith('_1'):
            return 'eddy covariance h2o high rate count'
        elif name.endswith('_2'):
            return 'eddy covariance co2 high rate count'
        elif name.endswith('_3'):
            return 'eddy covariance P high rate count'
        else:
            return 'eddy covariance momentum high rate count'        
    if any([prefix in name for prefix in ['SF_avg_1m_ue', 'SF_avg_2m_ue']]): # these are the only two 
        return 'snow flux'
    elif any([prefix in name for prefix in ['P_10m_', 'P_20m_']]):
        return 'pressure'
    elif any([prefix in name for prefix in ['dir_1m_','dir_2m_','dir_3m_','dir_5m_','dir_10m_','dir_15m_','dir_20m_']]):
        return 'wind direction'
    elif any([prefix in name for prefix in ['spd_1m_', 'spd_2m_', 'spd_3m_', 'spd_5m_', 'spd_10m_', 'spd_15m_', 'spd_20m_']]):
        return 'wind speed'
    elif any([prefix in name for prefix in ['u_1m_','u_2m_','u_3m_','u_5m_','u_10m_','u_15m_','u_20m_']]):
        return 'u'
    elif any([prefix in name for prefix in ['v_1m_','v_2m_','v_3m_','v_5m_','v_10m_','v_15m_','v_20m_']]):
        return 'v'
    elif any([prefix in name for prefix in ['w_1m_','w_2m_','w_3m_','w_5m_','w_10m_','w_15m_','w_20m_']]):
        return 'w'
    elif any([prefix in name for prefix in ['u_u__1m_', 'u_u__2m_', 'u_u__3m_', 'u_u__5m_', 'u_u__10m_', 'u_u__15m_', 'u_u__20m_']]):
        return 'u_u_'
    elif any([prefix in name for prefix in ['v_v__1m_', 'v_v__2m_', 'v_v__3m_', 'v_v__5m_', 'v_v__10m_', 'v_v__15m_', 'v_v__20m_']]):
        return 'v_v_'
    elif any([prefix in name for prefix in ['w_w__1m_', 'w_w__2m_', 'w_w__3m_', 'w_w__5m_', 'w_w__10m_', 'w_w__15m_', 'w_w__20m_']]):
        return 'w_w_'
    elif any([prefix in name for prefix in ['u_w__1m_','u_w__2m_','u_w__3m_','u_w__5m_','u_w__10m_','u_w__15m_','u_w__20m_']]):
        return 'u_w_'
    elif any([prefix in name for prefix in ['v_w__1m_','v_w__2m_','v_w__3m_','v_w__5m_','v_w__10m_','v_w__15m_','v_w__20m_']]):
        return 'v_w_'
    elif any([prefix in name for prefix in ['u_v__1m_','u_v__2m_','u_v__3m_','u_v__5m_','u_v__10m_','u_v__15m_','u_v__20m_']]):
        return 'u_v_'
    elif any([prefix in name for prefix in ['u_tc__1m_','u_tc__2m_','u_tc__3m_','u_tc__5m_','u_tc__10m_','u_tc__15m_','u_tc__20m_']]):
        return 'u_tc_'
    elif any([prefix in name for prefix in ['v_tc__1m_','v_tc__2m_','v_tc__3m_','v_tc__5m_','v_tc__10m_','v_tc__15m_','v_tc__20m_']]):
        return 'v_tc_'
    elif any([prefix in name for prefix in ['w_tc__1m_','w_tc__2m_','w_tc__3m_','w_tc__5m_','w_tc__10m_','w_tc__15m_','w_tc__20m_']]):
        return 'w_tc_'
    elif any([prefix in name for prefix in ['u_h2o__1m_','u_h2o__2m_','u_h2o__3m_','u_h2o__5m_','u_h2o__10m_','u_h2o__15m_','u_h2o__20m_']]):
        return 'u_h2o_'
    elif any([prefix in name for prefix in ['v_h2o__1m_','v_h2o__2m_','v_h2o__3m_','v_h2o__5m_','v_h2o__10m_','v_h2o__15m_','v_h2o__20m_']]):
        return 'v_h2o_'
    elif any([prefix in name for prefix in ['w_h2o__1m_','w_h2o__2m_','w_h2o__3m_','w_h2o__5m_','w_h2o__10m_','w_h2o__15m_','w_h2o__20m_']]):
        return 'w_h2o_'
    elif any([prefix in name for prefix in [
        'T_1m_', 'T_2m_', 'T_3m_', 'T_4m_', 'T_5m_', 'T_6m_', 'T_7m_', 'T_8m_', 'T_9m_', 'T_10m_', 
        'T_11m_', 'T_12m_', 'T_13m_', 'T_14m_', 'T_15m_', 'T_16m_', 'T_17m_', 'T_18m_', 'T_19m_', 'T_20m_'
    ]]):
        return 'temperature'
    elif any([prefix in name for prefix in [
        'RH_1m_', 'RH_2m_', 'RH_3m_', 'RH_4m_', 'RH_5m_', 'RH_6m_', 'RH_7m_', 'RH_8m_', 'RH_9m_', 'RH_10m_', 
        'RH_11m_', 'RH_12m_', 'RH_13m_', 'RH_14m_', 'RH_15m_', 'RH_16m_', 'RH_17m_', 'RH_18m_', 'RH_19m_', 'RH_20m_'
    ]]):
        return 'RH'
    elif any([prefix in name for prefix in ['tc_1m', 'tc_2m', 'tc_3m', 'tc_5m', 'tc_10m', 'tc_15m', 'tc_20m']]):
        return 'virtual temperature'
    elif any([prefix in name for prefix in [
        'Tsoil_3_1cm_d', 'Tsoil_8_1cm_d', 'Tsoil_18_1cm_d', 'Tsoil_28_1cm_d', 'Tsoil_4_4cm_d', 'Tsoil_9_4cm_d', 'Tsoil_19_4cm_d', 'Tsoil_29_4cm_d', 
        'Tsoil_0_6cm_d',  'Tsoil_10_6cm_d', 'Tsoil_20_6cm_d', 'Tsoil_30_6cm_d', 'Tsoil_1_9cm_d', 'Tsoil_11_9cm_d', 'Tsoil_21_9cm_d', 'Tsoil_31_9cm_d'
    ]]):
        return 'soil temperature'
    elif name == 'Gsoil_d':
        return 'ground heat flux'
    elif name == 'Qsoil_d':   
        return 'soil moisture'
    elif name == 'Rsw_in_9m_d':
        return 'shortwave radiation incoming'
    elif name == 'Rsw_out_9m_d':
        return 'shortwave radiation outgoing'
    elif name in ['Vtherm_c', 'Vtherm_d', 'Vtherm_ue', 'Vtherm_uw']:
        return "Vtherm"
    elif name in ['Vpile_c', 'Vpile_d', 'Vpile_ue', 'Vpile_uw']:
        return "Vpile"
    elif name in ['IDir_c', 'IDir_d', 'IDir_ue', 'IDir_uw']:
        return "IDir"
    elif any([prefix in name for prefix in [
        'Tsnow_0_4m_', 'Tsnow_0_5m_', 'Tsnow_0_6m_', 'Tsnow_0_7m_', 'Tsnow_0_8m_', 'Tsnow_0_9m_', 
        'Tsnow_1_0m_', 'Tsnow_1_1m_', 'Tsnow_1_2m_', 'Tsnow_1_3m_', 'Tsnow_1_4m_', 'Tsnow_1_5m_'
    ]]):
        return 'snow temperature'
    elif any([prefix in name for prefix in ['SWE_p1_c', 'SWE_p2_c', 'SWE_p3_c', 'SWE_p4_c']]): # these are the only two 
        return 'SWE'
    # VARIABLE NAMES THAT do not COME FROM THE SOSNOQC DATASETS but we add and use a naming schema consistent with SOSNOQC dataset naming schema
    elif any([prefix in name for prefix in [
        'Tpot_1m_', 'Tpot_2m_', 'Tpot_3m_', 'Tpot_4m_', 'Tpot_5m_', 'Tpot_6m_', 'Tpot_7m_', 'Tpot_8m_', 'Tpot_9m_', 'Tpot_10m_', 
        'Tpot_11m_', 'Tpot_12m_', 'Tpot_13m_', 'Tpot_14m_', 'Tpot_15m_', 'Tpot_16m_', 'Tpot_17m_', 'Tpot_18m_', 'Tpot_19m_', 'Tpot_20m_'
    ]]):
        return 'potential temperature'
    elif name == 'Rlw_in_9m_d':
        return 'longwave radiation incoming'
    elif name == 'Rlw_out_9m_d':
        return 'longwave radiation outgoing'
    elif name in ['Tsurf_c', 'Tsurf_d', 'Tsurf_ue', 'Tsurf_uw', 'Tsurf_rad_d', 'Tsurfplanck_c', 'Tsurfplanck_d', 'Tsurfplanck_ue', 'Tsurfplanck_uw']:
        return "surface temperature"
    elif any([prefix in name for prefix in ['tke_1m_',    'tke_2m_',    'tke_3m_',    'tke_5m_',    'tke_10m_',    'tke_15m_',    'tke_20m_']]):
        return "turbulent kinetic energy"
    elif name.startswith('Ri_'):
        return 'richardson number'
    elif name.startswith('RiB_'):
        return 'richardson number bulk'
    elif name.startswith('Tpotvirtual'):
        return 'potential virtual temperature'
    elif name.startswith('Tsurfairdensity'):
        return 'air density'
    elif name.startswith('Tsurfmixingratio'):
        return 'mixing ratio'
    # the order of the following two matters
    elif name.startswith('Tsurfpotvirtual'):
        return 'surface potential virtual temperature'
    elif name.startswith('Tsurfpot'):
        return 'surface potential temperature'
    elif name.startswith('Tsurfvaporpressure'):
        return 'vapor pressure'
    elif name.startswith('Tsurfvirtual'):
        return 'surface virtual temperature'
    elif name.startswith('Tvirtual'):
        return 'virtual temperature'
    elif name.startswith('airdensity'):
        return 'air density'
    elif name.startswith('mixingratio'):
        return 'mixing ratio'
    elif name.startswith('vaporpressure'):
        return 'vapor pressure'
    elif name.startswith('temp_gradient'):
        return 'temperature gradient'
    elif name.startswith('wind_gradient'):
        return 'wind gradient'
    elif name.startswith("u*_"):
        return 'shear velocity'
    elif name.startswith("L_"):
        return 'Obukhov length'
    elif name.startswith("omega_"):
        return 'Decoupling metric'
    elif name.startswith("SnowDepth_"):
        return 'snow depth'
    

def get_tidy_dataset(ds, variable_names):
    """Convert an SoS netcdf xr.Dataset into a dataframe with time, height, tower, and measurement 
    as indexes in a tidy dataset.

    Args:
        ds (xr.Dataset): Dataset to convert
        variable_names (list(str)): Variable names that you want operated on. Variables may not be supported.
    """
    if type(ds) == xr.Dataset:
        tidy_df = ds[variable_names].to_dataframe().reset_index().melt(id_vars='time', value_vars=variable_names)
    elif type(ds) == pd.DataFrame:
        tidy_df = ds[variable_names + ['time']].melt(id_vars='time', value_vars=variable_names)
    else:
        raise ValueError("wrong ds type")

    tidy_df['height'] = tidy_df['variable'].apply(_height_from_variable_name)
    tidy_df['tower'] = tidy_df['variable'].apply(_tower_from_variable_name)
    tidy_df['measurement'] = tidy_df['variable'].apply(_measurement_from_variable_name)
    return tidy_df


def tidy_df_add_variable(tidy_df_original, variable_new, variable, measurement, height, tower):
    """Add variable to a tidy dataset in the format returned by the `get_tidy_dataset` function.

    Args:
        tidy_df_original (pd.DataFrame): Tidy dataset to add data to.
        variable_new (np.array): Array of new-data values to be added.
        variable (str): Variable name for new data.
        measurement (str): Measurement name for new data.
        height (int): Height for new data.
        tower (str): Tower for new data.

    Returns:
        _type_: _description_
    """
    time_array = tidy_df_original.time.drop_duplicates().values
    assert len(variable_new) == len(time_array), f"Found {len(variable_new)} not {len(time_array)}"
    new_data_df = pd.DataFrame({
        'time': time_array,
        'value': variable_new
    }).assign(
        variable = variable,
        measurement = measurement,
        height = height,
        tower = tower
    )

    return pd.concat([tidy_df_original, new_data_df])