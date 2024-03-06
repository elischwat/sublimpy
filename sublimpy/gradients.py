"""
Classes to calculate wind and temperature gradients and Richardson numbers from SoS measurements.
"""
import numpy as np
from metpy.units import units
from scipy.optimize import curve_fit
import pint_xarray
import metpy.constants

"""
Z0 = 0.005 for snow surface from Lapo, Nijssen, and Lundquist, 2019
"""
Z0 = 0.005
    
class LogPolynomialWithRoughness:
    """This class contains functions for fitting temperature data to a log-polynomial function, which 
    enforces a surface temperature at the roughness height Z0, and calculating the gradient of that function.

    This is an original method inspired by Sun (2011) and Grachev et al. (2005). Analysis indicates that
    this method is best for fitting temperature profiles when surface temperatures are available over snow.

    Sun, J. Vertical Variations of Mixing Lengths under Neutral and Stable Conditions during CASES-99. 
        Journal of Applied Meteorology and Climatology 50, 2030–2041 (2011)
    Grachev, A. A., Fairall, C. W., Persson, P. O. G., Andreas, E. L. & Guest, P. S. Stable Boundary-Layer 
        Scaling Regimes: The Sheba Data. Boundary-Layer Meteorol 116, 201–235 (2005).

    """

    @staticmethod
    def function(z, a, b, c):
        return a*(np.log(z/Z0))**2 + b*np.log(z/Z0) + c
    
    @staticmethod
    def fit_function(values, heights):
        # remove nans
        valid = ~(np.isnan(values) | np.isnan(heights))
        values = np.array(values)[valid]
        heights = np.array(heights)[valid]
        # only grab values that are from above our estimate of snowdepth_height
        valid = (heights > 0)
        values = values[valid]
        heights = heights[valid]
        if len(values) > 2 and all([np.isfinite(v) for v in values]):
            [a,b,c], _ = curve_fit(LogPolynomialWithRoughness.function, heights, values)
            return a,b,c
        else:
            return np.nan, np.nan, np.nan
        
    @staticmethod
    def gradient(z, a_u, b_u, a_v, b_v):
        return np.sqrt(((2*a_u*np.log(z/Z0) + b_u)/z)**2 + ((2*a_v*np.log(z/Z0) + b_v)/z)**2)

    @staticmethod
    def gradient_single_component(z, a_u, b_u):
        return (2*a_u*np.log(z/Z0) + b_u)/z

    @staticmethod
    def calculate_temperature_gradient_for_height(
            ds, 
            calculation_height
        ):
        """ 
        These calculations are done by fitting log-polynomial curve to temperature measurements 
        on tower C.
        We DO include the boundary wall condition, applying the measured surface temperature
        at a roughness height (T=T_s at z=z0). Therefore, we adjust for snow depth in our calculations.
        """    
        
        obs_heights = [Z0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        temp_variables = ['Tsurfpotvirtual_c'] + [f'Tpotvirtual_{h}m_c' for h in obs_heights[1:]]
        
        snowdepth_variable = 'SnowDepth_c'

        # create_datasets for u and v data 
        temp_ds = ds[temp_variables + [snowdepth_variable]].to_dataframe().rename(
            columns=dict(zip(
                temp_variables, 
                obs_heights
            ))
        )
        
        # calculate fitted loglinear parameters
        temp_ds['params'] = temp_ds.apply(lambda row: LogPolynomialWithRoughness.fit_function(
            [
                row[h] for h in obs_heights
            ],
            [
                h - row[snowdepth_variable] if h != Z0 else h for h in obs_heights
            ]
        ), axis = 1)
        
        # split the params column into two separate columns - we don't need the third (c) parameter 
        # because the gradient doesn't require them.
        temp_ds['a'] = temp_ds['params'].apply(lambda tup: tup[0])
        temp_ds['b'] = temp_ds['params'].apply(lambda tup: tup[1])
        
        gradient = temp_ds.apply(
            lambda row: LogPolynomialWithRoughness.gradient_single_component(calculation_height, row['a'], row['b']),
            axis=1
        )
        return gradient
    
class LogPolynomial:
    """This class contains functions for fitting temperature and wind data to a log-polynomial function and 
    calculating the gradient of that function, as described in Grachev et al. (2005).

    Grachev, A. A., Fairall, C. W., Persson, P. O. G., Andreas, E. L. & Guest, P. S. Stable Boundary-Layer 
        Scaling Regimes: The Sheba Data. Boundary-Layer Meteorol 116, 201–235 (2005).

    """
    @staticmethod
    def function(z, a, b, c):
        return a*(np.log(z))**2 + b*np.log(z) + c
    
    @staticmethod
    def fit_function(values, heights):
        # remove nans
        valid = ~(np.isnan(values) | np.isnan(heights))
        values = np.array(values)[valid]
        heights = np.array(heights)[valid]
        # only grab values that are from above our estimate of snowdepth_height
        valid = (heights > 0)
        values = values[valid]
        heights = heights[valid]
        if len(values) > 2 and all([np.isfinite(v) for v in values]):
            [a,b,c], _ = curve_fit(LogPolynomial.function, heights, values)
            return a,b,c
        else:
            return np.nan, np.nan, np.nan
            
    @staticmethod
    def gradient(z, a_u, b_u, a_v, b_v):
        return np.sqrt(((2*a_u*np.log(z) + b_u)/z)**2 + ((2*a_v*np.log(z) + b_v)/z)**2)

    @staticmethod
    def gradient_single_component(z, a_u, b_u):
        return (2*a_u*np.log(z) + b_u)/z

    @staticmethod
    def calculate_wind_gradient_for_height(ds, calculation_height, tower, snow_depth_var):
        """ 
        These calculations are done by fitting log-polynomial curve to wind measurements 
        on tower C.

        We do NOT include the boundary wall condition (U=0 at z=0) and therefore
        we do NOT adjust for snow depth.
        """

        # identify two heights on either side of this height 
        if tower == 'c':
            heights = [2,3,5,10,15,20]
        else:
            heights = [1,3,10]  
         
        u_variables = [f'u_{h}m_{tower}' for h in heights] + [snow_depth_var]
        v_variables = [f'v_{h}m_{tower}' for h in heights] + [snow_depth_var]

        # create_datasets for u and v data 
        u_ds = ds[u_variables].to_dataframe().rename(columns=dict(zip(u_variables, heights)))
        v_ds = ds[v_variables].to_dataframe().rename(columns=dict(zip(v_variables, heights)))
        # calculate fitted loglinear parameters
        u_ds['params'] = u_ds.apply(lambda row: LogPolynomial.fit_function(
            [row[h] for h in heights],
            [h - row[snow_depth_var] for h in heights]
        ), axis = 1)
        v_ds['params'] = v_ds.apply(lambda row: LogPolynomial.fit_function(
            [row[h] for h in heights],
            [h - row[snow_depth_var] for h in heights]
        ), axis = 1)
        u_ds['a_u'] = u_ds['params'].apply(lambda tup: tup[0])
        u_ds['b_u'] = u_ds['params'].apply(lambda tup: tup[1])
        v_ds['a_v'] = v_ds['params'].apply(lambda tup: tup[0])
        v_ds['b_v'] = v_ds['params'].apply(lambda tup: tup[1])
        merged_parameters = u_ds[['a_u', 'b_u']].join(v_ds[['a_v', 'b_v']]).reset_index().drop_duplicates()
        gradient = merged_parameters.apply(
            lambda row: LogPolynomial.gradient(calculation_height, row['a_u'], row['b_u'], row['a_v'], row['b_v']),
            axis=1
        )
        return gradient

class Ri:
    """This class contains functions for calculating Richardson numbers. Methods are taken
    from Grachev et al. (2008).

    Grachev, A. A., Andreas, E. L., Fairall, C. W., Guest, P. S. & Persson, P. O. G. Turbulent 
    measurements in the stable atmospheric boundary layer during SHEBA: ten years after. Acta 
    Geophys. 56, 142–166 (2008).
    """
    
    @staticmethod
    def calculate_richardson_number(
        ds,
        height,
        tower,
        wind_gradient_prefix='wind_gradient',
        temp_gradient_prefix = 'temp_gradient'
    ):
        """The standard gradient richardson number. See Grachev et al. (2008) equation 3.
        This uses estimates of the temperature gradient and wind profile gradient to calculate the richardson number.
        """
        multiplier = metpy.constants.g.magnitude / (ds[f'Tpotvirtual_{height}m_c'] * units.celsius).pint.to(units.kelvin)
        Ri = multiplier * ds[f'{temp_gradient_prefix}_{height}m_{tower}'] / ds[f'{wind_gradient_prefix}_{height}m_{tower}']**2
        return Ri.pint.magnitude

    @staticmethod
    def calculate_richardson_number_bulk(
            ds,
            height,
            tower
    ):
        """The standard gradient richardson number. See Grachev et al. (2008) equation 5.
        We adjust for snow depth in our calculations.
        """
        pot_virt_temperature_at_height = ds[f'Tpotvirtual_{height}m_{tower}']
        pot_temperature_at_height = ds[f'Tpot_{height}m_{tower}']
        pot_temperature_at_surface = ds[f'Tsurfpot_{tower}']
        specific_humidity_at_height = ds[f'mixingratio_{height}m_{tower}']
        specific_humidity_at_surface = ds[f'Tsurfmixingratio_{tower}']
        wind_speed_at_height = ds[f'spd_{height}m_{tower}']

        snow_depth = ds['SnowDepth_c']
        adjusted_height = height - snow_depth
        
        term1 = (metpy.constants.g.magnitude * adjusted_height) / (pot_virt_temperature_at_height)
        term2_numer = (pot_temperature_at_surface - pot_temperature_at_height) + 0.61 * (
            pot_virt_temperature_at_height * ( specific_humidity_at_surface - specific_humidity_at_height)
        )
        term2_denom = wind_speed_at_height**2

        return -(term1 * term2_numer / term2_denom).values