import pandas as pd
import numpy as np

# Number of datapoints collected in the averaging time period
# For the SoS dataset, 5 minute averages of 20hz data have 6000 data points
# per sample
COUNTS_PER_DATAPOINT = 6000

def clean_eddy_covariance(
        ec_variable: np.array, 
        counts_variable: np.array,
        lower_threshold,
        upper_threshold,
        fraction_good_data_reqd: float = 0.9, 
) -> np.array:

    assert len(ec_variable) == len(counts_variable)

    df = pd.DataFrame({
        'ec': ec_variable,
        'count': counts_variable
    })

    # Remove datapoints that do not satisfy the fraction_good_data_reqd threshold
    df['ec'] = df.apply(
        lambda row: row['ec'] if (
                row['count'] >= fraction_good_data_reqd*COUNTS_PER_DATAPOINT
            ) else np.nan,
        axis = 1 
    )

    # Remove data points outside n standard deviations where n is stddev_multipler.
    df['ec'] = df['ec'].where(
        (df['ec'] < (upper_threshold))
        &
        (df['ec'] > (lower_threshold))
    )

    return df['ec'].values