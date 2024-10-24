# SPDX-FileCopyrightText: : 2024 Lukas Franken
#
# SPDX-License-Identifier: MIT

"""
Script to create the file 'data/roc_values.csv'.
Compiles renewable bidding prices for a larger numbers of days (here arond 600)
and estimates the ROC values for these and other plants of the same technology.
Assumes that wind farms are bidding their negative ROCs in the balancing market.
This assumptions is based on the most wind farms bidding at exactly the same
or slightly varying price.

Further, we only use data from accepted bids, therefore it is likely that the 
wind farms not represented in the data have higher ROC values.

The script samples these missing values from an estimated distribution of all
ROC values where we consider the existing values to be sampled from a lower
quantile of the distribution. See fct `estimate_normal_params_truncated`.

The script also estimates ROC levels for cascading hydro.
We note that our data suggests that cascading hydro plants are bidding
with a typical ROC-induced behaviour only until around March 2024.
Currently, no method is in place to account for this change in behaviour.

"""

import logging

logger = logging.getLogger(__name__)

import sys
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from scipy import stats
from scipy.stats import norm

sys.path.append(str(Path.cwd() / 'scripts'))
from _helpers import configure_logging



def estimate_normal_params_truncated(data, truncation_quantile=0.7):
    """
    Estimate the parameters of an original normal distribution given data
    truncated at a specified quantile using the method of moments.

    Parameters:
    - data: array-like, the observed truncated data
    - truncation_quantile: float, the quantile at which the data is truncated.

    Returns:
    - mu_est: float, estimated mean of the original normal distribution
    - sigma_est: float, estimated standard deviation of the original normal distribution
    """
    data = np.asarray(data)
    if data.size == 0:
        raise ValueError("Data array is empty.")

    x_bar = np.mean(data)
    s = np.std(data, ddof=1)

    q = truncation_quantile
    z_q = norm.ppf(q)
    alpha = z_q

    lambda_ = norm.pdf(alpha) / (1 - norm.cdf(alpha))

    beta = 1 - lambda_ * (lambda_ - alpha)
    sigma_est = s / np.sqrt(beta)
    mu_est = x_bar - sigma_est * lambda_

    return mu_est, sigma_est



if __name__ == '__main__':

    configure_logging(snakemake)

    bmus = pd.read_csv(snakemake.input.bmu_locations, index_col=0)
    bmus = bmus.loc[bmus['lat'] != 0]

    cfd = pd.read_csv(snakemake.input['cfd_strike_prices'], index_col=0).index

    bids = []
    for d in tqdm(snakemake.input):

        if d.endswith('.csv'):
            continue

        try:
            bids.append(
                pd.read_csv(Path(d) / 'bids.csv', index_col=[0,1]
                )
            )
        except FileNotFoundError:
            pass

    bids = pd.concat(bids)

    idx = pd.IndexSlice

    bid_prices = bids.loc[idx[:,'price'],:]
    bid_acceptances = len(bid_prices) - bid_prices.isna().sum()

    thresh = 48
    bid_acceptances = bid_acceptances.loc[bid_acceptances > thresh].index

    bid_stats = pd.concat((
        bid_prices.mean().rename('mean'),
        ),
        axis=1).loc[bid_acceptances]

    bid_stats = bid_stats.loc[bid_stats.index.intersection(bmus.index)]
    bid_stats['carrier'] = list(
        map(lambda name: bmus.loc[name, 'carrier'],
        bid_stats.index)
        )
    
    roc_values = []

    logger.info('Estimating ROC values for wind farms.')

    for carrier in ['onwind', 'offwind']:

        # obtain roc levels from bids where bid data is available
        all_roc_plants = bmus.loc[bmus['carrier'] == carrier]
        all_roc_plants = all_roc_plants.drop(all_roc_plants.index.intersection(cfd))

        roc_avail = all_roc_plants.loc[all_roc_plants.index.intersection(bid_stats.index)]

        roc_avail['roc_value'] = bid_stats.loc[roc_avail.index, 'mean']
        roc_values.append(roc_avail[['roc_value']])

        # estimate roc values for the remaining plants
        truncation_quantile = len(roc_avail) / len(all_roc_plants)

        mu_est, sigma_est = estimate_normal_params_truncated(
            roc_avail['roc_value'].values,
            truncation_quantile
            )

        ppf = stats.norm.ppf(truncation_quantile, loc=mu_est, scale=sigma_est)    

        needed = len(all_roc_plants) - len(roc_avail)
        sample_roc = np.random.normal(size=10_000, loc=mu_est, scale=sigma_est)
        
        sample_roc = sample_roc[sample_roc >= ppf][:needed]

        roc_values.append(
            pd.DataFrame(
                {'roc_value': sample_roc},
                index=all_roc_plants.index.difference(roc_avail.index),
            )
        )

    logger.info('Estimating ROC values for cascading hydro.')

    cascading_hydro = bmus.loc[bmus['carrier'] == 'cascade'].index

    def build_cascading_roc_prices(bids, name):

        end_date = pd.Timestamp('2024-03-01', tz='utc')
        bids = bids.copy().loc[:end_date, name]
        return bids.value_counts().index[0]

    bid_prices.index = pd.to_datetime(bid_prices.index.get_level_values(0))

    cascade_roc_values = pd.Series(
        {name: build_cascading_roc_prices(bid_prices, name) for name in cascading_hydro},
        name='roc_value'
    )

    roc_values.append(cascade_roc_values)

    pd.concat(roc_values).to_csv(snakemake.output[0])

    
