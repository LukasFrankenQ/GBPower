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

"""

import logging

logger = logging.getLogger(__name__)

import sys
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from scipy import stats, optimize

sys.path.append(str(Path.cwd() / 'scripts'))
from _helpers import configure_logging


def estimate_normal_params_truncated(data, truncation_quantile=0.7):
    """
    Estimate the parameters of an original normal distribution given data
    truncated at a specified quantile.

    Parameters:
    - data: array-like, the observed truncated data
    - truncation_quantile: float, the quantile at which the data is truncated (default is 0.7)

    Returns:
    - mu_est: float, estimated mean of the original normal distribution
    - sigma_est: float, estimated standard deviation of the original normal distribution
    """
    mu_init = np.mean(data)
    sigma_init = np.std(data, ddof=1)
    params_init = [mu_init, sigma_init]

    truncation_point = stats.norm.ppf(truncation_quantile)

    def neg_log_likelihood(params):
        mu, sigma = params
        if sigma <= 0:
            return np.inf  # sigma must be positive

        pdf_vals = stats.norm.pdf(data, loc=mu, scale=sigma)
        cdf_trunc = stats.norm.cdf(truncation_point, loc=mu, scale=sigma)

        epsilon = 1e-9
        pdf_vals = np.maximum(pdf_vals, epsilon)
        cdf_trunc = max(cdf_trunc, epsilon)

        nll = -np.sum(np.log(pdf_vals)) + len(data) * np.log(cdf_trunc)
        return nll

    result = optimize.minimize(
        neg_log_likelihood,
        params_init,
        method='L-BFGS-B',
        bounds=[(None, None), (1e-5, None)]
    )

    if result.success:
        mu_est, sigma_est = result.x
        return mu_est, sigma_est
    else:
        raise RuntimeError("Optimization failed: " + result.message)


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

    for carrier in ['onwind', 'offwind']:

        # obtain roc levels from bids where bid data is available
        all_roc_plants = bmus.loc[bmus['carrier'] == carrier]
        all_roc_plants = all_roc_plants.drop(all_roc_plants.index.intersection(cfd))

        # roc_avail = bid_stats.loc[bid_stats['carrier'] == carrier]
        # roc_avail = roc_avail.loc[roc_avail.index.intersection(bmus.index)]
        roc_avail = all_roc_plants.loc[all_roc_plants.index.intersection(bid_stats.index)]

        roc_avail['roc_value'] = bid_stats.loc[roc_avail.index, 'mean']
        roc_values.append(roc_avail[['roc_value']])

        # estimate roc values for the remaining plants
        truncation_quantile = len(roc_avail) / len(all_roc_plants)

        print(roc_avail)
        print(truncation_quantile)
        print(roc_avail['roc_value'].mean())
        print(roc_avail['roc_value'].std())
        
        max_tries = 100
        tries = 0
        
        # numerically fast but unstable, try multiple times
        while tries < max_tries:
            try:
                mu_est, sigma_est = estimate_normal_params_truncated(
                    roc_avail['roc_value'].values,
                    truncation_quantile
                    )
                break
            except RuntimeError:
                tries += 1
        
        if tries == max_tries:
            raise RuntimeError('Optimization failed too many times.')

        ppf = stats.norm.ppf(truncation_quantile, loc=mu_est, scale=sigma_est)    

        needed = len(all_roc_plants) - len(roc_avail)

        sample_roc = np.random.normal(size=10_000, loc=mu_est, scale=sigma_est)
        
        print('----------------------------')   
        print(len(all_roc_plants))
        print(len(roc_avail))

        print('----------------------------')
        print(len(sample_roc[sample_roc >= ppf]))
        sample_roc = sample_roc[sample_roc >= ppf][:needed]

        print('sample_roc')
        print(sample_roc)
        print(sample_roc.shape)

        print('missing idx')
        print(all_roc_plants.index.difference(roc_avail.index))
        print(all_roc_plants.index.difference(roc_avail.index).shape)


        roc_values.append(
            pd.DataFrame(
                {'mean': sample_roc},
                index=all_roc_plants.index.difference(roc_avail.index),
            )
        )

    roc_values = pd.concat(roc_values)    
    print(roc_values)
