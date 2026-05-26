"""
Auto-converted from notebooks/estimate_roc_levels.ipynb.
Edits should target this file directly; the .ipynb is the source-of-truth for exploratory work only.
"""

from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent

# --- cell 0 ---
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path 
import matplotlib.pyplot as plt

# --- cell 1 ---
carrier_mapper = {
    "PHS": "hydro",
    "hydro-scheme": "hydro",
    "hydro": "hydro",
    "dam": "hydro",
    "PHS": "hydro",
    "floating wind": "wind",
    "onwind": "wind",
    "wind": "wind",
    "offwind": "wind",
    "CCGT": "gas",
    "CHP": "gas",
    "biomass": "biomass",
    "gas": "gas",
    "gas-fired": "gas",
    "gas turbine": "gas",
    "coal": "coal",
    "powerstation": "gas",
    "cascade": "gas",
    "nuclear": "nuclear",
    "battery": "battery",
    'interconnector': 'imports',
    'PV': 'solar',
    'solar power station': 'solar',
}

# --- cell 2 ---
carrier_colors = {
    "wind": "#7ac677",
    "offshore wind": "#6895dd",
    "hydro": "purple",
    "coal": "#454546",
    "biomass": "#dbc263",
    "gas": "#f6986b",
    "nuclear": '#549ca2',
    "imports": "#dd75b0",
    "cascade": "#46caf0",
    "solar": "#f9d002",
    "battery": 'turquoise',
}

# --- cell 3 ---
bids = []

# start = pd.Timestamp('2022-01')
# end = pd.Timestamp('2022-07-01')

for d in tqdm(os.listdir(REPO_ROOT / 'data' / 'base')):

    # ts = pd.Timestamp(d)
    # if ts < start or ts > end:
    #     continue

    try:
        bids.append(
            pd.read_csv(
                REPO_ROOT / 'data' / 'base' / d / 'bids.csv', index_col=[0,1]
            )
        )

    except FileNotFoundError:
        pass

bids = pd.concat(bids)

# --- cell 4 ---
bmus = pd.read_csv(REPO_ROOT / 'data' / 'temp_located_bmus.csv', index_col=0)
bmus = bmus.loc[bmus['lat'] != 0.]

# --- cell 5 ---
idx = pd.IndexSlice

bid_prices = bids.loc[idx[:,'price'],:]
bid_acceptances = len(bid_prices) - bid_prices.isna().sum()

thresh = 48

bid_acceptances = bid_acceptances.loc[bid_acceptances > thresh].index

# --- cell 6 ---
windgen = bmus.dropna().loc[bmus.dropna()['carrier'].str.contains('wind')].index

# --- cell 7 ---
bid_prices.index = pd.to_datetime(bid_prices.index.get_level_values(0))

# --- cell 8 ---
import seaborn as sns

#  variances = bids.loc[idx[:,'price'],:].var().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(11, 3))

bid_prices[windgen.intersection(bid_prices.columns)].iloc[30000:31000, :30].plot(ax=ax, legend=False)

ax.set_ylabel('Bid Price (£/MWh)')
# ax.set_xticks([])
# ax.set_xlabel('Time')

ax.set_ylim(ax.get_ylim()[0], 130)

plt.savefig('wind_bid_prices.pdf', bbox_inches='tight')
plt.show()

# --- cell 9 ---
cfd = pd.read_csv(REPO_ROOT / 'data' / 'prerun' / 'cfd_strike_prices.csv', index_col=0).index

# --- cell 10 ---
bid_stats = pd.concat((
    bid_prices.mean().rename('mean'),
    bid_prices.std().rename('std')),
    axis=1).loc[bid_acceptances]
bid_stats = bid_stats.loc[bid_stats.index.intersection(bmus.index)]
bid_stats.sort_values('std', ascending=True, inplace=True)
bid_stats['carrier'] = list(map(lambda name: bmus.loc[name, 'carrier'], bid_stats.index))
# bid_stats.drop(cfd.intersection(bid_stats.index), inplace=True)

# --- cell 11 ---
import seaborn as sns

fig, ax = plt.subplots(figsize=(10, 6))

# wind_rocs = bid_stats.loc[bid_stats['carrier'].str.contains('wind')]
wind_rocs = bid_stats.loc[bid_stats['carrier'] == 'onwind']
sns.kdeplot(wind_rocs['std'], cmap='viridis', fill=True, ax=ax, label='Onshore')

wind_rocs = bid_stats.loc[bid_stats['carrier'] == 'offwind']
# sns.kdeplot(wind_rocs['mean'], cmap='viridis', shade=True, shade_lowest=False, ax=ax, label='Offshore')
sns.kdeplot(wind_rocs['std'], cmap='viridis', fill=True, ax=ax, label='Offshore')

ax.legend()
plt.show()

# --- cell 12 ---
b = bmus.dropna()
b = b.loc[b['carrier'].str.contains('wind')]
b.drop(cfd.intersection(b.index), inplace=True)

b

# --- cell 13 ---
from scipy.stats import norm

# --- cell 14 ---
w = bid_stats.dropna().loc[bid_stats.dropna()['carrier'].str.contains('wind')]

print(w.loc[cfd.intersection(w.index), ['mean', 'std']].mean())
print(w.drop(cfd.intersection(w.index))[['mean', 'std']].mean())

# --- cell 15 ---
import numpy as np
from scipy import stats, optimize

import numpy as np
from scipy.stats import norm

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

    # Sample mean and standard deviation of the truncated data
    x_bar = np.mean(data)
    s = np.std(data, ddof=1)

    # Standardized truncation point (alpha)
    q = truncation_quantile
    z_q = norm.ppf(q)
    alpha = z_q

    # Adjustment factor (lambda)
    lambda_ = norm.pdf(alpha) / (1 - norm.cdf(alpha))

    # Adjusted variance factor (beta)
    beta = 1 - lambda_ * (lambda_ - alpha)

    # Estimate sigma
    sigma_est = s / np.sqrt(beta)

    # Estimate mu
    mu_est = x_bar - sigma_est * lambda_

    return mu_est, sigma_est


# Example usage:
np.random.seed(0)
true_mu = 0
true_sigma = 1
sample_size = 1000
full_data = np.random.normal(loc=true_mu, scale=true_sigma, size=sample_size)
truncation_quantile = 0.7
truncation_point = stats.norm.ppf(truncation_quantile, loc=true_mu, scale=true_sigma)
truncated_data = full_data[full_data <= truncation_point]

mu_est, sigma_est = estimate_normal_params_truncated(truncated_data, truncation_quantile)

print(f"Estimated mu: {mu_est}")
print(f"Estimated sigma: {sigma_est}")
print(f"True mu: {true_mu}")
print(f"True sigma: {true_sigma}")

# --- cell 16 ---


for carrier in ['onwind', 'offwind']:

    all_roc_plants = bmus.loc[bmus['carrier'] == carrier]
    all_roc_plants = all_roc_plants.drop(all_roc_plants.index.intersection(cfd))

    print('{} roc plants of carrier {}'.format(len(all_roc_plants), carrier))
    
    roc_avail = bid_stats.loc[bid_stats['carrier'] == carrier]
    roc_avail = roc_avail.loc[roc_avail.index.intersection(bmus.index)]
    print('if those, we have {} available'.format(len(roc_avail)))

    truncation_quantile = len(roc_avail) / len(all_roc_plants)
    print(truncation_quantile)

    mu_est, sigma_est = estimate_normal_params_truncated(
        roc_avail['mean'].values,
        truncation_quantile
        )

    print(mu_est, sigma_est)

    ppf = stats.norm.ppf(truncation_quantile, loc=mu_est, scale=sigma_est)    

    print(ppf)

    needed = len(all_roc_plants) - len(roc_avail)

    sample_roc = np.random.normal(size=1000, loc=mu_est, scale=sigma_est)
    sample_roc = sample_roc[sample_roc >= ppf][:needed]

    print(roc_avail['mean'].mean(), roc_avail['mean'].std())
    print('number of needed roc values: {}'.format(needed))

    print(sample_roc.mean(), sample_roc.std())

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(roc_avail['mean'], fill=True, ax=ax, label='Avail')
    sns.kdeplot(roc_avail['mean'].tolist() + list(sample_roc), fill=True, ax=ax, label='New')

    ax.legend()

    plt.show()
