"""
Auto-converted from notebooks/energy_crisis_shift.ipynb.
Edits should target this file directly; the .ipynb is the source-of-truth for exploratory work only.
"""

from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent

# --- cell 0 ---
import pypsa
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt

from plotting_constants import nice_names, color_dict

# --- cell 1 ---
# start, end = '2022-01-01', '2023-06-30'
start, end = '2022-01-01', '2024-12-31'
# start, end = '2022-01-01', '2022-12-31'

date_range = pd.date_range(start, end).strftime('%Y-%m-%d')

# --- cell 2 ---
mp = pd.read_csv(REPO_ROOT / 'summaries' / 'marginal_prices_summary_flex.csv', index_col=0, parse_dates=True, header=[0,1])
sc = pd.read_csv(REPO_ROOT / 'summaries' / 'total_summary_flex.csv', index_col=[0,1], parse_dates=True)

# --- cell 3 ---
sc.index.get_level_values(1).unique()

# --- cell 4 ---
lw = pd.read_csv(REPO_ROOT / 'data' / 'prerun' / 'load_weights.csv', index_col=0)
lw.index = lw.index.astype(str)

# --- cell 5 ---
regions = gpd.read_file(REPO_ROOT / 'data' / 'regions_onshore_s.geojson')

# --- cell 6 ---
bmus = pd.read_csv(REPO_ROOT / 'data' / 'prerun' / 'prepared_bmus.csv', index_col=0)
foss = bmus.loc[bmus.loc[:, 'carrier'] == 'fossil']

foss

# --- cell 7 ---
'''
loads = []

for day in date_range:

    try:
        n = pypsa.Network(REPO_ROOT / 'results' / day / 'network_flex_s_national.nc')
        l = n.loads.query('carrier == "electricity"').index
        loads.append(n.loads_t.p_set[l])
    except FileNotFoundError:
        continue

loads = pd.concat(loads)
'''

# loads = pd.read_csv('hold_loads.csv', index_col=0, parse_dates=True)
loads = pd.read_csv(
    REPO_ROOT / 'summaries' / 'total_gb_load.csv', index_col=0, parse_dates=True
    ).loc[start:end]

# --- cell 8 ---
7_000_000_000 / loads.mul(0.5).sum().sum()

# --- cell 9 ---
780_000_000 * 3 / loads.mul(0.5).sum().sum()

# --- cell 10 ---
# loads.to_csv('hold_loads.csv')

# --- cell 11 ---
from tqdm import tqdm

pns = []

for day in tqdm(date_range):

    try:
        pns.append(
            pd.read_csv(
                REPO_ROOT / 'data' / 'base' / day / 'physical_notifications.csv',
                index_col=0
                )
            )
    except FileNotFoundError:
        continue

pns = pd.concat(pns)

# --- cell 12 ---
from tqdm import tqdm

dap = []

for day in tqdm(date_range):

    try:
        dap.append(
            pd.read_csv(
                REPO_ROOT / 'data' / 'base' / day / 'day_ahead_prices.csv',
                index_col=0
                )
            )
    except FileNotFoundError:
        continue

dap = pd.concat(dap).iloc[:,0]

# --- cell 13 ---
gas_gen = pns.loc[:, pns.columns.intersection(foss.index)].sum(axis=1)

# --- cell 14 ---
(dap <= 0).sum() / len(dap)

# --- cell 15 ---
idx = pd.IndexSlice

zmp = mp.loc[start:end, idx['zonal',:]]
nmp = mp.loc[start:end, idx['national',:]]

# --- cell 16 ---
n_values = zmp.apply(lambda x: x.nunique(), axis=1)

# --- cell 17 ---
import seaborn as sns

for i in n_values.unique()[:2]:

    ssz = zmp.loc[n_values.loc[n_values == i].index]
    ssn = nmp.loc[n_values.loc[n_values == i].index]

    fig, ax = plt.subplots(figsize=(7, 3))

    sns.kdeplot(np.random.choice(ssz.values.flatten(), 2000), ax=ax, label='zonal')
    sns.kdeplot(np.random.choice(ssn.values.flatten(), 2000), ax=ax, label='national')

    share_of_times = np.around(100 * (ssn.shape[0] / len(nmp.index)), 2)

    ax.set_title(f'{i} unique values ({share_of_times}% of times)')

    ax.legend()
    plt.show()

# --- cell 18 ---
(zmp < 0).any(axis=1).sum() / len(zmp)

# --- cell 19 ---
nmp.columns = nmp.columns.get_level_values(1)

# --- cell 20 ---
zmp.columns = zmp.columns.get_level_values(1)

# --- cell 21 ---
def classify_period(nat, lmp):

    nat = nat.copy()
    lmp = lmp.copy()

    nat_negative_mask = nat.loc[nat.iloc[:,0] <= 0].index

    nat_negative_prices = {
        'national': nat.loc[nat_negative_mask],
        'zonal': lmp.loc[nat_negative_mask],
    }

    lmp.drop(nat_negative_mask, inplace=True)
    nat.drop(nat_negative_mask, inplace=True)

    all_positive_mask = lmp.loc[(lmp > 0).all(axis=1)].index

    all_positive_prices = {
        'national': nat.loc[all_positive_mask],
        'zonal': lmp.loc[all_positive_mask],
    }

    lmp.drop(all_positive_mask, inplace=True)
    nat.drop(all_positive_mask, inplace=True)

    mixed_prices = {
        'national': nat,
        'zonal': lmp,
    }

    return nat_negative_prices, all_positive_prices, mixed_prices


neg, pos, mixed = classify_period(nmp, zmp)

# --- cell 22 ---

def bars_to_ax(ax, lmp, loads, weights, costs, index_groups):

    distributed_price = costs.sum().drop('wholesale') / loads.sum().sum() * 1e6 #/2

    total_load = loads.sum().sum()

    bar_kwargs = {
        'alpha': 0.8,
    }
    # Create distinct colors for each cost component.

    for j, group in enumerate(index_groups):

        cumulative_positive = 0
        cumulative_negative = 0

        for i, (cat, value) in enumerate(distributed_price.items()):

            if value >= 0:
                ax.bar(j-0.25, value, width=0.5, bottom=cumulative_positive,
                    color=color_dict[cat], align='edge', label=f"cost: {cat}", **bar_kwargs)
                ax.plot(
                    [j-0.25, j+0.25],
                    [cumulative_positive + value, cumulative_positive + value],
                    color='k',
                    alpha=0.5,
                    lw=0.5,
                )
                cumulative_positive += value
            else:
                ax.bar(j-0.25, value, width=0.5, bottom=cumulative_negative,
                    color=color_dict[cat], align='edge', label=f"cost: {cat}", **bar_kwargs)
                ax.plot(
                    [j-0.25, j+0.25],
                    [cumulative_negative + value, cumulative_negative + value],
                    color='k',
                    alpha=0.5,
                    lw=0.5,
                )
                cumulative_negative += value

        assert len(lmp[group].iloc[0].unique()) == 1, "There should be only one unique value in the group"
        lmp_group = lmp[group].mean(axis=1)

        avg = (lmp_group * loads.values.flatten()).sum() / total_load
        x_loc = j - 0.25
        bar_width = 0.5

        if avg >= 0:
            ax.bar(
                x_loc,
                avg,
                width=bar_width,
                bottom=cumulative_positive,
                align='edge',
                label="lmp_group",
                color=color_dict['wholesale'],
                **bar_kwargs,
            )
            ax.plot(
                [x_loc, x_loc + bar_width],
                [cumulative_positive + avg, cumulative_positive + avg],
                color='k',
                alpha=0.5,
                lw=0.5,
            )
        else:
            ax.bar(
                x_loc,
                avg,
                width=bar_width,
                bottom=cumulative_negative,
                align='edge',
                label="lmp_group",
                color=color_dict['wholesale'],
                **bar_kwargs,
            )
            ax.plot(
                [x_loc, x_loc + bar_width],
                [cumulative_negative + avg, cumulative_negative + avg],
                color='k',
                alpha=0.5,
                lw=0.5,
            )
        ax.plot(
            [x_loc, x_loc + bar_width],
            [cumulative_positive + cumulative_negative + avg, cumulative_positive + cumulative_negative + avg],
            color='r',
            alpha=0.5,
        )

# --- cell 23 ---
ss = mixed['zonal'].iloc[0]

low_cost_regions = ss[ss < 0].index
high_cost_regions = ss[ss > 0].index

# --- cell 24 ---
from shapely.ops import unary_union
import geopandas as gpd

def regions_to_ax(regions, ax, dark_regions):

    if not dark_regions.empty:
        gpd.GeoSeries([unary_union(regions.loc[dark_regions].geometry)]).plot(ax=ax, color='midnightblue', alpha=0.8, edgecolor='none', lw=0.5)
    gpd.GeoSeries([unary_union(regions.geometry)]).plot(ax=ax, facecolor='none', edgecolor='k', alpha=1, lw=0.3)
    ax.set_xticks([])
    ax.set_yticks([])

# --- cell 25 ---
from copy import deepcopy


def get_wholesale_costs(mps, loads, lw):

    loads = loads.copy() / 2
    lw = lw.copy().loc[mps.columns].values

    # print(mps.mean(axis=1))
    weighted_row_means = (mps.mul(lw, axis=1)).sum(axis=1)# / lw.sum()
    # print(weighted_row_means)

    if len(loads.shape) > 1:
        loads = loads.sum(axis=1).values
    else:
        loads = loads.values

    return (weighted_row_means * loads).sum() / loads.sum()# / lw.loc[mps.columns].sum()


print('Warning! Double check wholesale prices')

def get_total_prices(sc, mps, loads, lw, layout, aggregate_time=True):

    mps = deepcopy(mps)

    try:
        mps = mps[layout]
    except KeyError:
        pass

    costs = sc.loc[idx[mps.index, :], layout].unstack().drop(columns=['balancing_volume'])

    if aggregate_time:
        return costs.sum() / loads.loc[costs.index].sum().sum() * 1e6
    else:
        return costs.div(loads.loc[costs.index].values.flatten(), axis=0) * 1e6


get_total_prices(sc, neg['national'], loads, pd.Series(1, ['GB']), 'national', aggregate_time=False).head()

# --- cell 26 ---
def simple_bars_to_ax(ax, values):

    cumulative_positive = 0
    cumulative_negative = 0

    bar_kwargs = {
        'alpha': 0.8,
    }

    for i, (cat, value) in enumerate(values.items()):

        if value >= 0:
            ax.bar(-0.25, value, width=0.5, bottom=cumulative_positive,
                color=color_dict[cat], align='edge', label=f"cost: {cat}", **bar_kwargs)
            ax.plot(
                [-0.25, 0.25],
                [cumulative_positive + value, cumulative_positive + value],
                color='k',
                alpha=0.5,
                lw=0.5,
            )
            cumulative_positive += value
        else:
            ax.bar(-0.25, value, width=0.5, bottom=cumulative_negative,
                color=color_dict[cat], align='edge', label=f"cost: {cat}", **bar_kwargs)
            ax.plot(
                [-0.25, 0.25],
                [cumulative_negative + value, cumulative_negative + value],
                color='k',
                alpha=0.5,
                lw=0.5,
            )
            cumulative_negative += value
        
    total = cumulative_positive + cumulative_negative
    ax.plot(
        [-0.25, 0.25],
        [total, total],
        color='red',
        lw=2,
    )

    ax.plot(
        [-0.25, 0.25],
        [values.drop('congestion_rent').sum(), values.drop('congestion_rent').sum()],
        color='red',
        linestyle='-.',
        alpha=0.8,
        lw=2,
    )

# --- cell 27 ---
fp = pd.read_excel(
    REPO_ROOT / 'systemaveragepriceofgasdataset130225.xlsx',
    sheet_name=3,
    parse_dates=True,
    index_col=0,
    skiprows=1
    )

def process(df):

    df = (
        df.iloc[4:]
        .rename(
        columns={
            'Unnamed: 1': 'day_price',
            'Unnamed: 2': 'before_week_average',
        })
        .replace('[x]', np.nan)
        ['day_price']
        )
    df.index.name = 'date'
    df.index = pd.to_datetime(df.index)
    return df * 10

fp = process(fp)

# --- cell 28 ---
fp.loc['2022':'2024-12-31'].plot()

# --- cell 29 ---
fp.loc['2022':'2025-01-01'].tail()

# --- cell 30 ---
n = neg['zonal'].index

fp.loc[n.strftime('%Y-%m-%d')].head()

# --- cell 31 ---
def get_weighted_average_price(df, weights):

    weighted_sum = df.multiply(weights.loc[df.index].values, axis=0) \
                            .groupby(df.index.strftime('%Y-%m-%d')).sum()
        
    daily_weights = weights.loc[df.index] \
                        .groupby(df.index.strftime('%Y-%m-%d')).sum()

    return weighted_sum.divide(daily_weights.values, axis=0)

# --- cell 32 ---
import seaborn as sns

fig, axs = plt.subplots(3, 3, figsize=(7.5, 6), gridspec_kw={'width_ratios': [1, 2, 1]})

# Reordering the rows: 
# First row (index 0) = what was previously third row (index 2)
# Second row (index 1) = what was previously first row (index 0)
# Third row (index 2) = what was previously second row (index 1)

# What was previously in third row (pos) now in first row
bars_to_ax(
    axs[0,1],
    pos['zonal'],
    loads.loc[pos['zonal'].index],
    lw,
    sc.loc[idx[pos['zonal'].index, :], 'zonal'].unstack().drop(columns=['balancing_volume']),
    [low_cost_regions, high_cost_regions]
    )

bars_to_ax(
    axs[0,0],
    pos['national'],
    loads.loc[pos['national'].index],
    lw,
    sc.loc[idx[pos['national'].index, :], 'national'].unstack().drop(columns=['balancing_volume']),
    [['GB']]
    )

# What was previously in first row (mixed) now in second row
bars_to_ax(
    axs[1,1],
    mixed['zonal'],
    loads.loc[mixed['zonal'].index],
    lw,
    sc.loc[idx[mixed['zonal'].index, :], 'zonal'].unstack().drop(columns=['balancing_volume']),
    [low_cost_regions, high_cost_regions]
    )

bars_to_ax(
    axs[1,0],
    mixed['national'],
    loads.loc[mixed['national'].index],
    lw,
    sc.loc[idx[mixed['national'].index, :], 'national'].unstack().drop(columns=['balancing_volume']),
    [['GB']]
    )

# What was previously in second row (neg) now in third row
bars_to_ax(
    axs[2,1],
    neg['zonal'],
    loads.loc[neg['zonal'].index],
    lw,
    sc.loc[idx[neg['zonal'].index, :], 'zonal'].unstack().drop(columns=['balancing_volume']),
    [low_cost_regions, high_cost_regions]
    )

bars_to_ax(
    axs[2,0],
    neg['national'],
    loads.loc[neg['national'].index],
    lw,
    sc.loc[idx[neg['national'].index, :], 'national'].unstack().drop(columns=['balancing_volume']),
    [['GB']]
    )

neg_zonal = get_total_prices(sc, neg['zonal'], loads, lw.iloc[:,0], 'zonal')
neg_national = get_total_prices(sc, neg['national'], loads, pd.Series(1, ['GB']), 'national')

neg_diff = neg_zonal - neg_national
simple_bars_to_ax(
    axs[2,2],
    # neg_diff,
    - neg_diff,
)

pos_zonal = get_total_prices(sc, pos['zonal'], loads, lw.iloc[:,0], 'zonal')
pos_national = get_total_prices(sc, pos['national'], loads, pd.Series(1, ['GB']), 'national')

pos_diff = pos_zonal - pos_national
simple_bars_to_ax(
    axs[0,2],
    # pos_diff,
    - pos_diff,
)

mixed_zonal = get_total_prices(sc, mixed['zonal'], loads, lw.iloc[:,0], 'zonal')
mixed_national = get_total_prices(sc, mixed['national'], loads, pd.Series(1, ['GB']), 'national')

mixed_diff = mixed_zonal - mixed_national
simple_bars_to_ax(
    axs[1,2],
    # mixed_diff,
    - mixed_diff,
)

for ax in axs[:, 0]:
    ax.set_xlim(-0.5, 0.5)

for ax in axs[:, 1]:
    ax.set_xlim(-0.5, 1.5)


for ax in axs[:,1]:
    ax.set_yticklabels([])
    ax.spines['left'].set_visible(False)

for ax in axs[:,2]:
    ax.spines['left'].set_visible(False)

for ax in axs.flatten():
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', axis='y')
    ax.set_axisbelow(True)

for ax in axs[2,:2]:
    ax.set_ylim(-70, 70)
for ax in axs[0,:2]:
    ax.set_ylim(-20, 250)
for ax in axs[1,:2]:
    ax.set_ylim(-70, 220)

ax_inset = axs[2,0].inset_axes([0.65, -0.1, 0.35, 0.5])
regions_to_ax(
    regions.set_index('name'),
    ax_inset,
    pd.Index([])
)
ax_inset = axs[2,1].inset_axes([0.8, 0.015, 0.2, 0.3])
regions_to_ax(
    regions.set_index('name'),
    ax_inset,
    high_cost_regions,
)
ax_inset = axs[0,0].inset_axes([0.65, -0.1, 0.35, 0.5])
regions_to_ax(
    regions.set_index('name'),
    ax_inset,
    high_cost_regions.union(low_cost_regions),
)
ax_inset = axs[0,1].inset_axes([0.8, 0.015, 0.2, 0.3])
regions_to_ax(
    regions.set_index('name'),
    ax_inset,
    high_cost_regions.union(low_cost_regions),
)
ax_inset = axs[1,0].inset_axes([0.65, -0.1, 0.35, 0.5])
regions_to_ax(
    regions.set_index('name'),
    ax_inset,
    high_cost_regions.union(low_cost_regions),
)
ax_inset = axs[1,1].inset_axes([0.8, 0.015, 0.2, 0.3])
regions_to_ax(
    regions.set_index('name'),
    ax_inset,
    high_cost_regions,
)

for ax in axs[:2,2]:
    ax.set_xticks([])

for ax in axs[:,2]:
    ax.set_xlim(-0.5, 0.5)

neg_share = 100 * len(neg['national']) / len(zmp)
pos_share = 100 * len(pos['national']) / len(zmp)
mixed_share = 100 * len(mixed['national']) / len(zmp)

axs[2,0].set_ylabel(f'\nExtreme Wind Case\n({neg_share:.1f}% of time)')
axs[0,0].set_ylabel(f'\nLow Wind Case\n({pos_share:.1f}% of time)')
axs[1,0].set_ylabel(f'Consumer Price (£/MWh)\nHigh Wind Case\n({mixed_share:.1f}% of time)')

# labels, handles = 

for ax in axs[:,2]:
    position = ax.get_position()
    new_pos = [position.x0 + 0.025, position.y0, position.width, position.height]
    ax.set_position(new_pos)

# for ax in axs[:,3]:
#     position = ax.get_position()
#     new_pos = [position.x0 + 0.04, position.y0, position.width, position.height]
#     ax.set_position(new_pos)

title_pad = 10
axs[0,0].set_title('National Market', fontsize=10, pad=title_pad)
axs[0,1].set_title('Zonal Market', fontsize=10, pad=title_pad)
axs[0,2].set_title('Zonal Consumer Cost\nReduction (£/MWh)', fontsize=10, pad=title_pad)
# axs[0,3].set_title('Daily Price\nDifference (£/MWh)', fontsize=10, pad=title_pad, loc='left')

axs[2,1].set_xticks([0, 1])
axs[2,1].set_xticklabels(['Low Price\nRegions', 'High Price\nRegions'])
axs[2,1].tick_params(axis='x', which='both', length=0)

for ax in axs[:2,:2].flatten():
    ax.set_xticks([])

axs[2,0].set_xticks([])
axs[2,2].set_xticks([])

axs[0,2].set_ylim(-4, 4)

for ax in axs[:,:].flatten():
    ax.axhline(0, color='k', lw=1)
    ax.spines['bottom'].set_visible(False)

handles, labels = [], []

from matplotlib.patches import Patch
from matplotlib.lines import Line2D

for label, color in color_dict.items():

    if label in ['wholesale selling', 'wholesale buying']:
        continue

    handles.append(Patch(color=color, label=nice_names[label]))
    labels.append(nice_names[label])

handles.append(Line2D([0], [0], color='r', lw=1, linestyle='-', alpha=0.8))
labels.append("Total")

handles.append(Line2D([0], [0], color='r', lw=1, linestyle='-.', alpha=0.8))
labels.append("Total w/o Congestion Rent")

axs[2,0].legend(
    handles=handles,
    labels=labels,
    loc='upper left',
    bbox_to_anchor=(-1, -0.4),
    fontsize=8,
    ncol=4,
    frameon=False,
)

axs[0,0].text(
    -1.35,
    330,
    f'{start} - {end}',
    fontsize=9,
    ha='left',
    weight='bold',
)

plt.savefig(f'wind_cases_from_{start}_to_{end}.pdf', bbox_inches='tight')
plt.show()
