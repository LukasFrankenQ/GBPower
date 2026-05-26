"""
Auto-converted from notebooks/revenue_stabilisation.ipynb.
Edits should target this file directly; the .ipynb is the source-of-truth for exploratory work only.
"""

from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent

# --- cell 0 ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pypsa

from plotting_constants import nice_names, color_dict

# --- cell 1 ---
day = '2024-03-21'

# --- cell 2 ---
sc = pd.read_csv(REPO_ROOT / 'results' / f'{day}' /  'system_cost_summary_flex.csv', index_col=[0,1], parse_dates=[0])

# --- cell 3 ---
if day != 'total':
    fn_nat = str(REPO_ROOT / 'results' / '{}'  / 'bmu_revenues_flex_{}.csv').format(day, 'national')
    fn_zonal = str(REPO_ROOT / 'results' / '{}'  / 'bmu_revenues_flex_{}.csv').format(day, 'zonal')
else:
    fn_nat = str(REPO_ROOT / 'summaries' / 'total_summary_revenues_flex_{}.csv').format('national')
    fn_zonal = str(REPO_ROOT / 'summaries' / 'total_summary_revenues_flex_{}.csv').format('zonal')

# --- cell 4 ---
nat = pd.read_csv(
    fn_nat,
    index_col=0,
    header=[0,1,2],
    parse_dates=True
)
zon = pd.read_csv(
    fn_zonal,
    index_col=0,
    header=[0,1,2],
    parse_dates=True
)

# --- cell 5 ---
bids = pd.read_csv(REPO_ROOT / 'data' / 'base' / f'{day}' / 'bids.csv', index_col=[0,1])
offers = pd.read_csv(REPO_ROOT / 'data' / 'base' / f'{day}' / 'offers.csv', index_col=[0,1])

# --- cell 6 ---
def process_balancing_data(df):
    df = (
        df
        .stack()
        .unstack(1)
        .dropna()
        .groupby(level=1)
        .agg({'price': 'mean', 'vol': 'sum'})
        .sort_values('price')
    )
    return df


def get_weighted_avg_price(df):
    assert set(df.columns) == {'price', 'vol'}, 'Columns must be price and vol'
    assert not df.empty, 'DataFrame must not be empty'

    return (df['price'] * df['vol']).sum() / df['vol'].sum()

# --- cell 7 ---
backup = pd.read_csv(
    REPO_ROOT / 'data' / 'prerun' / 'balancing_prices' / '2024-week12.csv', index_col=0
)

# --- cell 8 ---
if not bids.empty:
    bid_price = get_weighted_avg_price(process_balancing_data(bids))
else:
    bid_price = backup.loc['bids', 'disp']

if not offers.empty:
    offer_price = get_weighted_avg_price(process_balancing_data(offers))
else:
    offer_price = backup.loc['offers', 'disp']

# --- cell 9 ---
nat_who = pypsa.Network(
    REPO_ROOT / 'results' / f'{day}' / 'network_flex_s_national_solved.nc'
)
nat_bal = pypsa.Network(
    REPO_ROOT / 'results' / f'{day}' / 'network_flex_s_national_solved_redispatch.nc'
)
zon_who = pypsa.Network(
    REPO_ROOT / 'results' / f'{day}' / 'network_flex_s_zonal_solved.nc'
)
zon_bal = pypsa.Network(
    REPO_ROOT / 'results' / f'{day}' / 'network_flex_s_zonal_solved_redispatch.nc'
)

# --- cell 10 ---
zon_who.buses_t.marginal_price.iloc[20].value_counts()

# --- cell 11 ---
import geopandas as gpd

gb = gpd.read_file(REPO_ROOT / 'data' / 'gb_shape.geojson').set_index('name')

gdf = gpd.GeoDataFrame(
        zon_who.buses,
        geometry=gpd.points_from_xy(
            zon_who.buses['x'], zon_who.buses['y']
            )
        ).set_crs('EPSG:4326')
    
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(6, 6))

gb.plot(ax=ax)
gdf.plot(ax=ax, color='r')

plt.show()

mask = gdf.within(gb.loc['GB', 'geometry'])
gb_buses = zon_who.buses.index[mask]

# --- cell 12 ---
zon_who.buses_t.marginal_price.loc[:,gb_buses].iloc[10].value_counts()

# --- cell 13 ---
zon_who.buses

# --- cell 14 ---
roc = pd.read_csv(REPO_ROOT / 'data' / 'prerun' / 'roc_values.csv', index_col=0)

cfd = pd.read_csv(REPO_ROOT / 'data' / 'prerun' / 'cfd_strike_prices.csv', index_col=0)
cfd.columns = pd.to_datetime(cfd.columns)
cfd = cfd.loc[:,:day].iloc[:,-1]

# --- cell 15 ---
def get_unit_revenues(unit, who, bal):
    """
    Calculate revenue timeseries for a generation unit from multiple sources.
    
    Parameters:
    - unit: Generator name/ID
    - who: Wholesale market model network
    - bal: Balancing market model network
    
    Returns:
    - DataFrame with timeseries of revenue components
    """
    # Time step in hours (half-hourly)
    dt = 0.5

    # Initialize dataframe with snapshots as index
    snapshots = who.snapshots
    revenues = pd.DataFrame(index=snapshots, columns=[
        'wholesale', 'roc', 'cfd', 'ftr', 'offer_cost', 'bid_cost'
    ], data=0.0)

    if unit in who.links.index:
        dispatch = who.links_t.p0[unit]
        price0 = who.buses_t.marginal_price[who.links.loc[unit, 'bus0']]
        price1 = who.buses_t.marginal_price[who.links.loc[unit, 'bus1']]

        price_diff = abs(price0 - price1)
        revenues['wholesale'] = dispatch * price_diff * dt

        # revenues['total'] = revenues.sum(axis=1)

        return revenues
    

    if unit in who.storage_units.index:
        # Get storage unit dispatch
        who_dispatch = who.storage_units_t.p_dispatch[unit] - who.storage_units_t.p_store[unit]
        who_prices = who.buses_t.marginal_price[who.storage_units.loc[unit, 'bus']]
        revenues['wholesale'] = who_dispatch * who_prices * dt
        
        # Add ROC revenue if applicable
        if unit in roc.index:
            roc_value = roc.loc[unit, 'roc_value']
            revenues['roc'] = who_dispatch * roc_value * dt
            
        # revenues['total'] = revenues.sum(axis=1)
        return revenues

    # 1. Wholesale market revenue
    who_dispatch = who.generators_t.p[unit] if unit in who.generators_t.p else pd.Series(0, index=snapshots)
    who_prices = who.buses_t.marginal_price[who.generators.loc[unit, 'bus']]
    revenues['wholesale'] = who_dispatch * who_prices * dt

    bal_dispatch = bal.generators_t.p[unit] if unit in bal.generators_t.p else pd.Series(0, index=snapshots)

    # 2. ROC revenue (if applicable)
    if unit in roc.index:
        roc_value = roc.loc[unit, 'roc_value']
        revenues['roc'] = bal_dispatch * roc_value * dt

    # 3. CfD revenue (if applicable)
    if 'cfd' in globals() and unit in cfd.index:
        strike_price = cfd.loc[unit]
        
        # Check for negative price periods (at least 6 hours)
        negative_price_periods = who_prices < 0
        # Create a rolling window of 12 periods (6 hours with half-hourly data)
        rolling_negative = negative_price_periods.rolling(window=12).sum()
        # Identify snapshots where we shouldn't pay topup (preceded by 6+ hours of negative prices)
        no_topup_periods = rolling_negative >= 12
        
        # Calculate CfD top-up only for eligible periods
        topup_rates = strike_price - who_prices
        # Set topup to zero for periods following 6+ hours of negative prices
        topup_rates[no_topup_periods] = 0
        
        revenues['cfd'] = topup_rates * bal_dispatch * dt
    
    # 4. FTR revenue
    if len(who.buses) >= 20:
        # Get the unit's bus
        unit_bus = who.generators.loc[unit, 'bus']
        
        # For each timestep, check if there are exactly two unique prices
        for snapshot in snapshots:
            # Get unique prices across all buses for this snapshot
            unique_prices = who.buses_t.marginal_price.loc[:, gb_buses].loc[snapshot].unique()
            
            high_price = max(unique_prices)
            
            unit_price = who.buses_t.marginal_price.loc[snapshot, unit_bus]
            price_diff = abs(high_price - unit_price)
            
            # If unit is in the lower price zone, it gets FTR revenue
            revenues.loc[snapshot, 'ftr'] = who_dispatch[snapshot] * price_diff * dt
    
    # 5. Balancing revenue (if dispatch differs between models)
    if unit in bal.generators_t.p and unit in who.generators_t.p:
        dispatch_diff = bal.generators_t.p[unit] - who.generators_t.p[unit]
        bal_prices = bal.buses_t.marginal_price[bal.generators.loc[unit, 'bus']]
        
        # Calculate separately for up and down balancing
        up_balancing = dispatch_diff.clip(lower=0)
        down_balancing = dispatch_diff.clip(upper=0)
        
        # Balancing offers (turning up)
        up_revenue = up_balancing * bal_prices * dt
        
        # Apply ROC and CfD adjustments for turning up
        if unit in roc.index:
            roc_value = roc.loc[unit, 'roc_value']
            # For turning up, they would pay their ROC value
            up_revenue -= up_balancing * roc_value * dt
        
        elif unit in cfd.index:
            strike_price = cfd.loc[unit]
            # For turning up, they would pay their topup
            topup_rates = strike_price - who_prices
            up_revenue -= up_balancing * topup_rates * dt
        
        else:
            up_revenue = up_balancing * offer_price * dt

        revenues['offer_cost'] = up_revenue

        # Apply ROC and CfD adjustments for turning down
        if unit in roc.index:
            roc_value = roc.loc[unit, 'roc_value']
            # For turning down, they receive their ROC value per MWh curtailed
            down_revenue = down_balancing.abs() * roc_value * dt  # Negative * negative = positive

        elif unit in cfd.index:
            strike_price = cfd.loc[unit]
            # For turning down, they are forgoing their topup, so they get paid for that
            topup_rates = strike_price - who_prices
            down_revenue = down_balancing.abs() * topup_rates * dt  # Negative * positive = negative
        
        else:
            down_revenue = down_balancing.abs() * offer_price * dt

        revenues['bid_cost'] = down_revenue
    
    # revenues['total'] = revenues.sum(axis=1)

    return revenues

# --- cell 16 ---
all_units = nat_who.generators.index[nat_who.generators.carrier != 'local_market'].union(
    nat_who.storage_units.index.union(
        nat_who.links.index[nat_who.links.carrier == 'interconnector']
    )
)

all_carriers = pd.concat((
    nat_who.generators.loc[nat_who.generators.carrier != 'local_market', 'carrier'],
    nat_who.storage_units['carrier'],
    nat_who.links.loc[nat_who.links.carrier == 'interconnector', 'carrier']
)) 

all_carriers

# --- cell 17 ---
from tqdm import tqdm

all_revenues = list()

for unit in tqdm(all_units):

    n = get_unit_revenues(unit, nat_who, nat_bal)
    z = get_unit_revenues(unit, zon_who, zon_bal)

    # Concatenate the DataFrames with a new 0th index layer
    n.columns = pd.MultiIndex.from_product([[unit], ['national'], n.columns])
    z.columns = pd.MultiIndex.from_product([[unit], ['zonal'], z.columns])

    all_revenues.extend([n, z])

all_revenues = pd.concat(all_revenues, axis=1)

# --- cell 18 ---
lv3_carriers = ['onwind', 'offwind', 'nuclear', 'hydro', 'cascade']

# --- cell 19 ---
idx = pd.IndexSlice

available_shares = pd.Series(np.nan, all_carriers.unique())

for carrier in lv3_carriers:

    print(carrier)

    national_total = all_revenues.sum().loc[idx[all_carriers.loc[all_carriers == carrier].index, 'national']].sum()
    zonal_total = all_revenues.sum().loc[idx[all_carriers.loc[all_carriers == carrier].index, 'zonal']].sum()
    available_shares.loc[carrier] = zonal_total / national_total

# --- cell 20 ---
available_shares

# --- cell 21 ---
hold = all_revenues.copy()

for carrier, share in available_shares.dropna().items():

    print(carrier)

    units = all_carriers.index[all_carriers == carrier]
    ss = hold.loc[:, idx[units, 'national']].T.groupby(level=0).sum().T * share

    ss.columns = pd.MultiIndex.from_product([ss.columns, ['equitable'], ['level3']])

    hold = pd.concat((
        hold, ss
    ), axis=1)


hold = hold.sort_index(axis=1)

# --- cell 22 ---
hold.loc[:,idx['CMSTW-1']].T.groupby(level=0).sum().T.plot()

# --- cell 23 ---
syscosts = pd.read_csv(REPO_ROOT / 'results' / f'{day}' / 'system_cost_summary_flex.csv', index_col=[0,1], parse_dates=[0])
sc = syscosts.loc[syscosts.index.get_level_values(1) != 'balancing_volume']

ss1 = sc['zonal'].unstack()
ss2 = sc['national'].unstack()

ss1 *= 1e3
ss2 *= 1e3

ss1['settlement_period'] = range(1, len(ss1) + 1)
ss2['settlement_period'] = range(1, len(ss2) + 1)

json_result = {
    "dataName": 'System Costs',
    "data": [
        {
            key1: {'zonal': value1, 'national': value2} for (key1, value1), (key2, value2) in zip(row1.items(), row2.items())
        }
        for (_, row1), (_, row2) in zip(ss1.iterrows(), ss2.iterrows())
    ]
}

import json

with open(f'system_costs.json', 'w') as f:
    json.dump(json_result, f, indent=2)

# --- cell 24 ---
import numpy as np
import pandas as pd

# Create a dummy DataFrame with MultiIndex for policy analysis
# This will be used to store and analyze different policy scenarios

# Define the levels for our MultiIndex
policies = ['baseline', 'fixed_price', 'cap_and_floor', 'two_sided_cfd']
technologies = ['wind', 'solar', 'nuclear', 'gas', 'storage']
metrics = ['revenue', 'volatility', 'cost_to_consumer', 'market_distortion']

# Create MultiIndex
index = pd.MultiIndex.from_product([policies, technologies], 
                                   names=['policy', 'technology'])

# Create columns MultiIndex
columns = pd.MultiIndex.from_product([['annual', 'monthly', 'daily'], metrics],
                                     names=['timeframe', 'metric'])

# Create the dummy DataFrame with random values
np.random.seed(42)  # For reproducibility
dummy_data = np.random.rand(len(index), len(columns)) * 100
dummy_df = pd.DataFrame(data=dummy_data, index=index, columns=columns)

# Add some realistic patterns to the data
# Higher volatility for renewables in baseline
volatility_cols = [(t, 'volatility') for t in ['annual', 'monthly', 'daily']]
for t in ['wind', 'solar']:
    dummy_df.loc[('baseline', t), volatility_cols] *= 2
    
# Lower volatility but higher cost for fixed price
for t in technologies:
    dummy_df.loc[('fixed_price', t), [(tf, 'volatility') for tf in ['annual', 'monthly', 'daily']]] *= 0.3
    dummy_df.loc[('fixed_price', t), [(tf, 'cost_to_consumer') for tf in ['annual', 'monthly', 'daily']]] *= 1.5

# Display the first few rows of the dummy DataFrame
dummy_df.head()

# --- cell 25 ---
idx = pd.IndexSlice

policy_mapper = {
    'lev1': idx['annual', ['revenue', 'volatility']],
}

dummy_df.loc[:, policy_mapper['lev1']]

# --- cell 26 ---
print(nat_who.generators.loc[nat_who.generators.carrier == 'nuclear'].index)

# --- cell 27 ---
# get_unit_revenues('GORDW-1, nat_who, nat_bal).head()
get_unit_revenues('SIZB-2', zon_who, zon_bal).head()

# --- cell 28 ---
print(nat_who.generators.loc[nat_who.generators.carrier == 'fossil'].index[:5])
ss = nat_who.generators.loc[nat_who.generators.carrier == 'fossil'].index

random_choice = np.random.choice(ss)
# get_unit_revenues(random_choice, nat_who, nat_bal)

# --- cell 29 ---
# Get offshore wind generators from both networks
offshore_wind_who = nat_who.generators[nat_who.generators.carrier.isin(['onwind', 'offwind'])]
offshore_wind_bal = nat_bal.generators[nat_bal.generators.carrier.isin(['onwind', 'offwind'])]

# Get the dispatch for each generator in both networks
who_dispatch = {}
bal_dispatch = {}
deviations = {}

for gen in offshore_wind_who.index:
    if gen in nat_who.generators_t.p.columns:
        who_dispatch[gen] = nat_who.generators_t.p[gen].sum() * 0.5

for gen in offshore_wind_bal.index:
    if gen in nat_bal.generators_t.p.columns:
        bal_dispatch[gen] = nat_bal.generators_t.p[gen].sum() * 0.5

# Calculate deviations between the two models
for gen in who_dispatch:
    if gen in bal_dispatch:
        deviations[gen] = abs(who_dispatch[gen] - bal_dispatch[gen])

# Get the bus coordinates for each generator
gen_bus_info = {}
for gen in deviations:
    bus_id = nat_bal.generators.loc[gen, 'bus']
    y_coord = nat_bal.buses.loc[bus_id, 'y']
    gen_bus_info[gen] = {
        'bus': bus_id,
        'latitude': y_coord,
        'deviation': deviations[gen]
    }

# Group generators by bus
bus_generators = {}
for gen, info in gen_bus_info.items():
    bus_id = info['bus']
    if bus_id not in bus_generators:
        bus_generators[bus_id] = []
    bus_generators[bus_id].append(gen)

# Sort buses by latitude (descending)
sorted_buses = sorted(bus_generators.keys(), 
                     key=lambda bus: nat_bal.buses.loc[bus, 'y'],
                     reverse=True)


# Print results by bus, sorted by latitude
print("Offshore wind generator dispatch deviations by bus (sorted by latitude):")
print("-" * 100)
print(f"{'Bus':<10} {'Latitude':<12} {'Generator':<15} {'Deviation (MWh)':<15} {'Support Type':<16}")
print("-" * 100)


dispatched_roc_units = []
curtailed_roc_units = []
dispatched_cfd_units = []
curtailed_cfd_units = []

for bus in sorted_buses:
    bus_lat = nat_bal.buses.loc[bus, 'y']
    for i, gen in enumerate(bus_generators[bus]):
        # Determine support type
        if 'roc' in globals() and gen in roc.index:
            support_type = "ROC"
        elif 'cfd' in globals() and gen in cfd.index:
            support_type = "CfD"
        else:
            support_type = "None"
        
        if support_type == "ROC":
            if deviations[gen] == 0:
                dispatched_roc_units.append(gen)
            else:
                curtailed_roc_units.append(gen)
        elif support_type == "CfD":
            if deviations[gen] == 0:
                dispatched_cfd_units.append(gen)
            else:
                curtailed_cfd_units.append(gen)

        if i == 0:
            # Print bus info only for the first generator of each bus
            print(f"{bus:<10} {bus_lat:<12.6f} {gen:<15} {deviations[gen]:<15.2f} {support_type:<16}")
        else:
            # For other generators of the same bus, don't repeat bus info
            print(f"{'':<10} {'':<12} {gen:<15} {deviations[gen]:<15.2f} {support_type:<16}")
    print("-" * 100)

# --- cell 30 ---
# these hold for 21.03.2024
dispatched_roc_unit = 'GORDW-1'
dispatched_cfd_unit = 'WLNYO-4'
curtailed_cfd_unit = 'DOREW-1'
curtailed_roc_unit = 'SGRWO-1'

# --- cell 31 ---
sc.loc[~sc.index.get_level_values(1).isin(['balancing_volume', 'congestion_rent'])].sum()

# --- cell 32 ---
from tqdm import tqdm

roc_rev_nat = pd.Series(0, index=nat_who.snapshots)
roc_rev_zon = pd.Series(0, index=zon_who.snapshots)

for bus in tqdm(sorted_buses):
    bus_lat = nat_bal.buses.loc[bus, 'y']

    for i, gen in enumerate(bus_generators[bus]):

        if nat_who.generators_t.p[gen].sum() != zon_who.generators_t.p[gen].sum():
            continue

        if gen in roc.index:
            roc_rev_nat += get_unit_revenues(gen, nat_who, nat_bal).mul(1e-3).sum(axis=1)
            roc_rev_zon += get_unit_revenues(gen, zon_who, zon_bal).mul(1e-3).sum(axis=1)

# --- cell 33 ---
fig, ax = plt.subplots(1, 1, figsize=(12, 4))

roc_rev_nat.plot(ax=ax, label='National')
roc_rev_zon.plot(ax=ax, label='Zonal')

ax.set_ylabel('ROC Revenue (k£/30min)')
ax.set_xlabel('Time')

ax.legend()
plt.show()

# --- cell 34 ---
color_dict['roc'] = color_dict['roc_payments']
color_dict['cfd'] = color_dict['cfd_payments']
color_dict['ftr'] = color_dict['congestion_rent']
color_dict

# --- cell 35 ---
from tqdm import tqdm
# total_cost = sc.loc[~sc.index.get_level_values(1).isin(['balancing_volume', 'congestion_rent'])].sum()
# gamma = total_cost['zonal'] / total_cost['national']

total_wind_revenue_national = 0
total_wind_revenue_zon = 0

for bus in tqdm(sorted_buses):
    bus_lat = nat_bal.buses.loc[bus, 'y']
    for i, gen in enumerate(bus_generators[bus]):

        rev_nat = get_unit_revenues(gen, nat_who, nat_bal).mul(1e-3).sum(axis=1)
        rev_zon = get_unit_revenues(gen, zon_who, zon_bal).mul(1e-3).sum(axis=1)

        # import sys
        # sys.exit()

        total_wind_revenue_national += rev_nat.sum()
        total_wind_revenue_zon += rev_zon.sum()

gamma = total_wind_revenue_zon / total_wind_revenue_national
print(gamma)

# --- cell 36 ---
from plotting_constants import policy_colors

national_color = policy_colors['national']
no_ftr_zonal_color = policy_colors['no_ftr_zonal'] # policy 1 in the paper
zonal_color = policy_colors['zonal'] # policy 2 in the paper
zonal_with_policy_color = policy_colors['zonal_with_policy'] # policy 3 in the paper

# --- cell 37 ---
def get_handles():
# Create a legend with nice names and consistent styling
    import matplotlib.patches as mpatches

    # Define nice display names for the legend
    nice_names = {
        'wholesale': 'Wholesale Revenue',
        'wholesale selling': 'Wholesale Selling',
        'roc_payments': 'ROC Payments',
        'cfd_payments': 'CFD Payments',
        'congestion_rent': 'Congestion Rent',
        'offer_cost': 'Offer Cost',
        'bid_cost': 'Bid Cost',
        'transmission_congestion_credits': 'Transmission Congestion Credits'
    }

    # Remove redundant entries (keep only unique colors with preferred labels)
    unique_entries = {
        'wholesale': '#F78C6B',
        'roc_payments': '#EF476F',
        'cfd_payments': '#06D6A0',
        'congestion_rent': '#FFD166',
        'offer_cost': '#073B4C',
        'bid_cost': '#118AB2'
    }

    # Create patches for the legend
    legend_handles = []
    for key, color in unique_entries.items():
        patch = mpatches.Patch(
            facecolor=color, 
            alpha=0.6,
            edgecolor=color,
            linewidth=1.5,
            label=nice_names.get(key, key)
        )
        legend_handles.append(patch)


    from matplotlib.lines import Line2D

    for _ in range(3):
        legend_handles.append(Line2D([0], [0], color='w', label=''))

    for label, color in zip(
        ['National Market Revenue', 
        # 'No FTR Zonal Market Revenue'
        'Policy 1 (Simple)'
        ],
        [national_color, no_ftr_zonal_color]
        ):
        legend_handles.append(Line2D([0], [0], color=color, lw=2, label=label))
    
    legend_handles.append(Line2D([0], [0], color='w', lw=2, label=''))

    for label, color, ls in zip(
        # ['Simple FTR Zonal Market Revenue', 'Equitable FTR Zonal Market Revenue'],
        ['Policy 2 (Production-Based FTRs)', 'Policy 3 (Grandfathered Merit Order)'],
        [zonal_color, zonal_with_policy_color],
        ['--', '-']
        ):
        legend_handles.append(Line2D([0], [0], color=color, lw=2, label=label, linestyle=ls))
    
    return legend_handles

# --- cell 38 ---

fig, axs = plt.subplots(3, 4, figsize=(15, 7))

for outer_i, (unit, ax_col) in enumerate(zip([
        dispatched_roc_unit,
        curtailed_roc_unit,
        dispatched_cfd_unit,
        curtailed_cfd_unit
    ],
    axs.T)):

    nat_rev = get_unit_revenues(unit, nat_who, nat_bal).mul(1e-3)
    zon_rev = get_unit_revenues(unit, zon_who, zon_bal).mul(1e-3)

    global_y_min = min(nat_rev.clip(upper=0).sum(axis=1).min(), zon_rev.clip(upper=0).sum(axis=1).min())
    global_y_max = max(nat_rev.clip(lower=0).sum(axis=1).max(), zon_rev.clip(lower=0).sum(axis=1).max())
    global_y_range = [global_y_min * 1.1, global_y_max * 1.1]


    for j, (ax, rev) in enumerate(zip(ax_col, [nat_rev, zon_rev])):
        
        if j == 1:
            ls = '--'
        else:
            ls = '-'

        ax.set_ylim(global_y_range)

        columns_to_plot = rev.columns[:6]
        base = global_y_min

        pos_cols = [col for col in columns_to_plot if (rev[col] >= 0).any()]
        neg_cols = [col for col in columns_to_plot if (rev[col] < 0).any()]

        pos_stack = pd.DataFrame(0, index=rev.index, columns=pos_cols)
        for col in pos_cols:
            pos_stack[col] = rev[col].clip(lower=0)  # Only keep positive values

        # Create cumulative sums for stacking
        pos_cumsum = pos_stack.cumsum(axis=1)

        # Plot each positive column as stacked area
        for i, col in enumerate(pos_cols):

            if rev[col].clip(lower=0).sum() == 0:
                continue

            # Calculate bottom of current stack
            bottom = 0 if i == 0 else pos_cumsum[pos_cols[i-1]]
            top = pos_cumsum[col]

            ax.fill_between(
                range(0, 48),
                bottom,
                top,
                alpha=0.3,
                label=col,
                color=color_dict[col]
            )
            ax.plot(
                range(0, 48),
                top,
                alpha=0.7,
                color=color_dict[col]
            )

        # Plot negative values stacked on each other
        neg_stack = pd.DataFrame(0, index=rev.index, columns=neg_cols)
        for col in neg_cols:
            neg_stack[col] = rev[col].clip(upper=0)  # Only keep negative values

        # Create cumulative sums for stacking
        neg_cumsum = neg_stack.cumsum(axis=1)

        # Plot each negative column as stacked area
        for i, col in enumerate(neg_cols):

            if rev[col].clip(upper=0).sum() == 0:
                continue

            # Calculate bottom of current stack
            bottom = 0 if i == 0 else neg_cumsum[neg_cols[i-1]]
            top = neg_cumsum[col]

            ax.fill_between(
                range(0, 48),
                bottom,
                top,
                alpha=0.3,
                label=col if col not in pos_cols else f"{col} (neg)",
                color=color_dict[col]
            )
            ax.plot(
                range(0, 48),
                top,
                alpha=0.7,
                color=color_dict[col]
            )
        # Plot total line

        if j == 0:
            total_color = national_color
        else:
            total_color = zonal_color
        
        # print(rev.head())
        if j == 1:
            ax.plot(range(0, 48), rev[
                ['wholesale', 'cfd', 'roc', 'offer_cost', 'bid_cost']].sum(axis=1),
                color=no_ftr_zonal_color, linewidth=2)

        if not (ls == '--' and outer_i >= 2):
            ax.plot(range(0, 48), rev.sum(axis=1), color=total_color, linewidth=2, label='Total', linestyle=ls)

        ax.set_xlim(0, 47)

        # Add horizontal line at y=0
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)

    for ax in ax_col[:2]:
        ylim = ax.get_ylim()[0]  # Get the lower y limit
        ax.text(
            47, ylim, f'1 unit: {unit}',
            ha='right',
            va='bottom',
            fontsize=10,
            weight='bold'
        )

    ax_col[1].plot(range(0, 48), nat_rev.sum(axis=1) * gamma, color=zonal_with_policy_color)
    # ax_col[1].fill_between(range(0, 48), 0, nat_rev['total'] * gamma, color=zonal_with_policy_color, alpha=0.1)


for group, ax in zip([
    dispatched_roc_units, curtailed_roc_units, dispatched_cfd_units, curtailed_cfd_units
], axs[2]):

    total_national = pd.Series(0, index=nat_who.snapshots)
    total_zonal = pd.Series(0, index=zon_who.snapshots)
    total_no_ftr_zon = pd.Series(0, index=zon_who.snapshots)
    total_zonal_with_policy = pd.Series(0, index=zon_who.snapshots)

    for unit in group:
        nat_rev = get_unit_revenues(unit, nat_who, nat_bal).mul(1e-3)
        zon_rev = get_unit_revenues(unit, zon_who, zon_bal).mul(1e-3)

        zon_with_policy = nat_rev.mul(gamma).sum(axis=1)

        nat_perc = (nat_rev.sum(axis=1) / nat_rev.sum(axis=1)).replace(np.nan, 0)
        zon_perc = (zon_rev.sum(axis=1) / nat_rev.sum(axis=1)).replace(np.nan, 0)
        no_ftr_zon_perc = (

            zon_rev[['wholesale', 'cfd', 'roc', 'offer_cost', 'bid_cost']].sum(axis=1) /
            nat_rev.sum(axis=1)).replace(np.nan, 0)

        zon_with_policy_perc = (zon_rev.sum(axis=1) * gamma / nat_rev.sum(axis=1)).replace(np.nan, 0)

        total_national = total_national.add(nat_rev.sum(axis=1))
        total_zonal = total_zonal.add(zon_rev.sum(axis=1))

        total_no_ftr_zon = total_no_ftr_zon.add(zon_rev[['wholesale', 'cfd', 'roc', 'offer_cost', 'bid_cost']].sum(axis=1))
        total_zonal_with_policy = total_zonal_with_policy.add(zon_with_policy)

    total_zonal /= total_national
    total_zonal_with_policy /= total_national
    total_no_ftr_zon /= total_national
    total_national /= total_national

    ax.plot(range(0, 48), total_national * 100, color=national_color, lw=2)
    # ax.plot(range(0, 48), total_national * 100, color='k', lw=0.5)
    ax.plot(range(0, 48), total_no_ftr_zon * 100, color=no_ftr_zonal_color, lw=2, linestyle='-')

    if outer_i < 2:
        ax.plot(range(0, 48), total_zonal.clip(upper=1) * 100, color=zonal_color, lw=2, linestyle='--')
    # ax.plot(range(0, 48), total_zonal * 100, color='k', lw=0.5)
    ax.plot(range(0, 48), total_zonal_with_policy * 100, color=zonal_with_policy_color, lw=2)
    # ax.plot(range(0, 48), total_zonal_with_policy * 100, color='k', lw=0.5)

    ax.set_ylim(-10, 110)


for ax in axs.flatten():
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

for ax, title in zip(axs[0], ['Dispatched ROC Unit', 'Curtailed ROC Unit', 'Dispatched CfD Unit', 'Curtailed CfD Unit']):
    ax.set_title(title)

for ax, prefix in zip(axs[:2,0], ['National Market', 'Zonal Market', 'Zonal Market']):
    ax.set_ylabel(f'{prefix}\nRevenue (k£/30min)')


for ax in axs[:2,:].flatten():
    ax.set_xticks([])

axs[0,0].text(
    4,
    11,
    f'{day}',
    ha='right',
    va='top',
    fontsize=11,
    weight='bold'
)

# Set x-tick labels for the bottom row of axes to show hours
for ax in axs[2]:
    # Create 5 evenly spaced time points
    tick_positions = [0, 12, 24, 36, 47]
    tick_labels = ['00:00', '06:00', '12:00', '18:00', '00:00']
    
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.tick_params(axis='x', rotation=0)
    ax.set_xlim(0, 47)
    ax.set_xlabel('Time')


axs[2,0].set_ylabel('Percentage of National\nMarket Revenue (%)')

for ax, group in zip(axs[2], [dispatched_roc_units, curtailed_roc_units, dispatched_cfd_units, curtailed_cfd_units]):
    ax.text(
        1, -9, f'Average across {len(group)} units',
        ha='left',
        va='bottom',
        fontsize=10,
        weight='bold'
    )

legend_handles = get_handles()

axs[2,1].legend(
    handles=legend_handles,
    bbox_to_anchor=(2.7, -0.3),
    ncol=5,
    frameon=False
    )

plt.savefig('revenue_stabilisation.pdf', bbox_inches='tight')
plt.show()
