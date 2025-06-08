import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from shapely.geometry import Point, Polygon


all_dates = pd.date_range('2022-01-01', '2024-12-31', freq='D').strftime('%Y-%m-%d')


def create_squares_date_range(index):

    dates = pd.date_range(start=index[0], end=index[-1], freq='D')
    
    dates_df = pd.DataFrame({'date': dates})
    dates_df['year'] = dates_df['date'].dt.year
    dates_df['month'] = dates_df['date'].dt.month
    dates_df['day'] = dates_df['date'].dt.day
    
    dates_df['year_month'] = dates_df['date'].dt.to_period('M')
    
    months = dates_df['year_month'].drop_duplicates().sort_values().tolist()
    month_indices = {month: idx for idx, month in enumerate(months)}
    
    polygons = []

    for _, row in dates_df.iterrows():

        month_idx = month_indices[row['year_month']]
        day_idx = row['day'] - 1  # Days start from 1, so subtract 1 for zero-based index
        
        x0 = day_idx      # Column (x-coordinate)
        y0 = month_idx    # Row (y-coordinate)
        
        square = Polygon([
            (x0, y0),
            (x0 + 1, y0),
            (x0 + 1, y0 + 1),
            (x0, y0 + 1),
            (x0, y0)
        ])
        polygons.append(square)
    
    return gpd.GeoSeries(
        polygons,
        index=index
    )

print('visualise data progress')
results_tracker = gpd.GeoDataFrame(
    False,
    index=all_dates,
    columns=['summary_flex', 'summary_static', 'network_flex', 'network_static', 'solved', 'base'],
    geometry=create_squares_date_range(all_dates)
)

file_mapper = {
    'summary_flex': str(Path.cwd() / 'results' / '{}' / 'system_cost_summary_flex.csv'),
    'summary_static': str(Path.cwd() / 'results' / '{}' / 'system_cost_summary_static.csv'),
    'network_flex': str(Path.cwd() / 'results' / '{}' / 'network_flex_s_nodal.nc'),
    'network_static': str(Path.cwd() / 'results' / '{}' / 'network_static_s_nodal.nc'),
    'solved': str(Path.cwd() / 'results' / '{}' / 'network_flex_s_national_solved_redispatch.nc'),
    'base': str(Path.cwd() / 'data' / 'base' / '{}' / 'offers.csv'),
}

for key, fn in file_mapper.items():
    results_tracker[key] = (
        results_tracker
        .index
        .map(
            lambda x: os.path.isfile(fn.format(x))
            )
        .astype(float)
    )

fig, axs = plt.subplots(3, 2, figsize=(10, 12))

kwargs = dict(edgecolor='black', linewidth=0.5)

for col, ax in zip(list(file_mapper), axs.flatten()):
    results_tracker.plot(column=col, ax=ax, cmap='RdYlGn', **kwargs)
    ax.set_title(col)

plt.savefig('overview_squares.png', dpi=400)
plt.show()


for mode in ['static', 'flex']:

    print('=======================================================================================')

    total_summary = []
    marginal_prices = []

    revenues = {
        'national': [],
        'zonal': [],
        'nodal': [],
        }

    dispatch = {
        'national': [],
        'zonal': [],
        'nodal': [],
        }

    # unit_revenues = {
    #     'national': [],
    #     'zonal': [],
    #     'nodal': [],
    #     }
    unit_revenues = []
    unit_dispatch = []
    intercon_dispatch = []

    path = Path.cwd() / 'results' / '{}'
    f_path = Path.cwd() / 'frontend' / '{}'

    print('Gathering summaries.')
    for day in tqdm(all_dates):

        try:
            total_summary.append(pd.read_csv(str(path / 'system_cost_summary_{}.csv').format(day, mode), index_col=[0,1]))
            marginal_prices.append(pd.read_csv(str(path / 'marginal_prices_{}.csv').format(day, mode), header=[0,1], index_col=0))

        except FileNotFoundError:
            total_summary.append(pd.DataFrame(columns=['nodal', 'zonal', 'national']))

        try:
            for l in ['zonal', 'nodal', 'national']:
                revenues[l].append(pd.read_csv(str(path / 'bmu_revenues_{}_{}.csv').format(day, mode, l), index_col=0, header=[0,1,2]))
        except:
            pass

        try:
            for l in ['zonal', 'nodal', 'national']:
                dispatch[l].append(pd.read_csv(str(path / 'bmu_dispatch_{}_{}.csv').format(day, mode, l), index_col=0, header=[0,1,2]))
        except:
            pass

        try:
            # for l in ['zonal', 'nodal', 'national']:
                # ss = ['wholesale_revenue', 'cfd_revenue', 'roc_revenue']
                # unit_revenues[l].append(pd.read_csv(str(path / 'bmu_revenues_detailed_{}_{}.csv').format(day, mode, l), index_col=0)[ss].stack().rename(day))

            unit_revenues.append(pd.read_csv(str(f_path / 'revenues_{}.csv').format(day, mode), index_col=0, parse_dates=True, header=[0,1,2]).sum().rename(day))
            unit_dispatch.append(pd.read_csv(str(f_path / 'dispatch_{}.csv').format(day, mode), index_col=0, parse_dates=True, header=[0,1,2]).sum().rename(day))
            intercon_dispatch.append(pd.read_csv(str(f_path / 'dispatch_flex_{}_intercon.csv').format(day, mode), index_col=0, parse_dates=True, header=[0,1]))
        except:
            pass

    for l in ['zonal', 'nodal', 'national']:
        pd.concat(revenues[l]).to_csv('summaries/total_summary_revenues_{}_{}.csv'.format(mode, l))
        pd.concat(dispatch[l]).to_csv('summaries/total_summary_dispatch_{}_{}.csv'.format(mode, l))
        # pd.concat(unit_revenues[l], axis=1).to_csv('summaries/total_unit_revenues_{}_{}.csv'.format(mode, l))

    pd.concat(unit_revenues, axis=1).to_csv('summaries/total_unit_revenues_{}.csv'.format(mode))
    pd.concat(unit_dispatch, axis=1).to_csv('summaries/total_unit_dispatch_{}.csv'.format(mode))
    pd.concat(intercon_dispatch).to_csv('summaries/total_intercon_dispatch_{}.csv'.format(mode))

    total_summary = pd.concat(total_summary)
    total_summary.index = pd.MultiIndex.from_tuples(total_summary.index)

    pd.concat(marginal_prices).to_csv('summaries/marginal_prices_summary_{}.csv'.format(mode))

    print(total_summary)
    num_quants = len(total_summary.index.get_level_values(1).unique())

    share_available = np.around(100 * len(total_summary) / (48 * num_quants * len(all_dates)), decimals=2)

    print(f'total share of {mode} available data: {share_available}%')

    print(f'total {mode} costs £m:')
    print(total_summary.sum().astype(int))

    total_summary.to_csv('summaries/total_summary_{}.csv'.format(mode))

