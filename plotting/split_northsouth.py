"""
Auto-converted from notebooks/split_northsouth.ipynb.
Edits should target this file directly; the .ipynb is the source-of-truth for exploratory work only.
"""

from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent

# --- cell 0 ---
import numpy as np
import pandas as pd
from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt

# --- cell 1 ---
bmus = pd.read_csv(REPO_ROOT / 'data' / 'prerun' / 'prepared_bmus.csv', index_col=0)

import yaml

with open(REPO_ROOT / 'data' / 'interconnection_helpers.yaml', 'r') as f:
    inter_bmu = yaml.safe_load(f)['interconnection_mapper']

intercon_lats = []

for i, (key, item) in enumerate(inter_bmu.items()):

    if key == 'Nemo':
        intercon_lats.append(0)
        continue

    lat = bmus.loc[bmus.index.str.startswith(item[0]+'-'), 'lat'].dropna().astype(float).mean()
    intercon_lats.append(lat)

intercon_lats = pd.Series(intercon_lats, index=inter_bmu.keys())

intercon_lats.loc['EastWest'] = 53.23
intercon_lats.loc['Moyle'] = 55.07
intercon_lats.loc['Nemo'] = 51.297

intercon_lats = intercon_lats.to_frame().rename(columns={0: 'lat'})

# --- cell 2 ---
bmus = pd.read_csv(
    REPO_ROOT / 'data' / 'prerun' / 'prepared_bmus.csv', index_col=0
)
bmus = bmus.loc[bmus['bus'] != 'distributed']

bmus.loc[:,'lat'] = bmus['lat'].astype(float)
bmus.loc[:,'lon'] = bmus['lon'].astype(float)

bmus = bmus[['lon', 'lat']]

# --- cell 3 ---
bmus

# --- cell 4 ---
z = REPO_ROOT / 'summaries' / 'total_unit_revenues_flex_zonal.csv'
n = REPO_ROOT / 'summaries' / 'total_unit_revenues_flex_national.csv'

z = pd.read_csv(z, parse_dates=True, index_col=[0,1])
n = pd.read_csv(n, parse_dates=True, index_col=[0,1])

n_total = n.replace(np.nan, 0).groupby(level=0).sum().sum(axis=1)
z_total = z.replace(np.nan, 0).groupby(level=0).sum().sum(axis=1)

n_total = n_total[n_total > 0]
zonal_total = z_total.loc[n_total.index]

diff = (zonal_total - n_total).clip(lower=-1, upper=1)

# diff = pd.concat([diff.to_frame(), bmus.loc[diff.index, ['lon', 'lat']]], axis=1)

# --- cell 5 ---
bmus

# --- cell 6 ---
intersec = bmus.index.intersection(diff.index)

gseries = gpd.GeoDataFrame(
    diff.loc[intersec],
    geometry=gpd.points_from_xy(
        bmus.loc[intersec, 'lon'],
        bmus.loc[intersec, 'lat']
        )
    ).rename(columns={0: 'diff'})

# --- cell 7 ---
gseries

# --- cell 8 ---
regions = gpd.read_file(
    REPO_ROOT / 'data' / 'gb_shape.geojson'
).set_index('name')

# --- cell 9 ---
gseries

# --- cell 10 ---
def classify_north_south(lon, lat):
    """Splits GB into north and south, where north represents regions 
    with diminished wholesale market prices"""

    lon = float(lon)
    lat = float(lat)

    m = 0.55
    b = 56.4

    if lat > m * lon + b:
        return 'north'
    else:
        return 'south'

# --- cell 11 ---
gseries['region']

# --- cell 12 ---
bmus['region'] = bmus.apply(lambda x: classify_north_south(x['lon'], x['lat']), axis=1)
gdf = gpd.GeoDataFrame(bmus, geometry=gpd.points_from_xy(bmus['lon'], bmus['lat']))

# --- cell 13 ---
fig, ax = plt.subplots(figsize=(6, 6))

# gseries.plot(ax=ax, column='diff', cmap='coolwarm')
gdf.plot(ax=ax, column='region', cmap='coolwarm', edgecolor='k', alpha=0.8)
regions.plot(ax=ax, facecolor='grey', zorder=0, alpha=0.8, edgecolor='k')


tangent_x = np.linspace(-6, 0, 2)
tangent_y = 0.55 * tangent_x + 56.4

ax.plot(tangent_x, tangent_y, 'k--', label='Tangent')

ax.set_axis_off()

plt.savefig('north_south_split.pdf')
plt.show()

# --- cell 14 ---
import os
import pandas as pd

for day in pd.date_range('2022-01-01', '2024-12-31', freq='d').strftime('%Y-%m-%d'):
    print(day)

    break
