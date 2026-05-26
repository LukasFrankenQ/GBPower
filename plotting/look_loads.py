"""
Auto-converted from notebooks/look_loads.ipynb.
Edits should target this file directly; the .ipynb is the source-of-truth for exploratory work only.
"""

from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent

# --- cell 0 ---
import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import yaml

# --- cell 1 ---
with open(REPO_ROOT / 'data' / 'interconnection_helpers.yaml', 'r') as f:
    interconnection_capacities = yaml.safe_load(f)

# --- cell 2 ---
carrier = 'wind'

bmus = pd.read_csv(REPO_ROOT / 'data' / 'prerun' / 'prepared_bmus.csv', index_col=0)
bmus

if carrier != 'interconnector':
    wind = bmus.loc[bmus.carrier.str.contains(carrier)].index
else:
    wind = pd.Index(list(interconnection_capacities['interconnection_mapper'].keys()))

# --- cell 3 ---
df = pd.read_csv(REPO_ROOT / 'summaries' / 'total_unit_dispatch_flex.csv', index_col=[0,1,2])

df.columns = pd.to_datetime(df.columns)
idx = pd.IndexSlice
df_redispatch = df.loc[idx[:, 'national', 'redispatch'], :].sum(axis=1)
df_wholesale = df.loc[idx[:, 'national', 'wholesale'], :].sum(axis=1)

df_wholesale.index = df_wholesale.index.get_level_values(0)
df_redispatch.index = df_redispatch.index.get_level_values(0)

total_wholesale = df_wholesale.loc[df_wholesale.index.intersection(wind)].sum()
total_redispatch = df_redispatch.loc[df_redispatch.index.intersection(wind)].sum()

diff_perc = (total_redispatch - total_wholesale) / total_wholesale
print(diff_perc)

diff_total = total_redispatch - total_wholesale
print(diff_total * 1e-6)

# --- cell 4 ---
df = pd.read_csv(REPO_ROOT / 'summaries' / 'total_unit_dispatch_flex.csv', index_col=[0,1,2])
df.columns = pd.to_datetime(df.columns)
df = df.loc[:, '2022':]

idx = pd.IndexSlice
df = df.loc[idx[:, :, 'redispatch'], :].sum(axis=1)
# df = df.loc[idx[:, :, 'wholesale'], :].sum(axis=1)

df

# --- cell 5 ---
diff = df.loc[idx[:, 'national', :]] - df.loc[idx[:, 'zonal', :]]
diff.index = diff.index.get_level_values(0)

diff = diff.loc[diff.index.intersection(wind)].sort_values() * 1e-3

# --- cell 6 ---
diff.sort_values()

# --- cell 7 ---
diff.sum()

# --- cell 8 ---

df.loc[idx[:, 'zonal', :]].sum() / df.loc[idx[:, 'national', :]].sum()

# --- cell 9 ---
hold = df.loc[idx[:, 'national', :]].mul(1e-3)
hold.index = hold.index.get_level_values(0)

fig, ax = plt.subplots(figsize=(10, 5))
diff.abs().div(hold.loc[diff.index]).sort_values(ascending=False).head(50).mul(100).plot.bar(ax=ax)

ax.set_ylabel('Additional Dispatch under Zonal 2022, 2023, 2024 [%]')
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.set_axisbelow(True)

# plt.savefig('dispatch_increase_perc.pdf', bbox_inches='tight')

# --- cell 10 ---
print('Total additional zonal dispatch in GWh')
abs(diff.sum())

# --- cell 11 ---
print('Total additional zonal dispatch in GWh per windfarms sorted in GWh')
fig, ax = plt.subplots(figsize=(10, 5))
# diff.groupby(diff.index.str[:5]).sum().sort_values().head(15).abs().plot.bar(ax=ax)
diff.sort_values().head(50).abs().mul(1/3).plot.bar(ax=ax)
# ax.set_ylabel('Additional zonal dispatch in GWh')
ax.set_ylabel('Additional Dispatch under Zonal [GWh/a]')

ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.set_axisbelow(True)

# plt.savefig('dispatch_increase_gwh.pdf', bbox_inches='tight')

# --- cell 12 ---
print('Grouped to physical plants in GWh: SGRWO is Seagreen, MOWEO is Moray')
diff.groupby(diff.index.str[:5]).sum().sort_values().abs().head(20)

# --- cell 13 ---
diff.groupby(diff.index.str[:5]).sum().sort_values().mul(80 * 1e3).abs().mul(1e-6).head(20)
