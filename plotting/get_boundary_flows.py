"""
Auto-converted from notebooks/get_boundary_flows.ipynb.
Edits should target this file directly; the .ipynb is the source-of-truth for exploratory work only.
"""

from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent

# --- cell 0 ---
import re
import sys
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from urllib import parse
from pytz.exceptions import NonExistentTimeError

sys.path.append(str(REPO_ROOT / "scripts"))

from build_base import (
    build_sp_register,
    dst_start_dates,
    dst_end_dates,
)

year_file = REPO_ROOT / "data" / "year-ahead-constraint-limits.csv"
two_year_file = REPO_ROOT / "data" / "24-months-ahead-constraint-limit_060924.csv"

ds = []
from tqdm import tqdm

year = '2024'

for day in tqdm(pd.date_range(f'{year}-01-01', f'{year}-10-08')):
    day = day.strftime("%Y-%m-%d")

    sp_register = build_sp_register(day)
    date_range = sp_register.index

    df = get_boundary_flow_day(date_range)
    ds.append(df)

pd.concat(ds).to_csv(f'constraint_limits_{year}.csv')

# --- cell 1 ---
nice = pd.read_csv('constraint_limits_2023.csv', index_col=0, parse_dates=True)

# --- cell 2 ---
nice.interpolate().plot()

# --- cell 3 ---
current = pd.read_csv(REPO_ROOT / 'data' / 'flow_constraints_2023.csv', index_col=0, parse_dates=True)
current.plot()

# --- cell 4 ---
nice.head()

# --- cell 5 ---
current.head()

# --- cell 6 ---
fixed = nice.copy()
fixed[fixed.isna()] = current

# --- cell 7 ---
fixed.interpolate().to_csv('fixed_2023.csv')

# --- cell 8 ---
from io import StringIO
from tqdm import tqdm

filler = '060924'
template = 'https://api.neso.energy/dataset/d515b4a9-60a1-489c-a126-004efc04f121/resource/3c359e33-3dac-4bdd-87d1-efbf4cbc2f07/download/24-months-ahead-constraint-limit_{}.csv'

response = requests.get(template.format(filler))
df = pd.read_csv(StringIO(response.text))
print(df.head())

for date in tqdm(pd.date_range('2023-10', '2025', freq='d')):
    url = template.format(date.strftime('%y%m%d'))
    response = requests.get(url)
    print(date, response.status_code)
    df = pd.read_csv(StringIO(response.text))
    print(url)
    print(df.head())

    break
    if response.status_code != 200:
        print(date)
        break



response = requests.get(url.format(filler))
print(response.status_code)

df

# --- cell 9 ---
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(15, 5))

a.drop(columns=['YEAR', 'Week No']).sort_index().plot(ax=ax)
ax.set_ylim(0, 15000)

plt.show()

# --- cell 10 ---
import sys
import pandas as pd
from pathlib import Path

sys.path.append(str(REPO_ROOT / "scripts"))

from build_flow_constraints import get_boundary_flow_day
from _constants import build_sp_register

constraints_base = pd.read_csv(
    REPO_ROOT / "data" / "flow_constraints_2023.csv",
    index_col=0,
    parse_dates=True
)

day = '2024-10-09'
date_register = build_sp_register(day)
date_range = date_register.index

# --- cell 11 ---
# constraints_base.loc[date_range]
df = get_boundary_flow_day(date_range)
df.plot()

# --- cell 12 ---
# method does not work if at any timesteps data is missing for all boundaries
assert not df.isna().all(axis=1).any()

df = df.fillna(
    pd.DataFrame(
            df.mean(axis=1).values[:, None] * constraints_base.mean().values / constraints_base.mean().mean(),
            index=df.index,
            columns=df.columns
        )
    )

df.plot()

# --- cell 13 ---
constraints_base.plot()

# --- cell 14 ---
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

# --- cell 15 ---
path = REPO_ROOT / 'data'
fti = gpd.read_file(path / 'fti_zones.geojson')

# --- cell 16 ---
fti.plot()

# --- cell 17 ---
fti['geometry'] = fti['geometry'].simplify(0.01)

# --- cell 18 ---
fti.plot()

# --- cell 19 ---
fti.to_file('fti_zones_simplified.geojson', driver='GeoJSON')
