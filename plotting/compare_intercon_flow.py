"""
Auto-converted from notebooks/compare_intercon_flow.ipynb.
Edits should target this file directly; the .ipynb is the source-of-truth for exploratory work only.
"""

from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent

# --- cell 0 ---
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import pypsa

# --- cell 1 ---
ics = yaml.safe_load(
    open(
        REPO_ROOT / 'data' / 'interconnection_helpers.yaml'
        )
    )['interconnection_mapper']

# --- cell 2 ---
years = range(2022, 2025)

days = [
    '{}-03-01',
    '{}-03-02',
    '{}-03-03',
    '{}-03-04',
    '{}-03-05',
    '{}-03-06',
    '{}-03-07',
    '{}-03-08',
    '{}-03-09',
    '{}-03-10',
]

# --- cell 3 ---
bmus = pd.read_csv(
    REPO_ROOT / 'data' / 'prerun' / 'prepared_bmus.csv',
    index_col=0
    )


bmus

# --- cell 4 ---
def get_intercon_flows(df):
    df = df.copy()

    flows = pd.DataFrame(index=df.index)

    for ic, shorthands in ics.items():
        if ic == 'Nemo':
            continue

        matching_cols = [col for col in df.columns if any(col.startswith(prefix) for prefix in shorthands)]
        flows[ic] = df[matching_cols].sum(axis=1)

    return flows

# --- cell 5 ---

pns = {}
nemos = {}

for year in years:
    pns[year] = list()
    nemos[year] = list()

    model_flow = list()

    for day in days:

        ss_pns = pd.read_csv(f'../data/base/{day.format(year)}/physical_notifications.csv', index_col=0, parse_dates=True)
        nemo_ss = pd.read_csv(f'../data/base/{day.format(year)}/nemo_powerflow.csv', index_col=0, parse_dates=True)

        pns[year].append(ss_pns)
        nemos[year].append(nemo_ss)

        n = pypsa.Network(
            REPO_ROOT / 'results' / day.format(year) / 'network_flex_s_national_solved.nc'
        )

        model_flow.append(n.links_t.p0.loc[:, n.links.index[n.links.carrier == 'interconnector']].sum(axis=1))


    pns[year] = pd.concat(pns[year])
    nemos[year] = pd.concat(nemos[year])

    model_flow = pd.concat(model_flow)
    
    flows = get_intercon_flows(pns[year])
    flows['nemo '] = nemos[year].iloc[:,0]

    flows = flows.mul(1e-3).sum(axis=1)
    model_flow = model_flow.mul(1e-3)
    
    # max absolute derivative for both
    max_abs_derivative = model_flow.diff().abs().max()
    max_abs_derivative_flows = flows.diff().abs().max()
    print(f'max absolute derivative model: {max_abs_derivative}')
    print(f'max absolute derivative flows: {max_abs_derivative_flows}')

    fig, ax = plt.subplots(figsize=(8, 3))

    flows.plot(ax=ax, label='Real Interconnector Flow')
    model_flow.plot(ax=ax, label='Modeled Interconnector Flow')

    # ax.set_title(f'week of March {year}, total interconnector flow')
    ax.set_ylabel('Flow [GW]')

    ax.legend(title=f'{year}')

    plt.savefig(f'prefix_{year}_intercon_flow.pdf', bbox_inches='tight')
    plt.show()

# --- cell 6 ---
n = pypsa.Network(
    REPO_ROOT / 'results' / '2022-03-01' / 'network_flex_s_national_solved.nc'
)

n.links_t.p0.loc[:, n.links.index[n.links.carrier == 'interconnector']].sum(axis=1).plot()

# --- cell 7 ---
n.generators.loc[n.generators.carrier == 'nuclear'].marginal_cost
