# SPDX-FileCopyrightText: : 2024 Lukas Franken
#
# SPDX-License-Identifier: MIT

"""
Calculate merit order slope factors based on historic physical notifications data.
These factors represent how much generation capacity is available at different price points.
The factors are used to estimate the cost of generation at different levels of net load.
"""

import logging

import sys
import yaml
import pypsa
import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy import stats
from pathlib import Path

sys.path.append(str(Path.cwd() / 'scripts'))
from _helpers import configure_logging

logger = logging.getLogger(__name__)

if __name__ == "__main__":

    configure_logging(snakemake)

    bmu_locations = pd.read_csv(snakemake.input.bmu_locations)
    n = pypsa.Network(snakemake.input.network)

    with open(snakemake.input.interconnection_helpers) as f:
        interconnectors = yaml.safe_load(f)
    interconnection_mapper = interconnectors['interconnection_mapper']

    logger.info('Estimating merit order slope factors.')

    pns = []
    prices = []
    nemos = []

    for d in tqdm(snakemake.input):

        if (
            'bmus_prepared.csv' in d or
            'helper_network.nc' in d or
            'interconnection_helpers.yaml' in d
            ):
            continue

        try:
            pns.append(
                pd.read_csv(
                    Path(d) /
                    'physical_notifications.csv',
                    index_col=0,
                    parse_dates=True
                )
            )
            price = pd.read_csv(Path(d) / 'day_ahead_prices.csv', index_col=0, parse_dates=True)
            nemo = pd.read_csv(Path(d) / 'nemo_powerflow.csv', index_col=0, parse_dates=True)

            prices.append(price)
            nemos.append(nemo)

        except FileNotFoundError:
            continue

    price_df = pd.concat(prices)
    pns_df = pd.concat(pns)# .clip(lower=0).sum(axis=1)
    nemo_df = pd.concat(nemos)

    real_int_flow = pd.DataFrame(index=pns_df.index)

    for name, bmu_names in interconnection_mapper.items():
        if name == 'Nemo':
            real_int_flow[name] = nemo_df.iloc[:,0]

        else:
            real_int_flow[name] = (
                pns_df[pns_df.columns[pns_df.columns.str.contains('|'.join(bmu_names))]]
                .sum(axis=1)
            )

    export = real_int_flow.clip(upper=0).sum(axis=1).mul(-1)

    storages = n.storage_units.index.intersection(pns_df.columns)
    charging = pns_df[storages].clip(upper=0).sum(axis=1).mul(-1)

    assert (charging >= 0).all(), 'Charging should be negative.'

    net_load = pns_df.clip(lower=0).sum(axis=1) - export - charging

    ms = pd.date_range('2022-01-01', '2026-01-01', freq='ME').strftime('%Y-%m')

    gradients = []
    errors = []

    for m in ms:
        ssload = net_load.loc[m]
        ssprice = price_df.loc[m]

        intersect = ssload.index.intersection(ssprice.index)
        ssload = ssload.loc[intersect]
        ssprice = ssprice.loc[intersect]

        assert not ssload.empty, 'No data for month {m}'
        assert not ssprice.empty, 'No data for month {m}'
    
        ssload = ssload.values.reshape(-1)
        ssprice = ssprice.values.reshape(-1)
    
        mask = ~np.isnan(ssload) & ~np.isnan(ssprice)
        ssload_clean = ssload[mask]
        ssprice_clean = ssprice[mask]

        slope, intercept, r_value, p_value, std_err = stats.linregress(ssload_clean, ssprice_clean)

        gradients.append(slope)
        errors.append(std_err * 1.96)

    result = pd.DataFrame(
        {
            'gradient': gradients,
            'error': errors
        },
        index=ms
    )

    assert result.notna().all().all(), 'Some values are NaN.'

    baseline_gradient = result.loc[result.index.str.contains('2025'), 'gradient'].mean()

    result['factor'] = result['gradient'] / baseline_gradient

    # 2019 gas price was 1.5 times higher than 2025 (https://www.ons.gov.uk/economy/inflationandpriceindices/timeseries/d7du/mm23 — re-check ratio against 2025 average when regenerating)
    result['factor_relative_to_2019'] = result['factor'] * 1.5

    logger.info(f'Saving merit order slope factors to {snakemake.output[0]}:')
    print(result.head())

    result.to_csv(snakemake.output[0])
