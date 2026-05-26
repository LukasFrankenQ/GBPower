# SPDX-FileCopyrightText: : 2024 Lukas Franken
#
# SPDX-License-Identifier: MIT

import logging

logger = logging.getLogger(__name__)

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path.cwd() / 'scripts'))
from _helpers import configure_logging


def process_cfd_register(file_path, mapping):
    # LCCC renamed the sheet in 2025 (Register -> "Sheet 1") and switched to snake_case columns.
    last_err = None
    for sheet in ('Register', 'Sheet 1'):
        try:
            df = pd.read_excel(file_path, sheet_name=sheet)
            break
        except Exception as e:
            last_err = e
    else:
        print(f"Failed to read any known sheet in {file_path}: {last_err}")
        return None, None

    df = df.rename(columns={
        # legacy schema (pre-2025-04)
        'Current strike price (field_cfd_current_strikeprice)': 'strike_price',
        'Unique Identifier (field_cfd_unique_id)': 'cfd_Id',
        # 2025+ schema
        'current_strike_price': 'strike_price',
        'contract_id': 'cfd_Id',
    })

    df.set_index('cfd_Id', inplace=True)

    current_mapping = pd.DataFrame(mapping.copy())

    # df may have duplicate cfd_Id rows in some registers; keep the first.
    strike_lookup = df['strike_price'][~df.index.duplicated(keep='first')]

    # Missing IDs (e.g. AR4 contracts in pre-AR4 registers) -> NaN, not KeyError.
    current_mapping['strike_price'] = current_mapping['CFD_Id'].map(strike_lookup)

    return current_mapping.loc[
        ~current_mapping.index.duplicated(keep='first'),
        'strike_price'
        ]


if __name__ == "__main__":

    configure_logging(snakemake)

    bmu_locations = pd.read_csv(snakemake.input.bmu_locations, index_col=0)
    bmu_locations = bmu_locations.loc[bmu_locations['lat'] != 0]

    bmu_mappings = pd.read_csv(
        snakemake.input.bmu_mappings,
        index_col=1,
    )
    bmu_mappings.index = list(map(lambda i: i.split('_')[-1], bmu_mappings.index))
    bmu_mappings = bmu_mappings.loc[
        bmu_mappings.index.intersection(bmu_locations.index),
        'CFD_Id'
        ]

    dates = [
        '2021-12-09',
        '2022-01-21',
        '2022-04-11',
        '2022-07-05',
        '2022-09-27',
        '2022-12-05',
        '2023-03-29',
        '2023-06-30',
        '2023-09-29',
        '2024-01-02',
        '2024-01-24',
        '2024-04-03',
        '2024-07-03',
        '2024-09-10',
        '2024-09-18',
        '2025-01-31',
        '2025-04-23',
        '2025-07-29',
        '2025-10-31',
        '2025-12-23',
    ]

    assert len(dates) + 2 == len(snakemake.input), 'Hard-coded date assignment, change code accordingly.'

    strike_prices = []

    for date, fn in zip(dates, snakemake.input):

        logger.info(f"Processing {date}:, {fn}")

        strike_prices.append(
            process_cfd_register(
                fn,
                bmu_mappings 
                ).rename(date)
            )

    pd.concat(strike_prices, axis=1).to_csv(snakemake.output['cfd_strike_prices'])
