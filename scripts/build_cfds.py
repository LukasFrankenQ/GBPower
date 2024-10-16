# SPDX-FileCopyrightText: : 2024 Lukas Franken
#
# SPDX-License-Identifier: MIT

import logging

logger = logging.getLogger(__name__)

import pandas as pd

from _helpers import configure_logging


def process_cfd_register(file_path, mapping):
    try:
        df = pd.read_excel(file_path, index_col=0, sheet_name='Register')
    except Exception as e:
        print(f"Failed to read with pd.read_excel: {e}")
        try:
            df = pd.read_excel(file_path, index_col=0, sheet_name='Register', engine='openpyxl')
        except Exception as e:
            print(f"Failed to read with openpyxl: {e}")
            return None, None

    df = df.rename(columns={
        'Current strike price (field_cfd_current_strikeprice)': 'strike_price',
        'Unique Identifier (field_cfd_unique_id)': 'cfd_Id',
    })

    df.set_index('cfd_Id', inplace=True)

    current_mapping = pd.DataFrame(mapping.copy())
    current_mapping['strike_price'] = (
        current_mapping['CFD_Id']
        .apply(lambda x: df.loc[x, 'strike_price'])
    )

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
        '2024-09-18'
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

    pd.concat(strike_prices, axis=1).T.to_csv(snakemake.output['cfd_strike_prices'])
