# SPDX-FileCopyrightText: : 2024 Lukas Franken
#
# SPDX-License-Identifier: MIT

"""
Script to create the file 'data/nuclear_bidding_cost.csv'.

Goes over day-ahead prices and retrieves an estimate of nuclear bidding cost to
replicate their always-on market behaviour.

To that end, the script obtains the lower value for day-ahead prices over the
evaluated time-span

"""

import logging

logger = logging.getLogger(__name__)

import sys
import pandas as pd
from tqdm import tqdm
from pathlib import Path

sys.path.append(str(Path.cwd() / 'scripts'))
from _helpers import configure_logging


if __name__ == '__main__':
    configure_logging(snakemake)

    prices = []
    for d in tqdm(snakemake.input):

        try:
            prices.append(
                pd.read_csv(Path(d) / 'day_ahead_prices.csv', index_col=0, parse_dates=True
                )
            )
        except FileNotFoundError:
            pass

    (
        pd.Series(
            pd.concat(prices).sort_index()['day-ahead-prices'].min(),
            index=['nuclear_marginal_cost']
        )
        .to_csv(snakemake.output[0])
    )
