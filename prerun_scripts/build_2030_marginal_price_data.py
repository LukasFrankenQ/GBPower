# -*- coding: utf-8 -*-
# Copyright 2024-2025 Lukas Franken
# SPDX-FileCopyrightText: : 2024-2025 Lukas Franken
#
# SPDX-License-Identifier: MIT

# processes the 2030 marginal price from the 2024 TYNDP assessment to be feasible to 
# be included in the repository
# 'https://2024-data.entsos-tyndp-scenarios.eu/files/scenarios-outputs/MMStandardOutputFile_NT2030_Plexos_CY2009_2.5_v40.xlsx.zip'
# and in particular the file

# 'MMStandardOutputFile_NT2030_Plexos_CY2009_2.5_v40.xlsx'

# if you would like to rerun this script, you need to download the excel file and insert
# it into the data folder as 'MMStandardOutputFile_NT2030_Plexos_CY2009_2.5_v40.xlsx'

import logging

logger = logging.getLogger(__name__)

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path.cwd() / 'scripts'))
from _helpers import configure_logging

if __name__ == '__main__':
    configure_logging(snakemake)

    fn = snakemake.input[0]

    logger.info(f'Reading {fn}...')
    data = pd.read_excel(fn, sheet_name='Hourly Market Data')

    mc = data.loc[:, data.loc[9] == 'Marginal Cost [€]'].iloc[10:]
    mc.columns = mc.iloc[0]
    mc = mc.iloc[2:]
    mc.index = data.iloc[12:,1]
    mc.index.name = 'time'
    mc.columns.name = 'region'

    load = data.loc[:, data.loc[9] == 'Demand [MW]\n(losses included)']
    load = load.iloc[12:]
    load.columns = mc.columns
    load.index = mc.index

    # Drop columns with no load
    valid_cols = load.sum() != 0
    mc = mc.loc[:, valid_cols]
    load = load.loc[:, valid_cols]

    # Group by first 2 letters of column names
    groups = mc.columns.str[:2]

    # First calculate weighted average across time for each column
    time_weighted = (
        mc.mul(load)
        .sum() / 
        load.sum()
    )

    # Then group and weight by load sums within each group
    group_sums = load.sum().groupby(groups).sum()
    total_sum = group_sums.sum()

    if total_sum == 0:
        print("No data available - all loads are zero")
    else:
        # Filter for selected countries and rename
        country_mapping = {
            'NO': 'Norway',
            'FR': 'France', 
            'DE': 'Germany',
            'BE': 'Belgium',
            'DK': 'Denmark',
            'NL': 'Netherlands',
            'IE': 'Ireland'
        }
    
    final_weighted_avg = (
        time_weighted
        .groupby(groups)
        .agg(lambda x: (x * load.sum().loc[x.index]).sum() / load.sum().loc[x.index].sum())
        .loc[country_mapping.keys()]
        .rename(index=country_mapping)
        .sort_values()
    ) / 1.15 # unit conversion

    logger.info(f'Saving 2030 average day-ahead prices to {snakemake.output}:')
    logger.info(final_weighted_avg)

    final_weighted_avg.name = 'avg_day_ahead_prices'
    final_weighted_avg.to_csv(snakemake.output[0])
