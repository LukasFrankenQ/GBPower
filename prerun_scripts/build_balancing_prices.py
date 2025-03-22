# -*- coding: utf-8 -*-
# Copyright 2024-2024 Lukas Franken (University of Edinburgh)
# SPDX-FileCopyrightText: : 2024-2024 Lukas Franken
#
# SPDX-License-Identifier: MIT

'''
Builds monthly default bid and offer prices for different groups of assets.
This is because the model may predict balancing for a day where the actual
balancing market had no activity.
'''

import logging

logger = logging.getLogger(__name__)

import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

sys.path.append(str(Path.cwd() / 'scripts'))
from _helpers import configure_logging, classify_north_south


def make_north_south_split(
        bmus,
        carrier,
        ):

    if not isinstance(carrier, str):
        mask = bmus['carrier'].isin(carrier)
    else:
        mask = bmus['carrier'].str.contains(carrier)
    
    coords = bmus[['lat', 'lon']]
    bmus['region'] = coords.apply(
        lambda row: classify_north_south(row['lon'], row['lat']), axis=1
        )

    north = bmus.loc[mask & (bmus['region'] == 'north')].index
    south = bmus.loc[mask & (bmus['region'] == 'south')].index

    return north, south


def process_balancing_data(df):
    df = (
        df
        .stack()
        .unstack(1)
        .dropna()
        .groupby(level=1)
        .agg({'price': 'mean', 'vol': 'sum'})
        .sort_values('price')
    )
    return df


def get_weighted_avg_price(df):
    assert set(df.columns) == {'price', 'vol'}, 'Columns must be price and vol'
    assert not df.empty, 'DataFrame must not be empty'

    return (df['price'] * df['vol']).sum() / df['vol'].sum()


if __name__ == '__main__':
    configure_logging(snakemake)

    bmus = pd.read_csv(snakemake.input.bmus, index_col=0)

    bmus = bmus.loc[bmus['lat'] != 'distributed']

    bmus['lat'] = bmus['lat'].astype(float)

    disp_north, disp_south = make_north_south_split(
        bmus,
        ['fossil', 'biomass', 'coal'],
    )
    disp = disp_north.union(disp_south)

    wind_north, wind_south = make_north_south_split(
        bmus,
        'wind'
    )
    wind = wind_north.union(wind_south)

    water_north, water_south = make_north_south_split(
        bmus,
        ['cascade', 'hydro', 'PHS']
    )
    water = water_north.union(water_south)

    offers = []
    bids = []

    for fn in snakemake.input:

        if 'offers.csv' in fn:
            offers.append(pd.read_csv(fn, index_col=[0,1], parse_dates=True))
        elif 'bids.csv' in fn:
            bids.append(pd.read_csv(fn, index_col=[0,1], parse_dates=True))

    offers = process_balancing_data(pd.concat(offers))
    bids = process_balancing_data(pd.concat(bids))

    logger.info(f'Taking weighted average of {len(offers)} offers.')
    logger.info(f'Taking weighted average of {len(bids)} bids.')

    results = pd.DataFrame(
        np.nan,
        index=['disp', 'wind', 'water'],
        columns=['offers', 'bids'],
    )

    for group in ['disp', 'wind', 'water']:

        o = offers.index.intersection(globals()[f'{group}'])
        if len(o):
            results.loc[group, 'offers'] = get_weighted_avg_price(offers.loc[o])
        
        b = bids.index.intersection(globals()[f'{group}'])
        if len(b):
            results.loc[group, 'bids'] = get_weighted_avg_price(bids.loc[b])
    
    results.to_csv(snakemake.output[0])
