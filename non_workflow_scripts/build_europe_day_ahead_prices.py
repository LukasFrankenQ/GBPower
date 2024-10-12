# SPDX-FileCopyrightText: : 2024 Lukas Franken
#
# SPDX-License-Identifier: MIT
"""
Script to be transparent about how the Europe day ahead prices are built.

Draws data from EMBER: https://ember-climate.org/data-catalogue/european-wholesale-electricity-price-data/

Cuts out data for countries that are connected to the UK grid via interconnectors.
Starts at 2022.
Also uses a time-series of dailyl Euro-Pound exchange rates to convert prices to GBP
taken from the ECB: https://data.ecb.europa.eu/data/datasets/EXR/EXR.D.GBP.EUR.SP00.A

In total, the script expects the following files:
- 'data/europe_day_ahead_prices/ECB Data Portal_20241012123932.csv'
- 'data/europe_day_ahead_prices/Norway.csv'
- 'data/europe_day_ahead_prices/Belgium.csv'
- 'data/europe_day_ahead_prices/France.csv'
- 'data/europe_day_ahead_prices/Germany.csv'
- 'data/europe_day_ahead_prices/Netherlands.csv'
- 'data/europe_day_ahead_prices/Ireland.csv'
- 'data/europe_day_ahead_prices/Denmark.csv'

Exports a file 'data/europe_day_ahead_prices_GBP.csv'

Expects to the run from the root of the project 'GBPower/'

"""

import numpy as np
import pandas as pd 
from pathlib import Path


if __name__ == '__main__':

    er = pd.read_csv(
        Path.cwd() /
        'data' /
        'europe_day_ahead_prices' /
        'ECB Data Portal_20241012123932.csv',
        index_col=0,
        parse_dates=True
    )
    er.rename(columns={'UK pound sterling/Euro (EXR.D.GBP.EUR.SP00.A)': 'rate'}, inplace=True)
    er = (
        er.reindex(
            pd.date_range(er.index.min(), er.index.max(), freq='D'),
            fill_value=np.nan)
        .interpolate()
        ['rate']
    )

    def preprocess_df(df):
        df = df[df['Datetime (UTC)'] >= '2022-01-01']
        df = df[['Datetime (UTC)', 'Country', 'Price (EUR/MWhe)']]

        df['Price (GBP/MWhe)'] = df.apply(
            lambda row: row['Price (EUR/MWhe)'] * er.at[row['Datetime (UTC)'][:10]],
            axis=1,
        )

        df.rename(columns={'Price (GBP/MWhe)': df.Country.iloc[0]}, inplace=True)
        df.drop(columns=['Country', 'Price (EUR/MWhe)'], inplace=True)

        return df.set_index('Datetime (UTC)')


    all_dfs = []
    for country in ['Norway', 'Belgium', 'France', 'Germany', 'Netherlands', 'Ireland', 'Denmark']:

        all_dfs.append(
            preprocess_df(
                pd.read_csv(Path.cwd() / 'data' / 'europe_day_ahead_prices' / f'{country}.csv')
            )
        )

    prices = pd.concat(all_dfs, axis=1)
    prices.to_csv(
        Path.cwd() /
        'data' /
        'europe_day_ahead_prices_GBP.csv',
        index=True)