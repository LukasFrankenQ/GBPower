# SPDX-FileCopyrightText: : 2024 Lukas Franken
#
# SPDX-License-Identifier: MIT
"""
build_base.py
=============
This script is designed to build various datasets related to the UK electricity market using data from the Elexon API. The script includes functions to fetch and process data for physical notifications, maximum export limits, day-ahead prices, and balancing actions (bids and offers). The processed data is then saved to CSV files.
Functions
---------
- `build_sp_register(day)`: Builds a settlement period register for a given day.
- `build_physical_notifications_period(date, period)`: Fetches and processes physical notifications data for a given date and period.
- `build_maximum_export_limits_period(date, period)`: Fetches and processes maximum export limits data for a given date and period.
- `build_day_ahead_prices(date_range)`: Fetches and processes day-ahead prices for a given date range.
- `build_bm_actions_period(action, volumes, trades, date, period)`: Processes balancing actions (bids or offers) for a given date and period.
- `build_offers_period(*args)`: Builds offers data for a given date and period.
- `build_bids_period(*args)`: Builds bids data for a given date and period.
- `build_europe_day_ahead_prices(date)`: Placeholder function for fetching interconnector prices.
- `build_boundary_flow_limits(date)`: Placeholder function for fetching boundary flow limits.
"""


import logging

logger = logging.getLogger(__name__)

import urllib
import requests
import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True) # related to .replace(..) replacing .fillna()

from tqdm import tqdm
from functools import wraps

from io import StringIO
from typing import Iterable, Tuple
from entsoe import EntsoePandasClient

from _helpers import (
    to_datetime,
    configure_logging,
)
from build_flow_constraints import get_boundary_flow_day
from _elexon_helpers import robust_request
from _constants import build_sp_register
from _tokens import ENTSOE_API_KEY


######################    PHYSICAL NOTIFICATIONS   ########################
pn_url = (
    "https://data.elexon.co.uk/bmrs/api/v1/datasets/PN"
    "?settlementDate={}&settlementPeriod={}&format=csv"
)

def build_physical_notifications_period(date, period):

    response = robust_request(requests.get, pn_url.format(date, period))
    df = pd.read_csv(StringIO(response.text))

    df['TimeFrom'] = pd.to_datetime(df['TimeFrom'])
    df['TimeTo'] = pd.to_datetime(df['TimeTo'])

    df['TimeLength'] = df['TimeTo'] - df['TimeFrom']

    df['AverageLevel'] = (
        df[['LevelFrom', 'LevelTo']]
        .mean(axis=1)
        .mul(
            df['TimeLength'].div(pd.Timedelta('30min'))
            )
        )

    return (
        df.groupby('NationalGridBmUnit')
        ["AverageLevel"].sum()
        .rename(pd.Timestamp(to_datetime(date, period)))
    )


######################    MAXIMUM EXPORT LIMIT    #########################
mels_url = (
    "https://data.elexon.co.uk/bmrs/api/v1/datasets/MELS"
    "?from={}T{}%3A{}Z&to={}T{}%3A{}Z&format=csv"
)

def build_maximum_export_limits_period(date, period):

    prep_time = lambda x: str(x).zfill(2)

    start = to_datetime(date, period)    
    end = start + pd.Timedelta('30min')

    response = robust_request(
        requests.get,
        mels_url.format(
            start.strftime('%Y-%m-%d'),
            prep_time(start.hour),
            prep_time(start.minute),
            start.strftime('%Y-%m-%d'),
            prep_time(start.hour),
            prep_time(start.minute)
        )
    )

    df = pd.read_csv(StringIO(response.text))

    if df.empty:
        logger.warning(f"No data for {date} period {period}.")
        return pd.Series(name=start)

    df['TimeFrom'] = pd.to_datetime(df['TimeFrom'])
    df['TimeTo'] = pd.to_datetime(df['TimeTo'])

    # correct for some entries only covering less than 30 minutes
    helper = df.groupby('NationalGridBmUnit')["TimeTo"].max()
    helper = helper.loc[helper < end]

    df.loc[df.NationalGridBmUnit.isin(helper.index), 'TimeTo'] = end
    df['TimeLength'] = df['TimeTo'] - df['TimeFrom']

    df['AverageLevel'] = (
        df[['LevelFrom', 'LevelTo']]
        .mean(axis=1)
        .mul(
            df['TimeLength'].div(pd.Timedelta('30min'))
            )
        )

    return (
        df.groupby('NationalGridBmUnit')
        ["AverageLevel"].sum()
        .rename(start)
    )


######################    DAY-AHEAD PRICES   ##############################
day_ahead_url = (
    'https://data.elexon.co.uk/bmrs/api/v1/balancing/pricing/'
    'market-index?from={}%3A00Z&to={}'
    '%3A00Z&dataProviders=N2EX&dataProviders=APX&format=csv'
)

def build_day_ahead_prices(
        date_range: Iterable[pd.Timestamp],
    ) -> Tuple[pd.Series, pd.Index]:

    date_range = date_range.copy().tz_convert('utc')

    start = date_range[0]

    response = (
        requests.get(
            day_ahead_url.format(
                start.strftime('%Y-%m-%dT%H'),
                (start + pd.Timedelta(days=1)).strftime('%Y-%m-%dT%H'),
                )
            )
        )
    data = StringIO(response.text)
    df = pd.read_csv(data, index_col=0, parse_dates=True).sort_index()

    # volume-weighted average of AP and N2E wholesale markets
    df = (
        (a := df.loc[df.DataProvider == "APXMIDP"])['Price'].mul(
            a.Volume.div(
                t := df.groupby(df.index)['Volume'].sum()
            )) +
        (n := df.loc[df.DataProvider == "N2EXMIDP"])['Price'].mul(
            n.Volume.div(t)
        )
    )

    df = pd.DataFrame(df.rename('day-ahead-prices')).iloc[:-1]

    # cleanup
    duplicates = df.index.duplicated(keep='first')
    df = df[~duplicates]
    df = df.reindex(date_range).interpolate()

    return df


######################    BALANCING ACTIONS    ##############################
# to get bm units that have been accepted by the system operator
accepts_url = (
    'https://data.elexon.co.uk/bmrs/api/v1/balancing/acceptances/' +
    'all?settlementDate={}&settlementPeriod={}&format=csv'
    )
# to get volumes of bids of offers for accepted bm units
volumes_url = (
    "https://data.elexon.co.uk/bmrs/api/v1/balancing/settlement/"
    "indicative/volumes/all/{}/{}/{}?format=json&{}"
)
# to get prices of bids and offers
trades_url = (
        'https://data.elexon.co.uk/bmrs/api/v1/balancing/bid-offer/' +
        'all?settlementDate={}&settlementPeriod={}&format=csv'
    )


def get_accepted_units(date, period, so_only=False):

    response = requests.get(accepts_url.format(date, period))
    response.raise_for_status()
    
    acc = pd.read_csv(StringIO(response.text))

    if so_only:
        acc = acc.loc[acc.SoFlag]

    corrected = []

    for bm in acc.NationalGridBmUnit.unique():

        ss = acc.loc[acc.NationalGridBmUnit == bm]
        ss = ss.loc[ss.AcceptanceTime == ss.AcceptanceTime.max()]
        corrected.append(ss)

    try:
        return pd.concat(corrected).NationalGridBmUnit.unique()
    except ValueError:
        return np.array([], dtype=str)


def get_volumes(date, period):

    units = get_accepted_units(date, period)
    unit_params = '&'.join(f"bmUnit={urllib.parse.quote_plus(unit)}" for unit in units)

    data = list()

    for mode in ['offer', 'bid']:
    
        response = requests.get(volumes_url.format(mode, date, period, unit_params))
        response.raise_for_status()
        
        volumes_json = response.json()
        if 'data' in volumes_json:
            volumes_df = pd.json_normalize(volumes_json['data'])
        else:
            volumes_df = pd.DataFrame()
        
        data.append(volumes_df)
    
    return pd.concat(data)
    

def get_trades(date, period):

    units = get_accepted_units(date, period)

    response = requests.get(trades_url.format(date, period))
    response.raise_for_status()
    trades_df = pd.read_csv(StringIO(response.text))

    return trades_df[trades_df['NationalGridBmUnit'].isin(units)]


_cache = {}
_calls_remaining = {}

# both bids and offers build on the same data, so we cache the volumes and trades
def _cache_volumes_trades(func):
    @wraps(func)
    def wrapper(date, period):
        key = (date, period)
        if key not in _cache:

            _cache[key] = {}
            _cache[key]['v'] = get_volumes(date, period)
            _cache[key]['t'] = get_trades(date, period)

            _calls_remaining[key] = {'get_volumes': True, 'get_trades': True}

        result = func(_cache[key]['v'], _cache[key]['t'], date, period)
        _cleanup(func.__name__, key)
        return result
    return wrapper


def _cleanup(func_name, key):
    _calls_remaining[key][func_name] = False
    if not any(_calls_remaining[key].values()):
        del _cache[key]
        del _calls_remaining[key]


def build_bm_actions_period(action, volumes, trades, date, period):

    vol_marker = {'bids': 'negative', 'offers': 'positive'}[action]
    
    def get_unit_trades(action, df):
        if action == 'bids':
            return df.loc[df.PairId < 0, 'Bid'].iloc[::-1]
        elif action == 'offers':
            return df.loc[df.PairId > 0, 'Offer']

    if volumes.empty:
        return pd.DataFrame(
            columns=pd.MultiIndex.from_product(
                [[to_datetime(date, period)], ['vol', 'price']]
            )
        )

    cols = volumes.columns[volumes.columns.str.contains(vol_marker)].tolist()
    detected_actions = list()

    for bm in volumes.nationalGridBmUnit.unique():

        unit_volumes = volumes.loc[volumes.nationalGridBmUnit == bm]
        unit_trades = trades.loc[trades.NationalGridBmUnit == bm]

        row = unit_volumes[['dataType'] + cols]
        row = (
            row.loc[row['dataType'] == 'Tagged', cols]
            .replace(np.nan, 0)
            .abs()
            .max()
        )

        bm_trades = get_unit_trades(action, unit_trades)

        for trade_price, trade_vol in zip(bm_trades, row.loc[row != 0].values):
            detected_actions.append(pd.Series({
                'vol': trade_vol,
                'price': trade_price
            }, name=bm))

    def process_multiples(df):
        return df.groupby(df.index).apply(
            lambda x: pd.Series({
                'vol': x['vol'].sum(),
                'price': (x['vol'] * x['price']).sum() / x['vol'].sum()
            })
        )

    try:
        detected_actions = process_multiples(
            pd.concat(detected_actions, axis=1).T
        )
    except ValueError:
        detected_actions = pd.DataFrame(columns=['vol', 'price'])

    if action == 'bids':
        detected_actions['price'] = -detected_actions['price']

    detected_actions.columns = (
        pd.MultiIndex.from_product(
            [[to_datetime(date, period)], detected_actions.columns]
            )
    )

    return detected_actions


@_cache_volumes_trades
def build_offers_period(*args):
    return build_bm_actions_period('offers', *args)


@_cache_volumes_trades
def build_bids_period(*args):
    return build_bm_actions_period('bids', *args)


def build_europe_day_ahead_prices(date_range):

    data = pd.read_csv(
        snakemake.input.europe_day_ahead_prices,
        index_col=0,
        parse_dates=True
    ).tz_localize('utc')

    if date_range[0] >= data.index[0] and date_range[-1] <= data.index[-1]:
        return (
            data
            .reindex(date_range, method='ffill')
            .loc[date_range]
        )
    elif date_range[0] > data.index[-1]:
        raise ValueError(
            "Queried date range {} is after the latest available date {}.".format(
                date_range[0], data.index[-1]
            )
        )
    elif date_range[-1] < data.index[0]:
        raise ValueError(
            "Queried date range {} is before the earliest available date {}.".format(
                date_range[-1], data.index[0]
            )
        )
    else:
        raise ValueError(
            "Queried date range {} is not fully covered by the available data range {} to {}.".format(
                date_range, data.index[0], data.index[-1]
            )
        )


def build_boundary_flow_constraints(date_range):
    
    year_data = pd.read_csv(
        snakemake.input.flow_constraints,
        index_col=0,
        parse_dates=True
    )

    try:
        return year_data.loc[date_range]
        
    except KeyError:
        df = get_boundary_flow_day(date_range)

    # method does not work if at any timesteps data is missing for all boundaries
    assert not df.isna().all(axis=1).any()

    # based on mean value of each boundary, fill missing 
    # values according to variation over the day
    df = df.fillna(
        pd.DataFrame(
                df.mean(axis=1).values[:, None] * year_data.mean().values / year_data.mean().mean(),
                index=df.index,
                columns=df.columns
            )
        )

    assert df.isna().sum().sum() == 0, "There are missing values in the flow constraint data."

    return df


def build_nemo_powerflow(date_range):
    '''Flowdata on the Nemo interconnector is not available from Elexon, and
    is retrieved through the ENTSOE API.'''

    client = EntsoePandasClient(api_key=ENTSOE_API_KEY, retry_delay=60)

    nemo_BEGB = client.query_crossborder_flows(
        'BE', 'GB',
        start=date_range[0] - pd.Timedelta('60min'),
        end=date_range[-1] + pd.Timedelta('60min'),
        )

    nemo_GBBE = client.query_crossborder_flows(
        'GB', 'BE',
        start=date_range[0] - pd.Timedelta('60min'),
        end=date_range[-1] + pd.Timedelta('60min'),
        )

    nemo_flow = (nemo_BEGB - nemo_GBBE).resample('30min').mean()

    if nemo_flow.index.tz is None:
        nemo_flow.index = nemo_flow.index.tz_localize(date_range.tz)
    else:
        nemo_flow.index = nemo_flow.index.tz_convert(date_range.tz)

    nemo_flow = nemo_flow.loc[date_range]

    assert nemo_flow.index.equals(date_range)

    return nemo_flow


if __name__ == '__main__':

    configure_logging(snakemake)

    day = snakemake.wildcards.day
    sp_register = build_sp_register(day)

    sp_register.to_csv(snakemake.output.date_register)

    logger.info(f"Building data base for {day}.")

    date_range = sp_register.index
    periods = sp_register.settlement_period.tolist()

    first_timestep = None
    last_timestep = None

    for quantity, target in snakemake.output.items():

        if quantity == 'date_register':
            continue

        logger.info(f"Building {quantity}.")

        if f"build_{quantity}_period" in globals():
            data = []

            for period in tqdm(periods):

                data.append(
                    globals()[f"build_{quantity}_period"](day, period)
                    )

            data = pd.concat(data, axis=1).T
 
        else:
            data = globals()[f'build_{quantity}'](date_range)

        # ensure consistency between datasets
        assert len(data.index.get_level_values(0).unique()) == len(date_range), 'Number of periods mismatch.'

        if first_timestep is None:
            first_timestep = data.index.get_level_values(0)[0]
        else:
            assert first_timestep == data.index.get_level_values(0)[0], 'First timestep mismatch.'

        if last_timestep is None:
            last_timestep = data.index.get_level_values(0)[-1]
        else:
            assert last_timestep == data.index.get_level_values(0)[-1], 'Last timestep mismatch.'

        data.to_csv(target)

