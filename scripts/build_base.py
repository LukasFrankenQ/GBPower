import urllib
import logging
import requests
import pandas as pd
from functools import wraps

from io import StringIO
from typing import Union, Tuple

# logger = logging.getLogger(__name__)

# configure_logging()

from _helpers import (
    to_date_period,
    to_datetime,
    to_daterange,
    # configure_logging,
)

from _elexon_helpers import robust_request



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

def build_maximum_export_period(date, period):

    prep_time = lambda x: str(x).zfill(2)

    start = to_datetime(date, period)
    end = start + pd.Timedelta('30min')

    assert start.tz == 'utc', 'Start time must be in UTC'

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
    'market-index?from={}T00%3A00Z&to={}'
    'T00%3A00Z&dataProviders=N2EX&dataProviders=APX&format=csv'
)
def build_day_ahead_prices(
        start: Union[str, pd.Timestamp],
        end: Union[str, pd.Timestamp]
    ) -> Tuple[pd.Series, pd.Index]:

    if isinstance(start, str):
        start = pd.Timestamp(start, tz='UTC')
        end = pd.Timestamp(end, tz='UTC')

    all_dates = list(map(pd.Timestamp, pd.date_range(start, end, freq='30min', tz='utc')[:-1]))

    dfs = []

    shift = pd.Timedelta(days=0)

    while start + shift < end:

        retrieve_start = (start + shift).strftime('%Y-%m-%d')
        retrieve_end = (start + shift+pd.Timedelta('7D'))
        retrieve_end = min(retrieve_end, end).strftime('%Y-%m-%d') 

        response = requests.get(day_ahead_url.format(retrieve_start, retrieve_end))
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

        dfs.append(df)
        shift += pd.Timedelta('7D')

    df = pd.concat(dfs).iloc[:-1]

    duplicates = df.index.duplicated(keep='first')
    df = df[~duplicates]

    problem_dates = pd.Index(all_dates).difference(df.index)
    
    df = df.reindex(all_dates).interpolate()
    
    return df, problem_dates


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

    acc = pd.concat(corrected)
    return acc.NationalGridBmUnit.unique()


def get_volumes(date, period):
    print('getting volumes for', date, period)

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
    print('getting trades for', date, period)

    units = get_accepted_units(date, period)

    response = requests.get(trades_url.format(date, period))
    response.raise_for_status()
    trades_df = pd.read_csv(StringIO(response.text))

    return trades_df[trades_df['NationalGridBmUnit'].isin(units)]


_cache = {}
_calls_remaining = {}

def cache_volumes_trades(func):
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
        print("Deleting volumes and trades for ", key)
        del _cache[key]
        del _calls_remaining[key]


@cache_volumes_trades
def get_offers(*args):
    return get_bm_actions('offers', *args)


@cache_volumes_trades
def get_bids(*args):
    return get_bm_actions('bids', *args)


def get_bm_actions(action, volumes, trades, date, period):
    print('Building', action, 'for', date, period)

    vol_marker = {'bids': 'negative', 'offers': 'positive'}[action]
    
    def get_unit_trades(action, df):
        if action == 'bids':
            return df.loc[df.PairId < 0, 'Bid'].iloc[::-1]
        elif action == 'offers':
            return df.loc[df.PairId > 0, 'Offer']

    cols = volumes.columns[volumes.columns.str.contains(vol_marker)].tolist()
    detected_actions = list()

    for bm in volumes.nationalGridBmUnit.unique():

        unit_volumes = volumes.loc[volumes.nationalGridBmUnit == bm]
        unit_trades = trades.loc[trades.NationalGridBmUnit == bm]

        row = unit_volumes[['dataType'] + cols]
        row = (
            row.loc[row['dataType'] == 'Tagged', cols]
            .fillna(value=0)
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


######################    GATHER DATA    ##############################
def gather_data(start, end):

    date_range = to_daterange(start, end)

    for quantity in snakemake.outputs.keys():

        if f"build_{quantity}_period" in globals():

            data = []
            
            for ts in date_range:
                date, period = to_date_period(ts)

                data.append(
                    globals()[f"build_{quantity}_period"](date, period)
                    )
            
            pd.concat(data, axis=1).to_csv(snakemake.output[quantity])
        
        else:

            globals()[f'build_{quantity}'](start, end)





