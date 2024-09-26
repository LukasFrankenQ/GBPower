import requests
import pandas as pd
from io import StringIO
from functools import wraps
import urllib

_cache = {}
_calls_remaining = {}

accepts_url = (
    'https://data.elexon.co.uk/bmrs/api/v1/balancing/acceptances/' +
    'all?settlementDate={}&settlementPeriod={}&format=csv'
    )

volumes_url = (
    "https://data.elexon.co.uk/bmrs/api/v1/balancing/settlement/"
    "indicative/volumes/all/{}/{}/{}?format=json&{}"
)

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


def cache_volumes_trades(func):
    @wraps(func)
    def wrapper(date, period):
        key = (date, period)
        if key not in _cache:
            print('building volumes, trades for ', date, period)

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

    vol_marker = {'bids': 'negative', 'offers': 'positive'}[action]
    
    def get_trades(action, df):
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

        trades = get_trades(action, unit_trades)

        for trade_price, trade_vol in zip(trades, row.loc[row != 0].values):
            detected_actions.append(pd.Series({
                'vol': trade_vol,
                'price': trade_price
            }, name=bm))

    return pd.concat(detected_actions, axis=1).T
    


'''
@cache_volumes_trades
def get_offers(volumes, trades, date, period):

    print('Building offers for ', date, period)
    cols = volumes.columns[volumes.columns.str.contains('positive')].tolist()
    detected_offers = list()

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

        trade = unit_trades.loc[unit_trades.PairId > 0, 'Offer']

        for trade_price, trade_vol in zip(trade, row.loc[row != 0].values):
            detected_offers.append(pd.Series({
                'vol': trade_vol,
                'price': trade_price
            }, name=bm))

    return pd.concat(detected_offers, axis=1).T


def get_bids(volumes, trades, date, period):

    print('Building bids for ', date, period)
    cols = volumes.columns[volumes.columns.str.contains('negative')].tolist()
    detected_bids = list()

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

        trade = unit_trades.loc[unit_trades.PairId > 0, 'Offer']

        for trade_price, trade_vol in zip(trade, row.loc[row != 0].values):
            detected_bids.append(pd.Series({
                'vol': trade_vol,
                'price': trade_price
            }, name=bm))

    return pd.concat(detected_bids, axis=1).T
'''





date, period = '2024-03-23', 24
print(date, period)

# print('-------------')
# print(_cache)
# print(_calls_remaining)
# print('-------------')
print(get_offers(date, period))
print(get_bids(date, period))

print('-------------')
print(date, period)
# print(_cache)
# print(_calls_remaining)
# print('-------------')

date, period = '2022-10-19', 30

print(get_offers(date, period))
print(get_bids(date, period))