import logging
import requests
import pandas as pd
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

pn_url = (
    "https://data.elexon.co.uk/bmrs/api/v1/datasets/PN"
    "?settlementDate={}&settlementPeriod={}&format=csv"
)

mels_url = (
    "https://data.elexon.co.uk/bmrs/api/v1/datasets/MELS"
    "?from={}T{}%3A{}Z&to={}T{}%3A{}Z&format=csv"
)

day_ahead_url = (
    'https://data.elexon.co.uk/bmrs/api/v1/balancing/pricing/'
    'market-index?from={}T00%3A00Z&to={}'
    'T00%3A00Z&dataProviders=N2EX&dataProviders=APX&format=csv'
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
            

        









# if __name__ == '__main__':


