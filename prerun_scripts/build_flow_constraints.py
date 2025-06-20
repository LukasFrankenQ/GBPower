# SPDX-FileCopyrightText: : 2024 Lukas Franken
#
# SPDX-License-Identifier: MIT

"""
build_boundary_flow_base.py
=============
Builds database for flow limits in 2022, 2023, 2024 from half-hourly data.
Instead of querying this data for each day, like the other data points, it is here
helpful to get all data at once to simplify the interpolation of missing data points.
"""

import logging

logger = logging.getLogger(__name__)

import re
import sys
import requests
import numpy as np
import pandas as pd

from tqdm import tqdm
from urllib import parse
from pathlib import Path
from pytz.exceptions import NonExistentTimeError

sys.path.append(str(Path.cwd() / 'scripts'))
from _helpers import configure_logging
from _constants import dst_start_dates, dst_end_dates, build_sp_register



def get_boundary_flow_day(date_range):
    """
    Fetches and processes boundary flow data for a given date range.
    This function queries the National Grid ESO API to retrieve boundary flow data
    for specified boundaries within the given date range. The data is then processed
    and returned as a DataFrame.
    Parameters:
    date_range (pd.DatetimeIndex): A range of dates for which to fetch the boundary flow data.
    Returns:
    pd.DataFrame: A DataFrame containing the boundary flow data with boundaries as columns
                  and the date range as the index. If no data is found, returns a DataFrame
                  filled with NaN values.
    Notes:
    - The function handles daylight saving time changes by adjusting the DataFrame index accordingly.
    - Boundaries with flow values greater than or equal to 15,000 MW are set to NaN.
    - The function ensures that the DataFrame columns match the specified boundaries.
    Example:
    >>> date_range = pd.date_range(start='2023-01-01', end='2023-01-07', freq='H')
    >>> df = get_boundary_flow_day(date_range)
    >>> print(df.head())
    """

    day = date_range[len(date_range) // 2].strftime('%Y-%m-%d')

    boundaries = pd.Index(['SSE-SP', 'SCOTEX', 'SSHARN', 'FLOWSTH', 'SEIMP'])

    # request to national grid ESO is made in London time
    start = (
        date_range[0]
        .tz_convert('Europe/London')
    )
    end = (
        date_range[-1]
        .tz_convert('Europe/London')
    )

    sql_query = (
        '''SELECT COUNT(*) OVER () AS _count, * FROM "38a18ec1-9e40-465d-93fb-301e80fd1352"'''+
        ''' WHERE "Date (GMT/BST)" >= '{}' '''.format(start.strftime("%Y-%m-%d %H:%M:%S")) +
        '''AND "Date (GMT/BST)" <= '{}' '''.format(end.strftime("%Y-%m-%d %H:%M:%S")) +
        '''ORDER BY "_id" ASC LIMIT 1000'''
    )

    params = {'sql': sql_query}

    response = requests.get(
        'https://api.nationalgrideso.com/api/3/action/datastore_search_sql',
        params=parse.urlencode(params)
        )

    data = response.json()["result"]

    if not data['records']:
        print(f"No data found for {date_range[0].strftime('%Y-%m-%d')}")
        return pd.DataFrame(np.nan, index=date_range, columns=boundaries)

    df = (
        pd.DataFrame(data["records"])
        .set_index("Constraint Group")
        [["_count", "Limit (MW)", "Flow (MW)", "Date (GMT/BST)"]]
        .rename(columns={
            "Limit (MW)": "limit",
            "Flow (MW)": "flow",
            "Date (GMT/BST)": "date",
            })
        .drop(columns=['_count'])
    )

    df.loc[df['limit'] >= 15_000, 'limit'] = np.nan

    try: # ironically crashes for some daylight saving endings
        df = df.set_index('date', append=True)['limit'].unstack().T

        # prevents error on days where daylight saving time starts or ends
        if day in dst_end_dates:
            df = pd.concat((
                df.iloc[:3],
                df.iloc[[2]],
                df.iloc[[3]],
                df.iloc[3:],
            ))
            df.index = date_range

        elif day in dst_start_dates:

            try:
                df.index = (
                    pd.to_datetime(df.index)
                    .tz_localize('Europe/London', ambiguous='infer')
                    .tz_convert('UTC')
                )
            except NonExistentTimeError:
                df = pd.concat((
                    df.iloc[:2], df.iloc[4:]
                ))
                df.index = date_range

        else:
            df.index = (
                pd.to_datetime(df.index)
                .tz_localize('Europe/London', ambiguous='infer')
                .tz_convert('UTC')
            )

    except ValueError:

        limits = []
        for b in df.index.unique():

            ss = df.loc[b, 'limit']
            ss.index = date_range[:len(ss)]
            ss = ss.reindex(date_range, method='ffill')
            ss.name = b

            limits.append(ss)

        df = pd.concat(limits, axis=1)


    number_split = lambda x: re.split(r'\d', x, maxsplit=1)[0]
    def match_and_cut(index, true_names):

        new_index = pd.Index(map(number_split, index)).intersection(true_names)
        old_index = pd.Index([i for i in index if number_split(i) in new_index])

        return {
            old: new for new, old in zip(new_index, old_index)
        }

    renamer = match_and_cut(df.columns, boundaries)

    return (
        pd.concat((
            df[renamer.keys()].rename(columns=renamer),
            pd.DataFrame(
                np.nan, index=df.index,
                columns=boundaries.difference(renamer.values())
                )
        ), axis=1).loc[:, boundaries]
    )


if __name__ == '__main__':

    configure_logging(snakemake)

    year = snakemake.wildcards.year

    ds = []

    for day in tqdm(pd.date_range(f'{year}-01-01', f'{year}-10-08')):
        day = day.strftime("%Y-%m-%d")

        sp_register = build_sp_register(day)
        date_range = sp_register.index

        df = get_boundary_flow_day(date_range)
        ds.append(df)

    df = pd.concat(ds)

    # some values seem unnaturally low, set them to nan and interpolate
    df = df.mask(df < df.mean() * 0.3).interpolate()

    threshold = 60 * 48 # two months of unchanging data are set to nan, and inferred from other constraints 
    final_mask = pd.DataFrame(False, index=df.index, columns=df.columns)

    for col in df.columns:

        constant_mask = df[col].eq(df[col].shift())

        streak_counter = (
            constant_mask
            .groupby((constant_mask != constant_mask.shift()).cumsum())
            .cumsum()
        )
        final_mask[col] = streak_counter >= (threshold - 1)
        valid_streaks = (
            (streak_counter >= (threshold - 1))
            .groupby((streak_counter == 0).cumsum())
            .transform('max')
        )
        final_mask[col] = valid_streaks

    df[final_mask] = np.nan

    # method does not work if at any timesteps data is missing for all boundaries
    assert not df.isna().all(axis=1).any()

    df = df.fillna(
        pd.DataFrame(
                df.mean(axis=1).values[:, None] * df.mean().values / df.mean().mean(),
                index=df.index,
                columns=df.columns
            )
        )

    assert df.isna().sum().sum() == 0, "There are missing values in the flow constraint data."

    df.to_csv(snakemake.output["flow_constraints"])