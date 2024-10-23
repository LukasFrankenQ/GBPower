# -*- coding: utf-8 -*-
# Copyright 2024-2024 Lukas Franken (University of Edinburgh, Octopus Energy)
# SPDX-FileCopyrightText: : 2024-2024 Lukas Franken
#
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def get_extremes(dispatch, mode='on', neighbour_filter=10):
    """
    Returns indices of timeseries of dispatch in which a generator is switched on or off.
    Works for high-cost peak-demand meeting generators 
    
    Parameters
    ----------
    dispatch : pd.Series
        Timeseries of dispatch
    mode : str, optional
        'on' or 'off', by default 'on'; on to return times of on-switching else off-switching
    """

    if isinstance(dispatch, pd.DataFrame):
        assert len(dispatch.columns) == 0, 'unclear how to handle multiple columns in dispatch dataframe'
        dispatch = dispatch.iloc[:,0]

    if dispatch.max() > 1.:
        dispatch = pd.Series(
            MinMaxScaler().fit_transform(dispatch.values.reshape(-1, 1)).flatten(),
            index=dispatch.index
        )
    
    rounded = dispatch.fillna(0.).round()

    deriv_window_size = 10
    deriv = (
        rounded
        .rolling(deriv_window_size, center=True)
        .apply(lambda x: x[deriv_window_size//2:].mean() - x[:deriv_window_size//2].mean())
    )

    if mode == 'on':
        extreme_func = np.less_equal
    elif mode == 'off':
        extreme_func = np.greater_equal
    else:
        raise ValueError('mode must be either "on" or "off"')

    extremes = np.array(argrelextrema(deriv.values, extreme_func, order=5)[0])

    extremes = extremes[deriv.iloc[extremes] != 0]

    # minor cleanup; remove neighbouring values
    mask = np.abs(np.roll(extremes, 1) - extremes) < neighbour_filter
    mask = mask + np.roll(mask, -1)

    extremes = extremes[~mask]

    return extremes


def get_price_distribution(dispatch, prices, visval=False, **kwargs):
    """
    For the dispatch of a single BMU, returns the average price of electricity
    during the time it is switched on. 

    Parameters
    ----------
    dispatch : pd.Series
        Timeseries of dispatch
    prices : pd.Series
        wholesale market price for electricity
    visval : bool, optional
        if True plot the dispatch and the inferred switch-on and switch-off times

    """

    full_index = pd.date_range(prices.index[0], prices.index[-1], freq='30min')

    prices = prices.copy().reindex(full_index).interpolate()
    dispatch = dispatch.copy().reindex(full_index).interpolate()

    dispatch = dispatch.loc[~dispatch.isna()]
    prices = prices.loc[dispatch.index]

    switchons = get_extremes(dispatch, mode='on', **kwargs)
    switchoffs = get_extremes(dispatch, mode='off', **kwargs)

    switchons = pd.Index([dispatch.index[n] for n in switchons])
    switchoffs = pd.Index([dispatch.index[n] for n in switchoffs])

    if visval:

        start = dispatch.index[0]
        if len(dispatch) > 1e3:
            end = dispatch.index[min(len(dispatch)-1, 1000)]

        _, ax = plt.subplots(1, 1, figsize=(10, 3.5))
        ax.plot(dispatch, color='k')
        for on in switchons:
            ax.axvline(on, color='r', linestyle='--')
        for off in switchoffs:
            ax.axvline(off, color='b', linestyle='--')
        ax.set_title(dispatch.name)
        
        ax.set_xlim(start, end)

        ax.set_ylabel('Dispatch [MW]')
        ax.set_xlabel('Time')
        ax.xaxis.set_tick_params(rotation=45)

        plt.show()

    avg_prices = list()

    for on in switchons:

        try:
            off = switchoffs[switchoffs > on][0]
        except IndexError:
            continue

        avg_prices.append(prices.loc[on:off].mean())
    
    if len(avg_prices) == 0:
        return np.nan
    else:
        return avg_prices