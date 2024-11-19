# -*- coding: utf-8 -*-
# Copyright 2024-2024 Lukas Franken
# SPDX-FileCopyrightText: : 2024-2024 Lukas Franken
#
# SPDX-License-Identifier: MIT
"""
Compares dispatch in wholesale market between national and nodal market layout

"""

import logging

logger = logging.getLogger(__name__)

import pypsa
import numpy as np
import pandas as pd

from _helpers import configure_logging


def get_wholesale_cost(n):

    wholesale_cost = 0

    for load, row in n.loads.iterrows():
        if not row.carrier == 'electricity':
            continue

        prices = n.buses_t.marginal_price[row.bus]
        activity = n.loads_t.p[load]

        wholesale_cost += prices * activity * 0.5
    
    return wholesale_cost


def get_congestion_rents(n):

    congestion_rent = 0

    for branch, getter in zip(['lines', 'links'], ['s', 'p']):

        for name, row in getattr(n, branch).iterrows():

            if branch == 'links':
                if row.carrier == 'interconnector':
                    continue

            bus0 = row.bus0
            bus1 = row.bus1
            flow = getattr(n, branch+f'_t')[f'{getter}0'][name]

            prices0 = n.buses_t.marginal_price[bus0]
            prices1 = n.buses_t.marginal_price[bus1]

            congestion_rent += (prices1 - prices0).abs() * flow.abs() * 0.5
    
    return congestion_rent


def get_roc_cost(bal, roc):
    '''
    Calculate the ROC cost for the given network and ROC prices.
    ROC payments are only done when the farms didnt curtail their production,
    therefore bal refers to the network after redispatch (in a
    nodal market insert the nodal wholesale market).
    '''
    roc_payments = 0

    for plant, roc_price in roc.items():

        if plant in bal.generators.index:
            activity = bal.generators_t.p[plant]

        elif plant in bal.storage_units.index:
            activity = bal.storage_units_t.p[plant]

        else:
            print(f'Plant {plant} not found in network')
            continue
            
        roc_payments += roc_price * activity * 0.5

    return roc_payments


def get_cfd_payments(n, strike_prices):

    strike_prices = strike_prices.copy()
    strike_prices.columns = pd.to_datetime(strike_prices.columns)

    strike_prices = (
        strike_prices
        .loc[:,:n.snapshots[0]]
        .iloc[:,-1]
    )

    cfd_payments = 0
    for plant, strike_price in strike_prices.items():

        price_gap = (
            strike_price - n.buses_t.marginal_price[n.generators.loc[plant, 'bus']]
        )
        cfd_payments += n.generators_t.p[plant] * price_gap * 0.5

    return cfd_payments


if __name__ == '__main__':

    configure_logging(snakemake)

    nod = pypsa.Network(snakemake.input.network_nodal) # nodal wholesale market
    nat = pypsa.Network(snakemake.input.network_national) # national wholesale market

    nat_bal = pypsa.Network(
        snakemake.input.network_national_redispatch
        ) # national system post balancing
    
    logger.info('Computing Balancing Costs')

    bids = pd.read_csv(snakemake.input.bids)
    offers = pd.read_csv(snakemake.input.offers)




    logger.info('Computing Congestion Rents')
