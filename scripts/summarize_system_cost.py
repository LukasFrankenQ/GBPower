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

    congestion_rent = pd.Series(0, index=n.snapshots)

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
    
    return congestion_rent.mul(-1)


def get_roc_cost(bal, roc):
    '''
    Calculate the ROC cost for the given network and ROC prices.
    ROC payments are only done when the farms didnt curtail their production,
    therefore bal refers to the network after redispatch (in a
    nodal market insert the nodal wholesale market).
    '''

    roc_payments = pd.Series(0, index=bal.snapshots)

    for plant, roc_price in roc.items():

        if plant in bal.generators.index:
            activity = bal.generators_t.p[plant]

        elif plant in bal.storage_units.index:
            activity = bal.storage_units_t.p[plant]

        else:
            continue

        roc_payments += roc_price * activity * 0.5

    return roc_payments


def get_cfd_payments(n, strike_prices):

    strike_prices = strike_prices.copy()
    strike_prices.columns = pd.to_datetime(strike_prices.columns)

    strike_prices = (
        strike_prices
        .loc[:,:n.snapshots[4]]
        .iloc[:,-1]
    )

    cfd_payments = pd.Series(0, index=n.snapshots)

    for plant, strike_price in strike_prices.loc[
        strike_prices.index.intersection(n.generators.index)
        ].items():

        price_gap = (
            strike_price - n.buses_t.marginal_price[n.generators.loc[plant, 'bus']]
        )
        cfd_payments += n.generators_t.p[plant] * price_gap * 0.5

    return cfd_payments


def get_bidding_volume(nat, bal):
    """
    Returns the bidding volume of the network.

    Parameters:
        nat(pypsa.Network): The (national or zonal) wholesale market dispatch optimum.
        bal(pypsa.Network): The network dispatch after balancing.
    """

    lat_bidding_threshold = 55.3 # groups dispatchable generators into north and south

    wind = nat.generators.index[nat.generators.carrier.str.contains('wind')]
    water = nat.storage_units.index[nat.storage_units.carrier.isin(['cascade', 'hydro'])]
    solar = nat.generators.index[nat.generators.carrier.str.contains('solar')]

    gen = bal.generators
    gen['y'] = gen.bus.map(bal.buses.y)

    dispatchable_south = gen.loc[
        (gen.carrier.isin(['fossil', 'biomass', 'coal'])) &
        (gen['y'] <= lat_bidding_threshold)
    ].index
    dispatchable_north = gen.loc[
        (gen.carrier.isin(['fossil', 'biomass', 'coal'])) &
        (gen['y'] > lat_bidding_threshold)
    ].index

    bidding_volume = pd.Series(0, nat.snapshots)

    for plant in [wind, solar, dispatchable_south, dispatchable_north]:
        
        bidding_volume += (
            nat.generators_t.p[plant].sum(axis=1) -
            bal.generators_t.p[plant].sum(axis=1)
        ).clip(lower=0)
    
    water = (
        water
        .intersection(bal.storage_units_t.p.columns)
        .intersection(nat.storage_units_t.p.columns)
    )
    bidding_volume += (
        nat.storage_units_t.p[water].sum(axis=1) -
        bal.storage_units_t.p[water].sum(axis=1)
    ).clip(lower=0)
    
    return bidding_volume * 0.5


def get_balancing_cost(
        wholesale,
        balanced,
        actual_bids,
        actual_offers
        ):

    bidding_volume = get_bidding_volume(wholesale, balanced)
    bidding_volume.index = pd.to_datetime(bidding_volume.index, utc=True)

    balancing_cost_shape = bidding_volume.copy().div(bidding_volume.sum())

    bid_default_cost = 50 # £/MWh
    offer_default_cost = 90 # £/MWh

    process_data = lambda df: (
        df
        .stack()
        .unstack(1)
        .dropna()
        .reset_index(drop=True)
        .sort_values('price')
    )

    try:
        actual_bids = process_data(actual_bids)
    except KeyError:
        actual_bids = pd.DataFrame(0, index=['dummy_bmu'], columns=['vol', 'price'])

    try:
        actual_offers = process_data(actual_offers)
    except KeyError:
        actual_offers = pd.DataFrame(0, index=['dummy_bmu'], columns=['vol', 'price'])

    def is_faulty_price(prices):

        nine_check = lambda num: str(int(num)).count('9') >= 3
        magnitude_check = lambda num: num > 5000

        return prices.apply(nine_check) | prices.apply(magnitude_check)

    # there appears to be a nonzero risk of faulty prices of vast magnitude to
    # sneak into bid/offer prices.
    # Appears to be a very rare occurance, but still needs correction
    # due to huge impact.
    actual_bids.loc[
        mask,
        'price'
    ] = actual_bids.loc[~(mask := is_faulty_price(actual_bids['price'])), 'price'].median()

    actual_offers.loc[
        mask,
        'price'
    ] = actual_offers.loc[~(mask := is_faulty_price(actual_offers['price'])), 'price'].median()


    actual_bids['cumvol'] = actual_bids['vol'].cumsum()
    actual_offers['cumvol'] = actual_offers['vol'].cumsum()

    model_vol = bidding_volume.sum()
    
    costs = {
        'bid': 0.,
        'offer': 0.,
    }

    for mode in ['bid', 'offer']:
        
        actual = actual_bids if mode == 'bid' else actual_offers
        default_cost = bid_default_cost if mode == 'bid' else offer_default_cost

        if actual['vol'].sum() == 0:
            costs[mode] += default_cost * model_vol
            break

        elif model_vol > actual['vol'].sum():
            costs[mode] += actual['price'].dot(actual['vol'])
            costs[mode] += actual['price'].iloc[-1] * (model_vol - actual['vol'].sum())

        else:
            ss = actual.loc[actual['cumvol'] < model_vol]
            remainder = actual.loc[actual['cumvol'] >= model_vol]

            costs[mode] += ss['price'].dot(ss['vol'])
            costs[mode] += (model_vol - ss['vol'].sum()) * remainder['price'].iloc[0]

    return balancing_cost_shape * costs['bid'], balancing_cost_shape * costs['offer']


if __name__ == '__main__':

    configure_logging(snakemake)

    nod = pypsa.Network(snakemake.input.network_nodal) # nodal wholesale market

    nat = pypsa.Network(snakemake.input.network_national) # national wholesale market
    nat_bal = pypsa.Network(snakemake.input.network_national_redispatch) # national system after redispatch

    zon = pypsa.Network(snakemake.input.network_zonal) # zonal wholesale market
    zon_bal = pypsa.Network(snakemake.input.network_zonal_redispatch) # zonal system after redispatch

    logger.info('Computing Balancing Costs')

    bids = pd.read_csv(snakemake.input.bids, index_col=[0,1], parse_dates=True)
    offers = pd.read_csv(snakemake.input.offers, index_col=[0,1], parse_dates=True)

    cfd_strike_prices = pd.read_csv(
        snakemake.input.cfd_strike_prices,
        index_col=0,
        parse_dates=True
        )

    roc_values = pd.read_csv(snakemake.input.roc_values, index_col=0).iloc[:,0]

    congestion_rent_compensation_share = 0.5
    # share of congestion rents that are not interpreted
    # and instead are assumed to be paid to the generators

    bidcosts, offercosts = get_balancing_cost(nat, nat_bal, bids, offers)
    total_national_costs = pd.DataFrame(
        {
            'wholesale': get_wholesale_cost(nat).values,
            'congestion_rent': get_congestion_rents(nat).values * congestion_rent_compensation_share,
            'cfd_payments': get_cfd_payments(nat, cfd_strike_prices).values,
            'roc_payments': get_roc_cost(nat_bal, roc_values).values,
            'offer_cost': offercosts.values,
            'bid_cost': bidcosts.values,
            'balancing_volume': get_bidding_volume(nat, nat_bal).values
        },
        index=nat.snapshots
    ).mul(1e-6)

    bidcosts, offercosts = get_balancing_cost(zon, zon_bal, bids, offers)
    total_zonal_costs = pd.DataFrame(
        {
            'wholesale': get_wholesale_cost(zon).values,
            'congestion_rent': get_congestion_rents(zon).values * congestion_rent_compensation_share,
            'cfd_payments': get_cfd_payments(zon, cfd_strike_prices).values,
            'roc_payments': get_roc_cost(zon_bal, roc_values).values,
            'offer_cost': offercosts.values,
            'bid_cost': bidcosts.values,
            'balancing_volume': get_bidding_volume(zon, zon_bal).values
        },
        index=nat.snapshots
    ).mul(1e-6)

    total_nodal_costs = pd.DataFrame(
        {
            'wholesale': get_wholesale_cost(nod).values,
            'congestion_rent': get_congestion_rents(nod).values * congestion_rent_compensation_share,
            'cfd_payments': get_cfd_payments(nod, cfd_strike_prices).values,
            'roc_payments': get_roc_cost(nod, roc_values).values,
            'offer_cost': 0,
            'bid_cost': 0,
            'balancing_volume': 0,
        },
        index=nat.snapshots
    ).mul(1e-6)

    pd.concat((
        total_nodal_costs.stack().rename('nodal'),
        total_zonal_costs.stack().rename('zonal'),
        total_national_costs.stack().rename('national')
    ), axis=1).to_csv(snakemake.output.system_cost_summary)


    logger.info("Storing marginal prices")
    other_countries = pd.Index([
        'Denmark', 'Belgium', 'Netherlands', 'France', 'Norway',
    ])

    def prep_marginal_prices(n, name):
        mp = n.buses_t.marginal_price
        mp.drop(columns=other_countries.intersection(mp.columns), inplace=True)

        return (
            mp
            .stack()
            .rename(name)
            .to_frame()
            .unstack(1)
        )

    pd.concat((
            prep_marginal_prices(nat, 'national'),
            prep_marginal_prices(zon, 'zonal'),
            prep_marginal_prices(nod, 'nodal'),
        ), axis=1).to_csv(snakemake.output.marginal_prices)
