# -*- coding: utf-8 -*-
# Copyright 2024-2025 Lukas Franken
# SPDX-FileCopyrightText: : 2024-2025 Lukas Franken
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

from _helpers import configure_logging, classify_north_south


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

        bus = n.generators.loc[plant, 'bus']
        bus_mp = n.buses_t.marginal_price[bus]
        # Identify snapshots where the marginal price has been negative for the last 6 hours (i.e. 12 consecutive 30-min steps)
        negative_streak = bus_mp.rolling(window=12, min_periods=12).max() < 0
        payment_mask = ~negative_streak.fillna(False)  # True if payment is allowed
        price_gap = strike_price - bus_mp
        cfd_payments += n.generators_t.p[plant] * price_gap * 0.5 * payment_mask

    return cfd_payments


def get_bidding_volume(nat, bal):
    """
    Returns the bidding volume of the network.

    Parameters:
        nat(pypsa.Network): The (national or zonal) wholesale market dispatch optimum.
        bal(pypsa.Network): The network dispatch after balancing.
    """

    wind = nat.generators.index[nat.generators.carrier.str.contains('wind')]
    water = nat.storage_units.index[nat.storage_units.carrier.isin(['cascade', 'hydro'])]
    solar = nat.generators.index[nat.generators.carrier.str.contains('solar')]

    gen = bal.generators

    coords = bal.buses.loc[gen.bus, ['x', 'y']]
    coords.index = gen.index

    gen['region'] = coords.apply(
        lambda row: classify_north_south(row['x'], row['y']), axis=1
        )

    dispatchable_south = gen.loc[
        (gen.carrier.isin(['fossil', 'biomass', 'coal', 'backup'])) &
        (gen['region'] == 'south')
    ].index
    dispatchable_north = gen.loc[
        (gen.carrier.isin(['fossil', 'biomass', 'coal', 'backup'])) &
        (gen['region'] == 'north')
    ].index

    bidding_volume = pd.Series(0, nat.snapshots)

    for plant in [wind, solar, dispatchable_south, dispatchable_north]:

        nat_dispatch = nat.generators_t.p[plant.intersection(nat.generators.index)].sum(axis=1)
        
        bidding_volume += (
            nat_dispatch -
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


def _curve_cost(model_vol, curve):
    """Area under a pooled marginal price-vs-cumulative-volume curve, 0 -> model_vol (£).

    `curve` has columns cumvol_MWh (ascending bin edges) and marginal_price_GBP_per_MWh,
    pooled from all 2022-2025 days (prerun_scripts/build_balancing_curves.py). Volume beyond
    the largest historical bin is held flat at the top-of-stack price (np.interp end-hold),
    so a model redispatch volume larger than any single historical day is still priced from
    real high-volume offers — not the single most expensive offer of one day (the old
    `actual['price'].iloc[-1]` extrapolation that dominated 2028-29 costs).
    """
    if model_vol <= 0:
        return 0.0
    v = curve['cumvol_MWh'].to_numpy(dtype=float)
    p = curve['marginal_price_GBP_per_MWh'].to_numpy(dtype=float)
    grid = np.linspace(0.0, model_vol, 4000)
    pg = np.interp(grid, v, p)        # flat-held at p[0] below v[0] and p[-1] above v[-1]
    return float(np.sum(0.5 * (pg[:-1] + pg[1:]) * np.diff(grid)))   # trapezoidal area


def get_balancing_cost(
        wholesale,
        balanced,
        offer_curve,
        bid_curve,
        ):
    """Cost of the model's redispatch volume, priced against the pooled 2022-2025 bid and
    offer curves (area under each curve up to the model volume). Returns per-snapshot
    (bid_cost, offer_cost) series, distributed across snapshots by the redispatch-volume shape.
    """
    bidding_volume = get_bidding_volume(wholesale, balanced)
    bidding_volume.index = pd.to_datetime(bidding_volume.index, utc=True)

    model_vol = bidding_volume.sum()
    if model_vol <= 0:
        zero = bidding_volume * 0.0
        return zero, zero

    balancing_cost_shape = bidding_volume.div(model_vol)

    bid_cost = _curve_cost(model_vol, bid_curve)
    offer_cost = _curve_cost(model_vol, offer_curve)

    return balancing_cost_shape * bid_cost, balancing_cost_shape * offer_cost


if __name__ == '__main__':

    configure_logging(snakemake)

    nod = pypsa.Network(snakemake.input.network_nodal) # nodal wholesale market

    nat = pypsa.Network(snakemake.input.network_national) # national wholesale market
    nat_bal = pypsa.Network(snakemake.input.network_national_redispatch) # national system after redispatch

    zon = pypsa.Network(snakemake.input.network_zonal) # zonal wholesale market
    zon_bal = pypsa.Network(snakemake.input.network_zonal_redispatch) # zonal system after redispatch

    logger.info('Computing Balancing Costs')

    # Pooled 2022-2025 bid/offer curves (price vs within-day cumulative volume) replace the
    # per-day stack + single-most-expensive-offer extrapolation. See build_balancing_curves.py.
    offer_curve = pd.read_csv(snakemake.input.offer_curve)
    bid_curve = pd.read_csv(snakemake.input.bid_curve)

    cfd_strike_prices = pd.read_csv(
        snakemake.input.cfd_strike_prices,
        index_col=0,
        parse_dates=True
        )

    roc_values = pd.read_csv(snakemake.input.roc_values, index_col=0).iloc[:,0]

    # 1 reflects that no money is yet allocated for grandfathering
    congestion_rent_compensation_share = 1.

    bidcosts, offercosts = get_balancing_cost(nat, nat_bal, offer_curve, bid_curve)
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

    bidcosts, offercosts = get_balancing_cost(zon, zon_bal, offer_curve, bid_curve)
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
