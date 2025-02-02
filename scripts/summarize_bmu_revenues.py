# -*- coding: utf-8 -*-
# Copyright 2024-2025 Lukas Franken
# SPDX-FileCopyrightText: : 2024-2025 Lukas Franken
#
# SPDX-License-Identifier: MIT

"""
Estimates the revenues for each BMUs including wholesale, (constraint
managing) balancing, cfd payments and (estimated) ROC revenue.

"""

import logging

logger = logging.getLogger(__name__)

import pypsa
import numpy as np
import pandas as pd

from _helpers import configure_logging

idx = pd.IndexSlice


def get_unit_wholesale_revenue(n, comp, unit):
    if isinstance(unit, str):
        return (
            getattr(n, comp+'_t').p[unit].multiply(
                0.5 * n.buses_t.marginal_price[getattr(n, comp).bus[unit]]
                )
            .rename(unit)
        )
    elif isinstance(unit, pd.Index):
        m_prices = n.buses_t.marginal_price[getattr(n, comp).bus[unit]]
        m_prices.columns = unit

        commodities = getattr(n, comp+'_t').p[unit]

        return (
            commodities.multiply(0.5 * m_prices)
        )
    
    else:
        assert False, 'unit must be either a string or a pd.Index'


def get_cfd_revenue(n, strike_prices):

    strike_prices = strike_prices.copy()
    strike_prices.columns = pd.to_datetime(strike_prices.columns)

    strike_prices = (
        strike_prices
        .loc[:,:n.snapshots[4]]
        .iloc[:,-1]
    )

    cfd_plants = strike_prices.index.intersection(n.generators.index)

    cfd_payments = pd.DataFrame(columns=cfd_plants, index=n.snapshots)

    for plant, strike_price in strike_prices.loc[cfd_plants].items():

        price_gap = (
            strike_price - n.buses_t.marginal_price[n.generators.loc[plant, 'bus']]
        )
        cfd_payments.loc[:, plant] = n.generators_t.p[plant] * price_gap * 0.5
    
    return cfd_payments


def process_daily_balancing_data(df):
    df = (
        df
        .stack()
        .unstack(1)
        .dropna()
        .groupby(level=1)
        .agg({'price': 'mean', 'vol': 'sum'})
        .sort_values('price')
    )
    return df


def make_north_south_split(
        n,
        carrier,
        comp,
        threshold=55.3
        ):

    comp_df = getattr(n, comp)
    comp_df['lat'] = comp_df.bus.map(n.buses.y)

    if not isinstance(carrier, str):
        mask = comp_df['carrier'].isin(carrier)
    else:
        mask = comp_df['carrier'].str.contains(carrier)

    north = comp_df.loc[mask & (comp_df['lat'] > threshold)].index
    south = comp_df.loc[mask & (comp_df['lat'] <= threshold)].index

    return north, south


def get_weighted_avg_price(df):
    assert set(df.columns) == {'price', 'vol'}, 'Columns must be price and vol'
    assert not df.empty, 'DataFrame must not be empty'

    return (df['price'] * df['vol']).sum() / df['vol'].sum()


if __name__ == '__main__':

    configure_logging(snakemake)

    actual_bids = pd.read_csv(snakemake.input['bids'], index_col=[0,1], parse_dates=True)

    if not actual_bids.empty:
        actual_bids = process_daily_balancing_data(actual_bids)
    else:
        actual_bids = pd.DataFrame(columns=['price', 'vol'])    


    actual_offers = pd.read_csv(snakemake.input['offers'], index_col=[0,1], parse_dates=True)

    if not actual_offers.empty:
        actual_offers = process_daily_balancing_data(actual_offers)
    else:
        actual_offers = pd.DataFrame(columns=['price', 'vol'])


    default_balancing_prices = pd.read_csv(
        snakemake.input['default_balancing_prices'],
        index_col=0
    )

    cfd_strike_prices = pd.read_csv(
        snakemake.input['cfd_strike_prices'],
        index_col=0,
        parse_dates=True
        )

    roc_values = pd.read_csv(
        snakemake.input['roc_values'],
        index_col=0).iloc[:,0]
    
    nodal_network = pypsa.Network(
        snakemake.input['network_nodal']
    )

    total_balancing_volumes = {}

    for layout in ['national', 'zonal']:

        who = pypsa.Network(
            snakemake.input[f'network_{layout}']
        )
        if not layout == 'nodal':
            bal = pypsa.Network(
                snakemake.input[f'network_{layout}_redispatch']
            )
        else:
            bal = who.copy()
        
        diff = pd.concat((
            bal.generators_t.p - who.generators_t.p,
        ), axis=1)

        total_balancing_volumes[layout] = {}

        total_balancing_volumes[layout]['offers'] = diff.clip(lower=0.).sum().mul(0.5).sum()
        total_balancing_volumes[layout]['bids'] = diff.clip(upper=0.).sum().mul(0.5).abs().sum()


    for layout in ['national', 'zonal', 'nodal']:

        logger.info(f'Processing revenues for {layout} layout.')

        who = pypsa.Network(
            snakemake.input[f'network_{layout}']
        )
        if not layout == 'nodal':
            bal = pypsa.Network(
                snakemake.input[f'network_{layout}_redispatch']
            )
        else:
            bal = who.copy()

        revenues = pd.DataFrame(
            0.,
            columns=pd.MultiIndex.from_product(
                [
                    ['north', 'south'],
                    ['wind', 'disp', 'water'],
                    ['wholesale', 'offers', 'bids', 'cfd', 'roc'],
                    ],
                ).append(
                    pd.MultiIndex.from_tuples(
                        [
                            ['total', 'intercon', 'wholesale']
                        ]
                    )
                ).append(
                    pd.MultiIndex.from_tuples(
                        [
                            ['total', 'load', 'wholesale']
                        ]
                    )
                ), index=who.snapshots)

        revenues.loc[:, idx['total', 'intercon', 'wholesale']] = (
            who
            .statistics
            .revenue(
                aggregate_time=False,
                comps='Link',
                groupby='carrier',
                )
            .loc['interconnector']
        )

        revenues.loc[:, idx['total', 'load', 'wholesale']] = (
            who
            .statistics
            .revenue(
                aggregate_time=False,
                comps='Load'
                )
            .loc['electricity']
        )

        # groups assets into north and south, dispatchable and non-dispatchable groups
        # revenue is assigned to groups
        wind_north, wind_south = make_north_south_split(
            bal,
            'wind',
            'generators',
            )

        disp_north, disp_south = make_north_south_split(
            bal,
            ['fossil', 'biomass', 'coal'],
            'generators',
            )

        water_north, water_south = make_north_south_split(
            nodal_network,
            ['cascade', 'hydro', 'PHS'],
            'storage_units',
            )

        # difference between wholesale and balancing model
        diff = pd.concat((
            bal.generators_t.p - who.generators_t.p,
        ), axis=1)

        ####              REVENUE OF SOUTHERN WIND GENERATORS

        # balancing offers of southern wind generators can be overestimated by
        # the model, we therefore ensure that excessive offers are 
        # reintepreted as offers contributed by dispatchable generators

        south_wind_offers = diff.loc[:, wind_south].sum(axis=1)
        south_wind_offers_total = south_wind_offers.sum()
        south_wind_offers_total_actual = (
            actual_offers
            .loc[idx[wind_south.intersection(actual_offers.index), 'vol']]
            .sum()
        )

        south_disp_offers = diff.loc[
            :, actual_offers.index.intersection(wind_south)
        ].sum(axis=1)

        if south_wind_offers_total > south_wind_offers_total_actual:

            keep_share = south_wind_offers_total_actual / south_wind_offers_total
            transfer_share = 1. - keep_share

        else:

            keep_share = south_wind_offers_total_actual / south_wind_offers_total
            transfer_share = 0.

        south_wind_transfer = south_wind_offers * transfer_share
        south_wind_offers *= keep_share

        try:
            south_wind_offers_price = get_weighted_avg_price(
                actual_offers.loc[wind_south.intersection(actual_offers.index)]
            )
            revenues.loc[:, idx['south', 'wind', 'offers']] = (
                south_wind_offers_price
                * south_wind_offers
                / 2
            )
        except:
            pass

        wind_south_dispatch = bal.generators_t.p.loc[
            :,
            wind_south.intersection(roc_values.index)
            ]

        wind_south_roc_revenue = wind_south_dispatch.multiply(
            roc_values.loc[wind_south_dispatch.columns] / 2
        )

        revenues.loc[:, idx['south', 'wind', 'roc']] = wind_south_roc_revenue.sum(axis=1)

        cfd = get_cfd_revenue(bal, cfd_strike_prices)
        cfd_revenue = cfd.loc[:, wind_south.intersection(cfd.columns)]
        revenues.loc[:, idx['south', 'wind', 'cfd']] = cfd_revenue.sum(axis=1).astype(np.float64)

        revenues.loc[:, idx['south', 'wind', 'wholesale']] = (
            get_unit_wholesale_revenue(
                who,
                'generators',
                wind_south)
            .sum(axis=1)
        )

        ####              REVENUE OF NORTHERN WIND GENERATORS

        revenues.loc[:, idx['north', 'wind', 'wholesale']] = (
            get_unit_wholesale_revenue(
                who,
                'generators',
                wind_north)
            .sum(axis=1)
        )

        logger.warning('Open question: Should CFD payments should be considered in the wholesale or balancing model?')
        cfd = get_cfd_revenue(who, cfd_strike_prices)
        cfd_revenue = cfd.loc[:, wind_north.intersection(cfd.columns)]

        revenues.loc[:, idx['north', 'wind', 'cfd']] = cfd_revenue.sum(axis=1).astype(np.float64)

        wind_north_dispatch = bal.generators_t.p.loc[
            :,
            wind_north.intersection(roc_values.index)
            ]

        wind_north_roc_revenue = wind_north_dispatch.multiply(
            roc_values.loc[wind_north_dispatch.columns] / 2
        )

        revenues.loc[:, idx['north', 'wind', 'roc']] = wind_north_roc_revenue.sum(axis=1)

        north_wind_bids = (
            diff
            .loc[:, wind_north]
            .sum(axis=1)
            .clip(upper=0)
            .abs()
        )

        if north_wind_bids.sum() > 0:

            actual_north_wind_bid_vol = actual_bids.loc[actual_bids.index.intersection(wind_north), 'vol']

            if actual_north_wind_bid_vol.sum() > 0:
                north_wind_bid_price = get_weighted_avg_price(
                    actual_bids.loc[actual_bids.index.intersection(wind_north), :]
                )

            else:
                north_wind_bid_price = default_balancing_prices.loc['wind', 'bids']
            
            revenues.loc[:, idx['north', 'wind', 'bids']] = (
                north_wind_bids
                * north_wind_bid_price
                / 2
            )

        north_wind_offers = (
            diff
            .loc[:, wind_north]
            .sum(axis=1)
            .clip(lower=0)
        )

        if north_wind_offers.sum() > 0:
            
            actual_north_wind_offer_vol = actual_offers.loc[
                actual_offers.index.intersection(wind_north), 'vol'
                ]

            if actual_north_wind_offer_vol.sum() > 0:
                north_wind_offer_price = get_weighted_avg_price(
                    actual_offers.loc[actual_offers.index.intersection(wind_north), :]
                )

            else:
                north_wind_offer_price = default_balancing_prices.loc['wind', 'offers']
            
            revenues.loc[:, idx['north', 'wind', 'offers']] = (
                north_wind_offers
                * north_wind_offer_price
                / 2
            )


        ####              REVENUE OF DISPATCHABLE GENERATORS

        disp_offers_actual = (
            actual_offers
            .loc[disp_south.union(disp_north).intersection(actual_offers.index)]
            )

        if not disp_offers_actual.empty:
            disp_offers_price = get_weighted_avg_price(disp_offers_actual)
        else:
            disp_offers_price = default_balancing_prices.loc['disp', 'offers']

        south_disp_offers = (
            diff.loc[
                # :, actual_offers.index.intersection(wind_south)
                :, disp_south
            ]
            .sum(axis=1)
            .clip(lower=0.)
        )

        assert (
            (south_wind_transfer >= 0).all(),
            'Negative transfer of wind offers to dispatchable generators'
        )

        south_disp_offers += south_wind_transfer

        revenues.loc[:, idx['south', 'disp', 'offers']] = (
            south_disp_offers * disp_offers_price / 2
        )

        disp_bids_actual = (
            actual_bids
            .loc[disp_south.union(disp_north).intersection(actual_bids.index)]
            )

        if not disp_bids_actual.empty:
            disp_bids_price = get_weighted_avg_price(disp_bids_actual)
        else:
            disp_bids_price = default_balancing_prices.loc['disp', 'bids']

        south_disp_bids = (
            diff.loc[
                # :, actual_bids.index.intersection(wind_south)
                :, disp_south
            ]
            .sum(axis=1)
            .clip(upper=0.)
            .abs()
        )

        revenues.loc[:, idx['south', 'disp', 'bids']] = (
            south_disp_bids * disp_bids_price / 2
        )

        north_disp_offers = (
            diff.loc[:, disp_north]
            .sum(axis=1)
            .clip(lower=0.)
        )

        north_disp_bids = (
            diff.loc[:, disp_north]
            .sum(axis=1)
            .clip(upper=0.)
            .abs()
        )

        revenues.loc[:, idx['north', 'disp', 'offers']] = (
            north_disp_offers * disp_offers_price / 2
        )

        revenues.loc[:, idx['north', 'disp', 'bids']] = (
            north_disp_bids * disp_bids_price / 2
        )

        revenues.loc[:, idx['south', 'disp', 'wholesale']] = (
            get_unit_wholesale_revenue(
                who,
                'generators',
                disp_south)
            .sum(axis=1)
        )

        revenues.loc[:, idx['north', 'disp', 'wholesale']] = (
            get_unit_wholesale_revenue(
                who,
                'generators',
                disp_north)
            .sum(axis=1)
        )


        def get_water_balancing_revenue(who, units, mode, actual_trades):
            '''
            who: wholesale model
            units: units to consider
            mode: 'offers' or 'bids'
            '''

            logger.warning((
                'Implementation of hydropower revenue assumes that hydro '
                'wholesale trading cant be reversed in the balancing market.'
                ))

            assert mode in ['offers', 'bids'], 'mode must be either offers or bids'

            actual = actual_trades.loc[actual_trades.index.intersection(units)]

            total_actual_volume = actual['vol'].sum()

            if total_actual_volume == 0:
                return pd.Series(0., index=who.snapshots)
            
            actual_revenue = (actual['price'] * actual['vol']).sum()

            if mode == 'offers':
                kwargs = {'lower': 0.}
            else:
                kwargs = {'upper': 0.}

            trading_profile = who.storage_units_t.p[units].sum(axis=1).clip(**kwargs).abs()

            if trading_profile.sum() == 0:
                return pd.Series(actual_revenue / len(who.snapshots), index=who.snapshots)
            
            trading_profile = trading_profile.div(trading_profile.sum())

            return (
                trading_profile
                .multiply(actual_revenue)
            )

        revenues.loc[:, idx['south', 'water', 'wholesale']] = (
            get_unit_wholesale_revenue(
                who,
                'storage_units',
                water_south)
            .sum(axis=1)
        )

        revenues.loc[:, idx['north', 'water', 'wholesale']] = (
            get_unit_wholesale_revenue(
                who,
                'storage_units',
                water_north)
            .sum(axis=1)
        )


        if not layout == 'nodal':

            bid_reduction_factor = (
                total_balancing_volumes[layout]['bids'] /
                total_balancing_volumes['national']['bids']
            )
            offer_reduction_factor = (
                total_balancing_volumes[layout]['offers'] /
                total_balancing_volumes['national']['offers']
            )

            revenues.loc[:, idx['north', 'water', 'bids']] = (
                get_water_balancing_revenue(
                    who,
                    water_north,
                    'bids',
                    actual_bids)
                    .mul(bid_reduction_factor)
            )

            revenues.loc[:, idx['north', 'water', 'offers']] = (
                get_water_balancing_revenue(
                    who,
                    water_north,
                    'offers',
                    actual_offers)
                    .mul(offer_reduction_factor)
            )

            revenues.loc[:, idx['south', 'water', 'bids']] = (
                get_water_balancing_revenue(
                    who,
                    water_south,
                    'bids',
                    actual_bids)
                    .mul(bid_reduction_factor)
            )

            revenues.loc[:, idx['south', 'water', 'offers']] = (
                get_water_balancing_revenue(
                    who,
                    water_south,
                    'offers',
                    actual_offers)
                    .mul(offer_reduction_factor)
            )

        def get_water_roc_revenue(
                n,
                actual_bids,
                actual_offers,
                units,
                roc_values
                ):
            '''
            n: network
            actual_bids: actual accepted bids of that day
            actual_offers: actual accepted offers of that day
            units: units to consider
            roc_values: ROC values
            '''

            units = units.copy().intersection(roc_values.index)

            if units.empty:
                return pd.Series(0., index=n.snapshots)

            bid_volume = actual_bids.loc[units.intersection(actual_bids.index)].vol.sum()
            offer_volume = actual_offers.loc[units.intersection(actual_offers.index)].vol.sum()

            units_dispatch = n.storage_units_t.p[units].clip(lower=0.)

            total_dispatch = units_dispatch.sum().sum() * 0.5

            units_dispatch *= (
                (total_dispatch + offer_volume - bid_volume) / total_dispatch
                )
            
            return units_dispatch.multiply(
                roc_values.loc[units_dispatch.columns] / 2
                ).sum(axis=1)

        revenues.loc[:, idx['south', 'water', 'roc']] = (
            get_water_roc_revenue(
                who,
                actual_bids,
                actual_offers,
                water_south,
                roc_values
                )
        )

        revenues.to_csv(snakemake.output[f'bmu_revenues_{layout}'])

        # --- Track Dispatch
        #
        # Create a dataframe with the same multi-index columns as revenues.
        dispatch = pd.DataFrame(
            0.,
            index=who.snapshots,
            columns=revenues.columns
        )

        # WIND:
        # Wholesale: use the wholesale model generator dispatch.
        dispatch.loc[:, idx['south', 'wind', 'wholesale']] = who.generators_t.p.loc[:, wind_south].sum(axis=1)
        dispatch.loc[:, idx['north', 'wind', 'wholesale']] = who.generators_t.p.loc[:, wind_north].sum(axis=1)

        # Balancing offers / bids:
        # For south wind, we already adjusted offers (and transferred some to disp).
        dispatch.loc[:, idx['south', 'wind', 'offers']] = south_wind_offers
        dispatch.loc[:, idx['south', 'wind', 'bids']] = diff.loc[:, wind_south].sum(axis=1).clip(upper=0)
        dispatch.loc[:, idx['north', 'wind', 'offers']] = diff.loc[:, wind_north].sum(axis=1).clip(lower=0)
        dispatch.loc[:, idx['north', 'wind', 'bids']] = diff.loc[:, wind_north].sum(axis=1).clip(upper=0)

        # CFD: use the dispatch of plants for which CFD strike prices exist.
        csouth = wind_south.intersection(cfd_strike_prices.index)
        cnorth = wind_north.intersection(cfd_strike_prices.index)
        if not csouth.empty:
            dispatch.loc[:, idx['south', 'wind', 'cfd']] = bal.generators_t.p.loc[:, csouth].sum(axis=1)
        else:
            dispatch.loc[:, idx['south', 'wind', 'cfd']] = 0.
        if not cnorth.empty:
            dispatch.loc[:, idx['north', 'wind', 'cfd']] = who.generators_t.p.loc[:, cnorth].sum(axis=1)
        else:
            dispatch.loc[:, idx['north', 'wind', 'cfd']] = 0.

        # ROC: use the balancing model’s dispatch for those units with ROC values.
        rnorth = wind_north.intersection(roc_values.index)
        rsouth = wind_south.intersection(roc_values.index)
        if not rsouth.empty:
            dispatch.loc[:, idx['south', 'wind', 'roc']] = bal.generators_t.p.loc[:, rsouth].sum(axis=1)
        else:
            dispatch.loc[:, idx['south', 'wind', 'roc']] = 0.
        if not rnorth.empty:
            dispatch.loc[:, idx['north', 'wind', 'roc']] = bal.generators_t.p.loc[:, rnorth].sum(axis=1)
        else:
            dispatch.loc[:, idx['north', 'wind', 'roc']] = 0.

        # DISPATCHABLE (disp):
        dispatch.loc[:, idx['south', 'disp', 'wholesale']] = who.generators_t.p.loc[:, disp_south].sum(axis=1)
        dispatch.loc[:, idx['north', 'disp', 'wholesale']] = who.generators_t.p.loc[:, disp_north].sum(axis=1)

        dispatch.loc[:, idx['south', 'disp', 'offers']] = diff.loc[:, disp_south].sum(axis=1).clip(lower=0) + south_wind_transfer
        dispatch.loc[:, idx['north', 'disp', 'offers']] = diff.loc[:, disp_north].sum(axis=1).clip(lower=0)

        dispatch.loc[:, idx['south', 'disp', 'bids']] = diff.loc[:, disp_south].sum(axis=1).clip(upper=0)
        dispatch.loc[:, idx['north', 'disp', 'bids']] = diff.loc[:, disp_north].sum(axis=1).clip(upper=0)

        # WATER:
        dispatch.loc[:, idx['south', 'water', 'wholesale']] = who.storage_units_t.p.loc[:, water_south].sum(axis=1)
        dispatch.loc[:, idx['north', 'water', 'wholesale']] = who.storage_units_t.p.loc[:, water_north].sum(axis=1)

        dispatch.loc[:, idx['south', 'water', 'offers']] = who.storage_units_t.p.loc[:, water_south].sum(axis=1).clip(lower=0)
        dispatch.loc[:, idx['south', 'water', 'bids']] = who.storage_units_t.p.loc[:, water_south].sum(axis=1).clip(upper=0)
        dispatch.loc[:, idx['north', 'water', 'offers']] = who.storage_units_t.p.loc[:, water_north].sum(axis=1).clip(lower=0)
        dispatch.loc[:, idx['north', 'water', 'bids']] = who.storage_units_t.p.loc[:, water_north].sum(axis=1).clip(upper=0)

        # For water ROC, mimic the scaling in get_water_roc_revenue but report the scaled dispatch.
        def compute_water_roc_dispatch(network, units, actual_bids, actual_offers):
            units = units.intersection(network.storage_units.index)

            if units.empty:
                return pd.Series(0., index=network.snapshots)

            units_dispatch = network.storage_units_t.p.loc[:, units].clip(lower=0)
            total_dispatch = units_dispatch.sum().sum() * 0.5

            if total_dispatch == 0:
                return pd.Series(0., index=network.snapshots)

            bid_volume = actual_bids.loc[units.intersection(actual_bids.index), 'vol'].sum() if not actual_bids.empty else 0.
            offer_volume = actual_offers.loc[units.intersection(actual_offers.index), 'vol'].sum() if not actual_offers.empty else 0.

            scale_factor = (total_dispatch + offer_volume - bid_volume) / total_dispatch
            scaled_dispatch = units_dispatch * scale_factor

            return scaled_dispatch.sum(axis=1)


        dispatch.loc[:, idx['south', 'water', 'roc']] = compute_water_roc_dispatch(who, water_south, actual_bids, actual_offers)
        dispatch.loc[:, idx['north', 'water', 'roc']] = compute_water_roc_dispatch(who, water_north, actual_bids, actual_offers)

        # TOTAL INTERCONNECTOR & LOAD:
        intercon_links = who.links.index[who.links.carrier == 'interconnector']
        if not intercon_links.empty:
            dispatch.loc[:, idx['total', 'intercon', 'wholesale']] = who.links_t.p0.loc[:, intercon_links].sum(axis=1)
        else:
            dispatch.loc[:, idx['total', 'intercon', 'wholesale']] = 0.

        if 'carrier' in who.loads.columns:
            load_units = who.loads.index[who.loads.carrier=='electricity']
            dispatch.loc[:, idx['total', 'load', 'wholesale']] = who.loads_t.p.loc[:, load_units].sum(axis=1)
        else:
            dispatch.loc[:, idx['total', 'load', 'wholesale']] = who.loads_t.p.sum(axis=1)

        dispatch.to_csv(snakemake.output[f'bmu_dispatch_{layout}'])
