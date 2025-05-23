# -*- coding: utf-8 -*-
# Copyright 2024-2025 Lukas Franken
# SPDX-FileCopyrightText: : 2024-2025 Lukas Franken
#
# SPDX-License-Identifier: MIT

"""
Summarizes the frontend data for the given day and IC.
"""

import logging

import pypsa
import numpy as np
import pandas as pd
from tqdm import tqdm
import geopandas as gpd
from _helpers import configure_logging, classify_north_south

logger = logging.getLogger(__name__)
idx = pd.IndexSlice


def process_balancing_data(df):
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


def get_weighted_avg_price(df):
    assert set(df.columns) == {'price', 'vol'}, 'Columns must be price and vol'
    assert not df.empty, 'DataFrame must not be empty'

    return (df['price'] * df['vol']).sum() / df['vol'].sum()


def get_unit_revenues(unit, who, bal):
    """
    Calculate revenue timeseries for a generation unit from multiple sources.
    
    Parameters:
    - unit: Generator name/ID
    - who: Wholesale market model network
    - bal: Balancing market model network
    
    Returns:
    - DataFrame with timeseries of revenue components
    """
    # Time step in hours (half-hourly)
    dt = 0.5

    # Initialize dataframe with snapshots as index
    snapshots = who.snapshots
    revenues = pd.DataFrame(index=snapshots, columns=[
        'wholesale', 'roc', 'cfd', 'ftr', 'offer_cost', 'bid_cost'
    ], data=0.0)

    if unit in who.links.index:
        dispatch = who.links_t.p0[unit]
        price0 = who.buses_t.marginal_price[who.links.loc[unit, 'bus0']]
        price1 = who.buses_t.marginal_price[who.links.loc[unit, 'bus1']]

        price_diff = abs(price0 - price1)
        revenues['wholesale'] = dispatch * price_diff * dt

        # revenues['total'] = revenues.sum(axis=1)

        return revenues
    

    if unit in who.storage_units.index:
        # Get storage unit dispatch
        who_dispatch = who.storage_units_t.p_dispatch[unit] - who.storage_units_t.p_store[unit]
        who_prices = who.buses_t.marginal_price[who.storage_units.loc[unit, 'bus']]
        revenues['wholesale'] = who_dispatch * who_prices * dt
        
        # Add ROC revenue if applicable
        if unit in roc.index:
            roc_value = roc.loc[unit, 'roc_value']
            revenues['roc'] = who_dispatch * roc_value * dt
            
        # revenues['total'] = revenues.sum(axis=1)
        return revenues

    # 1. Wholesale market revenue
    who_dispatch = who.generators_t.p[unit] if unit in who.generators_t.p else pd.Series(0, index=snapshots)
    who_prices = who.buses_t.marginal_price[who.generators.loc[unit, 'bus']]
    revenues['wholesale'] = who_dispatch * who_prices * dt

    bal_dispatch = bal.generators_t.p[unit] if unit in bal.generators_t.p else pd.Series(0, index=snapshots)

    # 2. ROC revenue (if applicable)
    if unit in roc.index:
        roc_value = roc.loc[unit, 'roc_value']
        revenues['roc'] = bal_dispatch * roc_value * dt

    # 3. CfD revenue (if applicable)
    if 'cfd' in globals() and unit in cfd.index:
        strike_price = cfd.loc[unit]
        
        # Check for negative price periods (at least 6 hours)
        negative_price_periods = who_prices < 0
        # Create a rolling window of 12 periods (6 hours with half-hourly data)
        rolling_negative = negative_price_periods.rolling(window=12).sum()
        # Identify snapshots where we shouldn't pay topup (preceded by 6+ hours of negative prices)
        no_topup_periods = rolling_negative >= 12
        
        # Calculate CfD top-up only for eligible periods
        topup_rates = strike_price - who_prices
        # Set topup to zero for periods following 6+ hours of negative prices
        topup_rates[no_topup_periods] = 0
        
        revenues['cfd'] = topup_rates * bal_dispatch * dt
    
    # 4. FTR revenue
    if len(who.buses) >= 20:
        # Get the unit's bus
        unit_bus = who.generators.loc[unit, 'bus']
        
        # For each timestep, check if there are exactly two unique prices
        for snapshot in snapshots:
            # Get unique prices across all buses for this snapshot
            unique_prices = who.buses_t.marginal_price.loc[:, gb_buses].loc[snapshot].unique()
            
            high_price = max(unique_prices)
            
            unit_price = who.buses_t.marginal_price.loc[snapshot, unit_bus]
            price_diff = abs(high_price - unit_price)
            
            # If unit is in the lower price zone, it gets FTR revenue
            revenues.loc[snapshot, 'ftr'] = who_dispatch[snapshot] * price_diff * dt
    
    # 5. Balancing revenue (if dispatch differs between models)
    if unit in bal.generators_t.p and unit in who.generators_t.p:
        dispatch_diff = bal.generators_t.p[unit] - who.generators_t.p[unit]
        bal_prices = bal.buses_t.marginal_price[bal.generators.loc[unit, 'bus']]
        
        # Calculate separately for up and down balancing
        up_balancing = dispatch_diff.clip(lower=0)
        down_balancing = dispatch_diff.clip(upper=0)
        
        # Balancing offers (turning up)
        up_revenue = up_balancing * bal_prices * dt
        
        # Apply ROC and CfD adjustments for turning up
        if unit in roc.index:
            roc_value = roc.loc[unit, 'roc_value']
            # For turning up, they would pay their ROC value
            up_revenue -= up_balancing * roc_value * dt
        
        elif unit in cfd.index:
            strike_price = cfd.loc[unit]
            # For turning up, they would pay their topup
            topup_rates = strike_price - who_prices
            up_revenue -= up_balancing * topup_rates * dt
        
        else:
            up_revenue = up_balancing * offer_price * dt

        revenues['offer_cost'] = up_revenue

        # Apply ROC and CfD adjustments for turning down
        if unit in roc.index:
            roc_value = roc.loc[unit, 'roc_value']
            # For turning down, they receive their ROC value per MWh curtailed
            down_revenue = down_balancing.abs() * roc_value * dt  # Negative * negative = positive

        elif unit in cfd.index:
            strike_price = cfd.loc[unit]
            # For turning down, they are forgoing their topup, so they get paid for that
            topup_rates = strike_price - who_prices
            down_revenue = down_balancing.abs() * topup_rates * dt  # Negative * positive = negative
        
        else:
            down_revenue = down_balancing.abs() * offer_price * dt

        revenues['bid_cost'] = down_revenue
    
    # revenues['total'] = revenues.sum(axis=1)

    return revenues


def calculate_dispatch_volumes(unit, who):
    """
    Calculate the total electricity volume sold by a unit in the wholesale market.
    
    Parameters
    ----------
    unit : str
        The name of the generation unit
    who : pypsa.Network
        The wholesale market network
        
    Returns
    -------
    pandas.Series
        Time series of electricity volumes sold in MWh
    """
    volumes = pd.Series(0, index=who.snapshots)
    
    # Check if unit is a generator
    if unit in who.generators_t.p:
        volumes = who.generators_t.p[unit] * 0.5
    
    # Check if unit is a storage unit
    elif unit in who.storage_units_t.p:
        volumes = who.storage_units_t.p[unit] * 0.5
    
    # Check if unit is an interconnector (link)
    elif unit in who.links_t.p0 and unit in who.links.index[who.links.carrier == 'interconnector']:
        volumes = who.links_t.p0[unit] * 0.5
    
    return volumes


if __name__ == "__main__":
    configure_logging(snakemake)

    day = snakemake.wildcards.day

    nat_who = pypsa.Network(snakemake.input.nat_who)
    nat_bal = pypsa.Network(snakemake.input.nat_bal)
    zon_who = pypsa.Network(snakemake.input.zon_who)
    zon_bal = pypsa.Network(snakemake.input.zon_bal)

    bids = pd.read_csv(snakemake.input.bids, index_col=[0,1])
    offers = pd.read_csv(snakemake.input.offers, index_col=[0,1])

    roc = pd.read_csv(snakemake.input.roc_values, index_col=0)
    cfd = pd.read_csv(snakemake.input.cfd_strike_prices, index_col=0)
    cfd.columns = pd.to_datetime(cfd.columns)
    cfd = cfd.loc[:,:day].iloc[:,-1]

    backup_balancing_prices = pd.read_csv(snakemake.input.default_balancing_prices, index_col=0)

    if not bids.empty:
        bid_price = get_weighted_avg_price(process_balancing_data(bids))
    else:
        bid_price = backup_balancing_prices.loc['disp', 'bids']

    if not offers.empty:
        offer_price = get_weighted_avg_price(process_balancing_data(offers))
    else:
        offer_price = backup_balancing_prices.loc['disp', 'offers']

    gb = gpd.read_file(snakemake.input.gb_shape).set_index('name')

    gdf = gpd.GeoDataFrame(
            zon_who.buses,
            geometry=gpd.points_from_xy(
                zon_who.buses['x'], zon_who.buses['y']
                )
            ).set_crs('EPSG:4326')

    mask = gdf.within(gb.loc['GB', 'geometry'])
    gb_buses = zon_who.buses.index[mask]


    all_units = nat_who.generators.index[nat_who.generators.carrier != 'local_market'].union(
        nat_who.storage_units.index.union(
            nat_who.links.index[nat_who.links.carrier == 'interconnector']
        )
    )

    all_carriers = pd.concat((
        nat_who.generators.loc[nat_who.generators.carrier != 'local_market', 'carrier'],
        nat_who.storage_units['carrier'],
        nat_who.links.loc[nat_who.links.carrier == 'interconnector', 'carrier']
    )) 

    all_revenues = list()

    for unit in tqdm(all_units):

        n = get_unit_revenues(unit, nat_who, nat_bal)
        z = get_unit_revenues(unit, zon_who, zon_bal)

        n.columns = pd.MultiIndex.from_product([[unit], ['national'], n.columns])
        z.columns = pd.MultiIndex.from_product([[unit], ['zonal'], z.columns])

        all_revenues.extend([n, z])

    all_revenues = pd.concat(all_revenues, axis=1)

    # refers to level 3 policy (equitable) from paper Locational Pricing Without Losers
    lv3_carriers = ['onwind', 'offwind', 'nuclear', 'hydro', 'cascade']

    available_shares = pd.Series(np.nan, all_carriers.unique())

    for carrier in lv3_carriers:

        national_total = all_revenues.sum().loc[
            idx[all_carriers.loc[all_carriers.isin(lv3_carriers)].index, 'national']].sum()
        zonal_total = all_revenues.sum().loc[idx[all_carriers.loc[all_carriers.isin(lv3_carriers)].index, 'zonal']].sum()

        available_shares.loc[carrier] = zonal_total / national_total

    for carrier in lv3_carriers:
        available_shares.loc[carrier] = zonal_total / national_total

    for carrier, share in available_shares.dropna().items():

        units = all_carriers.index[all_carriers == carrier]
        ss = all_revenues.loc[:, idx[units, 'national']].T.groupby(level=0).sum().T * share

        ss.columns = pd.MultiIndex.from_product([ss.columns, ['equitable'], ['level3']])

        all_revenues = pd.concat((
            all_revenues, ss
        ), axis=1)

    all_revenues = all_revenues.sort_index(axis=1)
    all_revenues.to_csv(snakemake.output.frontend_revenues)

    all_dispatch = pd.DataFrame(
        index=all_revenues.index,
        columns=pd.MultiIndex.from_product([all_units, ['national', 'zonal'], ['wholesale', 'redispatch']])
    )

    intercons = nat_who.links.index[nat_who.links.carrier == 'interconnector']
    intercon_dispatch = pd.DataFrame(
        index=all_revenues.index,
        columns=pd.MultiIndex.from_product([intercons, ['national', 'zonal']])
    )

    for unit in all_units:
        all_dispatch.loc[:, idx[unit, 'national', 'wholesale']] = calculate_dispatch_volumes(unit, nat_who)
        all_dispatch.loc[:, idx[unit, 'zonal', 'wholesale']] = calculate_dispatch_volumes(unit, zon_who)
        all_dispatch.loc[:, idx[unit, 'national', 'redispatch']] = calculate_dispatch_volumes(unit, nat_bal)
        all_dispatch.loc[:, idx[unit, 'zonal', 'redispatch']] = calculate_dispatch_volumes(unit, zon_bal)

        if unit in intercons:
            intercon_dispatch.loc[:, idx[unit, 'national']] = calculate_dispatch_volumes(unit, nat_bal)
            intercon_dispatch.loc[:, idx[unit, 'zonal']] = calculate_dispatch_volumes(unit, zon_bal)

    all_dispatch.sort_index(level=0, axis=1).to_csv(snakemake.output.frontend_dispatch)
    intercon_dispatch.sort_index(level=0, axis=1).to_csv(snakemake.output.frontend_dispatch_intercon)

    nat_who.generators_t.marginal_cost.to_csv(snakemake.output.frontend_marginal_costs)
    thermal_units = nat_who.generators.index[
        nat_who.generators.carrier.isin(['fossil', 'coal', 'biomass'])
        ]
    pd.concat({
        'national': nat_bal.generators_t.p.loc[:, thermal_units],
        'zonal': zon_bal.generators_t.p.loc[:, thermal_units]
    }, axis=1).to_csv(snakemake.output.frontend_thermal_dispatch)
