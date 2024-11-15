# -*- coding: utf-8 -*-
# Copyright 2024-2024 Lukas Franken (University of Edinburgh, Octopus Energy)
# SPDX-FileCopyrightText: : 2024-2024 Lukas Franken
#
# SPDX-License-Identifier: MIT
"""
Inserts generation, storage and loads into the network

"""


import logging

logger = logging.getLogger(__name__)

import yaml
import pypsa
import numpy as np
import pandas as pd

from _helpers import configure_logging


def scale_merit_order(
        n,
        dah,
        collective=False,
        smooth_dah=True,
        ):
    """
    Scale the marginal cost of fossil and biomass generators by the wholesale price

    if collective is True, the scaling factor is uniform over all network snapshots
    and given by the ratio of the 
        weighted average wholesale price to the
        weighted average marginal_cost of the marginal unit.

    if collective is False, the scaling factor is calculated for each period separately.

    smooth_dah applies a smoothing filter to the wholesale price time series
    to prevent erratic behaviour in the marginal cost of the fossil and biomass generators.
    """

    gb_buses = n.buses.loc[n.buses.country == 'GB'].index

    if smooth_dah:
        dah = (
            dah.rolling(8, center=True, min_periods=1)
            .mean()
        )

    # technologies that see their marginal price scaled by the wholesale price
    scalers = ['fossil', 'biomass']
    scaling_units = n.generators[n.generators.carrier.isin(scalers)].index

    marginal_unit_cost = pd.Series(np.nan, index=n.snapshots)

    for period, dt in enumerate(n.snapshots):

        ints_marginal_cost = []
        ints_marginal_cap = []
        found_countries = []

        for ic, country in interconnection_countries.items():
            try:
                ints_marginal_cost.append(n.generators_t.marginal_cost[country.lower() + "_local_market"].iloc[period])
                ints_marginal_cap.append(n.links.p_nom.loc[ic])
                found_countries.append(country)

            except KeyError:
                continue

        max_avail_gen = pd.concat((
            n.generators.loc[n.generators.bus.isin(gb_buses), ['marginal_cost', 'p_nom', 'carrier']],
            n.storage_units[['marginal_cost', 'p_nom', 'carrier']],
            pd.DataFrame({
                "marginal_cost": ints_marginal_cost,
                'p_nom': ints_marginal_cap,
                'carrier': ['interconnector'] * len(ints_marginal_cost)
                }, index=found_countries)
        ))

        intermittents = n.generators_t.p_max_pu.columns.intersection(max_avail_gen.index)
        max_avail_gen.loc[intermittents, "p_nom"] *= n.generators_t.p_max_pu[intermittents].iloc[period]

        max_avail_gen = max_avail_gen.sort_values(by=['marginal_cost', 'carrier'])

        max_avail_gen['cum_p_nom'] = max_avail_gen['p_nom'].cumsum()

        load = n.loads_t.p_set[gb_buses].iloc[period].sum()
        marginal_unit = max_avail_gen[max_avail_gen['cum_p_nom'] > load].iloc[0].name
        mc = max_avail_gen.loc[marginal_unit, 'marginal_cost']

        if isinstance(mc, pd.Series): 
            mc = mc.iloc[0]

        marginal_unit_cost.loc[dt] = mc

        if not collective:

            if mc <= 1 or dah.iloc[period, 0] <= 1:
                scaling_factor = 1.
            else:
                scaling_factor = dah.iloc[period, 0] / mc

            new_marginal_cost_units = (
                scaling_units
                .difference(n.generators_t.marginal_cost.columns) 
            )

            n.generators_t.marginal_cost = pd.concat((
                n.generators_t.marginal_cost,
                pd.DataFrame(0., index=n.snapshots, columns=new_marginal_cost_units)
            ), axis=1)

            n.generators_t.marginal_cost.loc[n.snapshots[period], scaling_units] = (
                n.generators.loc[scaling_units, 'marginal_cost'] * scaling_factor
            )

    if not collective:
        return

    # calculate the scaling factor
    total_demand = n.loads_t.p_set[gb_buses].sum(axis=1)

    weighted_avg_mc = (marginal_unit_cost * total_demand).sum() / total_demand.sum()
    weighted_avg_dah = (dah.iloc[:, 0] * total_demand).sum() / total_demand.sum()

    if weighted_avg_mc <= 1 or weighted_avg_dah <= 1:
        scaling_factor = 1.
    else:
        scaling_factor = weighted_avg_dah / weighted_avg_mc

    n.generators.loc[scaling_units, 'marginal_cost'] *= scaling_factor


def add_wind(
        n,
        bmus,
        pn,
        mel,
        cfds,
        rocs,
        carrier=None,
    ):

    plants = (
        bmus[bmus['carrier'] == carrier]
        .index
        .intersection(pn.columns)
    )

    logger.info(f'Adding {len(plants)} {carrier} generators...')

    n.add(
        "Generator",
        plants,
        bus=bmus.loc[plants, 'bus'],
        carrier=carrier,
        p_nom=pn[plants].max(),
        marginal_cost=np.nan,
        efficiency=1,
        p_max_pu=pn[plants].div(pn[plants].max()).replace(np.nan, 0),
    )

    roc_generators = rocs.index.intersection(plants)
    n.generators.loc[roc_generators, 'marginal_cost'] = - rocs[roc_generators]

    cfd_generators = cfds.index.intersection(plants)
    n.generators.loc[cfd_generators, 'marginal_cost'] = 0.


def add_onshore_wind(*args):
    add_wind(*args, carrier='onwind')


def add_offshore_wind(*args):
    add_wind(*args, carrier='offwind')


def add_solar(
        n,
        bmus,
        pn,
    ):

    plants = (
        bmus[bmus['carrier'] == 'solar']
        .index
        .intersection(pn.columns)
    )

    logger.info(f'Adding {len(plants)} solar generators...')

    n.add(
        "Generator",
        plants,
        bus=bmus.loc[plants, 'bus'],
        carrier='solar',
        p_nom=pn[plants].max(),
        marginal_cost=0.,
        efficiency=1,
        p_max_pu=pn[plants].div(pn[plants].max()).replace(np.nan, 0),
    )


def add_nuclear(
        n,
        bmus,
        pn,
        nuclear_wholesale_price,
    ):

    plants = bmus[bmus['carrier'] == 'nuclear'].index

    # some nuclear-labelled BMUs always import electricity
    # these are thrown out
    plants = plants.intersection(pn.columns[pn.mean() > 0])

    logger.info(f'Adding {len(plants)} nuclear generators...')
    n.add(
        "Generator",
        plants,
        bus=bmus.loc[plants, 'bus'],
        carrier='nuclear',
        p_nom=pn[plants].max(),
        marginal_cost=nuclear_wholesale_price,
        efficiency=1,
    )

    # For nuclear units that are not constant, a 'p_max_pu' is needed
    for plant in plants:
        if len(pn[plant].unique()) > 1:
            n.generators_t['p_max_pu'].loc[:,plant] = pn[plant].div(pn[plant].max())


def add_thermal(
        n,
        bmus,
        pn,
        mel,
        wholesale_prices,
    ):

    plants = (
        bmus[bmus['carrier'].isin(['biomass', 'fossil'])]
        .index
        .intersection(pn.columns)
        .intersection(mel.columns)
    )
    logger.info(f'Adding {len(plants)} thermal generators...')

    assert plants.isin(wholesale_prices.index).all(), 'Missing wholesale prices for some thermal plants.'    

    missing = plants.difference(wholesale_prices.index)
    if len(missing) > 0:
        logger.warning(f'Filling in wholesale prices for {", ".join(missing)}')

    wholesale_prices = pd.concat((
        wholesale_prices,
        pd.Series(wholesale_prices.mean(), index=missing)
    ))

    n.add(
        "Generator",
        plants,
        bus=bmus.loc[plants, 'bus'],
        carrier=bmus.loc[plants, 'carrier'],
        p_nom=mel[plants].max(),
        marginal_cost=wholesale_prices.loc[plants],
        p_max_pu=mel[plants].div(mel[plants].max()).replace(np.nan, 0),
    )


def add_temporal_flexibility(
        n,
        bmus,
        pn,
        mel,
        battery_phs_capacities,
        carrier='battery',
        ):

    # data shows that flexible assets are not used to their full potential
    # due to usage in other markets or myopic foresight.
    # To account for this, the storage capacity is reduced by a factor.
    damping_factor = 0.25

    assets = bmus[bmus['carrier'] == carrier].index

    assets = assets.intersection(pn.columns).intersection(mel.columns)

    logger.info(f'Adding {len(assets)} {carrier} storage units...')
    # times two because time step is 30 minutes and max_hours does not
    # refer to hours but time steps within the context of the network's
    # time scale
    max_hours = (
        battery_phs_capacities['energy_cap[MWh]']
        .div(battery_phs_capacities['power_cap[MW]'])
        .mul(2.)
    )

    n.add(
        "StorageUnit",
        assets,
        bus=bmus.loc[assets, 'bus'],
        carrier=carrier,
        p_nom=battery_phs_capacities.loc[assets, 'power_cap[MW]'] * damping_factor,
        max_hours=max_hours.loc[assets],
        marginal_cost=0.,
        e_cyclic=True,
        state_of_charge_initial=(
            battery_phs_capacities
            .loc[assets, 'energy_cap[MWh]']
            .div(3.) # start at 1/3 of capacity based on typical overnight charging
        ),
    )


def add_pumped_hydro(*args):
    add_temporal_flexibility(*args, carrier='PHS')


def add_batteries(*args):
    add_temporal_flexibility(*args, carrier='battery')


def add_hydropower(
        n,
        bmus,
        pn,
        roc_values, 
        carrier='hydro',
    ):

    assets = (
        bmus[bmus['carrier'] == carrier]
        .index
        .intersection(pn.columns)
    )

    logger.info(f'Adding {len(assets)} {carrier} generators...')

    if carrier == 'cascade':
        marginal_costs = - roc_values.loc[assets]
    elif carrier == 'hydro':
        marginal_costs = 0.

    n.add(
        "StorageUnit",
        assets,
        bus=bmus.loc[assets, 'bus'],
        carrier=carrier,
        p_nom=pn[assets].max(),
        marginal_cost=marginal_costs,
        e_cyclic=False,
        state_of_charge_initial=pn[assets].sum(),
        max_hours=pn[assets].sum().div(pn[assets].max()),
        p_min_pu=0.,
    )


def add_cascade(*args):
    add_hydropower(*args, carrier='cascade')


def add_dispatchable_hydro(*args):
    add_hydropower(*args, carrier='hydro')
    

def add_interconnectors(
        n,
        bmus,
        pn,
        europe_wholesale_prices,
        nemo,
        interconnection_mapper,
        interconnection_capacities,
        interconnection_countries,
        country_coords,
    ):

    logger.info(f'Adding {len(interconnection_mapper)} interconnectors...')

    for (ic, ic_bmunits) in interconnection_mapper.items():

        p_nom = interconnection_capacities[ic]
        country = interconnection_countries[ic]
        marginal_cost = europe_wholesale_prices.loc[:, country]

        # no data for Nemo at the moment
        if ic == 'Nemo':
            link_kwargs = {
                'bus1': '4975',
                'p_set': nemo.iloc[:,0]
            }

        else:
            inter_flow = bmus.loc[
                bmus.index.str.contains('|'.join(ic_bmunits)),
                'bus']

            if inter_flow.empty:
                logger.info(f'No interconnector flow data for {ic}')
                continue

            gb_bus = inter_flow.value_counts().index[0]

            flow = (
                pn[pn.columns[pn.columns.str.contains('|'.join(ic_bmunits))]]
                .sum(axis=1)
            )

            p_nom = max(p_nom, flow.max())
            link_kwargs = {
                'p_set': flow,
                'bus1': gb_bus,
                }

            if (flow == 0).all():
                logger.info(f'No interconnector flow data for {ic}')
                continue

        p_nom = max(p_nom, link_kwargs['p_set'].abs().max())

        # this setup simulates a local market for each country that
        # can either be supplied by local generators (if the local wholesale
        # price is lower than GB wholesale price) or by GB generators
        # via the interconnector (if the GB wholesale price is lower)
        if not country in n.buses.index:
            n.add(
                'Bus',
                country,
                carrier='electricity',
                x=country_coords[country][0],
                y=country_coords[country][1],
                country=country,
                )
            n.add(
                "Load",
                country.lower() + '_local_market',
                bus=country,
                p_set=pd.Series(0, index=pn.index),
                carrier=country,
                )
            n.add(
                "Generator",
                country.lower() + '_local_market',
                bus=country,
                p_nom=0.,
                marginal_cost=marginal_cost,
                carrier="local_market",
                )

        n.add(
            "Link",
            ic,
            bus0=country,
            p_nom=p_nom,
            efficiency=0.99,
            p_max_pu=1.,
            p_min_pu=-1.,
            carrier='interconnector',
            **link_kwargs,
            )

        n.generators.loc[country.lower() + '_local_market', 'p_nom'] += 2 * p_nom
        n.loads_t.p_set.loc[:, country.lower() + '_local_market'] += (
            pd.Series(p_nom, index=pn.index)
        )


def build_static_supply_curve(
        n,
        bmus,
        pn,
        mel,
        wholesale_prices,
        europe_wholesale_prices,
        nemo,
        cfd_strike_prices,
        roc_values,
        nuclear_wholesale_price,
        battery_phs_capacities,
        interconnection_mapper,
        interconnection_capacities,
        interconnection_countries,
        country_coords,
    ):
    """
    Builds one day of available power plants, storage units and their marginal costs.
    It is 'static' in not scaling marginal costs according to wholesale price (yet).
    """

    n.generators.loc[:, 'marginal_cost'] = np.nan
    n.storage_units.loc[:, 'marginal_cost'] = np.nan

    add_onshore_wind(n, bmus, pn, mel, cfd_strike_prices, roc_values)
    add_offshore_wind(n, bmus, pn, mel, cfd_strike_prices, roc_values)
    add_solar(n, bmus, pn)
    add_nuclear(n, bmus, pn, nuclear_wholesale_price)
    add_thermal(n, bmus, pn, mel, wholesale_prices)
    add_batteries(n, bmus, pn, mel, battery_phs_capacities)
    add_pumped_hydro(n, bmus, pn, mel, battery_phs_capacities)
    add_cascade(n, bmus, pn, roc_values)
    add_dispatchable_hydro(n, bmus, pn, roc_values)
    add_interconnectors(
        n,
        bmus,
        pn,
        europe_wholesale_prices,
        nemo,
        interconnection_mapper,
        interconnection_capacities,
        interconnection_countries,
        country_coords,
        )


def add_load(n, pns, weights, interconnection_mapper, nemo):

    # subtract for interconnector export
    real_int_flow = pd.DataFrame(index=n.snapshots)

    for name, bmu_names in interconnection_mapper.items():
        if name == 'Nemo':
            real_int_flow[name] = nemo.iloc[:,0]

        else:
            real_int_flow[name] = (
                pn[pn.columns[pn.columns.str.contains('|'.join(bmu_names))]]
                .sum(axis=1)
            )

    export = real_int_flow.clip(upper=0).sum(axis=1).mul(-1)

    assert (export >= 0).all(), 'Interconnector export should be negative.'

    # subtract charging of temporal flexibility assets
    storages = n.storage_units.index.intersection(pns.columns)
    charging = pns[storages].clip(upper=0).sum(axis=1).mul(-1)

    assert (charging >= 0).all(), 'Charging should be negative.'

    net_load = pn.clip(lower=0).sum(axis=1) - export - charging

    logger.info('Adding load...')

    p_set = pd.DataFrame(
        np.outer(net_load, weights['load_weight']),
        index=n.snapshots,
        columns=weights.index
        )

    n.add(
        "Load",
        weights.index,
        bus=weights.index,
        p_set=p_set,
        carrier='electricity',
    )


def add_carriers(n, bmus, interconnection_countries):
    n.add("Carrier", bmus['carrier'].unique())
    n.add("Carrier", "load")
    n.add("Carrier", "electricity")
    n.add("Carrier", "AC")
    n.add("Carrier", "DC")
    n.add("Carrier", "local_market")
    n.add("Carrier", list(set(interconnection_countries.values())))


if __name__ == '__main__':

    configure_logging(snakemake)    

    day = snakemake.wildcards['day']

    pn = pd.read_csv(
        snakemake.input['physical_notifications'],
        index_col=0,
        parse_dates=True
        )
    mel = pd.read_csv(
        snakemake.input['maximum_export_limits'],
        index_col=0,
        parse_dates=True
        )
    dah = pd.read_csv(
        snakemake.input['day_ahead_prices'],
        index_col=0,
        parse_dates=True
        )
    europe_wholesale_prices = pd.read_csv(
        snakemake.input['europe_day_ahead_prices'],
        index_col=0,
        parse_dates=True
        )
    nemo = pd.read_csv(
        snakemake.input['nemo_powerflow'],
        index_col=0,
        parse_dates=True
    )

    pn.index = pn.index.values
    mel.index = mel.index.values
    europe_wholesale_prices.index = europe_wholesale_prices.index.values
    dah.index = dah.index.values
    nemo.index = nemo.index.values

    thermal_generation_costs = (
        pd.read_csv(
            snakemake.input['thermal_generation_costs'],
            index_col=0
            )
        .iloc[:,0]
    )

    cfd_strike_prices = (
        pd.read_csv(
            snakemake.input['cfd_strike_prices'],
            index_col=0,
            parse_dates=True,
            )
        .loc[:, :day]
        .iloc[:, -1]
        .rename('cfd_strike_price')
    )

    roc_values = (
        pd.read_csv(
            snakemake.input['roc_values'],
            index_col=0
            )
        .iloc[:,0]
        .rename('roc_value')
    )

    nuclear_wholesale_price = pd.read_csv(
        snakemake.input['nuclear_marginal_price'],
        index_col=0).iloc[0,0]

    battery_phs_capacities = pd.read_csv(
        snakemake.input['battery_phs_capacities'],
        index_col=0
        )

    weights = pd.read_csv(snakemake.input['load_weights'], index_col=0)
    weights.index = weights.index.astype(str)

    bmus = pd.read_csv(snakemake.input['bmus'], index_col=0)
    bmus['bus'] = bmus['bus'].astype(str)

    with open(snakemake.input['interconnection_helpers'], 'r') as f:
        data = yaml.safe_load(f)

    interconnection_countries = data['interconnection_countries']
    interconnection_mapper = data['interconnection_mapper']
    interconnection_capacities = data['interconnection_capacities']
    country_coords = data['country_coords']

    n = pypsa.Network(snakemake.input['network'])

    add_carriers(n, bmus, interconnection_countries)

    n.set_snapshots(pn.index)

    add_load(
        n,
        pn,
        weights,
        interconnection_mapper,
        nemo
        )

    build_static_supply_curve(
        n,
        bmus,
        pn,
        mel,
        thermal_generation_costs,
        europe_wholesale_prices,
        nemo,
        cfd_strike_prices,
        roc_values,
        nuclear_wholesale_price,
        battery_phs_capacities,
        interconnection_mapper,
        interconnection_capacities,
        interconnection_countries,
        country_coords,
        )

    scale_merit_order(n, dah, collective=False)

    n.export_to_netcdf(snakemake.output['network'])
