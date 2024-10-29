# SPDX-FileCopyrightText: : 2024 Lukas Franken
#
# SPDX-License-Identifier: MIT

"""
Script to prepare the (nodal layout) network for one day.
"""

import logging

logger = logging.getLogger(__name__)


import sys
import pypsa
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path.cwd() / 'data'))

from _helpers import configure_logging
from interconnection_helpers import (
    interconnection_mapper,
    interconnection_capacities,
    interconnection_countries,
)


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
        bus='bus',
        carrier='onwind',
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
        bus='bus',
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
        bus='bus',
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

    n.add(
        "Generator",
        plants,
        bus='bus',
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
        bus='bus',
        carrier=carrier,
        p_nom=battery_phs_capacities.loc[assets, 'power_cap[MW]'],
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
        bus="bus",
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
        interconnection_mapper,
        interconnection_capacities,
        interconnection_countries,
    ):

    logger.info(f'Adding {len(interconnection_mapper)} interconnectors...')

    for (ic, bmu_names) in interconnection_mapper.items():

        p_nom = interconnection_capacities[ic]
        country = interconnection_countries[ic]
        marginal_cost = europe_wholesale_prices.loc[:, country]

        # this setup simulates a local market for each country that
        # can either be supplied by local generators (if the local wholesale
        # price is lower than GB wholesale price) or by GB generators
        # via the interconnector (if the GB wholesale price is lower)
        if not country in n.buses.index:
            n.add(
                'Bus',
                country,
                carrier='electricity',
                )

            n.add(
                "Load",
                country,
                bus=country,
                p_set=pd.Series(0, index=pn.index),
                carrier=country,
                )

            n.add(
                "Generator",
                country,
                bus=country,
                p_nom=0.,
                marginal_cost=marginal_cost,
                carrier=country,
                )

        n.generators.loc[country, 'p_nom'] += 2 * p_nom

        n.loads.loc[country, 'p_set'] += p_nom

        n.add(
            "Link",
            ic,
            bus0='bus',
            bus1=country,
            p_nom=p_nom,
            efficiency=0.99,
            p_max_pu=1.,
            p_min_pu=-1.,
            carrier='interconnector',
            )


def build_static_supply_curve(
        n,
        bmus,
        pn,
        mel,
        wholesale_prices,
        europe_wholesale_prices,
        cfd_strike_prices,
        roc_values,
        nuclear_wholesale_price,
        battery_phs_capacities,
        interconnection_mapper,
        interconnection_capacities,
        interconnection_countries,
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
        interconnection_mapper,
        interconnection_capacities,
        interconnection_countries,
        )
    

def add_load(n, pns):

    logger.info('Adding load...')
    n.add(
        "Load",
        "load",
        bus="bus",
        p_set=pns.clip(lower=0).sum(axis=1),
    )


def add_carriers(n, bmus, interconnection_countries):
    n.add("Carrier", bmus['carrier'].unique())
    n.add("Carrier", "load")
    n.add("Carrier", "electricity")
    n.add("Carrier", list(set(interconnection_countries.values())))



if __name__ == '__main__':

    configure_logging(snakemake)

    day = snakemake.wildcards["day"]

    bmus = pd.read_csv(snakemake.input["bmus"], index_col=0)

    pn = pd.read_csv(
        snakemake.input["physical_notifications"],
        index_col=0,
        parse_dates=True
        )
    mel = pd.read_csv(
        snakemake.input["maximum_export_limits"],
        index_col=0,
        parse_dates=True,
        )
    europe_day_ahead_prices = pd.read_csv(
        snakemake.input["europe_day_ahead_prices"],
        index_col=0,
        parse_dates=True
        )

    # pypsa does not like datetime indices and their timezones
    pn.index = pn.index.values
    mel.index = mel.index.values
    europe_day_ahead_prices.index = europe_day_ahead_prices.index.values

    thermal_costs = (
        pd.read_csv(
            snakemake.input["thermal_costs"],
            index_col=0
            )
        .iloc[:,0]
    )

    cfd_strike_prices = (
        pd.read_csv(
            snakemake.input["cfd_strike_prices"],
            index_col=0,
            parse_dates=True
        )
        .loc[:, :day]
        .iloc[:, -1]
        .rename('cfd_strike_price')
    )

    roc_values = (
        pd.read_csv(
            snakemake.input["roc_values"],
            index_col=0
        )
        .iloc[:,0]
        .rename('roc_value')
    )

    nuclear_wholesale_price = (
        pd.read_csv(
            snakemake.input["nuclear_marginal_price"],
            index_col=0
        )
        .iloc[0,0]
    )

    battery_phs_capacities = (
        pd.read_csv(
            snakemake.input["battery_phs_capacities"],
            index_col=0
        )
    )

    n = pypsa.Network()

    add_carriers(n, bmus, interconnection_countries)

    n.add("Bus", "bus", carrier="electricity")
    n.set_snapshots(pn.index)

    # 'static' refers to building a supply curve without tuning marginal costs
    # to actual observed day-ahead prices on that day
    build_static_supply_curve(
        n,
        bmus,
        pn,
        mel,
        thermal_costs,
        europe_day_ahead_prices,
        cfd_strike_prices,
        roc_values,
        nuclear_wholesale_price,
        battery_phs_capacities,
        interconnection_mapper,
        interconnection_capacities,
        interconnection_countries,
        )

    add_load(n, pn)

    n.export_to_netcdf(snakemake.output["network"])
