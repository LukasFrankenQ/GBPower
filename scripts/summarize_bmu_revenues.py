# -*- coding: utf-8 -*-
# Copyright 2024-2025 Lukas Franken
# SPDX-FileCopyrightText: : 2024-2025 Lukas Franken
#
# SPDX-License-Identifier: MIT

"""
Estimates the revenues for each BMU (balancing mechanism unit) including:
  - Wholesale revenues,
  - Balancing (constraint managing) market bids/offers,
  - CFD payments, and
  - (Estimated) ROC revenue.

This updated version splits the storage (formerly “water”) assets into:
  - One-way generation assets (labeled "hydro") which include carriers "cascade" and "hydro", and
  - Two-way storage assets (labeled "storage") which include carriers "PHS" and "battery".
"""

import logging

import numpy as np
import pandas as pd
import pypsa
from _helpers import configure_logging

logger = logging.getLogger(__name__)
idx = pd.IndexSlice


def get_unit_wholesale_revenue(n, comp, unit):
    if isinstance(unit, str):
        return (
            getattr(n, comp + "_t").p[unit].multiply(
                0.5 * n.buses_t.marginal_price[getattr(n, comp).bus[unit]]
            )
        ).rename(unit)
    elif isinstance(unit, pd.Index):
        m_prices = n.buses_t.marginal_price[getattr(n, comp).bus[unit]]
        m_prices.columns = unit
        commodities = getattr(n, comp + "_t").p[unit]
        return commodities.multiply(0.5 * m_prices)
    else:
        assert False, "unit must be either a string or a pd.Index"


def get_cfd_revenue(n, strike_prices):
    strike_prices = strike_prices.copy()
    strike_prices.columns = pd.to_datetime(strike_prices.columns)
    strike_prices = strike_prices.loc[:, : n.snapshots[4]].iloc[:, -1]
    cfd_plants = strike_prices.index.intersection(n.generators.index)
    cfd_payments = pd.DataFrame(columns=cfd_plants, index=n.snapshots)
    for plant, strike_price in strike_prices.loc[cfd_plants].items():
        bus = n.generators.loc[plant, "bus"]
        bus_mp = n.buses_t.marginal_price[bus]

        # Identify snapshots where the marginal price has been negative for the last 6 hours (i.e. 12 consecutive 30-min intervals)
        negative_streak = bus_mp.rolling(window=12, min_periods=12).max() < 0
        payment_mask = ~negative_streak.fillna(False)
        price_gap = strike_price - bus_mp
        cfd_payments.loc[:, plant] = n.generators_t.p[plant] * price_gap * 0.5 * payment_mask

    return cfd_payments


def process_daily_balancing_data(df):
    df = (
        df.stack()
        .unstack(1)
        .dropna()
        .groupby(level=1)
        .agg({"price": "mean", "vol": "sum"})
        .sort_values("price")
    )
    return df


def make_north_south_split(n, carrier, comp, threshold=55.3):
    comp_df = getattr(n, comp)
    comp_df["lat"] = comp_df.bus.map(n.buses.y)
    if not isinstance(carrier, str):
        mask = comp_df["carrier"].isin(carrier)
    else:
        mask = comp_df["carrier"].str.contains(carrier)
    north = comp_df.loc[mask & (comp_df["lat"] > threshold)].index
    south = comp_df.loc[mask & (comp_df["lat"] <= threshold)].index
    return north, south


def get_weighted_avg_price(df):
    assert set(df.columns) == {"price", "vol"}, "Columns must be price and vol"
    assert not df.empty, "DataFrame must not be empty"
    return (df["price"] * df["vol"]).sum() / df["vol"].sum()


# === RENAMED STORAGE FUNCTIONS (formerly "water") ===

def get_storage_balancing_revenue(who, units, mode, actual_trades):
    """
    Calculates the balancing revenue for storage units.

    who: wholesale model network.
    units: the storage units (a list/index of units).
    mode: either "offers" or "bids".
    actual_trades: a DataFrame of actual trade data with columns ["price", "vol"].

    Note: This implementation assumes that storage wholesale trading cannot be reversed in the balancing market.
    """
    logger.warning(
        "Implementation of storage balancing revenue assumes that storage wholesale trading "
        "cannot be reversed in the balancing market."
    )
    assert mode in ["offers", "bids"], 'mode must be either "offers" or "bids"'
    actual = actual_trades.loc[actual_trades.index.intersection(units)]
    total_actual_volume = actual["vol"].sum()
    if total_actual_volume == 0:
        return pd.Series(0.0, index=who.snapshots)
    actual_revenue = (actual["price"] * actual["vol"]).sum()
    kwargs = {"lower": 0.0} if mode == "offers" else {"upper": 0.0}
    trading_profile = (
        who.storage_units_t.p[units].sum(axis=1).clip(**kwargs).abs()
    )
    if trading_profile.sum() == 0:
        return pd.Series(actual_revenue / len(who.snapshots), index=who.snapshots)
    trading_profile = trading_profile.div(trading_profile.sum())
    return trading_profile.multiply(actual_revenue)


def get_storage_roc_revenue(n, actual_bids, actual_offers, units, roc_values):
    """
    Calculates ROC revenue for storage units.

    n: network.
    actual_bids: actual accepted bids DataFrame.
    actual_offers: actual accepted offers DataFrame.
    units: the storage units to consider.
    roc_values: ROC values (a Series with unit indices).
    """
    units = roc_values.index.intersection(units)
    if units.empty:
        return pd.Series(0.0, index=n.snapshots)
    bid_volume = actual_bids.loc[units.intersection(actual_bids.index)].vol.sum()
    offer_volume = actual_offers.loc[units.intersection(actual_offers.index)].vol.sum()
    units_dispatch = n.storage_units_t.p[units].clip(lower=0.0)
    total_dispatch = units_dispatch.sum().sum() * 0.5
    units_dispatch *= ((total_dispatch + offer_volume - bid_volume) / total_dispatch)
    return units_dispatch.multiply(roc_values.loc[units_dispatch.columns] / 2).sum(
        axis=1
    )


def compute_storage_roc_dispatch(network, units, actual_bids, actual_offers):
    """
    Computes the scaled dispatch for ROC revenue calculations for storage units.

    network: the wholesale network.
    units: the storage units to consider.
    actual_bids: actual bids DataFrame.
    actual_offers: actual offers DataFrame.
    """
    if not isinstance(units, pd.Index):
        units = pd.Index([units])

    units = units.intersection(network.storage_units.index)
    if units.empty:
        return pd.Series(0.0, index=network.snapshots)
    units_dispatch = network.storage_units_t.p.loc[:, units].clip(lower=0)
    total_dispatch = units_dispatch.sum().sum() * 0.5
    if total_dispatch == 0:
        return pd.Series(0.0, index=network.snapshots)
    bid_volume = (
        actual_bids.loc[units.intersection(actual_bids.index), "vol"].sum()
        if not actual_bids.empty
        else 0.0
    )
    offer_volume = (
        actual_offers.loc[units.intersection(actual_offers.index), "vol"].sum()
        if not actual_offers.empty
        else 0.0
    )
    scale_factor = (total_dispatch + offer_volume - bid_volume) / total_dispatch
    scaled_dispatch = units_dispatch * scale_factor
    return scaled_dispatch.sum(axis=1)


if __name__ == "__main__":
    configure_logging(snakemake)

    actual_bids = pd.read_csv(
        snakemake.input["bids"], index_col=[0, 1], parse_dates=True
    )
    if not actual_bids.empty:
        actual_bids = process_daily_balancing_data(actual_bids)
    else:
        actual_bids = pd.DataFrame(columns=["price", "vol"])

    actual_offers = pd.read_csv(
        snakemake.input["offers"], index_col=[0, 1], parse_dates=True
    )
    if not actual_offers.empty:
        actual_offers = process_daily_balancing_data(actual_offers)
    else:
        actual_offers = pd.DataFrame(columns=["price", "vol"])

    default_balancing_prices = pd.read_csv(
        snakemake.input["default_balancing_prices"], index_col=0
    )

    roc_values = pd.read_csv(snakemake.input["roc_values"], index_col=0).iloc[:, 0]

    nodal_network = pypsa.Network(snakemake.input["network_nodal"])

    total_balancing_volumes = {}

    for layout in ["national", "zonal"]:
        who = pypsa.Network(snakemake.input[f"network_{layout}"])
        if layout != "nodal":
            bal = pypsa.Network(snakemake.input[f"network_{layout}_redispatch"])
        else:
            bal = who.copy()

        diff = pd.concat((bal.generators_t.p - who.generators_t.p,), axis=1)
        total_balancing_volumes[layout] = {}
        total_balancing_volumes[layout]["offers"] = (
            diff.clip(lower=0.0).sum().mul(0.5).sum()
        )
        total_balancing_volumes[layout]["bids"] = (
            diff.clip(upper=0.0).sum().mul(0.5).abs().sum()
        )

    for layout in ["national", "zonal", "nodal"]:

        cfd_strike_prices = pd.read_csv(
            snakemake.input["cfd_strike_prices"],
            index_col=0,
            parse_dates=True,
        )

        logger.info(f"Processing revenues for {layout} layout.")
        who = pypsa.Network(snakemake.input[f"network_{layout}"])
        if layout != "nodal":
            bal = pypsa.Network(snakemake.input[f"network_{layout}_redispatch"])
        else:
            bal = who.copy()

        revenues = pd.DataFrame(
            0.0,
            columns=pd.MultiIndex.from_product(
                [
                    ["north", "south"],
                    ["wind", "disp", "hydro", "storage"],
                    ["wholesale", "offers", "bids", "cfd", "roc"],
                ]
            )
            .append(
                pd.MultiIndex.from_tuples(
                    [
                        ("total", "intercon", "wholesale buying"),
                        ("total", "intercon", "wholesale selling"),
                    ]
                )
            )
            .append(pd.MultiIndex.from_tuples([("total", "load", "wholesale")])),
            index=who.snapshots,
        )

        # Interconnectors
        inter = who.links[who.links.carrier == "interconnector"]
        p0_links = who.links_t.p0.loc[:, inter.index]
        p1_links = who.links_t.p1.loc[:, inter.index]
        price_bus0 = pd.DataFrame(
            {link: who.buses_t.marginal_price[bus] for link, bus in inter["bus0"].items()},
            index=who.snapshots,
        )
        price_bus1 = pd.DataFrame(
            {link: who.buses_t.marginal_price[bus] for link, bus in inter["bus1"].items()},
            index=who.snapshots,
        )
        wholesale_buying = (
            p0_links.clip(lower=0)
            .multiply(price_bus0 * 0.5)
            .sum(axis=1)
            .add(p1_links.clip(lower=0).multiply(price_bus1 * 0.5).sum(axis=1), axis=0)
            .multiply(-1)
        )
        revenues.loc[:, idx["total", "intercon", "wholesale buying"]] = wholesale_buying
        wholesale_selling = (
            p0_links.clip(upper=0)
            .multiply(price_bus0 * 0.5)
            .sum(axis=1)
            .add(p1_links.clip(upper=0).multiply(price_bus1 * 0.5).sum(axis=1), axis=0)
            .multiply(-1)
        )
        revenues.loc[:, idx["total", "intercon", "wholesale selling"]] = wholesale_selling

        # Total Load Cost
        revenues.loc[:, idx["total", "load", "wholesale"]] = (
            who.statistics.revenue(aggregate_time=False, comps="Load")
            .loc["electricity"]
            .mul(0.5)
        )

        # Group assets into north and south, and by technology.
        wind_north, wind_south = make_north_south_split(
            bal, "wind", "generators"
        )
        disp_north, disp_south = make_north_south_split(
            bal, ["fossil", "biomass", "coal"], "generators"
        )
        # Split storage assets into one-way (hydro) and two-way (storage)
        hydro_north, hydro_south = make_north_south_split(
            nodal_network, ["cascade", "hydro"], "storage_units"
        )
        storage_north, storage_south = make_north_south_split(
            nodal_network, ["PHS", "battery"], "storage_units"
        )

        diff = pd.concat((bal.generators_t.p - who.generators_t.p,), axis=1)

        # Revenue of southern wind generators
        south_wind_offers = diff.loc[:, wind_south].sum(axis=1)
        south_wind_offers_total = float(south_wind_offers.sum())
        south_wind_offers_total_actual = float(
            actual_offers.loc[idx[wind_south.intersection(actual_offers.index), "vol"]].sum()
        )
        if south_wind_offers_total:
            if south_wind_offers_total > south_wind_offers_total_actual:
                keep_share = south_wind_offers_total_actual / south_wind_offers_total
                transfer_share = 1.0 - keep_share
            else:
                keep_share = (
                    south_wind_offers_total_actual / south_wind_offers_total
                    if south_wind_offers_total != 0
                    else 1.0
                )
                transfer_share = 0.0
        else:
            keep_share = 1.0
            transfer_share = 0.0

        south_wind_transfer = south_wind_offers * transfer_share
        south_wind_offers *= keep_share

        try:
            south_wind_offers_price = get_weighted_avg_price(
                actual_offers.loc[wind_south.intersection(actual_offers.index)]
            )
            revenues.loc[:, idx["south", "wind", "offers"]] = (
                south_wind_offers_price * south_wind_offers / 2
            )
        except Exception as e:
            logger.error(f"Error computing south wind offers price: {e}")

        wind_south_dispatch = bal.generators_t.p.loc[
            :, wind_south.intersection(roc_values.index)
        ]
        wind_south_roc_revenue = wind_south_dispatch.multiply(
            roc_values.loc[wind_south_dispatch.columns] / 2
        )
        revenues.loc[:, idx["south", "wind", "roc"]] = wind_south_roc_revenue.sum(
            axis=1
        )
        cfd = get_cfd_revenue(bal, cfd_strike_prices)
        cfd_revenue = cfd.loc[:, wind_south.intersection(cfd.columns)]
        revenues.loc[:, idx["south", "wind", "cfd"]] = cfd_revenue.sum(
            axis=1
        ).astype(np.float64)
        revenues.loc[:, idx["south", "wind", "wholesale"]] = (
            get_unit_wholesale_revenue(who, "generators", wind_south).sum(axis=1)
        )

        # Revenue of northern wind generators
        revenues.loc[:, idx["north", "wind", "wholesale"]] = (
            get_unit_wholesale_revenue(who, "generators", wind_north).sum(axis=1)
        )
        logger.warning(
            "Open question: Should CFD payments be considered in the wholesale or balancing model?"
        )
        cfd = get_cfd_revenue(who, cfd_strike_prices)
        cfd_revenue = cfd.loc[:, wind_north.intersection(cfd.columns)]
        revenues.loc[:, idx["north", "wind", "cfd"]] = cfd_revenue.sum(
            axis=1
        ).astype(np.float64)
        wind_north_dispatch = bal.generators_t.p.loc[
            :, wind_north.intersection(roc_values.index)
        ]
        wind_north_roc_revenue = wind_north_dispatch.multiply(
            roc_values.loc[wind_north_dispatch.columns] / 2
        )
        revenues.loc[:, idx["north", "wind", "roc"]] = wind_north_roc_revenue.sum(
            axis=1
        )
        north_wind_bids = (
            diff.loc[:, wind_north].sum(axis=1).clip(upper=0).abs()
        )
        if north_wind_bids.sum() > 0:
            actual_north_wind_bid_vol = actual_bids.loc[
                actual_bids.index.intersection(wind_north), "vol"
            ]
            if actual_north_wind_bid_vol.sum() > 0:
                north_wind_bid_price = get_weighted_avg_price(
                    actual_bids.loc[actual_bids.index.intersection(wind_north), :]
                )
            else:
                north_wind_bid_price = default_balancing_prices.loc["wind", "bids"]
            revenues.loc[:, idx["north", "wind", "bids"]] = (
                north_wind_bids * north_wind_bid_price / 2
            )
        north_wind_offers = (
            diff.loc[:, wind_north].sum(axis=1).clip(lower=0)
        )
        if north_wind_offers.sum() > 0:
            actual_north_wind_offer_vol = actual_offers.loc[
                actual_offers.index.intersection(wind_north), "vol"
            ]
            if actual_north_wind_offer_vol.sum() > 0:
                north_wind_offer_price = get_weighted_avg_price(
                    actual_offers.loc[actual_offers.index.intersection(wind_north), :]
                )
            else:
                north_wind_offer_price = default_balancing_prices.loc["wind", "offers"]
                safe_value = 150.0
                if north_wind_offer_price > safe_value:
                    logger.warning(
                        "Crudely correcting for excessive offer price."
                    )
                north_wind_offer_price = min(north_wind_offer_price, safe_value)
            revenues.loc[:, idx["north", "wind", "offers"]] = (
                north_wind_offers * north_wind_offer_price / 2
            )

        # Revenue of dispatchable generators
        disp_offers_actual = actual_offers.loc[
            disp_south.union(disp_north).intersection(actual_offers.index)
        ]
        if not disp_offers_actual.empty:
            disp_offers_price = get_weighted_avg_price(disp_offers_actual)
        else:
            disp_offers_price = default_balancing_prices.loc["disp", "offers"]

        south_disp_offers = diff.loc[:, disp_south].sum(axis=1).clip(lower=0.0)
        south_disp_offers += south_wind_transfer
        revenues.loc[:, idx["south", "disp", "offers"]] = (
            south_disp_offers * disp_offers_price / 2
        )
        disp_bids_actual = actual_bids.loc[
            disp_south.union(disp_north).intersection(actual_bids.index)
        ]
        if not disp_bids_actual.empty:
            disp_bids_price = get_weighted_avg_price(disp_bids_actual)
        else:
            disp_bids_price = default_balancing_prices.loc["disp", "bids"]
        south_disp_bids = diff.loc[:, disp_south].sum(axis=1).clip(upper=0.0).abs()
        revenues.loc[:, idx["south", "disp", "bids"]] = (
            south_disp_bids * disp_bids_price / 2
        )
        north_disp_offers = diff.loc[:, disp_north].sum(axis=1).clip(lower=0.0)
        north_disp_bids = diff.loc[:, disp_north].sum(axis=1).clip(upper=0.0).abs()
        revenues.loc[:, idx["north", "disp", "offers"]] = (
            north_disp_offers * disp_offers_price / 2
        )
        revenues.loc[:, idx["north", "disp", "bids"]] = (
            north_disp_bids * disp_bids_price / 2
        )
        revenues.loc[:, idx["south", "disp", "wholesale"]] = (
            get_unit_wholesale_revenue(who, "generators", disp_south).sum(axis=1)
        )
        revenues.loc[:, idx["north", "disp", "wholesale"]] = (
            get_unit_wholesale_revenue(who, "generators", disp_north).sum(axis=1)
        )

        # Revenue of storage assets
        # ONE-WAY ASSETS ("hydro"): run-of-river / generation-only.
        revenues.loc[:, idx["south", "hydro", "wholesale"]] = (
            get_unit_wholesale_revenue(who, "storage_units", hydro_south).sum(axis=1)
        )
        revenues.loc[:, idx["north", "hydro", "wholesale"]] = (
            get_unit_wholesale_revenue(who, "storage_units", hydro_north).sum(axis=1)
        )
        if layout != "nodal":
            logger.warning("Logic of storage balancing volumes could warrant more thinking")
            bid_reduction_factor = 1.0
            offer_reduction_factor = 1.0
            revenues.loc[:, idx["north", "hydro", "bids"]] = (
                get_storage_balancing_revenue(who, hydro_north, "bids", actual_bids)
                .mul(bid_reduction_factor)
            )
            revenues.loc[:, idx["north", "hydro", "offers"]] = (
                get_storage_balancing_revenue(who, hydro_north, "offers", actual_offers)
                .mul(offer_reduction_factor)
            )
            revenues.loc[:, idx["south", "hydro", "bids"]] = (
                get_storage_balancing_revenue(who, hydro_south, "bids", actual_bids)
                .mul(bid_reduction_factor)
            )
            revenues.loc[:, idx["south", "hydro", "offers"]] = (
                get_storage_balancing_revenue(who, hydro_south, "offers", actual_offers)
                .mul(offer_reduction_factor)
            )
        revenues.loc[:, idx["south", "hydro", "roc"]] = get_storage_roc_revenue(
            who, actual_bids, actual_offers, hydro_south, roc_values
        )
        revenues.loc[:, idx["north", "hydro", "roc"]] = get_storage_roc_revenue(
            who, actual_bids, actual_offers, hydro_north, roc_values
        )

        # TWO-WAY ASSETS ("storage"): PHS and battery.
        revenues.loc[:, idx["south", "storage", "wholesale"]] = (
            get_unit_wholesale_revenue(who, "storage_units", storage_south).sum(axis=1)
        )
        revenues.loc[:, idx["north", "storage", "wholesale"]] = (
            get_unit_wholesale_revenue(who, "storage_units", storage_north).sum(axis=1)
        )
        if layout != "nodal":
            revenues.loc[:, idx["north", "storage", "bids"]] = (
                get_storage_balancing_revenue(who, storage_north, "bids", actual_bids)
                .mul(bid_reduction_factor)
            )
            revenues.loc[:, idx["north", "storage", "offers"]] = (
                get_storage_balancing_revenue(who, storage_north, "offers", actual_offers)
                .mul(offer_reduction_factor)
            )
            revenues.loc[:, idx["south", "storage", "bids"]] = (
                get_storage_balancing_revenue(who, storage_south, "bids", actual_bids)
                .mul(bid_reduction_factor)
            )
            revenues.loc[:, idx["south", "storage", "offers"]] = (
                get_storage_balancing_revenue(who, storage_south, "offers", actual_offers)
                .mul(offer_reduction_factor)
            )
        revenues.loc[:, idx["south", "storage", "roc"]] = get_storage_roc_revenue(
            who, actual_bids, actual_offers, storage_south, roc_values
        )
        revenues.loc[:, idx["north", "storage", "roc"]] = get_storage_roc_revenue(
            who, actual_bids, actual_offers, storage_north, roc_values
        )

        revenues.to_csv(snakemake.output[f"bmu_revenues_{layout}"])

        # --- Track Dispatch
        dispatch = pd.DataFrame(
            0.0, index=who.snapshots, columns=revenues.columns
        )

        # WIND:
        dispatch.loc[:, idx["south", "wind", "wholesale"]] = who.generators_t.p.loc[
            :, wind_south
        ].sum(axis=1)
        dispatch.loc[:, idx["north", "wind", "wholesale"]] = who.generators_t.p.loc[
            :, wind_north
        ].sum(axis=1)
        dispatch.loc[:, idx["south", "wind", "offers"]] = south_wind_offers
        dispatch.loc[:, idx["south", "wind", "bids"]] = diff.loc[:, wind_south].sum(
            axis=1
        ).clip(upper=0)
        dispatch.loc[:, idx["north", "wind", "offers"]] = diff.loc[:, wind_north].sum(
            axis=1
        ).clip(lower=0)
        dispatch.loc[:, idx["north", "wind", "bids"]] = diff.loc[:, wind_north].sum(
            axis=1
        ).clip(upper=0)
        csouth = wind_south.intersection(cfd_strike_prices.index)
        cnorth = wind_north.intersection(cfd_strike_prices.index)
        if not csouth.empty:
            dispatch.loc[:, idx["south", "wind", "cfd"]] = bal.generators_t.p.loc[
                :, csouth
            ].sum(axis=1)
        else:
            dispatch.loc[:, idx["south", "wind", "cfd"]] = 0.0
        if not cnorth.empty:
            dispatch.loc[:, idx["north", "wind", "cfd"]] = who.generators_t.p.loc[
                :, cnorth
            ].sum(axis=1)
        else:
            dispatch.loc[:, idx["north", "wind", "cfd"]] = 0.0
        rnorth = wind_north.intersection(roc_values.index)
        rsouth = wind_south.intersection(roc_values.index)
        if not rsouth.empty:
            dispatch.loc[:, idx["south", "wind", "roc"]] = bal.generators_t.p.loc[
                :, rsouth
            ].sum(axis=1)
        else:
            dispatch.loc[:, idx["south", "wind", "roc"]] = 0.0
        if not rnorth.empty:
            dispatch.loc[:, idx["north", "wind", "roc"]] = bal.generators_t.p.loc[
                :, rnorth
            ].sum(axis=1)
        else:
            dispatch.loc[:, idx["north", "wind", "roc"]] = 0.0

        # DISPATCHABLE (disp):
        dispatch.loc[:, idx["south", "disp", "wholesale"]] = who.generators_t.p.loc[
            :, disp_south
        ].sum(axis=1)
        dispatch.loc[:, idx["north", "disp", "wholesale"]] = who.generators_t.p.loc[
            :, disp_north
        ].sum(axis=1)
        dispatch.loc[:, idx["south", "disp", "offers"]] = (
            diff.loc[:, disp_south].sum(axis=1).clip(lower=0) + south_wind_transfer
        )
        dispatch.loc[:, idx["north", "disp", "offers"]] = diff.loc[:, disp_north].sum(
            axis=1
        ).clip(lower=0)
        dispatch.loc[:, idx["south", "disp", "bids"]] = diff.loc[:, disp_south].sum(
            axis=1
        ).clip(upper=0)
        dispatch.loc[:, idx["north", "disp", "bids"]] = diff.loc[:, disp_north].sum(
            axis=1
        ).clip(upper=0)

        # STORAGE ASSETS:
        # ONE-WAY ("hydro")
        dispatch.loc[:, idx["south", "hydro", "wholesale"]] = who.storage_units_t.p.loc[
            :, hydro_south
        ].sum(axis=1)
        dispatch.loc[:, idx["north", "hydro", "wholesale"]] = who.storage_units_t.p.loc[
            :, hydro_north
        ].sum(axis=1)
        dispatch.loc[:, idx["south", "hydro", "offers"]] = who.storage_units_t.p.loc[
            :, hydro_south
        ].sum(axis=1).clip(lower=0)
        dispatch.loc[:, idx["south", "hydro", "bids"]] = who.storage_units_t.p.loc[
            :, hydro_south
        ].sum(axis=1).clip(upper=0)
        dispatch.loc[:, idx["north", "hydro", "offers"]] = who.storage_units_t.p.loc[
            :, hydro_north
        ].sum(axis=1).clip(lower=0)
        dispatch.loc[:, idx["north", "hydro", "bids"]] = who.storage_units_t.p.loc[
            :, hydro_north
        ].sum(axis=1).clip(upper=0)

        # TWO-WAY ("storage")
        dispatch.loc[:, idx["south", "storage", "wholesale"]] = who.storage_units_t.p.loc[
            :, storage_south
        ].sum(axis=1)
        dispatch.loc[:, idx["north", "storage", "wholesale"]] = who.storage_units_t.p.loc[
            :, storage_north
        ].sum(axis=1)
        dispatch.loc[:, idx["south", "storage", "offers"]] = who.storage_units_t.p.loc[
            :, storage_south
        ].sum(axis=1).clip(lower=0)
        dispatch.loc[:, idx["south", "storage", "bids"]] = who.storage_units_t.p.loc[
            :, storage_south
        ].sum(axis=1).clip(upper=0)
        dispatch.loc[:, idx["north", "storage", "offers"]] = who.storage_units_t.p.loc[
            :, storage_north
        ].sum(axis=1).clip(lower=0)
        dispatch.loc[:, idx["north", "storage", "bids"]] = who.storage_units_t.p.loc[
            :, storage_north
        ].sum(axis=1).clip(upper=0)

        dispatch.loc[:, idx["south", "hydro", "roc"]] = compute_storage_roc_dispatch(
            who, hydro_south, actual_bids, actual_offers
        )
        dispatch.loc[:, idx["north", "hydro", "roc"]] = compute_storage_roc_dispatch(
            who, hydro_north, actual_bids, actual_offers
        )
        dispatch.loc[:, idx["south", "storage", "roc"]] = compute_storage_roc_dispatch(
            who, storage_south, actual_bids, actual_offers
        )
        dispatch.loc[:, idx["north", "storage", "roc"]] = compute_storage_roc_dispatch(
            who, storage_north, actual_bids, actual_offers
        )

        # TOTAL INTERCONNECTOR & LOAD:
        intercon_links = who.links.index[who.links.carrier == "interconnector"]
        p0_links = who.links_t.p0.loc[:, intercon_links]
        p1_links = who.links_t.p1.loc[:, intercon_links]
        dispatch.loc[:, idx["total", "intercon", "wholesale selling"]] = (
            p0_links.clip(lower=0)
            .sum(axis=1)
            .add(p1_links.clip(lower=0).sum(axis=1), axis=0)
            .multiply(0.5)
        )
        dispatch.loc[:, idx["total", "intercon", "wholesale buying"]] = (
            p0_links.clip(upper=0)
            .sum(axis=1)
            .add(p1_links.clip(upper=0).sum(axis=1), axis=0)
            .multiply(0.5)
        )
        if "carrier" in who.loads.columns:
            load_units = who.loads.index[who.loads.carrier == "electricity"]
            dispatch.loc[:, idx["total", "load", "wholesale"]] = who.loads_t.p.loc[
                :, load_units
            ].sum(axis=1)
        else:
            dispatch.loc[:, idx["total", "load", "wholesale"]] = who.loads_t.p.sum(
                axis=1
            )

        dispatch.to_csv(snakemake.output[f"bmu_dispatch_{layout}"])


        ### Track revenues of individual assets and keep revenue components (wholesale, CFD, and ROC) separate
        cfd_strike_prices = pd.read_csv(
            snakemake.input["cfd_strike_prices"],
            index_col=0,
            parse_dates=True,
        )
        cfd_strike_prices.columns = pd.to_datetime(cfd_strike_prices.columns)
        cfd_strike_prices = cfd_strike_prices.loc[:, :who.snapshots[4]].iloc[:,-1]

        revenue_rows = []

        # Process generator revenues: compute base wholesale and additional CFD revenue if applicable.

        for gen, bus in who.generators["bus"].items():
            if 'local_market' in gen:
                continue

            wholesale = 0.5 * (who.generators_t.p[gen] * who.buses_t.marginal_price[bus]).sum()

            cfd = 0.0
            if gen in cfd_strike_prices.index:
                strike_price = cfd_strike_prices.loc[gen]
                bus_mp = who.buses_t.marginal_price[bus]
                # Identify snapshots where the marginal price has been negative for the last 6 hours (i.e. 12 consecutive 30-min intervals)
                negative_streak = bus_mp.rolling(window=12, min_periods=12).max() < 0
                payment_mask = ~negative_streak.fillna(False)
                price_gap = strike_price - bus_mp
                cfd = 0.5 * (who.generators_t.p[gen] * price_gap * payment_mask).sum()

            roc = 0.0
            if gen in roc_values.index:
                roc = who.generators_t.p[gen].sum() * 0.5 * roc_values.loc[gen]

            revenue_rows.append({
                "asset": gen,
                "asset_type": "generator",
                "wholesale_revenue": wholesale,
                "cfd_revenue": cfd,
                "roc_revenue": roc
            })

        # Process storage unit revenues: compute base wholesale, additional CFD revenue, and ROC revenue if applicable.
        for unit, bus in who.storage_units["bus"].items():
            wholesale = 0.5 * (who.storage_units_t.p[unit] * who.buses_t.marginal_price[bus]).sum()

            cfd = 0.0
            if unit in cfd_strike_prices.index:
                strike_price = cfd_strike_prices.loc[unit]
                bus_mp = who.buses_t.marginal_price[bus]
                # Identify snapshots where the marginal price has been negative for the last 6 hours (i.e. 12 consecutive 30-min intervals)
                negative_streak = bus_mp.rolling(window=12, min_periods=12).max() < 0
                payment_mask = ~negative_streak.fillna(False)
                price_gap = strike_price - bus_mp
                cfd = 0.5 * (who.storage_units_t.p[unit] * price_gap * payment_mask).sum()

            roc = 0.0
            if unit in roc_values.index:
                unit_dispatch = who.storage_units_t.p[unit].clip(lower=0).sum()
                roc = unit_dispatch * 0.5 * roc_values.loc[unit]

            revenue_rows.append({
                "asset": unit,
                "asset_type": "storage",
                "wholesale_revenue": wholesale,
                "cfd_revenue": cfd,
                "roc_revenue": roc
            })

        # Process interconnector link revenues: these only have the wholesale component.
        intercon_links = who.links.index[who.links.carrier == "interconnector"]
    
        for link in intercon_links:
            bus0 = who.links.at[link, "bus0"]
            bus1 = who.links.at[link, "bus1"]
            price_bus0 = who.buses_t.marginal_price[bus0]
            price_bus1 = who.buses_t.marginal_price[bus1]
            # wholesale = 0.5 * ((who.links_t.p0[link] * price_bus0 + who.links_t.p1[link] * price_bus1).sum())
            wholesale = (
                0.5
                * (who.links_t.p0[link].abs() * (price_bus0 - price_bus1).abs())
                .sum()
            )
            revenue_rows.append({
                "asset": link,
                "asset_type": "interconnector",
                "wholesale_revenue": wholesale,
                "cfd_revenue": 0.0,
                "roc_revenue": 0.0
            })

        detailed_wholesale_revenue = pd.DataFrame(revenue_rows).set_index("asset")

        detailed_wholesale_revenue.to_csv(snakemake.output[f"bmu_revenues_detailed_{layout}"])

    l = who.loads.index[who.loads.carrier == 'electricity']
    who.loads_t.p_set.loc[:, l].sum(axis=1).to_csv(snakemake.output[f"gb_total_load"])
