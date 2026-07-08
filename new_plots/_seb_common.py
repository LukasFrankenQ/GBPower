"""Shared data loading for Figs 8 & 9 (waterfall + bottom-up socioeconomic benefit).

Ports the load cells (16 / 28) of plotting/get_total_results.py. Returns a namespace with
the summary frames, interconnector flows, European + GB marginal prices, the geometry
network, and the thermal/wind/all unit lists. The per-figure compute (get_surplus for the
waterfall; calculate_bottom_up_SEB for the SEB panels) lives in the fig scripts.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from types import SimpleNamespace

import yaml
import pypsa
import pandas as pd

from _helpers import (
    REPO_ROOT, SUMMARIES, get_consumer_cost, get_dispatch_data, get_revenue_data,
    get_marginal_costs_data, get_thermal_dispatch_data, get_all_bmus,
    get_interconnector_dispatch, get_europe_prices, get_marginal_prices,
)

BALANCING_MARKUP = 30  # £/MWh
GEOMETRY_NETWORK = REPO_ROOT / "results" / "2024-03-21" / "network_flex_s_nodal.nc"


def load():
    with open(REPO_ROOT / "data" / "interconnection_helpers.yaml") as f:
        interconnection_capacities = yaml.safe_load(f)

    consumer_cost = get_consumer_cost(SUMMARIES / "total_summary_flex.csv")
    dispatch = get_dispatch_data(SUMMARIES / "total_unit_dispatch_flex.csv")
    revenues = get_revenue_data(SUMMARIES / "total_unit_revenues_flex.csv")
    marginal_cost = get_marginal_costs_data(SUMMARIES / "total_marginal_costs_flex.csv")
    thermal_dispatch = get_thermal_dispatch_data(SUMMARIES / "total_thermal_dispatch_flex.csv")
    marginal_prices = get_marginal_prices(SUMMARIES / "marginal_prices_summary_flex.csv")
    cons = get_interconnector_dispatch(SUMMARIES / "total_intercon_dispatch_flex.csv")

    imports = cons.clip(lower=0)
    exports = cons.clip(upper=0).abs()

    europe_prices = get_europe_prices(REPO_ROOT / "data" / "base")
    network = pypsa.Network(GEOMETRY_NETWORK)
    bmus = get_all_bmus()
    all_units = dispatch.index.get_level_values(0).unique()

    def get_units(carrier):
        if isinstance(carrier, list):
            return bmus.loc[bmus.carrier.isin(carrier)].index
        if isinstance(carrier, str) and carrier != "interconnector":
            return bmus.loc[bmus.carrier.str.contains(carrier)].index
        return pd.Index(list(interconnection_capacities["interconnection_mapper"].keys()))

    # restrict to units actually present in the frames we index (some carrier BMUs
    # never appear in the model dispatch; without this the per-unit loops KeyError).
    disp_units = set(all_units)
    td_units = set(thermal_dispatch.columns.get_level_values(1))
    mc_units = set(marginal_cost.columns)
    thermal_units = get_units(["coal", "biomass", "fossil"])
    thermal_units = thermal_units[thermal_units.isin(disp_units & td_units & mc_units)]
    wind_units = get_units(["onwind", "offwind"])
    wind_units = wind_units[wind_units.isin(disp_units)]

    return SimpleNamespace(
        consumer_cost=consumer_cost, dispatch=dispatch, revenues=revenues,
        marginal_cost=marginal_cost, thermal_dispatch=thermal_dispatch,
        marginal_prices=marginal_prices, cons=cons, imports=imports, exports=exports,
        europe_prices=europe_prices, network=network, bmus=bmus,
        all_units=all_units, thermal_units=thermal_units, wind_units=wind_units,
        balancing_markup=BALANCING_MARKUP,
    )
