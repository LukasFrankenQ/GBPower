"""Shared loaders + constants for the 2022-2025 paper-figure rebuilds (new_plots/).

Adapted from plotting/data_getter.py and plotting/plotting_constants.py, with three
deliberate changes for the four-year (2022-2025) rebuild:

  1. FOUR-YEAR DEFAULTS. YEARS/N_YEARS span 2022-2025; get_europe_prices() ingests
     through 2025-12-31 (the original stopped at 2024-12-31).
  2. GAS-PRICE PROXY. The paper's thermal running-cost figures (Figs 5 & 7) relied on
     an external NG "System Average Price of Gas" spreadsheet that is not in the repo.
     get_gas_proxy() reconstructs an equivalent daily system-wide thermal cost series
     from the model's own marginal-cost assets (summaries/total_marginal_costs_flex.csv):
     the daily fleet-mean electricity marginal cost x eta. Because get_plant_expenses()
     divides the series by eta again, the net effect is expenses = dispatch x daily-mean
     thermal MC -- faithful to the model, but an approximation of the paper's gas index.
     GAS_PROXY_CAVEAT is printed onto Figs 5 & 7 so the substitution is visible.
  3. Output paths point at new_plots/figures_2022-2025/ with fresh filenames so nothing
     overwrites the paper's ../lmp_paper/imgs/ originals.

Every fig*.py script does `from _helpers import ...` after inserting its own dir on the
path, so the scripts are runnable from any working directory.
"""
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SUMMARIES = REPO_ROOT / "summaries"
FIG_DIR = Path(__file__).resolve().parent / "figures_2022-2025"

idx = pd.IndexSlice

# ---- four-year window ----------------------------------------------------------
YEARS = ["2022", "2023", "2024", "2025"]
N_YEARS = len(YEARS)
DATA_START = "2022-01-01"
DATA_END = "2025-12-31"

GAS_PROXY_CAVEAT = (
    "Thermal running cost derived from the model's own marginal-cost assets\n"
    "(daily fleet-mean thermal SRMC), not the NG gas index used in the paper.\n"
    "Absolute thermal-unit surplus levels are sensitive to this and to the\n"
    "assumed £30/MWh balancing premium; relative (zonal vs national) changes are robust."
)

# ================================================================================
# constants (from plotting/plotting_constants.py)
# ================================================================================
width_unit = 6
height_unit = 4

policy_colors = {
    "national": "red",
    "no_ftr_zonal": "rebeccapurple",   # policy 1
    "zonal": "lime",                    # policy 2
    "zonal_with_policy": "deepskyblue", # policy 3
}

nice_names = {
    "wholesale": "Wholesale Market",
    "wholesale buying": "Electricity Procurement",
    "roc_payments": "Renewable Obligation Certificates",
    "cfd_payments": "Contracts for Differences",
    "congestion_rent": "Intra-GB Congestion Rents",
    "offer_cost": "Redispatch - Offers",
    "bid_cost": "Redispatch - Bids",
}

color_dict = {
    "wholesale": "#F78C6B",
    "wholesale selling": "#F78C6B",
    "wholesale buying": "#c3763c",
    "roc_payments": "#EF476F",
    "cfd_payments": "#06D6A0",
    "congestion_rent": "#FFD166",
    "offer_cost": "#073B4C",
    "bid_cost": "#118AB2",
}

carrier_colors = {
    "onwind": "#7ac677", "offwind": "#6895dd", "hydro": "#0079c1",
    "biomass": "#dbc263", "fossil": "#f6986b", "nuclear": "orange",
    "imports": "#dd75b0", "cascade": "#0079c1", "PHS": "#46caf0",
    "solar": "#f9d002", "battery": "darkred", "interconnector": "plum",
}

nice_carrier_names = {
    "onwind": "Onshore Wind", "offwind": "Offshore Wind", "hydro": "Hydro",
    "biomass": "Biomass", "fossil": "Fossil", "nuclear": "Nuclear",
    "imports": "Imports", "cascade": "Cascade", "PHS": "Pumped Hydro Storage",
    "solar": "Solar", "battery": "Battery", "interconnector": "Interconnector",
}

bmus_alldata = pd.read_csv(REPO_ROOT / "data" / "prerun" / "prepared_bmus.csv", index_col=0)


def stack_to_ax(df, ax, text_y_offset=0.2):
    s = df.sum()
    x_loc = "{}-{}".format(df.index[0][0], df.index[0][1])
    pos = s.loc[s > 0]
    neg = s.loc[s < 0]
    pos_quants = [q for q in list(nice_names) if q in pos.index]
    neg_quants = [q for q in list(nice_names) if q in neg.index]
    pos = pos.loc[pos_quants]
    pos_bottom = [0] + pos.cumsum().tolist()[:-1]
    for (name, value), b in zip(pos.items(), pos_bottom):
        ax.bar(x_loc, bottom=b, height=value, label=nice_names[name], color=color_dict[name])
    neg = neg.loc[neg_quants]
    neg_bottom = [0] + neg.cumsum().tolist()[:-1]
    for (name, value), b in zip(neg.items(), neg_bottom):
        ax.bar(x_loc, bottom=0, height=value, label=nice_names[name], color=color_dict[name])
    ax.text(x_loc, pos.sum() + text_y_offset, "{}".format(np.around(s.sum(), decimals=1)), ha="center")
    ax.scatter([x_loc], [s.sum()], color="w", zorder=10, edgecolor="k", s=50)


def plot_stacked_horizontal_bar(series, ax, right_tick=True):
    pos_series = series[series > 0].copy().sort_values(ascending=False)
    neg_series = series[series < 0].copy().sort_values(ascending=True)
    total_neg = neg_series.sum()
    left = total_neg
    for i, val in neg_series.items():
        width = abs(val)
        left += width
        ax.barh(0, -width, left=left, height=0.5, label=i, color=color_dict.get(i, "gray"))
    left = 0
    for i, val in pos_series.items():
        ax.barh(0, val, left=left, height=0.5, label=i, color=color_dict.get(i, "gray"))
        left += val
    ax.scatter([total_neg + pos_series.sum()], [0], color="white", edgecolor="black", s=100, marker="o")
    xticks = ax.get_xticks()
    more_ticks = [0, total_neg]
    if right_tick:
        more_ticks.append(total_neg + pos_series.sum())
    xticks = np.unique(np.concatenate([xticks, more_ticks]))
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{x:.1f}" for x in xticks])
    ax.set_yticks([])
    ax.set_xlabel("(£bn)")
    ax.set_xlim(total_neg, pos_series.sum())
    ax.set_ylim(-0.25, 0.25)
    return ax


# ================================================================================
# getters (from plotting/data_getter.py)
# ================================================================================
def get_all_bmus():
    bmus = bmus_alldata.copy()
    return bmus.loc[bmus.lon != "distributed"]


def get_units(carrier):
    if isinstance(carrier, list):
        return bmus_alldata.loc[bmus_alldata.carrier.isin(carrier)].index
    return bmus_alldata.loc[bmus_alldata.carrier.str.contains(carrier)].index


def get_interconnector_names():
    return ["IFA1", "Moyle", "BritNed", "IFA2", "EastWest", "Viking", "ElecLink", "NSL", "Nemo"]


def get_consumer_cost(fn):
    cc = pd.read_csv(fn, index_col=[0, 1], parse_dates=True)
    return cc.loc[~(cc.index.get_level_values(1) == "balancing_volume")]


def get_model_balancing_volume(fn):
    bv = pd.read_csv(fn, index_col=[0, 1], parse_dates=True)
    return bv.loc[bv.index.get_level_values(1) == "balancing_volume"]


def get_revenue_data(fn):
    rev = pd.read_csv(fn, index_col=[0, 1, 2], parse_dates=True)
    keepers = ["bid_cost", "cfd", "offer_cost", "roc", "wholesale"]
    rev = rev.loc[idx[:, :, keepers], :].sort_index().replace(np.nan, 0)
    return rev.loc[~rev.index.get_level_values(0).isin(get_interconnector_names())]


def get_dispatch_data(fn):
    disp = pd.read_csv(fn, index_col=[0, 1, 2], parse_dates=True)
    return disp.loc[~disp.index.get_level_values(0).isin(get_interconnector_names())]


def get_unit_schedule(data, unit, layout, mode, time_slice):
    if not isinstance(data.columns, pd.DatetimeIndex):
        data.columns = pd.to_datetime(data.columns)
    who = data.loc[(unit, layout, "wholesale"), time_slice.intersection(data.columns)].sum()
    if mode == "wholesale":
        return who
    return data.loc[(unit, layout, "redispatch"), time_slice.intersection(data.columns)].sum() - who


def get_unit_revenue(data, unit, layout, time_slice):
    if not isinstance(data.columns, pd.DatetimeIndex):
        data.columns = pd.to_datetime(data.columns)
    return data.loc[idx[unit, layout, :], time_slice.intersection(data.columns)].sum().sum()


def get_thermal_dispatch_data(fn):
    return pd.read_csv(fn, index_col=[0], parse_dates=[0], header=[0, 1]).replace(np.nan, 0) * 0.5


def get_marginal_costs_data(fn):
    return pd.read_csv(fn, index_col=[0], parse_dates=[0])


def get_wholesale_expenses(dispatch, marginal_cost, unit, layout, time_slice):
    avail = time_slice.intersection(marginal_cost.index).intersection(dispatch.index)
    assert unit in dispatch.columns.get_level_values(1) and unit in marginal_cost.columns, "thermal units only"
    expenses = pd.Series(
        dispatch.loc[avail, idx[layout, unit]] * marginal_cost.loc[avail, unit],
        index=avail, dtype=float, name=unit,
    ).replace(np.nan, 0)
    return expenses.sum()


def get_balancing_expenses(daily_dispatch, hh_dispatch, marginal_cost, balancing_markup, unit, layout, time_slice):
    avail = time_slice.intersection(marginal_cost.index).intersection(hh_dispatch.index)
    assert unit in hh_dispatch.columns.get_level_values(1) and unit in marginal_cost.columns, "thermal units only"
    basic_running_cost = (
        (hh_dispatch.loc[avail, idx[layout, unit]] * marginal_cost.loc[avail, unit]).sum()
        / hh_dispatch.loc[avail, idx[layout, unit]].sum()
    )
    balancing_volume = get_unit_schedule(daily_dispatch, unit, layout, "redispatch", time_slice)
    return balancing_volume * (basic_running_cost + balancing_markup)


def get_europe_prices(base_path, start=DATA_START, end=DATA_END):
    days = pd.date_range(start=start, end=end, freq="D").strftime("%Y-%m-%d")
    prices = []
    for day in days:
        try:
            prices.append(pd.read_csv(base_path / day / "europe_day_ahead_prices.csv", index_col=0, parse_dates=True))
        except FileNotFoundError:
            pass
    europe_prices = pd.concat(prices)
    europe_prices.index = pd.to_datetime(europe_prices.index).tz_localize(None)
    return europe_prices


def get_interconnector_dispatch(fn):
    return pd.read_csv(fn, index_col=[0], parse_dates=True, header=[0, 1])


def get_marginal_prices(fn):
    return pd.read_csv(fn, index_col=[0], parse_dates=True, header=[0, 1])


def get_gas_proxy(eta=0.52, marginal_costs=None):
    """Daily system-wide thermal-cost proxy replacing the missing NG gas spreadsheet.

    Returns a daily series (DatetimeIndex) = fleet-mean thermal electricity SRMC x eta,
    so that get_plant_expenses (which divides by eta) yields dispatch x daily-mean SRMC.
    """
    if marginal_costs is None:
        marginal_costs = get_marginal_costs_data(SUMMARIES / "total_marginal_costs_flex.csv")
    daily_mc = marginal_costs.mean(axis=1)                      # mean across thermal units
    daily_mc = daily_mc.groupby(daily_mc.index.normalize()).mean()
    daily_mc.index = pd.to_datetime(daily_mc.index)
    return (daily_mc * eta).rename("gas_proxy")


def get_ic_congestion_rent(ic_flow, marginal_prices_gb, eu_prices, network, layout, time_slice):
    if not isinstance(ic_flow.index, pd.DatetimeIndex):
        ic_flow.index = pd.to_datetime(ic_flow.index)
    inter = ic_flow.index.intersection(time_slice).intersection(eu_prices.index)
    ics = ic_flow.columns.get_level_values(0).unique()
    ic_flow = ic_flow.loc[inter, idx[:, layout]].replace(np.nan, 0.0)
    ic_flow.columns = ic_flow.columns.droplevel(1)
    zonal_intercon_buses = dict(network.links.loc[ics, "bus1"]) if layout == "zonal" else {}
    intercon_countries = network.links.loc[ics, "bus0"]
    congestion_rent = 0
    for ic in ics:
        gb_bus = zonal_intercon_buses.get(ic, "GB")
        country = intercon_countries[ic]
        gb_price = marginal_prices_gb.loc[inter, idx[layout, gb_bus]]
        country_price = eu_prices.loc[inter, country]
        congestion_rent += 0.5 * ((gb_price - country_price) * ic_flow.loc[inter, ic]).abs().sum()
    return congestion_rent


def get_export_revenues(exports, gb_prices, time_slice, layout, network):
    exports = exports.replace(np.nan, 0.0)
    if not isinstance(exports.index, pd.DatetimeIndex):
        exports.index = pd.to_datetime(exports.index)
    inter = time_slice.intersection(gb_prices.index).intersection(exports.index)
    buses = dict(network.links.loc[exports.columns.get_level_values(0).unique(), "bus1"]) if layout == "zonal" else {}
    revenues = 0
    for col in exports.columns.get_level_values(0).unique():
        bus = buses.get(col, "GB")
        revenues += exports.loc[inter, idx[col, layout]].mul(gb_prices.loc[inter, idx[layout, bus]]).sum()
    return revenues


def get_import_cost(imports, eu_prices, time_slice, layout, network):
    imports = imports.replace(np.nan, 0.0)
    if not isinstance(imports.index, pd.DatetimeIndex):
        imports.index = pd.to_datetime(imports.index)
    inter = time_slice.intersection(eu_prices.index).intersection(imports.index)
    intercon_countries = network.links.loc[imports.columns.get_level_values(0).unique(), "bus0"]
    cost = 0
    for col in imports.columns.get_level_values(0).unique():
        country = intercon_countries[col]
        cost += imports.loc[inter, idx[col, layout]].mul(eu_prices.loc[inter, country]).sum()
    return cost
