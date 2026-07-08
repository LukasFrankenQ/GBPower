"""Shared data prep for Figs 5 & 7 (asset-level producer-surplus change, national->zonal).

Ports cells 1-17, 37-41, 45-46 of plotting/unit_revenues.py, with two changes:
  - the missing NG gas spreadsheet is replaced by the model-derived daily thermal-cost
    proxy from _helpers.get_gas_proxy();
  - three-year annualisation constants (/3) become /N_YEARS (=4).

build() returns a namespace with everything the two figure scripts plot:
  x, perc_change, perc_change_2, perc_change_3, sizes, n_total, zonal_total,
  zonal_overview, national_overview, cfd, bmu_carriers, used_carriers, thermal_units.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from types import SimpleNamespace

import yaml
import numpy as np
import pandas as pd

from _helpers import (
    REPO_ROOT, SUMMARIES, N_YEARS, idx, carrier_colors, get_gas_proxy,
)

THERMAL_ETA = 0.52
BALANCING_MARKUP = 30  # £/MWh


def _get_units(bmus_alldata, carrier):
    if isinstance(carrier, list):
        return bmus_alldata.loc[bmus_alldata.carrier.isin(carrier)].index
    return bmus_alldata.loc[bmus_alldata.carrier.str.contains(carrier)].index


def get_plant_expenses(dispatch, mode, fp, thermal_plant_eta, balancing_markup):
    ss = dispatch.loc[idx[:, mode, :], :].copy()
    ss.index = ss.index.droplevel(1)
    red_ss = ss.loc[idx[:, "redispatch"], :]
    red_ss.index = red_ss.index.get_level_values(0)
    who_ss = ss.loc[idx[:, "wholesale"], :]
    who_ss.index = who_ss.index.get_level_values(0)
    bal_ss = (red_ss - who_ss).abs()
    total_mwh_delivered = red_ss
    ss = ss.groupby(level=0).sum()
    hold_fp = fp.copy()
    hold_fp.index = fp.index.strftime("%Y-%m-%d")
    common_dates = ss.columns.intersection(hold_fp.index)
    result = pd.DataFrame(index=ss.index, columns=ss.columns)
    result.loc[:, common_dates] = (
        total_mwh_delivered.values
        * hold_fp.loc[common_dates].values.reshape(1, -1)
        / thermal_plant_eta
    )
    # Bug A fix: return FUEL cost only (raw £, positive). The £30/MWh balancing-markup
    # premium is added once, on upward redispatch volume, in build() below -- adding the
    # |bal|*markup term here as well double-counts it. (Previously both were included and
    # the whole result was later .mul(-1e-6)'d into oblivion; see build().)
    return result.sum(axis=1)


def build():
    # ---- revenue frames (national / zonal), per unit per stream --------------
    p = pd.read_csv(SUMMARIES / "total_unit_revenues_flex.csv", parse_dates=True, index_col=[0, 1, 2])
    keepers = ["bid_cost", "cfd", "offer_cost", "roc", "wholesale"]
    z = p.loc[idx[:, "zonal", keepers], :]
    n = p.loc[idx[:, "national", keepers], :]
    z.index = z.index.droplevel(1)
    n.index = n.index.droplevel(1)
    rename_dict = {"cfd": "cfd_revenue", "roc": "roc_revenue", "wholesale": "wholesale_revenue"}
    z.index = pd.MultiIndex.from_tuples([(a, rename_dict.get(r, r)) for a, r in z.index], names=z.index.names)
    n.index = pd.MultiIndex.from_tuples([(a, rename_dict.get(r, r)) for a, r in n.index], names=n.index.names)
    n.sort_index(inplace=True)
    z.sort_index(inplace=True)

    cfd = pd.read_csv(REPO_ROOT / "data" / "prerun" / "cfd_strike_prices.csv", index_col=0)

    # ---- interconnector latitudes -------------------------------------------
    bmus = pd.read_csv(REPO_ROOT / "data" / "prerun" / "prepared_bmus.csv", index_col=0)
    with open(REPO_ROOT / "data" / "interconnection_helpers.yaml") as f:
        inter_bmu = yaml.safe_load(f)["interconnection_mapper"]
    intercon_lats = []
    for key, item in inter_bmu.items():
        if key == "Nemo":
            intercon_lats.append(0)
            continue
        lat = bmus.loc[bmus.index.str.startswith(item[0] + "-"), "lat"].dropna().astype(float).mean()
        intercon_lats.append(lat)
    intercon_lats = pd.Series(intercon_lats, index=inter_bmu.keys())
    intercon_lats.loc["EastWest"] = 53.23
    intercon_lats.loc["Moyle"] = 55.07
    intercon_lats.loc["Nemo"] = 51.297
    intercon_lats = intercon_lats.to_frame().rename(columns={0: "lat"})

    # ---- latitudes + carriers ------------------------------------------------
    bmus = pd.read_csv(REPO_ROOT / "data" / "prerun" / "prepared_bmus.csv", index_col=0)
    bmus = bmus.loc[~(bmus["lat"] == "distributed")]
    bmus.loc[:, "lat"] = bmus["lat"].astype(float)
    bmu_carriers = pd.concat((bmus["carrier"], pd.Series("interconnector", intercon_lats.index)))
    bmus = pd.concat([bmus["lat"].to_frame(), intercon_lats])

    bmus_alldata = pd.read_csv(REPO_ROOT / "data" / "prerun" / "prepared_bmus.csv", index_col=0)
    thermal_units = _get_units(bmus_alldata, ["fossil", "biomass", "coal"])

    # ---- thermal expenses via model-derived gas proxy ------------------------
    dispatch = pd.read_csv(SUMMARIES / "total_unit_dispatch_flex.csv", index_col=[0, 1, 2])
    fp = get_gas_proxy(eta=THERMAL_ETA)
    thermal_in_disp = pd.Index(thermal_units).intersection(dispatch.index.get_level_values(0).unique())
    # Bug A fix: keep expenses in RAW £, matching revenue (n_total / zonal_total, also raw
    # £). profit = (revenue - expenses)*1e-6 below then converts once, consistently. The old
    # .mul(-1e-6) put expenses in £m AND flipped the sign, so fuel (~85% of thermal revenue)
    # was ~1e6x too small to register and thermal "surplus" collapsed to ~= revenue.
    zonal_expenses = get_plant_expenses(
        dispatch.loc[idx[thermal_in_disp, :, :], :], "zonal", fp, THERMAL_ETA, BALANCING_MARKUP
    )
    national_expenses = get_plant_expenses(
        dispatch.loc[idx[thermal_in_disp, :, :], :], "national", fp, THERMAL_ETA, BALANCING_MARKUP
    )

    # ---- totals + thermal overview -------------------------------------------
    n_total = n.replace(np.nan, 0).groupby(level=0).sum().sum(axis=1)
    n_total = n_total[n_total > 0]
    zonal_total = z.replace(np.nan, 0).groupby(level=0).sum().sum(axis=1)
    zonal_total.drop(intercon_lats.index.intersection(zonal_total.index), inplace=True)
    n_total.drop(intercon_lats.index.intersection(n_total.index), inplace=True)
    zonal_total = zonal_total.loc[n_total.index]

    thermal_national_revenues = n_total.loc[national_expenses.index.intersection(n_total.index)]
    thermal_zonal_revenues = zonal_total.loc[zonal_expenses.index.intersection(zonal_total.index)]
    national_overview = pd.concat(
        [thermal_national_revenues.rename("revenue"), national_expenses.rename("expenses")], axis=1
    ).dropna()
    zonal_overview = pd.concat(
        [thermal_zonal_revenues.rename("revenue"), zonal_expenses.rename("expenses")], axis=1
    ).dropna()

    def balancing_vol(layout):
        bal = pd.DataFrame(0, index=thermal_in_disp, columns=dispatch.columns)
        bal += dispatch.loc[idx[thermal_in_disp, layout, "redispatch"]].values
        bal -= dispatch.loc[idx[thermal_in_disp, layout, "wholesale"]].values
        return bal.clip(lower=0).sum(axis=1)

    zonal_overview["expenses"] += balancing_vol("zonal").loc[zonal_overview.index] * BALANCING_MARKUP
    national_overview["expenses"] += balancing_vol("national").loc[national_overview.index] * BALANCING_MARKUP
    zonal_overview["profit"] = (zonal_overview["revenue"] - zonal_overview["expenses"]).mul(1e-6).sort_index()
    national_overview["profit"] = (national_overview["revenue"] - national_overview["expenses"]).mul(1e-6).sort_index()

    # ---- % change vs latitude (thermal overridden by profit ratio) -----------
    perc_change = ((zonal_total / n_total) * 100 - 100).astype(float)
    x = bmus.loc[n_total.index.intersection(bmus.index), "lat"].astype(float)
    perc_change = perc_change.loc[x.index]
    common = zonal_overview.index.intersection(perc_change.index)
    perc_change.loc[common] = (zonal_overview["profit"].loc[common] / national_overview["profit"].loc[common]) * 100 - 100

    used_carriers = set(bmu_carriers[b] for b in x.index)
    sizes = n_total * 1e-6 / N_YEARS  # £mn/year

    # ---- policy 2 / 3 percentage changes (Fig 7) -----------------------------
    real_revs = pd.read_csv(SUMMARIES / "total_unit_revenues_flex.csv", index_col=[0, 1, 2])
    real_revs = real_revs.loc[~real_revs.index.get_level_values(0).isin(intercon_lats.index)]
    simple = real_revs.sum(axis=1).groupby(level=[0, 1]).sum().unstack().dropna()
    perc_change_2 = (100 * simple["zonal"].div(simple["national"]) - 100).clip(upper=0)
    perc_change_3 = (100 * simple["equitable"].div(simple["national"]) - 100).clip(upper=0)

    return SimpleNamespace(
        x=x, perc_change=perc_change, perc_change_2=perc_change_2, perc_change_3=perc_change_3,
        sizes=sizes, n_total=n_total, zonal_total=zonal_total,
        zonal_overview=zonal_overview, national_overview=national_overview,
        cfd=cfd, bmus=bmus, bmu_carriers=bmu_carriers, used_carriers=used_carriers,
        thermal_units=thermal_units,
    )
