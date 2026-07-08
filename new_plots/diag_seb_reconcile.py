"""Reconcile Fig 8 (top-down surplus) vs Fig 9 (bottom-up efficiency) welfare gain.

Computes both SEB totals over the full 2022-2025 window and decomposes the gap
term-by-term, to locate where they diverge.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm

from _helpers import (
    YEARS, DATA_START, DATA_END, idx,
    get_unit_revenue, get_wholesale_expenses, get_balancing_expenses,
    get_ic_congestion_rent, get_import_cost, get_export_revenues, get_unit_schedule,
)
from _seb_common import load

d = load()
markup = d.balancing_markup

ts = pd.date_range(f"{DATA_START}", f"{DATA_END} 23:30:00", freq="30min")
ts = ts.intersection(d.consumer_cost.index.get_level_values(0))

# =================== FIG 8 top-down ===================
def thermal_unit_surplus(unit, layout):
    revenue = get_unit_revenue(d.revenues, unit, layout, ts)
    wholesale = get_wholesale_expenses(d.thermal_dispatch, d.marginal_cost, unit, layout, ts)
    balancing = get_balancing_expenses(d.dispatch, d.thermal_dispatch, d.marginal_cost, markup, unit, layout, ts)
    s = revenue - wholesale - balancing
    return 0 if pd.isna(s) else s

def get_surplus(layout):
    thermal = nonthermal = 0.0
    th_rev = th_we = th_be = 0.0
    for unit in d.all_units:
        if unit in d.thermal_units:
            thermal += thermal_unit_surplus(unit, layout)
            th_rev += get_unit_revenue(d.revenues, unit, layout, ts)
            th_we += get_wholesale_expenses(d.thermal_dispatch, d.marginal_cost, unit, layout, ts)
            bb = get_balancing_expenses(d.dispatch, d.thermal_dispatch, d.marginal_cost, markup, unit, layout, ts)
            th_be += 0 if pd.isna(bb) else bb
        else:
            nonthermal += get_unit_revenue(d.revenues, unit, layout, ts)
    return dict(thermal=thermal, nonthermal=nonthermal, th_rev=th_rev, th_we=th_we, th_be=th_be)

print("computing producer surplus (zonal / national) ...")
zon = get_surplus("zonal")
nat = get_surplus("national")

zonal_ps = (zon["thermal"] + zon["nonthermal"]) * 1e-9
national_ps = (nat["thermal"] + nat["nonthermal"]) * 1e-9

cc_zonal = (d.consumer_cost.loc[idx[ts, :], "zonal"] * 1e-3).groupby(level=1).sum()
cc_national = (d.consumer_cost.loc[idx[ts, :], "national"] * 1e-3).groupby(level=1).sum()
cc_diff = cc_national - cc_zonal  # £bn, consumer benefit of zonal per component

net_consumer_benefit = cc_diff.sum()
producer_surplus = zonal_ps - national_ps
cfd_cons = cc_diff["cfd_payments"]
roc_cons = cc_diff["roc_payments"]
ic_rent_zon = get_ic_congestion_rent(d.cons, d.marginal_prices, d.europe_prices, d.network, "zonal", ts) * 1e-9
ic_rent_nat = get_ic_congestion_rent(d.cons, d.marginal_prices, d.europe_prices, d.network, "national", ts) * 1e-9
ic_rent = ic_rent_zon - ic_rent_nat

fig8_seb = net_consumer_benefit + producer_surplus - cfd_cons - roc_cons + ic_rent

print("\n===== FIG 8 (top-down) £bn =====")
print("consumer cost diff components (nat - zon):")
print(cc_diff.to_string())
print(f"  net_consumer_benefit         {net_consumer_benefit:8.3f}")
print(f"  producer_surplus (zon-nat)   {producer_surplus:8.3f}")
print(f"     zonal PS  thermal {zon['thermal']*1e-9:7.3f}  nonthermal {zon['nonthermal']*1e-9:7.3f}")
print(f"     nat   PS  thermal {nat['thermal']*1e-9:7.3f}  nonthermal {nat['nonthermal']*1e-9:7.3f}")
print(f"  - cfd_payments_consumer      {-cfd_cons:8.3f}")
print(f"  - roc_payments_consumer      {-roc_cons:8.3f}")
print(f"  + ic_congestion_rent         {ic_rent:8.3f}")
print(f"  => FIG8 SEB                   {fig8_seb:8.3f}")

# =================== FIG 9 bottom-up ===================
print("\ncomputing bottom-up (thermal loop) ...")
we_nat = we_zon = be_nat = be_zon = 0.0
for unit in tqdm(d.thermal_units, desc="thermal"):
    we_nat += get_wholesale_expenses(d.thermal_dispatch, d.marginal_cost, unit, "national", ts) or 0.0
    we_zon += get_wholesale_expenses(d.thermal_dispatch, d.marginal_cost, unit, "zonal", ts) or 0.0
    bn = get_balancing_expenses(d.dispatch, d.thermal_dispatch, d.marginal_cost, markup, unit, "national", ts)
    be_nat += 0 if pd.isna(bn) else bn
    bz = get_balancing_expenses(d.dispatch, d.thermal_dispatch, d.marginal_cost, markup, unit, "zonal", ts)
    be_zon += 0 if pd.isna(bz) else bz

ic_ts = ts.intersection(d.exports.index).intersection(d.imports.index)
import_cost_diff = (get_import_cost(d.imports, d.europe_prices, ic_ts, "national", d.network)
                    - get_import_cost(d.imports, d.europe_prices, ic_ts, "zonal", d.network)) * 1e-9
export_rev_diff = (get_export_revenues(d.exports, d.marginal_prices, ic_ts, "zonal", d.network)
                   - get_export_revenues(d.exports, d.marginal_prices, ic_ts, "national", d.network)) * 1e-9
cr_diff = (get_ic_congestion_rent(d.cons, d.marginal_prices, d.europe_prices, d.network, "zonal", ic_ts)
           - get_ic_congestion_rent(d.cons, d.marginal_prices, d.europe_prices, d.network, "national", ic_ts)) * 1e-9
prevented_we = (we_nat - we_zon) * 1e-9
prevented_be = (be_nat - be_zon) * 1e-9

fig9_seb = import_cost_diff + export_rev_diff + cr_diff + prevented_we + prevented_be
print("\n===== FIG 9 (bottom-up) £bn =====")
print(f"  prevented_wholesale (we_nat-we_zon)  {prevented_we:8.3f}")
print(f"  prevented_balancing (be_nat-be_zon)  {prevented_be:8.3f}")
print(f"  import_costs_diff                    {import_cost_diff:8.3f}")
print(f"  export_revenues_diff                 {export_rev_diff:8.3f}")
print(f"  congestion_rent_diff                 {cr_diff:8.3f}")
print(f"  => FIG9 SEB                          {fig9_seb:8.3f}")

# =================== reconciliation ===================
# Fig8_core - Fig9_core, IC rent term is shared (ic_rent vs cr_diff -- should match)
print("\n===== RECONCILIATION =====")
print(f"  IC rent: fig8 {ic_rent:.3f}  vs  fig9 cr_diff {cr_diff:.3f}  (should match)")
print(f"  FIG8 SEB {fig8_seb:.3f}   FIG9 SEB {fig9_seb:.3f}   GAP {fig8_seb - fig9_seb:.3f}")

# expand: Fig8_thermalPS contributes Δwe+Δbe + Δrev_th ; Fig9 has Δwe+Δbe
d_rev_th = (zon["th_rev"] - nat["th_rev"]) * 1e-9
d_we = (nat["th_we"] - zon["th_we"]) * 1e-9   # prevented sign
d_be = prevented_be
d_ps_nonthermal = (zon["nonthermal"] - nat["nonthermal"]) * 1e-9
print(f"\n  thermal rev change (zon-nat)   {d_rev_th:8.3f}")
print(f"  thermal prevented we (nat-zon) {d_we:8.3f}  (fig9 prevented_we {prevented_we:.3f})")
print(f"  nonthermal PS change (zon-nat) {d_ps_nonthermal:8.3f}")
print(f"  consumer benefit               {net_consumer_benefit:8.3f}")
print(f"  -cfd_cons -roc_cons            {-cfd_cons-roc_cons:8.3f}")
print(f"  import+export diff             {import_cost_diff+export_rev_diff:8.3f}")
# identity check: ΔCS + Δrev_th + ΔPS_nonthermal - Δcfd_cons - Δroc_cons  ?=  Δimport+Δexport
lhs = net_consumer_benefit + d_rev_th + d_ps_nonthermal - cfd_cons - roc_cons
rhs = import_cost_diff + export_rev_diff
print(f"\n  identity LHS (CS+drev_th+dPSnt-cfd-roc) {lhs:8.3f}")
print(f"  identity RHS (import+export diff)       {rhs:8.3f}")
print(f"  identity residual (=GAP)                {lhs - rhs:8.3f}")
