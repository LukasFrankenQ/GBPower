"""Decisive test: force the two leaking transfers (thermal bid_cost, CfD) to cancel in
Fig 8's top-down SEB, and check convergence to Fig 9's bottom-up SEB (full-window slice).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from _helpers import DATA_START, DATA_END, idx, get_unit_revenue
from _seb_common import load

d = load()
ts = pd.date_range(f"{DATA_START}", f"{DATA_END} 23:30:00", freq="30min")
ts = ts.intersection(d.consumer_cost.index.get_level_values(0))
rev = d.revenues
if not isinstance(rev.columns, pd.DatetimeIndex):
    rev.columns = pd.to_datetime(rev.columns)

FIG8 = 2.379   # from diag_seb_reconcile (top-down, full window)
FIG9 = 2.990   # from diag_seb_reconcile (bottom-up, full window)

# --- thermal bid_cost leak (Bug B): producer surplus wrongly credits +bid_cost on
#     thermal turndown. It enters Fig8 producer_surplus (zon-nat) but never Fig9.
th = d.thermal_units
bid_th_nat = rev.loc[idx[th, "national", "bid_cost"], rev.columns].sum().sum() * 1e-9
bid_th_zon = rev.loc[idx[th, "zonal", "bid_cost"], rev.columns].sum().sum() * 1e-9
# producer_surplus is (zonal - national); thermal bid_cost contributes (bid_th_zon - bid_th_nat)
ps_bid_contribution = bid_th_zon - bid_th_nat
print(f"thermal bid_cost  national £{bid_th_nat:.3f}bn  zonal £{bid_th_zon:.3f}bn")
print(f"  -> contribution to producer_surplus (zon-nat): {ps_bid_contribution:+.3f}bn")
print(f"  removing Bug B raises Fig8 by {-ps_bid_contribution:+.3f}bn -> {FIG8 - ps_bid_contribution:.3f}bn")

# --- CfD leak: producer cfd (redispatch basis) vs consumer cfd (wholesale basis).
cc_z = (d.consumer_cost.loc[idx[ts, :], "zonal"] * 1e-3).groupby(level=1).sum()
cc_n = (d.consumer_cost.loc[idx[ts, :], "national"] * 1e-3).groupby(level=1).sum()
cfd_cons_diff = (cc_n - cc_z)["cfd_payments"]                       # consumer nat-zon
cfd_prod_nat = rev.loc[idx[:, "national", "cfd"], rev.columns].sum().sum() * 1e-9
cfd_prod_zon = rev.loc[idx[:, "zonal", "cfd"], rev.columns].sum().sum() * 1e-9
cfd_prod_diff = cfd_prod_nat - cfd_prod_zon                        # producer nat-zon
# In Fig8: producer_surplus includes producer cfd (zon-nat) = -cfd_prod_diff;
# formula also subtracts cfd_payments_consumer (cfd_cons_diff). Net cfd in SEB currently:
cfd_in_seb_now = (-cfd_prod_diff) - cfd_cons_diff
# If cfd were a clean cancelling transfer it should contribute 0 to welfare.
print(f"\ncfd producer nat-zon {cfd_prod_diff:+.3f}bn   consumer nat-zon {cfd_cons_diff:+.3f}bn")
print(f"  cfd net contribution to Fig8 SEB now: {cfd_in_seb_now:+.3f}bn (should be ~0 for a transfer)")
print(f"  forcing cfd->0 changes Fig8 by {-cfd_in_seb_now:+.3f}bn")

fig8_corrected = FIG8 - ps_bid_contribution - cfd_in_seb_now
print(f"\nFig8 raw            {FIG8:.3f}bn")
print(f"Fig8 - Bug B - cfd  {fig8_corrected:.3f}bn")
print(f"Fig9 (bottom-up)    {FIG9:.3f}bn")
print(f"residual gap after removing both leaks: {fig8_corrected - FIG9:+.3f}bn")
