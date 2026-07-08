"""Measure transfer cancellation between top-down (Fig8) consumer cost and producer
revenue, stream by stream, to locate the top-down/bottom-up welfare gap.

For a clean identity, each consumer-cost stream Δ (nat-zon) should be offset by the
matching producer-revenue stream Δ (nat-zon), leaving only genuine efficiency terms.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from _helpers import YEARS, DATA_START, DATA_END, idx
from _seb_common import load

d = load()
ts = pd.date_range(f"{DATA_START}", f"{DATA_END} 23:30:00", freq="30min")
ts = ts.intersection(d.consumer_cost.index.get_level_values(0))

# --- consumer cost per stream, per layout (£bn) ---
cc_z = (d.consumer_cost.loc[idx[ts, :], "zonal"] * 1e-3).groupby(level=1).sum()
cc_n = (d.consumer_cost.loc[idx[ts, :], "national"] * 1e-3).groupby(level=1).sum()
cc_diff = cc_n - cc_z

# --- producer revenue per stream, per layout, summed over ALL units (£bn) ---
# revenues frame: index (unit, layout, stream), columns = days
rev = d.revenues
rev_cols = ts.intersection(pd.to_datetime(rev.columns)) if not isinstance(rev.columns, pd.DatetimeIndex) else ts.intersection(rev.columns)
if not isinstance(rev.columns, pd.DatetimeIndex):
    rev.columns = pd.to_datetime(rev.columns)
rev_cols = ts.normalize().unique().intersection(rev.columns)

streams = ["wholesale", "roc", "cfd", "offer_cost", "bid_cost"]
prod = {}
for layout in ["national", "zonal"]:
    s = {}
    for st in streams:
        try:
            s[st] = rev.loc[idx[:, layout, st], rev.columns].sum().sum() * 1e-9
        except KeyError:
            s[st] = np.nan
    prod[layout] = pd.Series(s)

prod_diff = prod["national"] - prod["zonal"]  # nat - zon, £bn

print("Producer revenue per stream (all units), £bn:")
comp = pd.DataFrame({"prod_national": prod["national"], "prod_zonal": prod["zonal"],
                     "prod_diff(n-z)": prod_diff})
print(comp.to_string())

print("\nTransfer cancellation check (consumer Δ should ~= producer Δ, nat-zon), £bn:")
# consumer cost stream names -> producer stream names
cmap = {"wholesale": "wholesale", "roc_payments": "roc", "cfd_payments": "cfd",
        "offer_cost": "offer_cost", "bid_cost": "bid_cost"}
rows = []
for cc_name, pr_name in cmap.items():
    c = cc_diff.get(cc_name, np.nan)
    p = prod_diff.get(pr_name, np.nan)
    rows.append((cc_name, c, p, c - p))
tbl = pd.DataFrame(rows, columns=["stream", "consumer_diff", "producer_diff", "leak(c-p)"]).set_index("stream")
print(tbl.to_string())
print(f"\ncongestion_rent (consumer only, no producer counterpart): {cc_diff.get('congestion_rent', np.nan):.3f}")
print(f"total consumer-only congestion leak stays in top-down as pure consumer benefit")
print(f"\nSUM of per-stream leaks (excl congestion): {tbl['leak(c-p)'].sum():.3f}")
