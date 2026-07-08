"""Aggregate per-day model outputs into summaries/*.csv for the 2022-2025 figures.

A fresh replacement for docs/gather_all.py (which is hardcoded to 2024, drops its
`mode` arg, and crashes when frontend/ is missing). This iterates over every solved
day present on disk (skipping the handful of incomplete days) and writes the exact
summary files consumed by the new_plots/fig*.py scripts.

Run once, after new_plots/run_frontend_all.sh has populated frontend/:

    python new_plots/build_summaries.py

Outputs (all under summaries/):
  results/-derived : total_summary_flex.csv, marginal_prices_summary_flex.csv,
                     total_gb_load.csv
  frontend/-derived: total_unit_revenues_flex.csv, total_unit_dispatch_flex.csv,
                     total_intercon_dispatch_flex.csv, total_thermal_dispatch_flex.csv,
                     total_marginal_costs_flex.csv
"""
import re
import glob
from pathlib import Path

import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"
FRONTEND = REPO_ROOT / "frontend"
SUMMARIES = REPO_ROOT / "summaries"
SUMMARIES.mkdir(exist_ok=True)

# solved days = those with a system cost summary; sorted, spans 2022-2025
DAYS = sorted(
    re.search(r"(\d{4}-\d{2}-\d{2})", f).group(1)
    for f in glob.glob(str(RESULTS / "*" / "system_cost_summary_flex.csv"))
)
print(f"solved days on disk: {len(DAYS)}  ({DAYS[0]} .. {DAYS[-1]})")


def _read(path, **kw):
    try:
        return pd.read_csv(path, **kw)
    except FileNotFoundError:
        return None


# ---- results/-derived (per-day CSVs, no frontend needed) -----------------------
total_summary, marginal_prices, gb_load = [], [], []
# ---- frontend/-derived ---------------------------------------------------------
unit_revenues, unit_dispatch = [], []
intercon_dispatch, thermal_dispatch, marginal_costs = [], [], []

miss_results, miss_frontend = [], []

for day in tqdm(DAYS, desc="aggregating"):
    rp, fp = RESULTS / day, FRONTEND / day

    sc = _read(rp / "system_cost_summary_flex.csv", index_col=[0, 1])
    mp = _read(rp / "marginal_prices_flex.csv", header=[0, 1], index_col=0)
    ld = _read(rp / "gb_total_load_flex.csv", index_col=0)
    if sc is None or mp is None or ld is None:
        miss_results.append(day)
    else:
        total_summary.append(sc)
        marginal_prices.append(mp)
        gb_load.append(ld)

    rev = _read(fp / "revenues_flex.csv", index_col=0, header=[0, 1, 2])
    dis = _read(fp / "dispatch_flex.csv", index_col=0, header=[0, 1, 2])
    ic = _read(fp / "dispatch_flex_flex_intercon.csv", index_col=0, header=[0, 1])
    th = _read(fp / "thermal_dispatch_flex.csv", index_col=0, header=[0, 1])
    mc = _read(fp / "marginal_costs_flex.csv", index_col=0)
    if any(x is None for x in (rev, dis, ic, th, mc)):
        miss_frontend.append(day)
    else:
        # per-day unit revenue/dispatch -> one column per day, index (unit,layout,stream)
        unit_revenues.append(rev.sum().rename(day))
        unit_dispatch.append(dis.sum().rename(day))
        # time-series frontends -> stack rows across days
        intercon_dispatch.append(ic)
        thermal_dispatch.append(th)
        marginal_costs.append(mc)

print(f"missing results-day inputs : {len(miss_results)} -> {miss_results[:8]}")
print(f"missing frontend-day inputs: {len(miss_frontend)} -> {miss_frontend[:8]}")


def _write(obj, name):
    obj.to_csv(SUMMARIES / name)
    print(f"  wrote {name:42s} shape={tuple(obj.shape)}")


print("writing summaries ...")
_write(pd.concat(total_summary), "total_summary_flex.csv")
_write(pd.concat(marginal_prices), "marginal_prices_summary_flex.csv")
_write(pd.concat(gb_load), "total_gb_load.csv")

_write(pd.concat(unit_revenues, axis=1), "total_unit_revenues_flex.csv")
_write(pd.concat(unit_dispatch, axis=1), "total_unit_dispatch_flex.csv")
_write(pd.concat(intercon_dispatch), "total_intercon_dispatch_flex.csv")
_write(pd.concat(thermal_dispatch), "total_thermal_dispatch_flex.csv")
_write(pd.concat(marginal_costs), "total_marginal_costs_flex.csv")

print("done.")
