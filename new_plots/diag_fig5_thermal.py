"""Diagnostic for Fig 5 (annual unit revenues) -- thermal-generator surplus path.

Fig 5 plots each unit's producer-SURPLUS change (zonal vs national, %) vs latitude.
Non-thermal units use a pure revenue ratio; thermal units are OVERRIDDEN by a
profit ratio (revenue - fuel - balancing). This diagnostic isolates the thermal
override and shows, unit by unit, what the figure actually computes vs what a
unit-consistent computation gives.

KEY BUG (inherited from plotting/unit_revenues.py cell 37-39): thermal `expenses`
are built in £-million (get_plant_expenses(...).mul(-1e-6)) but concatenated with
`revenue` in raw £ (n_total, no 1e-6). Profit = (revenue - expenses)*1e-6 then
annihilates the fuel term by ~1e6, so national_profit ~= revenue and the thermal
"surplus change" is really a REVENUE ratio. Fuel is ~85% of thermal revenue.

Dispatch units are FINE: total_unit_dispatch_flex already carries *0.5 (MWh) from
summarize_frontend_data.calculate_dispatch_volumes -- so this is NOT a 0.5 issue,
it is a pure £/£m scaling+sign mismatch.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _helpers import SUMMARIES, FIG_DIR, get_gas_proxy, idx
from _unit_surplus import build, THERMAL_ETA, BALANCING_MARKUP

FIG_DIR.mkdir(parents=True, exist_ok=True)

disp = pd.read_csv(SUMMARIES / "total_unit_dispatch_flex.csv", index_col=[0, 1, 2])
d = build()
no, zo = d.national_overview, d.zonal_overview
th = no.index

# ---- code path (as plotted) ----
code_pc = (zo["profit"].astype(float) / no["profit"].astype(float) * 100 - 100)

# ---- corrected path: unit-consistent raw £, fuel included, single balancing markup ----
fp = get_gas_proxy(eta=THERMAL_ETA)
hold = fp.copy(); hold.index = fp.index.strftime("%Y-%m-%d")


def components(layout, rev):
    ss = disp.loc[idx[th, layout, :], :].copy(); ss.index = ss.index.droplevel(1)
    red = ss.loc[idx[:, "redispatch"], :]; red.index = red.index.get_level_values(0)
    who = ss.loc[idx[:, "wholesale"], :]; who.index = who.index.get_level_values(0)
    com = red.columns.intersection(hold.index)
    fuel = pd.Series(np.nansum(red[com].values * hold.loc[com].values.reshape(1, -1) / THERMAL_ETA, 1),
                     index=red.index)                                   # £ (dispatch already MWh)
    upbal = (red[com] - who[com]).clip(lower=0).sum(axis=1) * BALANCING_MARKUP  # £
    profit = (rev.loc[th] - fuel - upbal) * 1e-6                        # £m
    return fuel, upbal, profit


fuel_n, bal_n, prof_n = components("national", d.n_total)
fuel_z, bal_z, prof_z = components("zonal", d.zonal_total)
corr_pc = (prof_z / prof_n * 100 - 100)

rev_n = (d.n_total.loc[th] * 1e-6)                                      # £m
order = rev_n.sort_values(ascending=False).index
xi = np.arange(len(order))
carr = [d.bmu_carriers.get(u, "fossil") for u in order]
ccol = {"fossil": "#f6986b", "biomass": "#dbc263", "coal": "dimgray"}
bar_c = [ccol.get(c, "#f6986b") for c in carr]

panels = []
panels.append(("1. Thermal revenue vs fuel cost vs balancing markup (£m, national, per unit)  -- fuel dominates",
    lambda ax: (ax.bar(xi - 0.25, rev_n.loc[order], width=0.25, color="#2c7fb8", label="revenue"),
                ax.bar(xi, (fuel_n * 1e-6).loc[order], width=0.25, color="#e34a33", label="fuel cost"),
                ax.bar(xi + 0.25, (bal_n * 1e-6).loc[order], width=0.25, color="#31a354", label="balancing markup"),
                ax.legend(fontsize=7, ncol=3), ax.set_ylabel("£m (4 yr)")),
    f"fuel is ~{100*(fuel_n.sum()/d.n_total.loc[th].sum()):.0f}% of thermal revenue; the code drops it entirely."))

panels.append(("2. National thermal surplus: code (fuel dropped) vs corrected (fuel in), £m per unit",
    lambda ax: (ax.bar(xi - 0.2, no["profit"].astype(float).loc[order], width=0.4, color="#fdae6b", label="code (£%.0fbn)" % (no["profit"].astype(float).sum()/1e3)),
                ax.bar(xi + 0.2, prof_n.loc[order], width=0.4, color="#756bb1", label="corrected (£%.0fbn)" % (prof_n.sum()/1e3)),
                ax.legend(fontsize=7), ax.set_ylabel("£m surplus (4 yr)")),
    "code ~= revenue (fuel gone); corrected is the thin true margin -> ~6.6x smaller."))

panels.append(("3. Plotted quantity: thermal surplus %-change (zonal vs national) -- code vs corrected",
    lambda ax: (ax.bar(xi - 0.2, code_pc.loc[order], width=0.4, color="#fdae6b", label=f"code (median {code_pc.median():.0f}%)"),
                ax.bar(xi + 0.2, corr_pc.loc[order].clip(upper=800), width=0.4, color="#756bb1", label=f"corrected (median {corr_pc.median():.0f}%, clipped 800)"),
                ax.axhspan(-58, 47, color="grey", alpha=0.15, label="current fig y-range"),
                ax.legend(fontsize=7, ncol=3), ax.set_ylabel("% change")),
    "the bug COMPRESSES %-change into the plot window; corrected values fly off-scale (thin margins)."))

panels.append(("4. Gas proxy (daily fleet-mean thermal SRMC x eta, £/MWh) -- the sensitivity driver",
    lambda ax: (ax.plot(fp.index, fp.values, color="k", lw=0.6),
                ax.set_ylabel("£/MWh"), ax.axhline(fp.mean(), color="r", lw=0.6, ls="--")),
    f"mean {fp.mean():.0f}, range {fp.min():.0f}-{fp.max():.0f}; corrected thermal margin (hence %-change) rides on this."))

n = len(panels)
fig, axs = plt.subplots(n, 1, figsize=(13, 2.5 * n))
for ax, (title, draw, note) in zip(axs, panels):
    draw(ax)
    ax.set_title(title, fontsize=9, loc="left", weight="bold")
    ax.text(0.004, 0.02, "note: " + note, transform=ax.transAxes, fontsize=7,
            style="italic", color="#444", va="bottom")
    ax.grid(True, ls="--", lw=0.4, alpha=0.6)
    ax.set_axisbelow(True)
for ax in axs[:3]:
    ax.set_xticks(xi[::3]); ax.set_xticklabels(order[::3], rotation=90, fontsize=5)
fig.suptitle("Fig 5 thermal audit -- fuel-cost annihilation bug (£m vs raw £) and its impact", fontsize=12, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.99])

out = FIG_DIR / "diag_fig5_thermal.pdf"
fig.savefig(out, bbox_inches="tight")
fig.savefig(FIG_DIR / "diag_fig5_thermal.png", dpi=130, bbox_inches="tight")
print("wrote", out)
print(f"thermal national surplus: code £{no['profit'].astype(float).sum()/1e3:.1f}bn  corrected £{prof_n.sum()/1e3:.1f}bn")
print(f"thermal %-change median : code {code_pc.median():.0f}%  corrected {corr_pc.median():.0f}%")
