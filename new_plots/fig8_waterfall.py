"""Figure 8 (2022-2025): top-down welfare-gain waterfall (national -> zonal).

Rebuild of paper Fig 8 (`waterfall_chart.pdf`) over four years. Builds the zonal-market
socioeconomic benefit top-down = consumer surplus - producer-surplus loss + IC congestion
rent change, as a waterfall; top panel = 2022-2025 total, bottom panels = per year.

Producer surplus is computed per unit (thermal via revenue - SRMC expenses using the
model-derived marginal costs; others via revenue). Heavy: loops all units x (1 + N_YEARS).

Inputs : summaries/total_{summary,unit_revenues,unit_dispatch,marginal_costs,thermal_dispatch,
         intercon_dispatch}_flex.csv, marginal_prices_summary_flex.csv, data/base europe prices
Output : new_plots/figures_2022-2025/fig8_waterfall_chart_2022-2025.pdf
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from _helpers import (
    FIG_DIR, YEARS, DATA_START, DATA_END, idx, color_dict,
    get_unit_revenue, get_wholesale_expenses, get_balancing_expenses, get_ic_congestion_rent,
)
from _seb_common import load

FIG_DIR.mkdir(parents=True, exist_ok=True)
d = load()
markup = d.balancing_markup


def thermal_unit_surplus(unit, layout, time_slice):
    revenue = get_unit_revenue(d.revenues, unit, layout, time_slice)
    wholesale = get_wholesale_expenses(d.thermal_dispatch, d.marginal_cost, unit, layout, time_slice)
    balancing = get_balancing_expenses(d.dispatch, d.thermal_dispatch, d.marginal_cost, markup, unit, layout, time_slice)
    surplus = revenue - wholesale - balancing
    return 0 if pd.isna(surplus) else surplus


def get_surplus(layout, time_slice):
    surplus = pd.Series(0.0, index=d.all_units, name=layout)
    for unit in d.all_units:
        if unit in d.thermal_units:
            surplus.loc[unit] = thermal_unit_surplus(unit, layout, time_slice) * 1e-9
        else:
            surplus.loc[unit] = get_unit_revenue(d.revenues, unit, layout, time_slice) * 1e-9
    return surplus


years = ["total"] + YEARS
zonal_tracker, national_tracker = [], []
ic_rent_zonal, ic_rent_national = [], []

for year in tqdm(years, desc="surplus by year"):
    if year != "total":
        ts = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:30:00", freq="30min")
    else:
        ts = pd.date_range(f"{DATA_START}", f"{DATA_END} 23:30:00", freq="30min")
    ts = ts.intersection(d.consumer_cost.index.get_level_values(0))
    zonal_tracker.append(get_surplus("zonal", ts))
    national_tracker.append(get_surplus("national", ts))
    ic_rent_zonal.append(get_ic_congestion_rent(d.cons, d.marginal_prices, d.europe_prices, d.network, "zonal", ts) * 1e-9)
    ic_rent_national.append(get_ic_congestion_rent(d.cons, d.marginal_prices, d.europe_prices, d.network, "national", ts) * 1e-9)

# total IC rent = sum of per-year
ic_rent_zonal[0] = sum(ic_rent_zonal[1:])
ic_rent_national[0] = sum(ic_rent_national[1:])

N = len(YEARS)
fig, axs = plt.subplots(2, N, figsize=(4 * N, 7), gridspec_kw={"height_ratios": [1, 1]})
gs = axs[0, 0].get_gridspec()
for ax in axs[0, :]:
    ax.remove()
top_ax = fig.add_subplot(gs[0, :])

components = [
    "offer_cost", "bid_cost", "wholesale", "congestion_rent",
    "cfd_payments_consumer", "roc_payments_consumer", "net_consumer_benefit",
    "producer_surplus", "cfd_payments_producer", "roc_payments_producer",
    "ic_congestion_rent", "socioeconmic_benefit",
]
friendly_labels = {
    "offer_cost": "Congestion\nManagement\nOffers", "bid_cost": "Congestion\nManagement\nBids",
    "wholesale": "Wholesale\nMarket", "congestion_rent": "Intra-GB\nCongestion\nRent",
    "cfd_payments_consumer": "CfD\nPayments\n(Consumers)", "roc_payments_consumer": "RO\nPayments\n(Consumers)",
    "net_consumer_benefit": "Consumer\nSurplus", "producer_surplus": "Producer\nSurplus",
    "cfd_payments_producer": "CfD\nPayments\n(Producers)", "roc_payments_producer": "RO\nPayments\n(Producers)",
    "ic_congestion_rent": "GB Share of\nIC Congestion\nRent", "socioeconmic_benefit": "GB\nSocioeconomic\nBenefit",
}
fallback_colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99", "#d9b3ff", "#b3ffcc",
                   "#c6e2ff", "#ffe6b3", "#d9b3ff", "#b3ffcc", "#ffb3b3"]

for i, year in enumerate(years):
    ax = top_ax if i == 0 else axs[1, i - 1]

    if year != "total":
        ts = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:30:00", freq="30min")
        zonal_surplus = zonal_tracker[i]
        national_surplus = national_tracker[i]
        cr_zonal, cr_national = ic_rent_zonal[i], ic_rent_national[i]
    else:
        ts = pd.date_range(f"{DATA_START}", f"{DATA_END} 23:30:00", freq="30min")
        zonal_surplus = sum(zonal_tracker[1:])
        national_surplus = sum(national_tracker[1:])
        cr_zonal, cr_national = ic_rent_zonal[0], ic_rent_national[0]

    ts = ts.intersection(d.consumer_cost.index.get_level_values(0))

    cc_zonal = (d.consumer_cost.loc[idx[ts, :], "zonal"] * 1e-3).groupby(level=1).sum()
    cc_national = (d.consumer_cost.loc[idx[ts, :], "national"] * 1e-3).groupby(level=1).sum()
    cc_diff = cc_national - cc_zonal

    wf = pd.Series(index=components, dtype=float)
    wf.loc[wf.index.intersection(cc_diff.index)] = cc_diff.loc[wf.index.intersection(cc_diff.index)]
    wf.loc["cfd_payments_consumer"] = cc_diff.loc["cfd_payments"]
    wf.loc["cfd_payments_producer"] = -cc_diff.loc["cfd_payments"]
    wf.loc["roc_payments_consumer"] = cc_diff.loc["roc_payments"]
    wf.loc["roc_payments_producer"] = -cc_diff.loc["roc_payments"]
    wf.loc["net_consumer_benefit"] = cc_diff.sum()
    wf.loc["producer_surplus"] = zonal_surplus.sum() - national_surplus.sum()
    wf.loc["ic_congestion_rent"] = cr_zonal - cr_national
    wf.loc["socioeconmic_benefit"] = (
        wf.loc["net_consumer_benefit"] + wf.loc["producer_surplus"]
        - wf.loc["cfd_payments_consumer"] - wf.loc["roc_payments_consumer"] + wf.loc["ic_congestion_rent"]
    )

    values = wf.values
    colors_wf = [color_dict[c] if c in color_dict else fallback_colors[j % len(fallback_colors)]
                 for j, c in enumerate(components)]
    bottoms = np.zeros_like(values)
    for j in range(1, 6):
        bottoms[j] = bottoms[j - 1] + values[j - 1]
    bottoms[6] = 0
    for j in range(7, 11):
        bottoms[j] = bottoms[j - 1] + values[j - 1]
    bottoms[11] = 0

    width = 0.6
    for j, (component, value, bottom, color) in enumerate(zip(components, values, bottoms, colors_wf)):
        ax.bar(j, value, bottom=bottom, color=color, edgecolor="black", width=width)
        y_pos = bottom + value + (0.05 if value >= 0 else -0.05)
        ax.text(j, y_pos, f"{value:.2f}", ha="center", va="bottom" if value >= 0 else "top")
        # waterfall connector: horizontal line at the TOP of the previous bar
        # (= bottoms[j-1] + values[j-1]), which is exactly where bar j starts to stack.
        # The old code tracked a running_total that was never seeded at j=0, so every
        # consumer-block connector (j=1..5) sat too low by values[0] (offer_cost).
        if j > 0:
            connector_y = bottoms[j - 1] + values[j - 1]
            ax.plot([j - 1 + width / 2, j + width / 2], [connector_y, connector_y], "r", linewidth=0.5)

    ax.set_ylabel("Zonal Market Benefit [£bn]")
    if i == 0:
        ax.set_xticks(range(len(components)))
        ax.set_xticklabels([friendly_labels[c] for c in components])
        ax.text(0.02, 0.95, f"{YEARS[0]}-{YEARS[-1]} Total", transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top", ha="left")
        ax.set_ylim(0, max(wf.max() * 1.2, 0.1))
    else:
        ax.set_xticks([])
        ax.text(0.05, 0.95, f"{year}", transform=ax.transAxes, fontsize=12, fontweight="bold", va="top", ha="left")
        ax.set_ylim(0, max(3.9, wf.max() * 1.2))
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    print(f"[fig8] {year}: SEB (top-down) = {wf.loc['socioeconmic_benefit']:.3f} bn"
          f"  (consumer {wf.loc['net_consumer_benefit']:.3f}, producer {wf.loc['producer_surplus']:.3f})")

plt.tight_layout()
out = FIG_DIR / "fig8_waterfall_chart_2022-2025.pdf"
plt.savefig(out, bbox_inches="tight")
print(f"wrote {out}  (balancing markup £{markup}/MWh)")
