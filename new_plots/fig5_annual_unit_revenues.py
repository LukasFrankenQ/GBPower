"""Figure 5 (2022-2025): asset-level producer-surplus change, national -> zonal.

Rebuild of paper Fig 5 (`annual_unit_revenues.pdf`) over four years. Scatter of each
asset's % producer-surplus change (zonal vs national) against latitude; circle area is
annual national-market revenue (£mn/yr, now averaged over 4 years); red crosses mark CfD
units. Thermal-unit surplus uses the model-derived gas proxy (see on-figure caveat).

Inputs : summaries/total_unit_{revenues,dispatch,marginal_costs}_flex.csv (+ shipped data)
Output : new_plots/figures_2022-2025/fig5_annual_unit_revenues_2022-2025.pdf
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from _helpers import FIG_DIR, carrier_colors, nice_carrier_names, GAS_PROXY_CAVEAT
from _unit_surplus import build, BALANCING_MARKUP

FIG_DIR.mkdir(parents=True, exist_ok=True)
d = build()
x, perc_change, sizes = d.x, d.perc_change, d.sizes

CIRCLE_SCALE = 1.5
scaled_sizes = CIRCLE_SCALE * np.sqrt(sizes / sizes.min())
colors = [carrier_colors[d.bmu_carriers[b]] for b in x.index]

fig, ax = plt.subplots(1, 1, figsize=(10, 3.5))
ax2 = fig.add_axes([0, 0, 1, 1])
ax2.set_axis_off()

ax.scatter(x, perc_change, s=scaled_sizes, color=colors, alpha=0.7)
cfd_gens = x.index.intersection(d.cfd.index)
ax.scatter(x.loc[cfd_gens], perc_change.loc[cfd_gens], color="r",
           s=scaled_sizes[cfd_gens], alpha=0.7, marker="x", linewidth=0.6)

ax.grid(True, linestyle="--", color="grey", alpha=0.5)
ax.set_axisbelow(True)
ax.set_ylabel("Wholesale, RO, CfD & Balancing\nProducer Surplus Change (%)")
ax.axhline(0.0, c="k", zorder=0)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

ylim_upper, ylim_lower = 47, -58
ax.set_ylim(ylim_lower, ylim_upper)
ax.set_xlim(49.5, 58.9)
ax.fill_between([49.5, 58.9], 0, ylim_upper, color="lightgreen", alpha=0.1, zorder=0)
ax.fill_between([49.5, 58.9], ylim_lower, 0, color="tomato", alpha=0.1, zorder=0)

legend_handles, legend_labels = [], []
for carrier in sorted(d.used_carriers):
    if carrier in ["cascade", "solar"]:
        continue
    color = carrier_colors.get(carrier, "gray")
    legend_handles.append(Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=color,
                                 markersize=6, markeredgecolor=color, alpha=0.7))
    legend_labels.append(nice_carrier_names[carrier])
legend_handles.append(Line2D([0], [0], marker="x", color="r", markersize=6,
                             markeredgecolor="r", alpha=0.7, linewidth=0.6, linestyle="None"))
legend_labels.append("CFD Generator")
ax2.legend(legend_handles, legend_labels, loc="center", bbox_to_anchor=(0.5, 0.96), ncol=5, frameon=False)

for lon_x, city in [(51.5, "London"), (53.48, "Manchester"), (55.95, "Edinburgh"), (57.48, "Inverness")]:
    ax.axvline(lon_x, c="k", linestyle="--", alpha=0.5)
    ax.text(lon_x + 0.05, -55.8, city, ha="left", va="center", fontsize=8, color="k", fontstyle="italic")

legend_sizes = [10, 50, 100]
legend_scatter = [ax.scatter([], [], s=CIRCLE_SCALE * np.sqrt(s / sizes.min()), c="gray", alpha=0.7)
                  for s in legend_sizes]
ax.legend(legend_scatter, [f"£{s}mn/year" for s in legend_sizes], loc="lower left",
          title="Annual Revenue", frameon=True, scatterpoints=1)
ax.set_xlabel("Latitude")

ax.text(0.99, 0.02, GAS_PROXY_CAVEAT, transform=ax.transAxes, ha="right", va="bottom",
        fontsize=5.5, color="dimgray", bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", boxstyle="round,pad=0.3"))

out = FIG_DIR / "fig5_annual_unit_revenues_2022-2025.pdf"
plt.savefig(out, bbox_inches="tight")
print(f"wrote {out}  (balancing markup £{BALANCING_MARKUP}/MWh)")
