"""Figure 7 (2022-2025): northern-generator surplus stabilisation under Policies 2 & 3.

Rebuild of paper Fig 7 (`surplus_changes_30_after_policy.pdf`) over four years.
  a) Policy 2 (production-based FTRs, zonal)      b) Policy 3 (grandfathered merit order,
  the `equitable` layout)                          c) histograms of surplus changes under
Policies 1/2/3. Thermal-unit surplus uses the model-derived gas proxy (on-figure caveat).

Inputs : summaries/total_unit_{revenues,dispatch,marginal_costs}_flex.csv (+ shipped data)
Output : new_plots/figures_2022-2025/fig7_surplus_changes_after_policy_2022-2025.pdf
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from _helpers import FIG_DIR, carrier_colors, nice_carrier_names, policy_colors, GAS_PROXY_CAVEAT
from _unit_surplus import build, BALANCING_MARKUP

FIG_DIR.mkdir(parents=True, exist_ok=True)
d = build()
x, perc_change = d.x, d.perc_change
perc_change_2, perc_change_3 = d.perc_change_2, d.perc_change_3
colors = [carrier_colors[d.bmu_carriers[b]] for b in x.index]


def plot_carrier_share(series, base_ax, coords):
    new_ax = base_ax.inset_axes(coords)
    cumulative = 0
    for carrier, share in series.items():
        seg = share * 100
        new_ax.barh(0, width=seg, left=cumulative, height=1, color=carrier_colors.get(carrier, "gray"), edgecolor="none")
        new_ax.axvline(cumulative, c="k", lw=0.5)
        cumulative += seg
    new_ax.set_xlim(0, 100)
    new_ax.set_xticks(list(range(0, 101, 20)))
    new_ax.set_yticks([])


fig = plt.figure(figsize=(10, 6.6))
gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 1])
axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])]
hist_ax = fig.add_subplot(gs[:, 1])

# carrier-share insets (gainers / losers), decorative
diff = d.zonal_total - d.n_total
hold = diff.to_frame().rename(columns={0: "diff"})
hold["carrier"] = d.bmu_carriers[hold.index]
phold = hold.loc[hold["diff"] > 0].groupby("carrier")["diff"].sum()
nhold = hold.loc[hold["diff"] < 0].groupby("carrier")["diff"].sum().abs()
try:
    if phold.sum() > 0 and len(phold) > 1:
        plot_carrier_share(phold / phold.sum(), axs[0], [0.68, 0.9, 0.235, 0.05])
    if len(nhold) > 1:
        plot_carrier_share(nhold / nhold.sum(), axs[0], [0.1, 0.233, 0.255, 0.05])
except Exception as e:
    print("carrier-share inset skipped:", e)

axs[0].text(0.02, 0.98, "a", transform=axs[0].transAxes, fontweight="bold", fontsize=12, va="top")
axs[1].text(0.02, 0.98, "b", transform=axs[1].transAxes, fontweight="bold", fontsize=12, va="top")
hist_ax.text(-0.0, 1.05, "c", transform=hist_ax.transAxes, fontweight="bold", fontsize=12, va="top")

for ax, policy_values, c, name in zip(
    axs,
    [perc_change_2, perc_change_3],
    [policy_colors["zonal"], policy_colors["zonal_with_policy"]],
    ["Production-Based FTRs\n(Policy 2)", "Grandfathered Merit\nOrder (Policy 3)"],
):
    ax.scatter(x, perc_change, color=colors, s=30, alpha=0.5)
    offset = 1.6
    intersection = perc_change.index.intersection(policy_values.index)
    for x0, lower, upper in zip(x.loc[intersection], perc_change.loc[intersection], policy_values.loc[intersection]):
        if lower > 0:
            continue
        ax.plot([x0, x0], [lower + offset, upper - offset], c=c, alpha=0.5, linestyle="--", linewidth=0.4)
        ax.scatter([x0], [upper], alpha=0.5, marker="o", facecolors="none", color=c)
    cfd_gens = x.index.intersection(d.cfd.index)
    ax.scatter(x.loc[cfd_gens], perc_change.loc[cfd_gens], color="r", s=30, alpha=0.4, marker="x", linewidth=0.6)

    ax.grid(True, linestyle="--", color="grey", alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_ylabel("Wholesale, RO, CfD & Balancing\nProducer Surplus Change (%)")
    ax.axhline(0.0, c="k", zorder=0)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_ylim(-58, 35)
    ax.set_xlim(49.5, 58.9)
    ax.fill_between([49.5, 58.9], 0, 35, color="lightgreen", alpha=0.1, zorder=0)
    ax.fill_between([49.5, 58.9], -58, 0, color="tomato", alpha=0.1, zorder=0)
    for lon_x in [51.5, 53.48, 55.95, 57.48]:
        ax.axvline(lon_x, c="k", linestyle="--", alpha=0.5)
    tb = ax.text(0.2, 0.3, name, transform=ax.transAxes, ha="center", va="center", fontsize=10, weight="bold")
    tb.set_bbox(dict(facecolor="white", alpha=0.8, edgecolor="gray", boxstyle="round,pad=0.5"))

# histograms of negative surplus changes
bins = np.linspace(-60, 0, 30)
bin_centers = (bins[:-1] + bins[1:]) / 2


def _hist(series):
    h, _ = np.histogram(series[series < 0], bins=bins)
    return h / h.max() if h.max() > 0 else h


for h, color, label, alpha in [
    (_hist(perc_change), "gray", "Policy 1", 0.5),
    (_hist(perc_change_2), policy_colors["zonal"], "Policy 2", 0.5),
    (_hist(perc_change_3), policy_colors["zonal_with_policy"], "Policy 3", 0.8),
]:
    hist_ax.barh(bin_centers, h, height=(bins[1] - bins[0]), alpha=alpha, color=color,
                 label=label, edgecolor="black", linewidth=0.5)

hist_ax.axvspan(0, hist_ax.get_xlim()[1], ymin=0, ymax=1, color="red", alpha=0.1)
hist_ax.set_ylim(-50, 0)
hist_ax.set_ylabel("Distribution of Surplus Changes [%]")
hist_ax.legend(loc="lower center", bbox_to_anchor=(0.3, -0.1), ncol=3, frameon=False, fontsize=8)
hist_ax.spines["right"].set_visible(False)
hist_ax.spines["top"].set_visible(False)
hist_ax.axhline(0, c="k", linestyle="-", alpha=1, lw=2)
hist_ax.set_xticklabels([])

for lon_x, city in [(51.55, "London"), (53.53, "Manchester"), (56.0, "Edinburgh"), (57.53, "Inverness")]:
    axs[1].text(lon_x, -55.8, city, ha="left", va="center", fontsize=8, color="k", fontstyle="italic")

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
axs[0].legend(legend_handles, legend_labels, loc="upper center", bbox_to_anchor=(0.6, 1.3), ncol=5, frameon=False)

axs[0].set_xticklabels("")
axs[1].set_xlabel("Latitude")
axs[1].text(0.99, 0.02, GAS_PROXY_CAVEAT, transform=axs[1].transAxes, ha="right", va="bottom",
            fontsize=5.5, color="dimgray", bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", boxstyle="round,pad=0.3"))

plt.subplots_adjust(hspace=0.1)
out = FIG_DIR / "fig7_surplus_changes_after_policy_2022-2025.pdf"
plt.savefig(out, bbox_inches="tight")
print(f"wrote {out}  (balancing markup £{BALANCING_MARKUP}/MWh)")
