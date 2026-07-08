"""Figure 4 (2022-2025): consumer-cost stack by market design and wind availability.

Rebuild of paper Fig 4 (`wind_cases_*.pdf`) over four years. Settlement periods are
classified by the presence of zonal price-splitting into Low / High / Extreme wind cases
(here: national-negative, all-positive-zonal, mixed), and consumer-cost stacks are shown
for national vs zonal plus the zonal reduction, with inset price-split maps.

Only summary + shipped data are needed (the notebook's gas-price / physical-notification
cells were exploratory and are omitted).

Inputs : summaries/{marginal_prices_summary_flex,total_summary_flex,total_gb_load}.csv,
         data/prerun/load_weights.csv, data/regions_onshore_s.geojson,
         data/prerun/prepared_bmus.csv
Output : new_plots/figures_2022-2025/fig4_wind_cases_2022-2025.pdf
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from copy import deepcopy

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from shapely.ops import unary_union

from _helpers import REPO_ROOT, SUMMARIES, FIG_DIR, DATA_START, DATA_END, idx, nice_names, color_dict

start, end = DATA_START, DATA_END
FIG_DIR.mkdir(parents=True, exist_ok=True)

mp = pd.read_csv(SUMMARIES / "marginal_prices_summary_flex.csv", index_col=0, parse_dates=True, header=[0, 1])
sc = pd.read_csv(SUMMARIES / "total_summary_flex.csv", index_col=[0, 1], parse_dates=True)
lw = pd.read_csv(REPO_ROOT / "data" / "prerun" / "load_weights.csv", index_col=0)
lw.index = lw.index.astype(str)
regions = gpd.read_file(REPO_ROOT / "data" / "regions_onshore_s.geojson")
loads = pd.read_csv(SUMMARIES / "total_gb_load.csv", index_col=0, parse_dates=True).loc[start:end]

zmp = mp.loc[start:end, idx["zonal", :]]
nmp = mp.loc[start:end, idx["national", :]]
nmp.columns = nmp.columns.get_level_values(1)
zmp.columns = zmp.columns.get_level_values(1)

# Restrict to GB buses. The "zonal" price frame also carries the foreign AC bus
# `Ireland` (an interconnector node, not a GB wind region). Ireland's price is <= 0
# in ~40% of all SPs, and on its own it flips the high-wind share from ~35% to 63%
# -- it accounts for EVERY "exactly one wind-set zone" period. Wind-case labels and
# region binning must see GB zones only. GB buses are the numeric cluster IDs;
# foreign buses (Ireland, France, Netherlands, Belgium, Norway) are named.
gb_zonal_buses = [c for c in zmp.columns if str(c).isdigit()]
zmp = zmp[gb_zonal_buses]


def classify_period(nat, lmp):
    nat, lmp = nat.copy(), lmp.copy()
    # Extreme wind: the unconstrained national price itself collapses to <=0 --
    # a system-wide renewable glut that even an unconstrained network cannot absorb.
    nat_negative_mask = nat.loc[nat.iloc[:, 0] <= 0].index
    nat_negative = {"national": nat.loc[nat_negative_mask], "zonal": lmp.loc[nat_negative_mask]}
    lmp.drop(nat_negative_mask, inplace=True)
    nat.drop(nat_negative_mask, inplace=True)
    # High wind: genuine price-splitting -- at least one zone is wind-set, i.e. its
    # price is <= 0. ROC turbines bid strictly negative (below zero to avoid
    # curtailment); CfD turbines have no such incentive and bid down to £0 -- so a
    # zone pinned at exactly £0 is *also* wind setting the local price and belongs in
    # the high-wind case. (A stricter "< 0" would drop all the CfD-wind splitting.)
    high_mask = lmp.loc[(lmp <= 0).any(axis=1)].index
    mixed = {"national": nat.loc[high_mask], "zonal": lmp.loc[high_mask]}
    lmp.drop(high_mask, inplace=True)
    nat.drop(high_mask, inplace=True)
    # Low wind: no zone below zero -> thermal sets the price everywhere, no splitting.
    all_positive = {"national": nat, "zonal": lmp}
    return nat_negative, all_positive, mixed


neg, pos, mixed = classify_period(nmp, zmp)


def bars_to_ax(ax, lmp, loads, weights, costs, index_groups, sign_split=False):
    # `loads` is MW per settlement period; energy served = load * 0.5 h (30-min SPs).
    # The cost components in `costs` are already energy-based £, so they must be
    # spread over MWh, not MW -> divide by (sum of load) * 0.5. (The wholesale bar
    # below is a load-weighted average PRICE and needs no 0.5.)
    distributed_price = costs.sum().drop("wholesale") / (loads.sum().sum() * 0.5) * 1e6
    w = loads.values.flatten()
    if sign_split:
        # Per-period partition (High-wind and Extreme rows): the low-cost bar is the
        # price of the zones that are actually wind-set (<= 0) in each period, the
        # high-cost bar the price of the thermal (> 0) zones. This makes the low-cost
        # bar track the genuine wind-set price (~£0) rather than the multi-year mean
        # of a frozen region set (which is wind-set only part of the time -> ~£70).
        group_series = [lmp.where(lmp <= 0).mean(axis=1), lmp.where(lmp > 0).mean(axis=1)]
    else:
        # Fixed geographic partition (Low-wind row): zonal buses within a price zone
        # share one price; a group may span >1 zone over 4 years, so average.
        group_series = [lmp[g].mean(axis=1) for g in index_groups]
    bar_kwargs = {"alpha": 0.8}
    for j, lmp_group in enumerate(group_series):
        cumulative_positive = 0
        cumulative_negative = 0
        for cat, value in distributed_price.items():
            base = cumulative_positive if value >= 0 else cumulative_negative
            ax.bar(j - 0.25, value, width=0.5, bottom=base, color=color_dict[cat], align="edge", **bar_kwargs)
            ax.plot([j - 0.25, j + 0.25], [base + value, base + value], color="k", alpha=0.5, lw=0.5)
            if value >= 0:
                cumulative_positive += value
            else:
                cumulative_negative += value
        # load-weighted average price of the group, ignoring periods where the group
        # is empty (e.g. no thermal zone in a fully wind-set period).
        m = lmp_group.notna().values
        avg = (lmp_group.values[m] * w[m]).sum() / w[m].sum()
        base = cumulative_positive if avg >= 0 else cumulative_negative
        ax.bar(j - 0.25, avg, width=0.5, bottom=base, align="edge", color=color_dict["wholesale"], **bar_kwargs)
        ax.plot([j - 0.25, j + 0.25], [base + avg, base + avg], color="k", alpha=0.5, lw=0.5)
        ax.plot([j - 0.25, j + 0.25],
                [cumulative_positive + cumulative_negative + avg] * 2, color="r", alpha=0.5)


def simple_bars_to_ax(ax, values):
    cumulative_positive = 0
    cumulative_negative = 0
    for cat, value in values.items():
        base = cumulative_positive if value >= 0 else cumulative_negative
        ax.bar(-0.25, value, width=0.5, bottom=base, color=color_dict[cat], align="edge", alpha=0.8)
        ax.plot([-0.25, 0.25], [base + value, base + value], color="k", alpha=0.5, lw=0.5)
        if value >= 0:
            cumulative_positive += value
        else:
            cumulative_negative += value
    total = cumulative_positive + cumulative_negative
    ax.plot([-0.25, 0.25], [total, total], color="red", lw=2)
    ax.plot([-0.25, 0.25], [values.drop("congestion_rent").sum()] * 2, color="red", linestyle="-.", alpha=0.8, lw=2)


def regions_to_ax(regions, ax, dark_regions):
    # some zonal bus IDs (e.g. future-network buses) have no onshore-region polygon
    dark_regions = dark_regions.intersection(regions.index)
    if not dark_regions.empty:
        gpd.GeoSeries([unary_union(regions.loc[dark_regions].geometry)]).plot(
            ax=ax, color="midnightblue", alpha=0.8, edgecolor="none", lw=0.5)
    gpd.GeoSeries([unary_union(regions.geometry)]).plot(ax=ax, facecolor="none", edgecolor="k", alpha=1, lw=0.3)
    ax.set_xticks([])
    ax.set_yticks([])


def get_total_prices(sc, mps, loads, lw, layout):
    mps = deepcopy(mps)
    try:
        mps = mps[layout]
    except KeyError:
        pass
    costs = sc.loc[idx[mps.index, :], layout].unstack().drop(columns=["balancing_volume"])
    # energy served = load * 0.5 h; components are energy-based £ -> divide by MWh.
    return costs.sum() / (loads.loc[costs.index].sum().sum() * 0.5) * 1e6


# Region binning (used for the inset maps and the Low-wind row): a zone is a
# "low-cost region" if it is wind-set (price <= 0) in a non-trivial share of the
# high-wind periods. This is a tight, persistent northern cluster -- 62 buses vs
# 225 never wind-set (corr with latitude 0.82; mean 56.3 deg N vs 52.6 deg N),
# i.e. Scotland. The split is bimodal, so the 5% cut-off is safe.
# NB: in the High-wind and Extreme rows the wholesale bars are NOT drawn from this
# fixed set but from a per-period sign split (see bars_to_ax sign_split=True), so
# the low-cost bar reflects the genuinely wind-set price (~£0) rather than the
# ~£70 four-year average of these buses (which are wind-set only part of the time).
windset_frequency = (mixed["zonal"] <= 0).mean()
low_cost_regions = windset_frequency.index[windset_frequency >= 0.05]
high_cost_regions = windset_frequency.index[windset_frequency < 0.05]

fig, axs = plt.subplots(3, 3, figsize=(7.5, 6), gridspec_kw={"width_ratios": [1, 2, 1]})

# rows: Low-wind (pos), High-wind (mixed), Extreme-wind (neg)
for row, case in zip(range(3), [pos, mixed, neg]):
    # Low-wind row (row 0): no wind-set zones, so show the fixed north/south cluster
    # (both positive -> illustrates that without wind there is no splitting benefit).
    # High-wind (1) and Extreme (2): partition zones per period by wind-set status.
    sign_split = row > 0
    bars_to_ax(axs[row, 1], case["zonal"], loads.loc[case["zonal"].index], lw,
               sc.loc[idx[case["zonal"].index, :], "zonal"].unstack().drop(columns=["balancing_volume"]),
               [low_cost_regions, high_cost_regions], sign_split=sign_split)
    bars_to_ax(axs[row, 0], case["national"], loads.loc[case["national"].index], lw,
               sc.loc[idx[case["national"].index, :], "national"].unstack().drop(columns=["balancing_volume"]),
               [["GB"]])
    zonal_p = get_total_prices(sc, case["zonal"], loads, lw.iloc[:, 0], "zonal")
    national_p = get_total_prices(sc, case["national"], loads, pd.Series(1, ["GB"]), "national")
    simple_bars_to_ax(axs[row, 2], -(zonal_p - national_p))

for ax in axs[:, 0]:
    ax.set_xlim(-0.5, 0.5)
for ax in axs[:, 1]:
    ax.set_xlim(-0.5, 1.5)
    ax.set_yticklabels([])
    ax.spines["left"].set_visible(False)
for ax in axs[:, 2]:
    ax.spines["left"].set_visible(False)
for ax in axs.flatten():
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", axis="y")
    ax.set_axisbelow(True)

for ax in axs[2, :2]:
    ax.set_ylim(-70, 70)
for ax in axs[0, :2]:
    ax.set_ylim(-20, 250)
for ax in axs[1, :2]:
    ax.set_ylim(-70, 220)

for r, c, coords, dark in [
    (2, 0, [0.65, -0.1, 0.35, 0.5], pd.Index([])),
    (2, 1, [0.8, 0.015, 0.2, 0.3], high_cost_regions),
    (0, 0, [0.65, -0.1, 0.35, 0.5], high_cost_regions.union(low_cost_regions)),
    (0, 1, [0.8, 0.015, 0.2, 0.3], high_cost_regions.union(low_cost_regions)),
    (1, 0, [0.65, -0.1, 0.35, 0.5], high_cost_regions.union(low_cost_regions)),
    (1, 1, [0.8, 0.015, 0.2, 0.3], high_cost_regions),
]:
    regions_to_ax(regions.set_index("name"), axs[r, c].inset_axes(coords), dark)

for ax in axs[:2, 2]:
    ax.set_xticks([])
for ax in axs[:, 2]:
    ax.set_xlim(-0.5, 0.5)

neg_share = 100 * len(neg["national"]) / len(zmp)
pos_share = 100 * len(pos["national"]) / len(zmp)
mixed_share = 100 * len(mixed["national"]) / len(zmp)
axs[2, 0].set_ylabel(f"\nExtreme Wind Case\n({neg_share:.1f}% of time)")
axs[0, 0].set_ylabel(f"\nLow Wind Case\n({pos_share:.1f}% of time)")
axs[1, 0].set_ylabel(f"Consumer Price (£/MWh)\nHigh Wind Case\n({mixed_share:.1f}% of time)")

for ax in axs[:, 2]:
    position = ax.get_position()
    ax.set_position([position.x0 + 0.025, position.y0, position.width, position.height])

axs[0, 0].set_title("National Market", fontsize=10, pad=10)
axs[0, 1].set_title("Zonal Market", fontsize=10, pad=10)
axs[0, 2].set_title("Zonal Consumer Cost\nReduction (£/MWh)", fontsize=10, pad=10)

axs[2, 1].set_xticks([0, 1])
axs[2, 1].set_xticklabels(["Low Price\nRegions", "High Price\nRegions"])
axs[2, 1].tick_params(axis="x", which="both", length=0)
for ax in axs[:2, :2].flatten():
    ax.set_xticks([])
axs[2, 0].set_xticks([])
axs[2, 2].set_xticks([])
axs[0, 2].set_ylim(-4, 4)

for ax in axs.flatten():
    ax.axhline(0, color="k", lw=1)
    ax.spines["bottom"].set_visible(False)

handles, labels = [], []
for label, color in color_dict.items():
    if label in ["wholesale selling", "wholesale buying"]:
        continue
    handles.append(Patch(color=color, label=nice_names[label]))
    labels.append(nice_names[label])
handles.append(Line2D([0], [0], color="r", lw=1, linestyle="-", alpha=0.8))
labels.append("Total")
handles.append(Line2D([0], [0], color="r", lw=1, linestyle="-.", alpha=0.8))
labels.append("Total w/o Congestion Rent")
axs[2, 0].legend(handles=handles, labels=labels, loc="upper left",
                 bbox_to_anchor=(-1, -0.4), fontsize=8, ncol=4, frameon=False)

axs[0, 0].text(-1.35, 330, f"{start} - {end}", fontsize=9, ha="left", weight="bold")

out = FIG_DIR / "fig4_wind_cases_2022-2025.pdf"
plt.savefig(out, bbox_inches="tight")
print(f"wrote {out}")
print(f"case shares: low(pos) {pos_share:.1f}%  high(mixed) {mixed_share:.1f}%  extreme(neg) {neg_share:.1f}%")
