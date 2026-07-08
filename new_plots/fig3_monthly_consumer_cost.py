"""Figure 3 (2022-2025): total monthly consumer cost, national vs zonal, + differential.

Rebuild of paper Fig 3 (`total_monthly_costs_w_comparison.pdf`) over four years.

The paper's published Fig 3 compares the *national* and *zonal* market designs (see its
caption). The maintained plotting/get_total_results.py cell instead compares national vs
nodal ("Optimal Operation RNP"); here we default to zonal to match the paper. Flip
COMPARISON_LAYOUT to 'nodal' to reproduce the RNP variant.

Inputs : summaries/total_summary_flex.csv   (built by new_plots/build_summaries.py)
Output : new_plots/figures_2022-2025/fig3_total_monthly_costs_2022-2025.pdf
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt

from _helpers import (
    SUMMARIES, FIG_DIR, idx, nice_names, color_dict,
    get_consumer_cost, stack_to_ax, plot_stacked_horizontal_bar,
)

COMPARISON_LAYOUT = "zonal"   # 'zonal' matches the paper; 'nodal' = RNP variant
FIG_DIR.mkdir(parents=True, exist_ok=True)

consumer_cost = get_consumer_cost(SUMMARIES / "total_summary_flex.csv")

fig, axs = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
totals = {}

for col, ax, t in zip(
    ["national", COMPARISON_LAYOUT],
    axs[:2],
    ["Current Market (National)", f"{COMPARISON_LAYOUT.capitalize()} Market"],
):
    df_unstack = consumer_cost.loc[idx[:, :], col].unstack()
    df = df_unstack.groupby([df_unstack.index.year, df_unstack.index.month]).sum().mul(1e-3)

    for m in df.index:
        stack_to_ax(df.loc[[m]], ax)

    ax.set_ylabel("Consumer Cost [£bn]")
    ax.grid(axis="y", linestyle="--")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(-1, len(df))

    total_cost = df.sum().sum()
    totals[col] = total_cost
    ax.text(1, ax.get_ylim()[1] * 0.9, t, ha="center", va="center", fontsize=12, fontweight="bold")

    new_ax = ax.figure.add_axes([
        ax.get_position().x0 + 0.32,
        ax.get_position().y0 + ax.get_position().height * 0.8 + 0.04,
        ax.get_position().width * 0.6,
        ax.get_position().height * 0.09,
    ])
    plot_stacked_horizontal_bar(df.sum(), new_ax)
    new_ax.set_title(f"Total: {np.around(total_cost, decimals=2)} £bn")

# difference row: national - comparison
ax = axs[2]
ss = (consumer_cost["national"] - consumer_cost[COMPARISON_LAYOUT]).unstack()
ss = ss.groupby([ss.index.year, ss.index.month]).sum().mul(1e-3)
for m in ss.index:
    stack_to_ax(ss.loc[[m]], ax, text_y_offset=0.03)

ax.set_ylabel("Consumer Cost Difference [£bn]\n← Higher under comparison | Higher under National →")
ax.grid(axis="y", linestyle="--")
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xlim(-1, len(ss))

total_diff = totals["national"] - totals[COMPARISON_LAYOUT]
new_ax = ax.figure.add_axes([
    ax.get_position().x0 + 0.32,
    ax.get_position().y0 + ax.get_position().height * 0.8,
    ax.get_position().width * 0.6,
    ax.get_position().height * 0.09,
])
plot_stacked_horizontal_bar(ss.sum(), new_ax, right_tick=False)
new_ax.set_title(f"Total: {np.around(total_diff, decimals=2)} £bn")
ax.text(1, ax.get_ylim()[1] * 0.9, "Difference", ha="center", va="center", fontweight="bold", fontsize=12)

axs[2].tick_params(axis="x", labelrotation=45)

handles, labels = [], []
for name, nice_name in nice_names.items():
    if name == "wholesale buying":
        continue
    handles.append(plt.Rectangle((0, 0), 1, 1, color=color_dict[name]))
    labels.append(nice_name)
axs[0].legend(handles, labels, bbox_to_anchor=(0.44, 1.2), frameon=False, ncol=3)

plt.tight_layout()
out = FIG_DIR / "fig3_total_monthly_costs_2022-2025.pdf"
plt.savefig(out, bbox_inches="tight")
print(f"wrote {out}")
print(f"national total £{totals['national']:.2f}bn  {COMPARISON_LAYOUT} £{totals[COMPARISON_LAYOUT]:.2f}bn  diff £{total_diff:.2f}bn")
