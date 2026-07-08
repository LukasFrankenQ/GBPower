"""Diagnostic for Fig 4 (wind cases): decompose every quantity the figure consumes
into an n x 1 stack of monthly time series (2022-2025), each judged for plausibility.

Fig 4 aggregates all four years into 3 static bars per case. That hides *when* each
ingredient is large/small, so here every ingredient is shown as a monthly series.

All £/MWh values are computed at the PHYSICALLY CORRECT scale:
    £/MWh  =  (component in £)  /  (energy served in MWh)
where energy_MWh = load_MW * 0.5  (30-min settlement periods).

NB: total_summary_flex stores components in £-million and already includes the *0.5
(energy) factor (proven: sc.wholesale.sum == sum(price*load*0.5)). The figure divides
by loads.sum() in MW (no *0.5) -> every sc-derived component in Fig 4 is HALF its true
£/MWh, while the wholesale bar (a load-weighted average PRICE) is at full scale. This
script uses the correct denominator so magnitudes can be judged on their own merits.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _helpers import REPO_ROOT, SUMMARIES, FIG_DIR, DATA_START, DATA_END, idx, color_dict

FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- load ingredients
sc = pd.read_csv(SUMMARIES / "total_summary_flex.csv", index_col=[0, 1], parse_dates=True)
mp = pd.read_csv(SUMMARIES / "marginal_prices_summary_flex.csv",
                 index_col=0, parse_dates=True, header=[0, 1])
loads = pd.read_csv(SUMMARIES / "total_gb_load.csv", index_col=0, parse_dates=True).loc[DATA_START:DATA_END]
loads = loads.iloc[:, 0]                                   # MW per settlement period

nat = mp.loc[DATA_START:DATA_END, ("national", "GB")]      # £/MWh
zon = mp.loc[DATA_START:DATA_END, idx["zonal", :]].copy()
zon.columns = zon.columns.get_level_values(1)

idx_t = nat.index
L = loads.reindex(idx_t)
month = idx_t.to_period("M")
energy_mwh = L * 0.5                                        # MWh per period

def monthly_price_wavg(price_series):
    """Load-weighted average PRICE per month (£/MWh) -- no 0.5 needed for an average."""
    num = (price_series * L).groupby(month).sum()
    den = L.groupby(month).sum()
    return num / den

def monthly_cost_per_mwh(layout, component):
    """Component (£m, energy-based) distributed over energy served -> £/MWh, per month."""
    s = sc.xs(component, level=1)[layout].reindex(idx_t)
    num = (s * 1e6).groupby(month).sum()                   # £m -> £
    den = energy_mwh.groupby(month).sum()                  # MWh
    return num / den

# ---- classification (same logic as the figure) ----------------------------------
neg_mask = idx_t[nat <= 0]
rem = idx_t.difference(neg_mask)
zrem = zon.loc[rem]
pos_mask = zrem.index[(zrem > 0).all(axis=1)]
mixed_mask = zrem.index.difference(pos_mask)
case_of = pd.Series("mixed", index=idx_t)
case_of.loc[pos_mask] = "pos"
case_of.loc[neg_mask] = "neg"
case_monthly = (case_of.groupby([month, case_of]).size()
                .unstack(fill_value=0))
case_share = case_monthly.div(case_monthly.sum(axis=1), axis=0) * 100

# zonal price spread: share of zones with negative price, per period -> monthly mean
frac_neg_zone = (zon < 0).mean(axis=1)

months = case_share.index.to_timestamp()

# ---------------------------------------------------------------- assemble panels
panels = []

# 1. national price (classification driver + national wholesale bar)
nat_m = monthly_price_wavg(nat)
panels.append(("1. National price  (load-wtd £/MWh)  — drives classification & national wholesale bar",
               lambda ax: (ax.plot(months, nat_m.values, color="k"),
                           ax.axhline(0, color="grey", lw=.6),
                           ax.set_ylabel("£/MWh")),
               "2022 crisis spike to ~£250 then normalises to £50-80. frac<=0 = 0.3% -> tiny 'Extreme' case. Plausible."))

# 2. national vs zonal wholesale price gap
zon_m = monthly_price_wavg((zon * 1).mul(L, axis=0).sum(axis=1) / zon.notna().mul(L, axis=0).sum(axis=1))
# simpler: load-weighted mean across zones each period, then monthly wavg
zon_period = zon.mean(axis=1)                              # unweighted zone mean (zones ~ price areas)
zon_m = monthly_price_wavg(zon_period)
gap = nat_m - zon_m
panels.append(("2. Wholesale price:  national vs zonal (£/MWh) and gap  — the headline reduction",
               lambda ax: (ax.plot(months, nat_m.values, color="k", label="national"),
                           ax.plot(months, zon_m.values, color="tab:blue", label="zonal"),
                           ax.plot(months, gap.values, color="tab:red", lw=.9, label="national-zonal"),
                           ax.axhline(0, color="grey", lw=.6),
                           ax.legend(fontsize=6, ncol=3, loc="upper right"),
                           ax.set_ylabel("£/MWh")),
               "Gap should be >=0 (splitting lowers avg zonal price). If gap goes negative, investigate."))

# 3. case shares over time
panels.append(("3. Wind-case classification shares (% of periods / month)",
               lambda ax: (ax.stackplot(months, case_share.get("pos", 0), case_share.get("mixed", 0),
                                        case_share.get("neg", 0),
                                        labels=["Low (all-pos)", "High (mixed)", "Extreme (nat<=0)"],
                                        colors=["#6bab6b", "#4a7fb5", "#c0504d"], alpha=.85),
                           ax.legend(fontsize=6, ncol=3, loc="lower center"),
                           ax.set_ylim(0, 100), ax.set_ylabel("% of time")),
               "Mixed (High) should dominate, Extreme rare. Seasonality: more splitting in windy winter."))

# 4. GB demand (the denominator; units matter)
dem_twh = (energy_mwh.groupby(month).sum()) / 1e6
panels.append(("4. GB transmission demand (TWh / month)  — the £/MWh denominator (uses *0.5 MW->MWh)",
               lambda ax: (ax.plot(months, dem_twh.values, color="tab:purple"),
                           ax.set_ylabel("TWh")),
               "~15-25 TWh/month transmission-level, winter-peaking. If flat/2x off, the ×0.5 units bug bites."))

# 5. congestion rent (zonal) -- negative by design (rebated to consumers)
cr_zon = monthly_cost_per_mwh("zonal", "congestion_rent")
cr_nat = monthly_cost_per_mwh("national", "congestion_rent")
panels.append(("5. Congestion rent £/MWh  (zonal; national≡0)  — stored NEGATIVE = rebated to consumers",
               lambda ax: (ax.plot(months, cr_zon.values, color="tab:orange", label="zonal"),
                           ax.plot(months, cr_nat.values, color="k", lw=.8, label="national"),
                           ax.axhline(0, color="grey", lw=.6),
                           ax.legend(fontsize=6, loc="lower left"),
                           ax.set_ylabel("£/MWh")),
               "National must be ~0 (single price). Zonal <0 by design. Magnitude grows with splitting/crisis."))

# 6. subsidies ROC + CfD (national)
roc = monthly_cost_per_mwh("national", "roc_payments")
cfd = monthly_cost_per_mwh("national", "cfd_payments")
panels.append(("6. Subsidy top-ups £/MWh (national):  ROC + CfD  — layout-independent consumer levy",
               lambda ax: (ax.plot(months, roc.values, color="tab:green", label="ROC"),
                           ax.plot(months, cfd.values, color="tab:olive", label="CfD"),
                           ax.axhline(0, color="grey", lw=.6),
                           ax.legend(fontsize=6, loc="upper left"),
                           ax.set_ylabel("£/MWh")),
               "ROC ~5-15 £/MWh. CfD can go NEGATIVE (generators pay back) when wholesale>strike (2022). Check sign."))

# 7. balancing offer/bid (national vs zonal)
off_n = monthly_cost_per_mwh("national", "offer_cost")
bid_n = monthly_cost_per_mwh("national", "bid_cost")
off_z = monthly_cost_per_mwh("zonal", "offer_cost")
bid_z = monthly_cost_per_mwh("zonal", "bid_cost")
panels.append(("7. Balancing £/MWh:  offer (up) & bid (down), national vs zonal  — zonal should need LESS",
               lambda ax: (ax.plot(months, off_n.values, color="firebrick", label="offer nat"),
                           ax.plot(months, off_z.values, color="firebrick", ls="--", label="offer zon"),
                           ax.plot(months, bid_n.values, color="steelblue", label="bid nat"),
                           ax.plot(months, bid_z.values, color="steelblue", ls="--", label="bid zon"),
                           ax.axhline(0, color="grey", lw=.6),
                           ax.legend(fontsize=6, ncol=2, loc="upper right"),
                           ax.set_ylabel("£/MWh")),
               "Zonal offer_cost should be <= national (less redispatch). bid_cost ~<=0 (paid to turn down)."))

# ---------------------------------------------------------------- render
n = len(panels)
fig, axs = plt.subplots(n, 1, figsize=(11, 2.15 * n), sharex=True)
for ax, (title, draw, note) in zip(axs, panels):
    draw(ax)
    ax.set_title(title, fontsize=8.5, loc="left", weight="bold")
    ax.text(0.004, 0.04, "plausibility: " + note, transform=ax.transAxes,
            fontsize=6.6, style="italic", color="#444", va="bottom")
    ax.margins(x=0.01)
    ax.grid(True, ls="--", lw=.4, alpha=.6)
axs[-1].set_xlabel("month")
fig.suptitle("Fig 4 ingredient audit — monthly quantities, 2022-2025 (all £/MWh at correct MWh scale)",
             fontsize=11, weight="bold", y=0.997)
fig.tight_layout(rect=[0, 0, 1, 0.99])

out_pdf = FIG_DIR / "diag_fig4_ingredients.pdf"
out_png = FIG_DIR / "diag_fig4_ingredients.png"
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, dpi=130, bbox_inches="tight")
print("wrote", out_pdf)

# ---------------------------------------------------------------- console summary
print("\n=== whole-window averages (correct £/MWh, energy-weighted) ===")
def wm(layout, comp):
    s = sc.xs(comp, level=1)[layout].reindex(idx_t)
    return (s * 1e6).sum() / energy_mwh.sum()
for lay in ["national", "zonal"]:
    parts = {c: wm(lay, c) for c in ["congestion_rent", "offer_cost", "bid_cost", "cfd_payments", "roc_payments"]}
    wprice = (nat if lay == "national" else zon_period)
    wprice = (wprice * L).sum() / L.sum()
    print(f"{lay:9s} wholesale(price)={wprice:7.2f}  " + "  ".join(f"{k}={v:7.2f}" for k, v in parts.items()))
print("\ncase shares:", {k: f"{100*len(v)/len(idx_t):.1f}%"
                         for k, v in [("pos", pos_mask), ("mixed", mixed_mask), ("neg", neg_mask)]})
