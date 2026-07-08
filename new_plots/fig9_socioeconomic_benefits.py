"""Figure 9 (2022-2025): bottom-up socioeconomic benefit, dispatch diff, storage correlation.

Rebuild of paper Fig 9 (`socioeconomic_benefits.pdf`) over four years.
  top row    : monthly bottom-up SEB (£mn), national vs zonal, with per-year totals;
  middle row : dispatch differences by technology (GWh);
  bottom row : north vs south storage wholesale-position correlation (example days).

Heavy: calculate_bottom_up_SEB loops thermal+wind units for each of the 48 months.

Inputs : summaries/total_{unit_dispatch,marginal_costs,thermal_dispatch,intercon_dispatch}_flex.csv,
         marginal_prices_summary_flex.csv, data/base europe prices, example-day networks
Output : new_plots/figures_2022-2025/fig9_socioeconomic_benefits_2022-2025.pdf
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pypsa
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

from _helpers import (
    REPO_ROOT, FIG_DIR, YEARS, idx,
    get_unit_schedule, get_wholesale_expenses, get_balancing_expenses,
    get_ic_congestion_rent, get_import_cost, get_export_revenues,
)
from _seb_common import load

FIG_DIR.mkdir(parents=True, exist_ok=True)
d = load()
markup = d.balancing_markup
imports, exports = d.imports, d.exports


def calculate_bottom_up_SEB(month_year):
    system_tracker = pd.Series(0.0, index=[
        "fossil_generation_change", "import_volume_change", "wind_generation_change", "export_volume_change"])
    bottom_up_seb = pd.Series(0.0, index=[
        "import_costs_diff", "export_revenues_diff", "congestion_rent_diff",
        "prevented_wholesale_expenses", "prevented_balancing_expenses"])

    month_start = pd.to_datetime(month_year)
    time_slice = pd.date_range(month_start, month_start + pd.offsets.MonthEnd(1) + pd.Timedelta(hours=23, minutes=30), freq="30min")
    ic_ts = time_slice.intersection(exports.index).intersection(imports.index)

    system_tracker.loc["import_volume_change"] = (
        imports.loc[ic_ts, idx[:, "zonal"]].sum().sum() - imports.loc[ic_ts, idx[:, "national"]].sum().sum()) * 1e-3
    bottom_up_seb.loc["import_costs_diff"] = (
        get_import_cost(imports, d.europe_prices, ic_ts, "national", d.network)
        - get_import_cost(imports, d.europe_prices, ic_ts, "zonal", d.network)) * 1e-9
    bottom_up_seb.loc["export_revenues_diff"] = (
        get_export_revenues(exports, d.marginal_prices, ic_ts, "zonal", d.network)
        - get_export_revenues(exports, d.marginal_prices, ic_ts, "national", d.network)) * 1e-9
    bottom_up_seb.loc["congestion_rent_diff"] = (
        get_ic_congestion_rent(d.cons, d.marginal_prices, d.europe_prices, d.network, "zonal", ic_ts)
        - get_ic_congestion_rent(d.cons, d.marginal_prices, d.europe_prices, d.network, "national", ic_ts)) * 1e-9

    we_nat = we_zon = be_nat = be_zon = 0.0
    for unit in d.thermal_units:
        we_nat += get_wholesale_expenses(d.thermal_dispatch, d.marginal_cost, unit, "national", time_slice) or 0.0
        we_zon += get_wholesale_expenses(d.thermal_dispatch, d.marginal_cost, unit, "zonal", time_slice) or 0.0
        bn = get_balancing_expenses(d.dispatch, d.thermal_dispatch, d.marginal_cost, markup, unit, "national", time_slice)
        be_nat += 0 if pd.isna(bn) else bn
        bz = get_balancing_expenses(d.dispatch, d.thermal_dispatch, d.marginal_cost, markup, unit, "zonal", time_slice)
        be_zon += 0 if pd.isna(bz) else bz
    bottom_up_seb.loc["prevented_wholesale_expenses"] = (we_nat - we_zon) * 1e-9
    bottom_up_seb.loc["prevented_balancing_expenses"] = (be_nat - be_zon) * 1e-9

    nat_td = zon_td = 0.0
    for unit in d.thermal_units:
        nat_td += get_unit_schedule(d.dispatch, unit, "national", "wholesale", time_slice)
        nat_td += get_unit_schedule(d.dispatch, unit, "national", "redispatch", time_slice)
        zon_td += get_unit_schedule(d.dispatch, unit, "zonal", "wholesale", time_slice)
        zon_td += get_unit_schedule(d.dispatch, unit, "zonal", "redispatch", time_slice)
    system_tracker.loc["fossil_generation_change"] = (zon_td - nat_td) * 1e-3

    nat_wd = zon_wd = 0.0
    for unit in d.wind_units:
        nat_wd += get_unit_schedule(d.dispatch, unit, "national", "wholesale", time_slice)
        nat_wd += get_unit_schedule(d.dispatch, unit, "national", "redispatch", time_slice)
        zon_wd += get_unit_schedule(d.dispatch, unit, "zonal", "wholesale", time_slice)
        zon_wd += get_unit_schedule(d.dispatch, unit, "zonal", "redispatch", time_slice)
    system_tracker.loc["wind_generation_change"] = (zon_wd - nat_wd) * 1e-3
    system_tracker.loc["export_volume_change"] = -system_tracker.sum()
    return bottom_up_seb, system_tracker


months = pd.date_range(f"{YEARS[0]}-01", f"{YEARS[-1]}-12", freq="MS")
seb_df, system_df = [], []
for month in tqdm(months, desc="monthly SEB"):
    seb, st = calculate_bottom_up_SEB(month.strftime("%Y-%m"))
    system_df.append(st.rename(month))
    seb_df.append(seb.rename(month))
seb = pd.concat(seb_df, axis=1).T
system = pd.concat(system_df, axis=1).T


# ---- north/south split ----
def classify_north_south(lon, lat):
    lon, lat = float(lon), float(lat)
    return "north" if lat > 0.55 * lon + 56.4 else "south"


def make_north_south_split(bmus, carrier):
    mask = bmus["carrier"].isin(carrier) if not isinstance(carrier, str) else bmus["carrier"].str.contains(carrier)
    bmus["region"] = bmus[["lat", "lon"]].apply(lambda r: classify_north_south(r["lon"], r["lat"]), axis=1)
    return (bmus.loc[mask & (bmus["region"] == "north")].index,
            bmus.loc[mask & (bmus["region"] == "south")].index)


bmus = pd.read_csv(REPO_ROOT / "data" / "prerun" / "prepared_bmus.csv", index_col=[0])
bmus = bmus.loc[bmus["lon"] != "distributed"]
nb, sb = make_north_south_split(bmus, "battery")
nph, sph = make_north_south_split(bmus, "phs")
north, south = nb.union(nph), sb.union(sph)

# ================================ plotting ================================
component_labels = {
    "prevented_wholesale_expenses": "Prevented Thermal Wholesale SRMC",
    "prevented_balancing_expenses": "Prevented Thermal Balancing SRMC",
    "import_costs_diff": "Import Cost", "export_revenues_diff": "Export Revenue",
    "congestion_rent_diff": "Interconnector Congestion Rent",
}
positive_colors = {
    "prevented_wholesale_expenses": "#9c27b0", "prevented_balancing_expenses": "#00e676",
    "congestion_rent_diff": "#651fff", "import_costs_diff": "#ff9100", "export_revenues_diff": "#ff1744",
}
year_ints = [int(y) for y in YEARS]

fig, axs = plt.subplots(3, 1, figsize=(14, 13), gridspec_kw={"height_ratios": [1, 1, 1]})

# --- top: monthly SEB ---
ax = axs[0]
seb = seb[["prevented_balancing_expenses", "prevented_wholesale_expenses", "congestion_rent_diff",
           "import_costs_diff", "export_revenues_diff"]]
pos_data = seb.clip(lower=0)
neg_data = seb.clip(upper=0)
pos_data.mul(1e3).plot(kind="bar", stacked=True, ax=ax, width=0.8,
                       color=[positive_colors.get(c, "#333333") for c in pos_data.columns])
neg_data.mul(1e3).plot(kind="bar", stacked=True, ax=ax, width=0.8,
                       color=[positive_colors.get(c, "#333333") for c in neg_data.columns], legend=False)
total_by_month = seb.sum(axis=1)
ax.scatter(range(len(total_by_month)), total_by_month.values * 1e3, color="white", edgecolor="black", s=60, zorder=10)

for year_idx, year in enumerate(year_ints):
    if year_idx % 2 == 0:
        s = (year - year_ints[0]) * 12
        ax.axvspan(s - 0.5, s + 11 + 0.5, color="lightgrey", alpha=0.3, zorder=-1)
yearly_totals = seb.sum(axis=1).groupby(seb.index.year).sum()
for i, year in enumerate(year_ints):
    if year not in yearly_totals.index:
        continue
    ax.text(i * 12 + 5.5, ax.get_ylim()[1] * 0.9, f"{year}: £{yearly_totals[year] * 1000:.0f}mn",
            ha="center", va="center", fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", boxstyle="round,pad=0.5"))

ax.set_ylabel("Monthly Socioeconomic Benefits (£mn)\n← National Benefits | Zonal Benefits →")
ax.set_xlabel("")
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
dot_handle = plt.scatter([], [], color="white", edgecolor="black", s=60)
handles = [by_label[c] for c in seb.columns] + [dot_handle]
labels = [component_labels.get(c, c) for c in seb.columns] + ["Total"]
ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.19), ncol=3, frameon=False)
ax.grid(axis="y", linestyle="--", alpha=0.7)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(labelbottom=False)

# --- middle: dispatch difference ---
ax = axs[1]
carrier_display_names = {"fossil": "Fossil + Biomass", "wind": "Wind", "interconnector": "IC Import Increases"}
colors = {"fossil": "firebrick", "wind": "royalblue", "interconnector": "#ff9100"}
colors_display = {carrier_display_names[c]: col for c, col in colors.items()}
column_mapping = {"fossil_generation_change": carrier_display_names["fossil"],
                  "import_volume_change": carrier_display_names["interconnector"],
                  "wind_generation_change": carrier_display_names["wind"]}
dispatch_df = system.rename(columns=column_mapping)[[carrier_display_names[c] for c in ["fossil", "wind", "interconnector"]]]
dispatch_df["IC Export Reductions"] = -dispatch_df.sum(axis=1)
colors_display["IC Export Reductions"] = "#ff1744"
dispatch_df.plot(kind="bar", stacked=True, ax=ax, color=[colors_display[c] for c in dispatch_df.columns], width=0.8)
for year_idx, year in enumerate(year_ints):
    if year_idx % 2 == 0:
        s = (year - year_ints[0]) * 12
        ax.axvspan(s - 0.5, s + 11 + 0.5 - 0.05, color="lightgrey", alpha=0.3, zorder=-1)
ax.grid(axis="y", linestyle="--", alpha=0.7)
ax.set_ylabel("Dispatch Difference (GWh)\n← More under National | More under Zonal →")
ax.set_xlabel("")
ax.set_xticklabels([m.strftime("%Y-%m") for m in dispatch_df.index], rotation=45)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
h, l = ax.get_legend_handles_labels()
ax.legend(handles=h, labels=l, loc="upper center", bbox_to_anchor=(0.5, 1.085), ncol=5, frameon=False)
print("Yearly SEB totals (£bn):")
print(yearly_totals)

# --- bottom: storage correlation (example days) ---
days = ["2024-01-26", "2024-01-27", "2024-01-28"]
nat_data, zon_data = [], []
def _ns_sum(net):
    cols = net.storage_units_t.p.columns
    return pd.DataFrame({"North": net.storage_units_t.p[north.intersection(cols)].sum(axis=1),
                         "South": net.storage_units_t.p[south.intersection(cols)].sum(axis=1)})


for day in days:
    nat = pypsa.Network(REPO_ROOT / "results" / day / "network_flex_s_national_solved.nc")
    zon = pypsa.Network(REPO_ROOT / "results" / day / "network_flex_s_zonal_solved.nc")
    nat_data.append(_ns_sum(nat))
    zon_data.append(_ns_sum(zon))
nat_df, zon_df = pd.concat(nat_data), pd.concat(zon_data)
nat_corr = nat_df["North"].corr(nat_df["South"])
zon_corr = zon_df["North"].corr(zon_df["South"])

axs[2].remove()
ax3 = fig.add_subplot(3, 2, 5)
ax4 = fig.add_subplot(3, 2, 6)
text_props = dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8, edgecolor="black")

for a, df, corr, color, title, ylabels in [
    (ax3, nat_df, nat_corr, "#ff0050", "National Market", True),
    (ax4, zon_df, zon_corr, "#008bfa", "Zonal Market", False),
]:
    a.scatter(df["North"], df["South"], alpha=0.7, c=color, edgecolor="k", zorder=10)
    a.set_xlabel("North Storage Power (MW)\n← Charging (Import) | Discharging (Export) →", fontsize=10)
    if ylabels:
        a.set_ylabel("South Storage Power (MW)\n← Charging (Import) | Discharging (Export) →", fontsize=10)
    else:
        a.set_yticklabels([])
    a.grid(alpha=0.3, linestyle="--")
    a.set_axisbelow(True)
    a.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    a.axvline(x=0, color="gray", linestyle="-", alpha=0.3)
    a.text(0.05, 0.95, f"{title}; r = {corr:.3f}", transform=a.transAxes, fontsize=12,
           fontweight="bold", va="top", bbox=text_props, zorder=11)
    slope, intercept, *_ = stats.linregress(df["North"], df["South"])
    xr = np.linspace(df["North"].min(), df["North"].max(), 100)
    a.plot(xr, intercept + slope * xr, color="black", linewidth=1.5, linestyle="--")
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)

fig.text(0.5, 0.08, "Settlement Periods during 26th, 27th, and 28th Jan 2024", ha="center", va="bottom",
         fontsize=11, bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="black"), zorder=12)

plt.tight_layout()
plt.subplots_adjust(hspace=0.2)
out = FIG_DIR / "fig9_socioeconomic_benefits_2022-2025.pdf"
plt.savefig(out, bbox_inches="tight")
print(f"wrote {out}  (balancing markup £{markup}/MWh)")
