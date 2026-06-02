"""
plotting/validation.py — model vs. outturn comparison 2022-2025.

Loads all completed per-day model outputs across 2022-2025, aggregates to annual
totals (scaled by the number of sample days), and overlays against published
outturn/projection sources from NESO, DESNZ, CCC, Ember, Cornwall Insight.

Usage:
    pixi run python plotting/validation.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'plotting'))
from plotting_constants import carrier_colors

# The historic-outturn plot is dedicated to the window where real data exists.
YEAR_MIN, YEAR_MAX = 2022, 2025
# The future-horizon plot validates the projection years against expectation (NESO CP30 etc).
FUT_MIN, FUT_MAX = 2026, 2030

# ===== Published projections =====================================================
# Compiled from NESO Clean Power 2030 Annex 1 (Dec 2024), DESNZ EEP 2023-2050,
# CCC Seventh Carbon Budget (Feb 2025), NESO 2025 Annual Balancing Costs Report,
# Cornwall Insight GB Power Market Outlook, Ember.
PREDICTIONS = {
    'annual_demand_TWh': {
        'NESO Clean Power 2030 Annex 1': {2023: 263, 2030: 287},
        'CCC Seventh Carbon Budget (Balanced)': {2023: 274, 2050: 692},
        'Outturn (NESO/DUKES)': {2022: 294, 2023: 283, 2024: 281, 2025: 280},
    },
    'offshore_wind_generation_TWh': {
        'NESO CP30 Annex 1 (midpoint)': {2023: 48, 2030: 177},
        'NESO CP30 Annex 1 (low/FFR)': {2030: 167},
        'NESO CP30 Annex 1 (high/ND)':  {2030: 187},
        'Outturn (DUKES/NESO)': {2022: 38, 2023: 49, 2024: 55, 2025: 60},
    },
    'onshore_wind_generation_TWh': {
        'NESO CP30 Annex 1': {2023: 31, 2030: 58},
        'Outturn (DUKES/NESO)': {2022: 27, 2023: 31, 2024: 34, 2025: 35},
    },
    'solar_generation_TWh': {
        'NESO CP30 Annex 1': {2023: 13, 2030: 47},
        'Outturn (DUKES/NESO)': {2022: 13.5, 2023: 13.3, 2024: 13.8, 2025: 15},
    },
    'nuclear_generation_TWh': {
        'NESO CP30 Annex 1 (midpoint)': {2030: 27},
        'NESO CP30 Annex 1 (low)':  {2030: 25},
        'NESO CP30 Annex 1 (high)': {2030: 29},
        'Outturn (DUKES/NESO)': {2022: 46, 2023: 37, 2024: 37, 2025: 41},
    },
    'gas_generation_TWh': {
        'NESO CP30 Annex 1 (FFR)':  {2023: 79, 2030: 14},
        'NESO CP30 Annex 1 (ND high)': {2030: 18},
        'DESNZ EEP 2023-2050': {2022: 122, 2040: 48},
        'Outturn (DUKES/NESO)': {2022: 98, 2023: 80, 2024: 71, 2025: 68},
    },
    'net_imports_TWh': {
        'NESO CP30 Annex 1':   {2023: 23, 2030: 0},
        'Outturn (NESO/Ember)':  {2022: 9, 2023: 24, 2024: 24, 2025: 31},
    },
    'wholesale_price_GBP_per_MWh': {
        'Cornwall Insight Q4-2023 outlook': {2024: 82, 2025: 84},
        'Outturn (ENTSO-E day-ahead avg)': {2022: 211, 2023: 98, 2024: 72, 2025: 80},
    },
    'balancing_cost_GBP_bn': {
        # MODEL SCOPE: GBPower's "balancing cost" is purely redispatch to relieve transmission/
        # thermal constraints (bids to turn down north + offers to turn up south). NESO's headline
        # TOTAL balancing cost (£2.5bn FY23/24, £2.7bn FY24/25) also covers reserve, response,
        # reactive, Black Start and energy imbalance — none represented here — so the model is NOT
        # comparable to it. The apples-to-apples outturn is NESO's THERMAL-CONSTRAINT cost (2025
        # Annual Balancing Costs Report, Fig 12), which already bundles the southern turn-up action.
        # Fig 12 is financial-year (Apr-Mar); mapped to the calendar year each FY begins in.
        'NESO thermal-constraint outturn (Fig 12; FY->CY)': {2022: 1.50, 2023: 1.04, 2024: 1.71},
        'NESO total balancing (context; out of model scope)': {2023: 2.5, 2024: 2.7},
        # Projections below are TOTAL balancing (the oft-quoted ~£8bn-by-2030 is total, not
        # constraints); shown only on the extended-horizon plot, filtered out of the 2022-2025 view.
        'NESO total balancing (FES central projection)': {
            2026: 4.0, 2027: 5.0, 2028: 6.5, 2029: 7.5, 2030: 8.0,
            2031: 6.0, 2032: 4.5, 2033: 4.0, 2034: 4.0, 2035: 4.5,
        },
        'NESO constraint cost (no-build counterfactual)': {2030: 12.7},
    },
    'wind_curtailment_TWh': {
        'NESO 2025 Balancing Costs (outturn)': {
            2018: 1.8, 2019: 2.0, 2020: 2.6, 2021: 3.0,
            2022: 4.0, 2023: 5.5, 2024: 9.5,
        },
        'Ember (industry press, 2025)': {2025: 10.0},
    },
    'co2_emissions_MtCO2': {
        'CCC 6th Carbon Budget (carbon intensity × demand)': {
            2019: 56, 2030: 14, 2035: 3, 2050: 1,
        },
        'CCC Seventh Carbon Budget (Balanced)': {2023: 38, 2040: 5},
    },
}

PRED_COLORS = {
    'Outturn': '#d62728',   # ground-truth observed (historic years) — red, stands out
    'NESO': '#1f77b4',
    'CCC':  '#9467bd',
    'DESNZ': '#8c564b',
    'Cornwall': '#e377c2',
    'Ember': '#2ca02c',
}

def pred_color(src_name):
    for k, c in PRED_COLORS.items():
        if k in src_name:
            return c
    return 'gray'


# ===== Transmission-share adjustment ============================================
# The model only covers transmission-connected BMUs. Published total-system
# projections include distribution-connected (embedded) capacity that the model
# treats as netted demand. To make the comparison apples-to-apples we scale each
# external projection by the T-share of that carrier in the relevant year.
#
# Sources: NESO Embedded Capacity Register via GridCog (2024 baseline);
# DESNZ/NESO Clean Power 2030 Action Plan — Connections Reform Annex (2030 target).
# Linear interpolation between 2024 and 2030, flat after 2030.
T_SHARES = {
    'offshore_wind': {2024: 1.00, 2030: 1.00},
    'onshore_wind':  {2024: 0.70, 2030: 0.55},
    'solar':         {2024: 0.10, 2030: 0.23},
    'nuclear':       {2024: 1.00, 2030: 1.00},
    'gas':           {2024: 1.00, 2030: 1.00},
    'demand':        {2024: 1.00, 2030: 1.00},   # demand is reported gross; model is too
    'imports':       {2024: 1.00, 2030: 1.00},   # ICs are 100% transmission
    'curtailment':   {2024: 1.00, 2030: 1.00},   # mostly transmission-driven
    'co2':           {2024: 1.00, 2030: 1.00},
    'balancing':     {2024: 1.00, 2030: 1.00},
    'price':         {2024: 1.00, 2030: 1.00},
}

def t_share(carrier, year):
    """Linear-interpolate the carrier's transmission share at the given year."""
    s = T_SHARES.get(carrier)
    if s is None:
        return 1.0
    if year <= 2024:
        return s[2024]
    if year >= 2030:
        return s[2030]
    return s[2024] + (s[2030] - s[2024]) * (year - 2024) / (2030 - 2024)


# ===== Model output aggregation ==================================================

def network_path(day, layout='national', solved=True, redispatch=False):
    suffix = '_solved_redispatch' if redispatch else ('_solved' if solved else '')
    return REPO_ROOT / 'results' / day / f'network_flex_s_{layout}{suffix}.nc'


def gb_buses(n):
    return n.buses.index[n.buses.country == 'GB']


def per_day_metrics(day):
    """Return a dict of per-day metrics for the given day, in MWh / £ / etc."""
    p = network_path(day, 'national', True)
    if not p.exists():
        return None
    n = pypsa.Network(p)
    out = {}

    gb = gb_buses(n)
    gens_gb = n.generators[n.generators.bus.isin(gb)]
    p_gen = n.generators_t.p[gens_gb.index]
    dispatch_mwh = p_gen * 0.5

    # Dispatch by carrier (MWh)
    by_carrier = dispatch_mwh.sum().groupby(gens_gb.carrier).sum()
    out['dispatch_MWh'] = by_carrier.to_dict()

    # Demand (MWh)
    gb_loads = n.loads.index[n.loads.bus.isin(gb)]
    out['demand_MWh'] = (n.loads_t.p_set[gb_loads].sum() * 0.5).sum()

    # Wholesale price (load-weighted, £/MWh) — same approach as dashboard
    load_t = n.loads_t.p_set[gb_loads].copy()
    load_t.columns = n.loads.loc[load_t.columns, 'bus']
    load_t = load_t.T.groupby(level=0).sum().T
    mp = n.buses_t.marginal_price[gb]
    common = mp.columns.intersection(load_t.columns)
    out['price_GBP_per_MWh'] = (mp[common] * load_t[common]).sum().sum() / load_t[common].sum().sum() if not common.empty else np.nan

    # Net interconnector imports (MWh) — sum over all ICs, positive = import to GB
    gbb = set(gb)
    ics = n.links[n.links.carrier == 'interconnector']
    flow_mwh = 0.0
    for ic, row in ics.iterrows():
        p0 = n.links_t.p0[ic]
        # p0 flows from bus0 → bus1; positive = bus0 sends to bus1
        if row['bus1'] in gbb:
            flow_mwh += (p0 * 0.5).sum()
        elif row['bus0'] in gbb:
            flow_mwh -= (p0 * 0.5).sum()
    out['net_imports_MWh'] = flow_mwh

    # Curtailment requires the redispatch (constrained) solution. The national WHOLESALE
    # solve is a copperplate (no internal constraints), so wind curtailment is ~0 there;
    # real GB curtailment is constraint-driven and only appears after redispatch.
    re_gens = gens_gb.index[gens_gb.carrier.isin(['onwind', 'offwind', 'solar'])]
    rp = network_path(day, 'national', redispatch=True)
    out['curtailment_MWh'] = 0.0
    out['balancing_cost_GBPm'] = 0.0

    # Balancing cost: use the SAME method as the paper's system-cost figure, i.e. the
    # redispatched volume priced against the day's actual ELEXON bid/offer stacks
    # (summarize_system_cost.get_balancing_cost → offer_cost/bid_cost, national layout,
    # already in £m). NOT the LP objective gap, which ignores real bid/offer prices and is
    # contaminated by the extendable backup generators added before the redispatch solve.
    sc = REPO_ROOT / 'results' / day / 'system_cost_summary_flex.csv'
    if sc.exists():
        df = pd.read_csv(sc, index_col=[0, 1])['national']
        by_comp = df.groupby(level=1).sum()
        out['balancing_cost_GBPm'] = float(
            by_comp.get('offer_cost', 0.0) + by_comp.get('bid_cost', 0.0)
        )

    if rp.exists():
        nr = pypsa.Network(rp)
        if len(re_gens):
            pmax = n.generators_t.p_max_pu.reindex(columns=re_gens).fillna(1.0)
            avail = (pmax.multiply(gens_gb.p_nom.reindex(re_gens), axis=1) * 0.5).sum().sum()
            realised_rd = (nr.generators_t.p.reindex(columns=re_gens).fillna(0.0) * 0.5).sum().sum()
            out['curtailment_MWh'] = max(avail - realised_rd, 0.0)
    elif len(re_gens):
        # fallback: economic curtailment only (rare)
        pmax = n.generators_t.p_max_pu.reindex(columns=re_gens).fillna(1.0)
        avail = (pmax.multiply(gens_gb.p_nom.reindex(re_gens), axis=1) * 0.5).sum().sum()
        out['curtailment_MWh'] = max(avail - (p_gen[re_gens] * 0.5).sum().sum(), 0.0)
    return out


def load_stratified_days():
    """Return the set of wind-stratified sample days (YYYY-MM-DD), or None if unavailable.

    The stratified set spans each year's wind distribution, removing the seasonal/weather
    sampling bias that a naive 'all solved days' glob would carry. When present we aggregate
    over exactly this set so cross-year comparisons are apples-to-apples.
    """
    import re
    days = set()
    # Historic (2022-2025) and future (2026-2030) stratified target lists. The future set reuses
    # the 2025 weather days mapped onto each future year (via fix_year), so the seasonal/wind
    # sampling is identical across both windows.
    for fname in ('gbpower_stratified_targets.txt', 'gbpower_future_targets.txt'):
        f = Path('/tmp') / fname
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            m = re.search(r'results/(\d{4}-\d{2}-\d{2})/', line)
            if m:
                days.add(m.group(1))
    return days or None


def aggregate_annual(year_min=YEAR_MIN, year_max=YEAR_MAX, stratified=None):
    """Walk results/ for [year_min, year_max], return DataFrame indexed by year with annual metrics.

    `stratified` is an optional set of YYYY-MM-DD days to restrict aggregation to (de-biasing
    the seasonal/weather sample). Pass None to aggregate every solved day in the window.
    """
    if stratified is not None:
        in_window = {d for d in stratified if year_min <= int(d[:4]) <= year_max}
        print(f"Aggregating over {len(in_window)} stratified days in {year_min}-{year_max}")
    rows = []
    for year in range(year_min, year_max + 1):
        per_day = []
        for d in sorted((REPO_ROOT / 'results').glob(f'{year}-*-*')):
            day = d.name
            if stratified is not None and day not in stratified:
                continue
            try:
                m = per_day_metrics(day)
            except Exception as e:
                print(f"  warn: {day} failed → {e}")
                continue
            if m is not None:
                per_day.append(m)
        if not per_day:
            print(f"{year}: NO completed days")
            continue
        n_days = len(per_day)
        scale = 365.25 / n_days

        # Aggregate dispatch by carrier
        carrier_totals = {}
        for d in per_day:
            for c, mwh in d['dispatch_MWh'].items():
                carrier_totals[c] = carrier_totals.get(c, 0.0) + mwh
        carrier_totals = {c: v * scale / 1e6 for c, v in carrier_totals.items()}   # TWh/yr

        # ---- Sampling uncertainty ---------------------------------------------------
        # Every annual figure is an estimate built from n_days sample days. For a sum-type
        # metric the annual estimate is 365.25 * mean_per_day, so its standard error is
        # 365.25 * std_per_day / sqrt(n) (simple-random-sampling SE; the wind-stratified
        # design makes the true SE somewhat tighter, so this is a conservative band). For the
        # load-weighted price (a per-day mean we average, not scale) the SE is std/sqrt(n).
        def sum_se(vals, unit_scale):
            a = np.asarray(vals, float)
            if len(a) < 2:
                return np.nan
            return 365.25 * a.std(ddof=1) / np.sqrt(len(a)) * unit_scale

        def mean_se(vals):
            a = np.asarray([v for v in vals if not np.isnan(v)], float)
            if len(a) < 2:
                return np.nan
            return a.std(ddof=1) / np.sqrt(len(a))

        agg = {
            'year': year,
            'n_days': n_days,
            'annual_demand_TWh':     sum(d['demand_MWh'] for d in per_day) * scale / 1e6,
            'net_imports_TWh':       sum(d['net_imports_MWh'] for d in per_day) * scale / 1e6,
            'curtailment_TWh':       sum(d['curtailment_MWh'] for d in per_day) * scale / 1e6,
            'wholesale_price_GBP_per_MWh': np.nanmean([d['price_GBP_per_MWh'] for d in per_day]),
            'balancing_cost_GBP_bn': (sum(d['balancing_cost_GBPm'] for d in per_day) * scale / 1000),
            'annual_demand_TWh_se':  sum_se([d['demand_MWh'] for d in per_day], 1 / 1e6),
            'net_imports_TWh_se':    sum_se([d['net_imports_MWh'] for d in per_day], 1 / 1e6),
            'curtailment_TWh_se':    sum_se([d['curtailment_MWh'] for d in per_day], 1 / 1e6),
            'wholesale_price_GBP_per_MWh_se': mean_se([d['price_GBP_per_MWh'] for d in per_day]),
            'balancing_cost_GBP_bn_se': sum_se([d['balancing_cost_GBPm'] for d in per_day], 1 / 1000),
        }
        agg.update({f'gen_{c}_TWh': v for c, v in carrier_totals.items()})
        # per-carrier SE: align each day's dispatch onto the full carrier set (0 if absent)
        all_carriers = {c for d in per_day for c in d['dispatch_MWh']}
        for c in all_carriers:
            agg[f'gen_{c}_TWh_se'] = sum_se(
                [d['dispatch_MWh'].get(c, 0.0) for d in per_day], 1 / 1e6)
        rows.append(agg)
        print(f"{year}: {n_days} days aggregated")
    return pd.DataFrame(rows).set_index('year') if rows else None


# ===== Plot ======================================================================

def panel_metric(ax, model_y, model_series, predictions, title, ylabel,
                 model_label='model (transmission-only, scaled to annual)',
                 t_share_key=None, model_se=None, year_min=YEAR_MIN, year_max=YEAR_MAX):
    if model_series is not None and len(model_series):
        ax.plot(model_y, model_series, marker='o', color='black', lw=2.2, label=model_label, zorder=10)
        # 95% sampling band from running only n_days/year (1.96 × SE of the annual estimate).
        if model_se is not None:
            ms = np.asarray(model_series, float)
            se = np.asarray(model_se, float)
            ok = ~np.isnan(se)
            if ok.any():
                lo, hi = ms - 1.96 * se, ms + 1.96 * se
                ax.fill_between(np.asarray(model_y)[ok], lo[ok], hi[ok],
                                color='black', alpha=0.15, lw=0, zorder=9,
                                label='95% sampling band')
    # Overlay published sources. If `t_share_key` is set, also draw a transmission-only
    # adjusted version of each source in solid form (the apples-to-apples comparison).
    for src, series in predictions.items():
        years = sorted(y for y in series if year_min <= y <= year_max)
        if not years:
            continue
        vals = [series[y] for y in years]
        col = pred_color(src)
        if len(years) == 1:
            ax.scatter(years, vals, s=70, color=col, marker='s', zorder=4,
                       edgecolor='black', linewidth=0.5, label=f'{src} (total system)')
        else:
            ax.plot(years, vals, marker='s', color=col, lw=1.0, linestyle=':',
                    markersize=4, label=f'{src} (total system)', alpha=0.55)
        if t_share_key:
            adj_vals = [v * t_share(t_share_key, y) for v, y in zip(vals, years)]
            if len(years) == 1:
                ax.scatter(years, adj_vals, s=70, color=col, marker='o', zorder=5,
                           edgecolor='black', linewidth=0.8, label=f'{src} × T-share')
            else:
                ax.plot(years, adj_vals, marker='o', color=col, lw=1.6, linestyle='--',
                        markersize=5, label=f'{src} × T-share')
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlim(year_min - 0.5, year_max + 0.5)
    ax.set_xticks(range(year_min, year_max + 1))
    # Anchor the y-axis at 0 for context; keep room for any negative values (e.g. net exports).
    ax.set_ylim(bottom=min(0.0, ax.get_ylim()[0]))
    ax.grid(alpha=0.3)
    ax.legend(loc='best', fontsize=6, framealpha=0.9)


def build_validation_plot(df_model, out_path, year_min=YEAR_MIN, year_max=YEAR_MAX,
                          suptitle='GBPower vs outturn (2022-2025)'):
    fig, axes = plt.subplots(4, 3, figsize=(22, 21))
    fig.suptitle(suptitle, fontsize=16, y=0.995)

    y = df_model.index.values if df_model is not None else []
    win = dict(year_min=year_min, year_max=year_max)

    def gen(carrier_key):
        col = f'gen_{carrier_key}_TWh'
        return df_model[col].values if df_model is not None and col in df_model else None

    def se(col):
        c = f'{col}_se'
        return df_model[c].values if df_model is not None and c in df_model else None

    panel_metric(axes[0,0], y, df_model['annual_demand_TWh'].values if df_model is not None else None,
                 PREDICTIONS['annual_demand_TWh'], 'Annual GB demand', 'TWh/yr', t_share_key='demand',
                 model_se=se('annual_demand_TWh'), **win)
    panel_metric(axes[0,1], y, gen('offwind'),
                 PREDICTIONS['offshore_wind_generation_TWh'], 'Offshore wind generation', 'TWh/yr', t_share_key='offshore_wind',
                 model_se=se('gen_offwind_TWh'), **win)
    panel_metric(axes[0,2], y, gen('onwind'),
                 PREDICTIONS['onshore_wind_generation_TWh'], 'Onshore wind generation', 'TWh/yr', t_share_key='onshore_wind',
                 model_se=se('gen_onwind_TWh'), **win)
    panel_metric(axes[1,0], y, gen('solar'),
                 PREDICTIONS['solar_generation_TWh'], 'Solar generation', 'TWh/yr', t_share_key='solar',
                 model_se=se('gen_solar_TWh'), **win)
    panel_metric(axes[1,1], y, gen('nuclear'),
                 PREDICTIONS['nuclear_generation_TWh'], 'Nuclear generation', 'TWh/yr', t_share_key='nuclear',
                 model_se=se('gen_nuclear_TWh'), **win)
    panel_metric(axes[1,2], y, gen('fossil'),
                 PREDICTIONS['gas_generation_TWh'], 'Gas generation', 'TWh/yr', t_share_key='gas',
                 model_se=se('gen_fossil_TWh'), **win)
    panel_metric(axes[2,0], y, df_model['net_imports_TWh'].values if df_model is not None else None,
                 PREDICTIONS['net_imports_TWh'], 'Net interconnector imports', 'TWh/yr', t_share_key='imports',
                 model_se=se('net_imports_TWh'), **win)
    panel_metric(axes[2,1], y, df_model['wholesale_price_GBP_per_MWh'].values if df_model is not None else None,
                 PREDICTIONS['wholesale_price_GBP_per_MWh'], 'GB wholesale price (load-wtd)', '£/MWh', t_share_key='price',
                 model_se=se('wholesale_price_GBP_per_MWh'), **win)
    panel_metric(axes[2,2], y, df_model['balancing_cost_GBP_bn'].values if df_model is not None else None,
                 PREDICTIONS['balancing_cost_GBP_bn'], 'Thermal-constraint cost (bids+offers; model scope)', '£bn/yr', t_share_key='balancing',
                 model_se=se('balancing_cost_GBP_bn'), **win)
    panel_metric(axes[3,0], y, df_model['curtailment_TWh'].values if df_model is not None else None,
                 PREDICTIONS['wind_curtailment_TWh'], 'Wind curtailment (post-redispatch)', 'TWh/yr', t_share_key='curtailment',
                 model_se=se('curtailment_TWh'), **win)
    axes[3,1].set_visible(False)
    axes[3,2].set_visible(False)

    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    print(f"Saved {out_path}")


# Maps each model column to its outturn reference series + the T-share carrier used to
# put the (total-system) outturn on the model's transmission-only footing.
_ASSESS_SPEC = [
    ('annual_demand_TWh',            'annual_demand_TWh',              'demand',       'GB demand (TWh)'),
    ('gen_offwind_TWh',              'offshore_wind_generation_TWh',   'offshore_wind','Offshore wind (TWh)'),
    ('gen_onwind_TWh',               'onshore_wind_generation_TWh',    'onshore_wind', 'Onshore wind (TWh)'),
    ('gen_solar_TWh',                'solar_generation_TWh',           'solar',        'Solar (TWh)'),
    ('gen_nuclear_TWh',              'nuclear_generation_TWh',         'nuclear',      'Nuclear (TWh)'),
    ('gen_fossil_TWh',               'gas_generation_TWh',             'gas',          'Gas (TWh)'),
    ('net_imports_TWh',              'net_imports_TWh',                'imports',      'Net imports (TWh)'),
    ('wholesale_price_GBP_per_MWh',  'wholesale_price_GBP_per_MWh',    'price',        'Wholesale price (£/MWh)'),
    ('balancing_cost_GBP_bn',        'balancing_cost_GBP_bn',          'balancing',    'Balancing cost (£bn)'),
    ('curtailment_TWh',              'wind_curtailment_TWh',           'curtailment',  'Curtailment (TWh)'),
]


def _outturn_series(pred_key):
    """Return the {year: value} dict of the outturn source for a predictions group, or None."""
    for src, series in PREDICTIONS[pred_key].items():
        if 'outturn' in src.lower():
            return series
    return None


def assess_sampling(df):
    """Quantify how much of each model-vs-outturn gap the finite sample could plausibly explain.

    z = (model − outturn×Tshare) / SE_annual; SE is the standard error of the annual estimate
    from n sample days. |z| < 2 → the discrepancy is within what sampling noise alone could
    produce; |z| ≥ 2 → a structural model gap the sample size cannot account for. p is the
    two-sided probability of a gap this large arising from sampling if the model were unbiased.
    """
    import math
    print("\n=== Sampling-noise assessment (95% = |z|<2) =========================================")
    print("Outturn put on transmission-only footing via T-share. SE from per-day spread / sqrt(n).")
    print(f"{'metric':<24}{'yr':>5}{'model':>9}{'±95%':>8}{'outturn*Ts':>12}{'z':>7}{'p_sampling':>12}  verdict")
    n_struct = n_noise = 0
    for col, pred_key, tkey, label in _ASSESS_SPEC:
        obs = _outturn_series(pred_key)
        if obs is None or col not in df:
            continue
        for yr in df.index:
            if yr not in obs:
                continue
            model = df.at[yr, col]
            sec = f'{col}_se'
            se = df.at[yr, sec] if sec in df else np.nan
            ref = obs[yr] * t_share(tkey, yr)
            dev = model - ref
            if not (se and se > 0 and not np.isnan(se)):
                continue
            z = dev / se
            p = math.erfc(abs(z) / math.sqrt(2))
            if abs(z) < 2:
                verdict = 'within sampling noise'; n_noise += 1
            else:
                verdict = 'STRUCTURAL gap'; n_struct += 1
            print(f"{label:<24}{yr:>5}{model:>9.1f}{1.96*se:>8.1f}{ref:>12.1f}"
                  f"{z:>7.1f}{p:>12.3f}  {verdict}")
    print(f"-> {n_noise} metric-years explainable by sampling, {n_struct} structural "
          f"(real model gaps the {int(df['n_days'].iloc[0])}-day/yr sample cannot explain).")


# For the future horizon there is no outturn; the model is judged against the published
# expectation (NESO Clean Power 2030). For each metric this gives the central CP30-2030 series
# to compare to, and a (low, high) band where CP30 publishes a range.
_FUTURE_EXPECT = {
    # col: (pred_key, central_src_substr, [range_src_substrs], tkey)
    'annual_demand_TWh':           ('annual_demand_TWh',            'NESO Clean Power 2030', [], 'demand'),
    'gen_offwind_TWh':             ('offshore_wind_generation_TWh', 'midpoint', ['low', 'high'], 'offshore_wind'),
    'gen_onwind_TWh':              ('onshore_wind_generation_TWh',  'NESO CP30 Annex 1', [], 'onshore_wind'),
    'gen_solar_TWh':               ('solar_generation_TWh',         'NESO CP30 Annex 1', [], 'solar'),
    'gen_nuclear_TWh':             ('nuclear_generation_TWh',       'midpoint', ['low', 'high'], 'nuclear'),
    'gen_fossil_TWh':              ('gas_generation_TWh',           'FFR', ['ND'], 'gas'),
    'net_imports_TWh':             ('net_imports_TWh',              'NESO CP30 Annex 1', [], 'imports'),
}


def assess_future(df, target_year=2030):
    """Judge the future-horizon endpoint against published expectation (NESO CP30 2030).

    No outturn exists, so 'verdict' here is overlap of the model's 95% sampling band with the
    CP30-2030 expectation (put on transmission-only footing via T-share). 'matches' = band
    contains the central expectation; 'consistent' = band overlaps the published range; else
    'above'/'below' expectation.
    """
    if df is None or target_year not in df.index:
        print(f"\n(no model data for {target_year}; skipping future-expectation assessment)")
        return
    print(f"\n=== Future-horizon assessment: model {target_year} vs NESO CP30 expectation ========")
    print("Expectation × T-share (transmission-only footing). Verdict = overlap of model 95% band.")
    print(f"{'metric':<24}{'model':>9}{'±95%':>8}{'CP30*Ts':>10}{'range':>16}  verdict")
    for col, (pred_key, central_sub, range_subs, tkey) in _FUTURE_EXPECT.items():
        if col not in df:
            continue
        series_by_src = PREDICTIONS[pred_key]
        ts = t_share(tkey, target_year)
        central = None
        for src, series in series_by_src.items():
            if central_sub in src and target_year in series:
                central = series[target_year] * ts
                break
        if central is None:
            continue
        lo = hi = central
        for src, series in series_by_src.items():
            if target_year in series and any(s in src for s in range_subs):
                lo = min(lo, series[target_year] * ts)
                hi = max(hi, series[target_year] * ts)
        model = df.at[target_year, col]
        sec = f'{col}_se'
        se = df.at[target_year, sec] if sec in df else np.nan
        band = 1.96 * se if (se and not np.isnan(se)) else 0.0
        m_lo, m_hi = model - band, model + band
        if m_lo <= central <= m_hi:
            verdict = 'matches expectation'
        elif m_hi >= lo and m_lo <= hi:
            verdict = 'consistent w/ range'
        elif model > hi:
            verdict = 'ABOVE expectation'
        else:
            verdict = 'BELOW expectation'
        rng = f'[{lo:.0f},{hi:.0f}]' if hi > lo else '—'
        print(f"{col:<24}{model:>9.1f}{band:>8.1f}{central:>10.1f}{rng:>16}  {verdict}")


def main():
    strat = load_stratified_days()
    # ---- Historic window (real outturn) ----
    df = aggregate_annual(YEAR_MIN, YEAR_MAX, stratified=strat)
    if df is None:
        print("No completed model days under results/ — cannot build validation plot.")
        return
    print("\nModel-aggregated annual values (historic):")
    print(df.round(2).to_string())
    assess_sampling(df)
    build_validation_plot(df, REPO_ROOT / 'plotting' / 'dashboards' / 'validation_2022-2025.pdf')

    # ---- Future window (vs expectation) ----
    df_fut = aggregate_annual(FUT_MIN, FUT_MAX, stratified=strat)
    if df_fut is not None:
        print("\nModel-aggregated annual values (future horizon):")
        print(df_fut.round(2).to_string())
        assess_future(df_fut, target_year=2030)
        build_validation_plot(
            df_fut, REPO_ROOT / 'plotting' / 'dashboards' / 'validation_2026-2030.pdf',
            year_min=FUT_MIN, year_max=FUT_MAX,
            suptitle='GBPower vs NESO Clean Power 2030 expectation (2026-2030)')
    else:
        print(f"\n(no solved days in {FUT_MIN}-{FUT_MAX} yet; future plot skipped)")


if __name__ == '__main__':
    main()
