"""
plotting/validation.py — model vs. published-projection comparison 2026-2035.

Loads all completed per-day model outputs across 2026-2035, aggregates to annual
totals (scaled by the number of sample days), and overlays against published
projections from NESO, DESNZ, CCC, Ember, Cornwall Insight.

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

# ===== Published projections =====================================================
# Compiled from NESO Clean Power 2030 Annex 1 (Dec 2024), DESNZ EEP 2023-2050,
# CCC Seventh Carbon Budget (Feb 2025), NESO 2025 Annual Balancing Costs Report,
# Cornwall Insight GB Power Market Outlook, Ember.
PREDICTIONS = {
    'annual_demand_TWh': {
        'NESO Clean Power 2030 Annex 1': {2023: 263, 2030: 287},
        'CCC Seventh Carbon Budget (Balanced)': {2023: 274, 2050: 692},
    },
    'offshore_wind_generation_TWh': {
        'NESO CP30 Annex 1 (midpoint)': {2023: 48, 2030: 177},
        'NESO CP30 Annex 1 (low/FFR)': {2030: 167},
        'NESO CP30 Annex 1 (high/ND)':  {2030: 187},
    },
    'onshore_wind_generation_TWh': {
        'NESO CP30 Annex 1': {2023: 31, 2030: 58},
    },
    'solar_generation_TWh': {
        'NESO CP30 Annex 1': {2023: 13, 2030: 47},
    },
    'nuclear_generation_TWh': {
        'NESO CP30 Annex 1 (midpoint)': {2030: 27},
        'NESO CP30 Annex 1 (low)':  {2030: 25},
        'NESO CP30 Annex 1 (high)': {2030: 29},
    },
    'gas_generation_TWh': {
        'NESO CP30 Annex 1 (FFR)':  {2023: 79, 2030: 14},
        'NESO CP30 Annex 1 (ND high)': {2030: 18},
        'DESNZ EEP 2023-2050': {2022: 122, 2040: 48},
    },
    'net_imports_TWh': {
        'NESO CP30 Annex 1':   {2023: 23, 2030: 0},
        'Ember outturn 2025':  {2025: 31},
    },
    'wholesale_price_GBP_per_MWh': {
        'Cornwall Insight Q4-2023 outlook': {2024: 82, 2025: 84},
    },
    'balancing_cost_GBP_bn': {
        'NESO 2025 Annual Balancing Costs (outturn)': {2023: 2.5, 2024: 2.7},
        'NESO 2025 Balancing Costs (FES central)': {
            2026: 4.0, 2027: 5.0, 2028: 6.5, 2029: 7.5, 2030: 8.0,
            2031: 6.0, 2032: 4.5, 2033: 4.0, 2034: 4.0, 2035: 4.5,
        },
        'NESO 2025 Balancing Costs (no-build counterfactual)': {2030: 12.7},
        'NESO 2025 Balancing Costs (CP30 with network)': {2030: 4.0},
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


# ===== Model output aggregation ==================================================

def network_path(day, layout='national', solved=True):
    suffix = '_solved' if solved else ''
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

    # Wind+solar curtailment (MWh) — available - realised
    re_gens = gens_gb.index[gens_gb.carrier.isin(['onwind', 'offwind', 'solar'])]
    if len(re_gens):
        pmax = n.generators_t.p_max_pu.reindex(columns=re_gens).fillna(1.0)
        avail = (pmax.multiply(gens_gb.p_nom.reindex(re_gens), axis=1) * 0.5).sum().sum()
        realised = (p_gen[re_gens] * 0.5).sum().sum()
        out['curtailment_MWh'] = max(avail - realised, 0.0)
    else:
        out['curtailment_MWh'] = 0.0

    # Daily system cost (£) — offer + bid as proxy for balancing cost
    sc_path = REPO_ROOT / 'results' / day / 'system_cost_summary_flex.csv'
    if sc_path.exists():
        sc = pd.read_csv(sc_path, index_col=[0, 1])
        daily = sc.groupby(level=1).sum()
        # CSV values are in £m
        out['offer_cost_GBPm'] = daily.loc['offer_cost', 'national'] if 'offer_cost' in daily.index else 0.0
        out['bid_cost_GBPm']   = daily.loc['bid_cost',   'national'] if 'bid_cost'   in daily.index else 0.0
    else:
        out['offer_cost_GBPm'] = 0.0
        out['bid_cost_GBPm']   = 0.0
    return out


def aggregate_annual():
    """Walk results/ for 2026-2035, return DataFrame indexed by year with annual-scaled metrics."""
    rows = []
    for year in range(2026, 2036):
        per_day = []
        for d in sorted((REPO_ROOT / 'results').glob(f'{year}-*-*')):
            day = d.name
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

        agg = {
            'year': year,
            'n_days': n_days,
            'annual_demand_TWh':     sum(d['demand_MWh'] for d in per_day) * scale / 1e6,
            'net_imports_TWh':       sum(d['net_imports_MWh'] for d in per_day) * scale / 1e6,
            'curtailment_TWh':       sum(d['curtailment_MWh'] for d in per_day) * scale / 1e6,
            'wholesale_price_GBP_per_MWh': np.nanmean([d['price_GBP_per_MWh'] for d in per_day]),
            'balancing_cost_GBP_bn': (sum(d['offer_cost_GBPm'] + d['bid_cost_GBPm'] for d in per_day) * scale / 1000),
        }
        agg.update({f'gen_{c}_TWh': v for c, v in carrier_totals.items()})
        rows.append(agg)
        print(f"{year}: {n_days} days aggregated")
    return pd.DataFrame(rows).set_index('year') if rows else None


# ===== Plot ======================================================================

def panel_metric(ax, model_y, model_series, predictions, title, ylabel,
                 model_label='model (scaled from sample days)'):
    if model_series is not None and len(model_series):
        ax.plot(model_y, model_series, marker='o', color='black', lw=2.0, label=model_label)
    # Overlay sources
    for src, series in predictions.items():
        years = sorted(series.keys())
        vals = [series[y] for y in years]
        if len(years) == 1:
            ax.scatter(years, vals, s=80, color=pred_color(src), label=src, zorder=5,
                        edgecolor='black', linewidth=0.5)
        else:
            ax.plot(years, vals, marker='s', color=pred_color(src), lw=1.2, label=src,
                     linestyle='--', markersize=5)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlim(2022.5, 2036.5)
    ax.grid(alpha=0.3)
    ax.legend(loc='best', fontsize=6, framealpha=0.9)


def build_validation_plot(df_model, out_path):
    fig, axes = plt.subplots(3, 3, figsize=(22, 16))
    fig.suptitle('GBPower vs published projections — 2026-2035', fontsize=16, y=0.995)

    y = df_model.index.values if df_model is not None else []

    def gen(carrier_key):
        col = f'gen_{carrier_key}_TWh'
        return df_model[col].values if df_model is not None and col in df_model else None

    panel_metric(axes[0,0], y, df_model['annual_demand_TWh'].values if df_model is not None else None,
                 PREDICTIONS['annual_demand_TWh'], 'Annual GB demand', 'TWh/yr')
    # combine offshore + onshore wind model
    total_wind = None
    if df_model is not None:
        ow = df_model.get('gen_offwind_TWh')
        on = df_model.get('gen_onwind_TWh')
        if ow is not None and on is not None:
            total_wind = (ow.fillna(0) + on.fillna(0)).values
    panel_metric(axes[0,1], y, gen('offwind'),
                 PREDICTIONS['offshore_wind_generation_TWh'], 'Offshore wind generation', 'TWh/yr')
    panel_metric(axes[0,2], y, gen('onwind'),
                 PREDICTIONS['onshore_wind_generation_TWh'], 'Onshore wind generation', 'TWh/yr')
    panel_metric(axes[1,0], y, gen('solar'),
                 PREDICTIONS['solar_generation_TWh'], 'Solar generation', 'TWh/yr')
    panel_metric(axes[1,1], y, gen('nuclear'),
                 PREDICTIONS['nuclear_generation_TWh'], 'Nuclear generation', 'TWh/yr')
    panel_metric(axes[1,2], y, gen('fossil'),
                 PREDICTIONS['gas_generation_TWh'], 'Gas generation', 'TWh/yr')
    panel_metric(axes[2,0], y, df_model['net_imports_TWh'].values if df_model is not None else None,
                 PREDICTIONS['net_imports_TWh'], 'Net interconnector imports', 'TWh/yr')
    panel_metric(axes[2,1], y, df_model['wholesale_price_GBP_per_MWh'].values if df_model is not None else None,
                 PREDICTIONS['wholesale_price_GBP_per_MWh'], 'GB wholesale price (load-wtd)', '£/MWh')
    panel_metric(axes[2,2], y, df_model['balancing_cost_GBP_bn'].values if df_model is not None else None,
                 PREDICTIONS['balancing_cost_GBP_bn'], 'Balancing cost (offer + bid)', '£bn/yr')

    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    print(f"Saved {out_path}")


def main():
    df = aggregate_annual()
    if df is None:
        print("No completed model days under results/ — cannot build validation plot.")
        return
    print("\nModel-aggregated annual values:")
    print(df.round(2).to_string())
    out_path = REPO_ROOT / 'plotting' / 'dashboards' / 'validation_2026-2035.pdf'
    build_validation_plot(df, out_path)


if __name__ == '__main__':
    main()
