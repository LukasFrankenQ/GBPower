"""
plotting/dashboard.py — single-day model-run overview.

A 3×3 grid of diagnostic panels intended for sanity-checking and debugging a day's
solve across all three layouts (national / zonal / nodal) plus the redispatch step.

Usage:
    pixi run python plotting/dashboard.py 2025-03-21
    pixi run python plotting/dashboard.py 2024-03-21 --out dashboard_dryrun.pdf

Outputs:
    plotting/dashboards/dashboard_{day}.pdf  (or whatever --out specifies)

Panels:
    (0,0) Load-weighted GB wholesale price per layout
    (0,1) Dispatch stack — national wholesale
    (0,2) Dispatch stack — nodal wholesale
    (1,0) Balancing volume by carrier (national redispatch - national wholesale)
    (1,1) Interconnector flows (positive = import into GB)
    (1,2) Storage state of charge (battery + PHS) across layouts
    (2,0) System cost breakdown per layout (wholesale + redispatch)
    (2,1) Energy balance — generation by carrier, daily totals per layout
    (2,2) Renewable curtailment — potential vs realised (wholesale, per layout)
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'plotting'))
from plotting_constants import carrier_colors, nice_carrier_names

LAYOUTS = ('national', 'zonal', 'nodal')
LAYOUT_COLORS = {'national': '#d62728', 'zonal': '#2ca02c', 'nodal': '#1f77b4'}


# ---------- data loading ----------

def network_path(day, layout, redispatch=False, ic='flex'):
    suffix = '_solved_redispatch' if redispatch else '_solved'
    return REPO_ROOT / 'results' / day / f'network_{ic}_s_{layout}{suffix}.nc'


def load_networks(day):
    """Return dict keyed by (layout, 'wholesale'|'redispatch')."""
    nets = {}
    for layout in LAYOUTS:
        for variant, redisp in (('wholesale', False), ('redispatch', True)):
            p = network_path(day, layout, redisp)
            if p.exists():
                try:
                    nets[(layout, variant)] = pypsa.Network(p)
                except Exception as e:
                    print(f"  warning: failed to load {p.name}: {e}")
    return nets


# ---------- helpers ----------

def gb_buses(n):
    return n.buses.index[n.buses.country == 'GB']


def gb_load_weighted_price(n):
    """£/MWh, load-weighted across GB buses."""
    buses = gb_buses(n)
    load_cols = n.loads.index[n.loads.bus.isin(buses)]
    load_t = n.loads_t.p_set[load_cols].copy()
    load_t.columns = n.loads.loc[load_t.columns, 'bus']
    lmp = n.buses_t.marginal_price[buses]
    # align columns
    common = lmp.columns.intersection(load_t.columns)
    lmp = lmp[common]
    # collapse duplicate bus columns (pandas dropped groupby(axis=1) — transpose-trick)
    load_t = load_t.loc[:, common].T.groupby(level=0).sum().T
    return (lmp * load_t).sum(axis=1) / load_t.sum(axis=1)


def dispatch_by_carrier(n, mwh=True):
    """GB generators only. Returns (snapshots × carrier) in MW or MWh per snapshot."""
    buses = gb_buses(n)
    gens = n.generators[n.generators.bus.isin(buses)]
    p = n.generators_t.p[gens.index]
    factor = 0.5 if mwh else 1.0
    by_carrier = p.T.groupby(gens.carrier).sum().T * factor
    # Order carriers consistently
    order = [c for c in carrier_colors if c in by_carrier.columns]
    extras = [c for c in by_carrier.columns if c not in order]
    return by_carrier[order + extras]


def storage_by_carrier(n):
    """Net storage dispatch (positive = discharge) and SoC, per carrier, GB only."""
    buses = gb_buses(n)
    su = n.storage_units[n.storage_units.bus.isin(buses)]
    if su.empty:
        return pd.DataFrame(), pd.DataFrame()
    p = n.storage_units_t.p[su.index].T.groupby(su.carrier).sum().T
    soc = n.storage_units_t.state_of_charge[su.index].T.groupby(su.carrier).sum().T
    return p, soc


def interconnector_flow(n):
    """Per-IC flow over snapshots; positive = import to GB. MWh per snapshot."""
    gbb = set(gb_buses(n))
    ics = n.links[n.links.carrier == 'interconnector']
    if ics.empty:
        return pd.DataFrame()
    # flow into GB = p0 if bus1 is GB; -p0 if bus0 is GB (export)
    flows = {}
    for ic, row in ics.iterrows():
        p0 = n.links_t.p0[ic]
        if row['bus1'] in gbb:
            flows[ic] = p0 * 0.5            # positive p0 means power flowing from bus0 -> bus1 = import to GB
        elif row['bus0'] in gbb:
            flows[ic] = -p0 * 0.5
        else:
            flows[ic] = p0 * 0.5
    return pd.DataFrame(flows)


def system_costs(day):
    """Read system_cost_summary_flex.csv if available."""
    p = REPO_ROOT / 'results' / day / 'system_cost_summary_flex.csv'
    if not p.exists():
        return None
    return pd.read_csv(p, index_col=0)


# ---------- panels ----------

def plot_price_comparison(ax, nets):
    for layout in LAYOUTS:
        n = nets.get((layout, 'wholesale'))
        if n is None:
            continue
        s = gb_load_weighted_price(n)
        ax.plot(s.index, s.values, label=layout, color=LAYOUT_COLORS[layout], lw=1.6)
    ax.set_title('Wholesale price — GB load-weighted')
    ax.set_ylabel('£/MWh')
    ax.legend(loc='upper left', fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.grid(alpha=0.3)


def plot_dispatch_stack(ax, n, title):
    if n is None:
        ax.text(0.5, 0.5, '(no network)', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return
    disp = dispatch_by_carrier(n, mwh=False)  # MW for instantaneous plot
    pos = disp.clip(lower=0)
    neg = disp.clip(upper=0)
    cols = [c for c in pos.columns if pos[c].abs().sum() > 0]
    colors = [carrier_colors.get(c, 'gray') for c in cols]
    labels = [nice_carrier_names.get(c, c) for c in cols]
    if cols:
        ax.stackplot(pos.index, pos[cols].T.values, labels=labels, colors=colors)
        # negative part (storage charging etc.)
        ax.stackplot(neg.index, neg[cols].T.values, colors=colors)
    ax.set_title(title)
    ax.set_ylabel('MW')
    ax.legend(loc='upper left', fontsize=7, ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.grid(alpha=0.3)


def plot_balancing_volume(ax, nets, layout='national'):
    w = nets.get((layout, 'wholesale'))
    r = nets.get((layout, 'redispatch'))
    if w is None or r is None:
        ax.text(0.5, 0.5, f'(no redispatch for {layout})', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'Balancing volume — {layout}')
        return
    diff = dispatch_by_carrier(r, mwh=False) - dispatch_by_carrier(w, mwh=False)
    cols = [c for c in diff.columns if diff[c].abs().sum() > 0]
    for c in cols:
        ax.plot(diff.index, diff[c].values, label=nice_carrier_names.get(c, c),
                color=carrier_colors.get(c, 'gray'), lw=1.2)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_title(f'Balancing volume — {layout} (redispatch − wholesale)')
    ax.set_ylabel('MW')
    ax.legend(loc='upper left', fontsize=7, ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.grid(alpha=0.3)


def plot_interconnectors(ax, nets):
    n = nets.get(('nodal', 'wholesale')) or next(iter(nets.values()), None)
    if n is None:
        ax.set_title('Interconnector flows')
        return
    flows = interconnector_flow(n)
    if flows.empty:
        ax.text(0.5, 0.5, '(no interconnectors)', ha='center', va='center', transform=ax.transAxes)
    else:
        for ic in flows.columns:
            ax.plot(flows.index, flows[ic].values / 0.5, label=ic, lw=1.0)  # back to MW
        ax.axhline(0, color='k', lw=0.5)
        ax.legend(loc='upper left', fontsize=7, ncol=2)
    ax.set_title('Interconnector flow — nodal wholesale (+ = import)')
    ax.set_ylabel('MW')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.grid(alpha=0.3)


def plot_storage(ax, nets):
    for layout in LAYOUTS:
        n = nets.get((layout, 'wholesale'))
        if n is None:
            continue
        _, soc = storage_by_carrier(n)
        if soc.empty:
            continue
        total_soc = soc.sum(axis=1)
        ax.plot(total_soc.index, total_soc.values, label=f'{layout}', color=LAYOUT_COLORS[layout], lw=1.4)
    ax.set_title('Storage state of charge (battery + PHS, GB only)')
    ax.set_ylabel('MWh')
    ax.legend(loc='upper left', fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.grid(alpha=0.3)


def plot_system_costs(ax, day):
    p = REPO_ROOT / 'results' / day / 'system_cost_summary_flex.csv'
    if not p.exists():
        ax.text(0.5, 0.5, '(no system_cost_summary_flex.csv)', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('System cost')
        return
    df = pd.read_csv(p, index_col=[0, 1])
    # Aggregate snapshot-level costs to a daily total per (component, layout)
    daily = df.groupby(level=1).sum()
    # balancing_volume is an MWh quantity, not a cost — drop from this panel
    daily = daily.drop(index='balancing_volume', errors='ignore')
    daily.plot.bar(ax=ax, color={'national': LAYOUT_COLORS['national'],
                                  'zonal': LAYOUT_COLORS['zonal'],
                                  'nodal': LAYOUT_COLORS['nodal']}, width=0.8)
    ax.set_title('Daily cost components by layout')
    ax.set_ylabel('£m (per day)')
    ax.tick_params(axis='x', rotation=20)
    ax.axhline(0, color='k', lw=0.5)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3, axis='y')


def plot_energy_balance(ax, nets):
    rows = []
    for layout in LAYOUTS:
        n = nets.get((layout, 'wholesale'))
        if n is None:
            continue
        disp = dispatch_by_carrier(n, mwh=True).sum()  # GWh -> MWh totals
        rows.append(disp.rename(layout) / 1000)         # GWh
    if not rows:
        ax.set_title('Energy balance')
        return
    df = pd.concat(rows, axis=1).fillna(0)
    cols = df.index.tolist()
    colors = [carrier_colors.get(c, 'gray') for c in cols]
    df.T.plot.bar(stacked=True, ax=ax, color=colors, width=0.7)
    ax.set_title('Daily generation by carrier (GB only, wholesale)')
    ax.set_ylabel('GWh')
    ax.legend(loc='upper left', fontsize=7, ncol=2)
    ax.tick_params(axis='x', rotation=0)
    ax.grid(alpha=0.3, axis='y')


def _producer_surplus_by_carrier(day, layout, balancing_markup=30.0):
    """Mirror get_total_results.py's get_thermal_unit_surplus, aggregated by carrier (£m).

    surplus = wholesale_rev + cfd_rev + roc_rev - production_cost - balancing_cost
    production_cost is dispatch × marginal_cost (thermal generators only).
    balancing_cost is redispatch_volume × (basic_running_cost + balancing_markup) — thermals only.
    """
    rev_path = REPO_ROOT / 'results' / day / f'bmu_revenues_detailed_flex_{layout}.csv'
    net_w_path = REPO_ROOT / 'results' / day / f'network_flex_s_{layout}_solved.nc'
    net_r_path = REPO_ROOT / 'results' / day / f'network_flex_s_{layout}_solved_redispatch.nc'
    if not (rev_path.exists() and net_w_path.exists()):
        return None

    revs = pd.read_csv(rev_path, index_col=0)
    revs['total_rev'] = revs[['wholesale_revenue', 'cfd_revenue', 'roc_revenue']].fillna(0).sum(axis=1)

    n_w = pypsa.Network(net_w_path)
    n_r = pypsa.Network(net_r_path) if net_r_path.exists() else None

    # carrier per asset (lookup against network generators)
    gens = n_w.generators[['carrier']].copy()
    revs = revs.join(gens, how='left')

    # production cost (£) per asset, thermal only — dispatch × marginal_cost summed over snapshots
    mc_static = n_w.generators.marginal_cost.fillna(0)
    mc_t = n_w.generators_t.marginal_cost
    p = n_w.generators_t.p
    cost = pd.Series(0.0, index=n_w.generators.index)
    for g in p.columns:
        if g in mc_t.columns and not mc_t[g].isna().all():
            cost[g] = (p[g] * mc_t[g]).sum() * 0.5
        else:
            cost[g] = (p[g] * mc_static.get(g, 0)).sum() * 0.5

    # balancing cost for thermal (uses redispatch). Approx: redispatch volume × (mean MC + markup)
    bal_cost = pd.Series(0.0, index=n_w.generators.index)
    if n_r is not None:
        common = p.columns.intersection(n_r.generators_t.p.columns)
        bv = (n_r.generators_t.p[common] - p[common]).abs() * 0.5  # MWh per snapshot
        # avg MC per generator over the day
        mean_mc = pd.Series(0.0, index=common)
        for g in common:
            if g in mc_t.columns and not mc_t[g].isna().all():
                mean_mc[g] = mc_t[g].mean()
            else:
                mean_mc[g] = mc_static.get(g, 0)
        bal_cost.loc[common] = (bv.sum() * (mean_mc + balancing_markup))

    # only apply prod/bal cost to thermal carriers (fossil, biomass)
    thermal = revs['carrier'].isin(['fossil', 'biomass'])
    revs['prod_cost'] = 0.0
    revs.loc[thermal, 'prod_cost'] = cost.reindex(revs.index[thermal]).fillna(0).values
    revs['bal_cost'] = 0.0
    revs.loc[thermal, 'bal_cost'] = bal_cost.reindex(revs.index[thermal]).fillna(0).values
    revs['surplus'] = revs['total_rev'] - revs['prod_cost'] - revs['bal_cost']

    # group by carrier, convert to £m
    return revs.dropna(subset=['carrier']).groupby('carrier')['surplus'].sum() / 1e6


def plot_producer_surplus(ax, day):
    """Daily producer surplus by carrier per layout (£m)."""
    rows = []
    for layout in LAYOUTS:
        s = _producer_surplus_by_carrier(day, layout)
        if s is None:
            continue
        rows.append(s.rename(layout))
    if not rows:
        ax.text(0.5, 0.5, '(no surplus data)', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Producer surplus by carrier')
        return
    df = pd.concat(rows, axis=1).fillna(0)
    # Order carriers consistently
    ordered = [c for c in carrier_colors if c in df.index]
    extras = [c for c in df.index if c not in ordered]
    df = df.loc[ordered + extras]
    colors = [carrier_colors.get(c, 'gray') for c in df.index]
    df.T.plot.bar(stacked=True, ax=ax, color=colors, width=0.7)
    ax.set_title('Producer surplus by carrier (£m, daily)')
    ax.set_ylabel('£m / day')
    ax.tick_params(axis='x', rotation=0)
    ax.axhline(0, color='k', lw=0.5)
    ax.legend(loc='best', fontsize=7, ncol=2)
    ax.grid(alpha=0.3, axis='y')


def plot_seb_delta(ax, day):
    """SEB-style decomposition: daily Δ vs national, per system-cost component (£m).

    Higher consumer-side cost is a negative SEB contribution. We show
    `(national − layout)` so positive means the layout is cheaper than national for that component.
    """
    p = REPO_ROOT / 'results' / day / 'system_cost_summary_flex.csv'
    if not p.exists():
        ax.text(0.5, 0.5, '(no system_cost_summary)', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('SEB delta vs national')
        return
    df = pd.read_csv(p, index_col=[0, 1])
    daily = df.groupby(level=1).sum()
    daily = daily.drop(index='balancing_volume', errors='ignore')
    # Sign convention: positive = layout saves money vs national (i.e. national_cost - layout_cost)
    deltas = pd.DataFrame({
        'zonal vs national': (daily['national'] - daily['zonal']).round(3),
        'nodal vs national': (daily['national'] - daily['nodal']).round(3),
    })
    deltas.plot.bar(ax=ax, color=[LAYOUT_COLORS['zonal'], LAYOUT_COLORS['nodal']], width=0.75)
    # totals
    for i, col in enumerate(deltas.columns):
        total = deltas[col].sum()
        ax.annotate(f'Σ = {total:+.2f}', (i / len(deltas.columns) + 0.05, 0.95),
                    xycoords='axes fraction', fontsize=8,
                    color=LAYOUT_COLORS['zonal' if 'zonal' in col else 'nodal'],
                    weight='bold')
    ax.set_title('SEB-style decomp — Δ vs national (positive = cheaper than national)')
    ax.set_ylabel('£m / day')
    ax.axhline(0, color='k', lw=0.5)
    ax.tick_params(axis='x', rotation=20)
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3, axis='y')


def plot_curtailment(ax, nets):
    """Available wind/solar (p_nom * p_max_pu) minus realised dispatch, per layout."""
    rows = []
    for layout in LAYOUTS:
        n = nets.get((layout, 'wholesale'))
        if n is None:
            continue
        buses = gb_buses(n)
        gens = n.generators[n.generators.bus.isin(buses) & n.generators.carrier.isin(['onwind', 'offwind', 'solar'])]
        if gens.empty:
            continue
        pmax = n.generators_t.p_max_pu.reindex(columns=gens.index).fillna(1.0)
        available = (pmax.multiply(gens.p_nom, axis=1) * 0.5).sum().sum() / 1000   # GWh
        realised = (n.generators_t.p[gens.index] * 0.5).sum().sum() / 1000          # GWh
        rows.append({'layout': layout, 'available': available, 'realised': realised, 'curtailed': available - realised})
    if not rows:
        ax.set_title('Renewable curtailment')
        return
    df = pd.DataFrame(rows).set_index('layout')
    df[['realised', 'curtailed']].plot.bar(stacked=True, ax=ax,
                                            color=['#7ac677', 'lightgray'], width=0.6)
    ax.set_title('Wind + solar — realised vs curtailed (wholesale)')
    ax.set_ylabel('GWh')
    ax.tick_params(axis='x', rotation=0)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(alpha=0.3, axis='y')


# ---------- main ----------

def build_dashboard(day, out_path):
    print(f'Loading networks for {day}…')
    nets = load_networks(day)
    if not nets:
        sys.exit(f'No solved networks found under results/{day}/ — has the pipeline run?')
    print(f'  loaded: {sorted(nets)}')

    fig, axes = plt.subplots(4, 3, figsize=(22, 20))
    fig.suptitle(f'GBPower dashboard — {day}', fontsize=18, y=0.995)

    plot_price_comparison(axes[0, 0], nets)
    plot_dispatch_stack(axes[0, 1], nets.get(('national', 'wholesale')), 'Dispatch — national wholesale')
    plot_dispatch_stack(axes[0, 2], nets.get(('nodal', 'wholesale')), 'Dispatch — nodal wholesale')
    plot_balancing_volume(axes[1, 0], nets, layout='national')
    plot_interconnectors(axes[1, 1], nets)
    plot_storage(axes[1, 2], nets)
    plot_system_costs(axes[2, 0], day)
    plot_energy_balance(axes[2, 1], nets)
    plot_curtailment(axes[2, 2], nets)
    plot_producer_surplus(axes[3, 0], day)
    plot_seb_delta(axes[3, 1], day)
    axes[3, 2].axis('off')

    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_path, bbox_inches='tight')
    print(f'Saved {out_path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('day', help='YYYY-MM-DD')
    parser.add_argument('--out', default=None, help='Output PDF path (default: plotting/dashboards/dashboard_{day}.pdf)')
    args = parser.parse_args()
    if args.out:
        out = Path(args.out)
    else:
        out_dir = REPO_ROOT / 'plotting' / 'dashboards'
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f'dashboard_{args.day}.pdf'
    build_dashboard(args.day, out)


if __name__ == '__main__':
    main()
