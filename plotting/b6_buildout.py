"""
plotting/b6_buildout.py — why constraint-management costs peak 2027-29.

Plots, with year on the x-axis:
  * total installed wind+solar GENERATION capacity each year, split North vs South of
    the B6 (Anglo-Scottish) boundary — taken from the solved future networks by the bus
    each unit attaches to (i.e. how the model itself locates generation, which is what
    drives its power flows), and
  * north-south TRANSMISSION capacity across B6 (SCOTEX baseline + the Eastern Green Link
    HVDC bootstraps EGL1 in 2029 and EGL2 in 2030).

Northern wind+solar reaches ~14 GW against a B6 boundary of only 5-9 GW, so on windy
periods Scottish output cannot all flow south and is redispatched at high cost. The
SCOTEX/SSE-SP boundaries run at ~0.85 peak utilisation in 2028-29 and ease in 2030 once
the EGLs land — tracking the model's £17-20 bn -> £14 bn constraint-cost profile.

Usage:  pixi run python plotting/b6_buildout.py
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / 'results'
YEARS = list(range(2026, 2031))
B6_LAT = 55.0   # Anglo-Scottish boundary latitude

# B6 transmission: SCOTEX baseline (data/prerun/flow_constraints_2025.csv ≈ 5.1 GW) plus the
# Eastern Green Link HVDC bootstraps. data/transmission_boundaries.yaml maps EGL1->SCOTEX (2029)
# and EGL2->SSHARN (2030); both move Scottish surplus south, so they are cumulative B6 relief.
B6_BASELINE_GW = 5.1
B6_ADDITIONS = {2029: 2.0, 2030: 2.0}   # EGL1, EGL2 (GW each)


def _first_solved_day(year):
    days = sorted(p.name for p in RESULTS.glob(f'{year}-*-*')
                  if (p / 'network_flex_s_national_solved_redispatch.nc').exists())
    return days[0] if days else None


def gen_capacity_by_side():
    """Total wind+solar capacity (GW) North vs South of B6, per year, from the networks."""
    rows = []
    for year in YEARS:
        day = _first_solved_day(year)
        if day is None:
            continue
        n = pypsa.Network(RESULTS / day / 'network_flex_s_national_solved_redispatch.nc')
        gb = n.buses.index[n.buses.country == 'GB']
        north_buses = set(n.buses.loc[gb, 'y'].index[n.buses.loc[gb, 'y'] > B6_LAT])
        g = n.generators[n.generators.bus.isin(gb)
                         & n.generators.carrier.isin(['offwind', 'onwind', 'solar'])]
        north = g[g.bus.isin(north_buses)].p_nom.sum() / 1000
        south = g[~g.bus.isin(north_buses)].p_nom.sum() / 1000
        rows.append(dict(year=year, north_GW=north, south_GW=south))
    return pd.DataFrame(rows).set_index('year')


def b6_capacity():
    cum, out = 0.0, {}
    for y in YEARS:
        cum += B6_ADDITIONS.get(y, 0.0)
        out[y] = B6_BASELINE_GW + cum
    return pd.Series(out)


def main():
    cap = gen_capacity_by_side()
    b6 = b6_capacity()
    print("Wind+solar capacity each side of B6 (GW) and B6 N-S transmission capacity (GW):")
    print(cap.assign(B6_capacity=b6).round(2).to_string())

    fig, ax = plt.subplots(figsize=(11, 7))
    x = np.array(cap.index)
    w = 0.38
    ax.bar(x - w / 2, cap['north_GW'], w, color='#1f4e79',
           label='Wind+solar NORTH of B6 (Scotland)')
    ax.bar(x + w / 2, cap['south_GW'], w, color='#9dc3e6',
           label='Wind+solar SOUTH of B6 (England/Wales)')

    ax.plot(x, b6.values, color='#c00000', marker='o', lw=2.8, zorder=6,
            label='B6 north-south transmission capacity')
    ax.axhline(B6_BASELINE_GW, color='#c00000', ls=':', lw=1.2, alpha=0.7,
               label=f'B6 baseline (SCOTEX ≈ {B6_BASELINE_GW} GW)')
    for y, lbl in [(2029, 'EGL1\n+2 GW'), (2030, 'EGL2\n+2 GW')]:
        ax.annotate(lbl, (y, b6[y]), textcoords='offset points', xytext=(6, 6),
                    color='#c00000', fontsize=9, fontweight='bold')

    ax.axvspan(2026.5, 2029.5, color='orange', alpha=0.08, zorder=0)
    ax.text(2028, ax.get_ylim()[1] * 0.97,
            'constraint-cost peak\n(model: £17-20 bn/yr;\nSCOTEX util. ≈ 0.85)',
            ha='center', va='top', fontsize=9, color='#b35900')

    ax.set_xlabel('Year')
    ax.set_ylabel('GW')
    ax.set_title('B6 bottleneck: northern wind+solar build-out vs north-south transmission\n'
                 '(installed capacity by the bus each unit attaches to in the model)')
    ax.set_xticks(x)
    ax.grid(alpha=0.3, axis='y')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.95)
    ax.set_ylim(bottom=0)

    out = REPO / 'plotting' / 'dashboards' / 'b6_buildout_2026-2030.pdf'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight')
    print(f"Saved {out}")


if __name__ == '__main__':
    main()
