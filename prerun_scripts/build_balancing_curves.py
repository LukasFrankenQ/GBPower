"""
prerun_scripts/build_balancing_curves.py

Build pooled 2022-2025 balancing OFFER and BID price-vs-cumulative-volume curves.

Motivation: get_balancing_cost() priced any model redispatch volume beyond a single
day's real offer stack at that day's most expensive offer (`actual['price'].iloc[-1]`).
For future years the model redispatches ~4x the historical daily offer volume, so ~80%
of the 2028-29 balancing cost was this single-price extrapolation rather than anything
grounded in real offers. Instead we pool every 2022-2025 day's offer/bid stack into one
canonical marginal-price curve P(cumvol): at within-day cumulative volume v, the median
marginal offer/bid price observed across all historical days. The model's turn-up/down
cost is then the area under this curve up to the model volume — rising with volume from
real data, with no single-day-max flatline.

Output: data/prerun/balancing_offer_curve.csv, data/prerun/balancing_bid_curve.csv
        columns: cumvol_MWh, marginal_price_GBP_per_MWh   (5 GWh bins)

Run:  pixi run python prerun_scripts/build_balancing_curves.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
BASE = REPO / 'data' / 'base'
OUT = REPO / 'data' / 'prerun'
BIN_MWH = 5000.0


def _process(df):
    """Daily ELEXON bid/offer table -> (price, vol) rows sorted ascending by price."""
    return (df.stack().unstack(1).dropna().reset_index(drop=True).sort_values('price'))


def _clean_prices(s):
    """Replace faulty prices (>|5000| or with >=3 nines) by the median of the clean ones."""
    faulty = s.apply(lambda n: str(int(abs(n))).count('9') >= 3) | (s.abs() > 5000)
    s = s.copy()
    if (~faulty).any():
        s[faulty] = s[~faulty].median()
    return s


def _curve_from_points(pts):
    """Pool a list of per-day (cumvol, price) frames into one median marginal-price curve."""
    allpts = pd.concat(pts, ignore_index=True)
    bins = np.arange(0, allpts['cumvol'].max() + BIN_MWH, BIN_MWH)
    allpts['bin'] = pd.cut(allpts['cumvol'], bins)
    curve = allpts.groupby('bin', observed=True)['price'].median()
    curve.index = [int(iv.right) for iv in curve.index]
    return curve.sort_index(), allpts['cumvol'].max(), len(pts)


def build_curves_by_year(kind):
    """kind is 'offers'|'bids'. Returns {year: median price-vs-cumvol Series}.

    Pooled PER YEAR so each price regime is preserved (e.g. the 2022 gas-crisis offer levels
    are not diluted by cheap 2023-25 offers). Future years (2026+) are priced on the 2025 curve
    downstream, consistent with fix_year sourcing 2025 base data for the future overlay.
    """
    by_year = {}
    for d in sorted(BASE.glob('2*-*-*')):
        f = d / f'{kind}.csv'
        if not f.exists():
            continue
        try:
            a = _process(pd.read_csv(f, index_col=[0, 1], parse_dates=True))
        except (KeyError, ValueError):
            continue
        if a.empty:
            continue
        a['price'] = _clean_prices(a['price'])
        a = a.sort_values('price')
        a['cumvol'] = a['vol'].cumsum()
        by_year.setdefault(d.name[:4], []).append(a[['cumvol', 'price']])
    out = {}
    for year, pts in sorted(by_year.items()):
        curve, vmax, n = _curve_from_points(pts)
        out[year] = curve
        print(f"  {kind} {year}: pooled {n} days, max within-day vol {vmax/1e3:.0f} GWh, "
              f"marginal range £{curve.min():.0f}-£{curve.max():.0f}/MWh")
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for kind, stem in [('offers', 'balancing_offer_curve'), ('bids', 'balancing_bid_curve')]:
        for year, curve in build_curves_by_year(kind).items():
            df = pd.DataFrame({'cumvol_MWh': curve.index,
                               'marginal_price_GBP_per_MWh': curve.values})
            df.to_csv(OUT / f'{stem}_{year}.csv', index=False)
        print(f"  saved {stem}_{{2022..2025}}.csv")


if __name__ == '__main__':
    main()
