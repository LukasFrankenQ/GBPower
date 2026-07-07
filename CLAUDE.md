# GBPower — Repository Guide for Claude

## What this repo is

**GBPower** is a [PyPSA](https://github.com/pypsa/pypsa)-based, unit-level electricity market model of Great Britain. It was built to analyse what would have happened to wholesale and balancing markets in 2022–2024 if GB had implemented a zonal market design instead of the national one. The accompanying paper is *Risk and Reward of Transitioning from a National to a Zonal Electricity Market in Great Britain* (Franken, Lyden, Friedrich — arXiv:2506.04107).

**Paper source location.** The LaTeX working copy of this paper lives *outside* the repo at `../lmp_paper/` (relative to the repo root): root `main.tex`, body sections in `sections/`, and all figure images in `imgs/`. Figures are numbered by order of appearance — Fig 3 `total_monthly_costs_w_comparison.pdf`, Fig 4 `wind_cases_*.pdf`, Fig 5 `annual_unit_revenues.pdf`, Fig 7 `surplus_changes_30_after_policy.pdf`, Fig 8 `waterfall_chart.pdf`, Fig 9 `socioeconomic_benefits.pdf`. Four-year (2022–2025) rebuilds of these six figures live in `new_plots/` (see that dir's scripts; original 3-year builders are in `notebooks/` and their auto-converted twins in `plotting/`).

The model is **data-driven**: for each historical day it pulls physical notifications, MELs, day-ahead prices, balancing actions, interconnector data and constraint volumes from ELEXON, ENTSO-E and NESO APIs, calibrates the network, then solves wholesale and balancing markets under three layouts (national, zonal, nodal).

Structurally inspired by **PyPSA-Eur**: a Snakemake workflow over a `rules/` + `scripts/` layout, NetCDF networks flowing between rules, post-processing into CSVs.

## How to run

**Preferred — pixi** (declarative, lockfile-backed):

```bash
pixi shell                                      # drops you into the project env
# put ENTSOE_API_KEY = '...' into scripts/_tokens.py (gitignored)
touch scripts/_tokens.py

# example day end-to-end (substantial curtailment) — also exposed as a pixi task:
pixi run example
# equivalent to:
snakemake -call --configfile config.yaml -- results/2024-03-21/system_cost_summary_flex.csv
```

The pixi spec is `pixi.toml` (channels: conda-forge, bioconda; platforms: linux-64). It mirrors `envs/environment.yaml` and additionally pulls in `entsoe-py` from PyPI, which `scripts/build_base.py` imports but which is missing from the conda env file.

**Alternative — conda/mamba** (original path, still supported):

```bash
mamba env create -f envs/environment.yaml
conda activate gbpower
pip install entsoe-py                            # add the missing dep
touch scripts/_tokens.py
snakemake -call --configfile config.yaml -- results/2024-03-21/system_cost_summary_flex.csv
```

Default solver is HiGHS (`highspy`, pip-installed). GLPK and SCIP are also available via the env.

## Top-level layout

```
Snakefile               # entry point; declares wildcard constraints and includes rule files
config.yaml             # NOTE: header says "HAS NO EFFECT ON THE MODEL; WORK IN PROGRESS"
                        # Only `tech_colors`, `nice_names`, `countries_cost_slopes` are read.
rules/
  retrieve.smk          # build_base: pulls a day's raw data from ELEXON/ENTSO-E
  prerun_rules.smk      # expensive build rules whose outputs are shipped pre-built
                        # (rule outputs are commented out by design — see file header)
  run.smk               # core pipeline: add_electricity -> simplify -> cluster ->
                        # calibrate_line_capacities -> (prepare_future_network) -> solve_network
  postprocess.smk       # summarize_bmu_revenues,
                        # summarize_system_cost, summarize_frontend_data
scripts/                # all rule scripts + helpers (see below)
prerun_scripts/         # one-off builders for shipped data in data/prerun/
data/                   # shipped inputs (shapes, fleets, CfD registers, prerun outputs)
  raw/                  # base network (lmp_base.nc), BMU locations, ESO zones
  prerun/               # outputs of prerun_rules.smk (shipped so users don't rerun them)
  base/{day}/           # build_base outputs per day (gitignored)
  cfd_registers/        # historical CfD register snapshots, used by build_cfds
notebooks/              # exploratory + paper-figure notebooks (gitignored as *.ipynb)
docs/                   # paper images + gather_all.py (collects per-day results into summaries/)
envs/environment.yaml   # conda env spec
```

`results/` (per-day networks + summaries), `frontend/` (summarized CSVs for downstream apps), `summaries/`, and `logs/` are all generated and gitignored.

## Wildcards (see `Snakefile`)

- `day` — `YYYY-MM-DD`
- `layout` — `national | zonal | nodal` (also `fti`/`eso` appear in some rules)
- `ic` — `flex` only (README marks the redundant `static` value as a candidate for removal — 🐮 task)
- `future` — `off | 2025..2029 | 2030`

## Pipeline (per day)

1. **`build_base`** (`scripts/build_base.py`) — fetches from ELEXON (PN, MEL, bids, offers, NEMO power flow, boundary flow constraints) and ENTSO-E (day-ahead prices for GB and Europe, generation). Writes `data/base/{day}/*.csv`.
2. **`add_electricity`** (`scripts/add_electricity.py`, ~930 lines) — populates `data/raw/lmp_base.nc` with generators, storage, loads, interconnectors. Uses prerun assets (ROC values, prepared BMUs, load weights, CfD strike prices, nuclear marginal cost, battery/PHS capacities, merit-order slope factors, weekly thermal generator prices) and the day's base data. Includes `scale_merit_order(...)` which tunes fossil/biomass marginal cost so wholesale prices reproduce observed day-ahead. Output: `results/{day}/network_flex.nc`.
3. **`simplify_network`** (`scripts/simplify_network.py`) — PyPSA-Eur-style simplification → `network_flex_s.nc`.
4. **`cluster_network`** (`scripts/cluster_network.py`) — clusters into the chosen `layout` (national / zonal / nodal) → `network_flex_s_{layout}.nc`.
5. **`calibrate_line_capacities`** (`scripts/calibrate_line_capacities.py`) — tunes interzonal line capacities so model balancing volumes match the real GB system; writes a single `calibration_factor` YAML.
6. **`prepare_future_network`** (`scripts/prepare_future_network.py`) — only for `future != off`; injects 2025–2030 fleet additions (CfD AR4/AR5/AR6, UK build pipeline, transmission additions) and forward Europe prices. For days >= 2025 it reuses 2024 inputs via the local `fix_year` helper in `rules/run.smk`.
7. **`solve_network`** (`scripts/solve_network.py`) — solves wholesale then redispatch for the national, zonal, and nodal layouts. Note: `add_backup_generators(...)` adds expensive extendable backups at every AC bus (provisional, see commit `e1c85c8`).
8. **Post-processing** (`rules/postprocess.smk`):
   - `summarize_bmu_revenues` — per-BMU revenue + dispatch broken down by layout, including CfD/ROC top-ups via `data/prerun/balancing_prices/{year}-week{week}.csv`.
   - `summarize_system_cost` — `system_cost_summary_{ic}.csv` (the README's smoke-test target).
   - `summarize_frontend_data` — flattens results for an external frontend.

## Key script files (worth knowing)

- `scripts/_helpers.py` — settlement-period ↔ datetime conversions (`to_date_period`, `to_datetime`, `to_daterange`), `configure_logging`, `path_provider` (imported by `Snakefile`), `set_nested_attr`. All times are handled in `Europe/London` then converted to UTC.
- `scripts/_constants.py` — DST start/end tables and `build_sp_register(day)` returning the 46/48/50 settlement periods per day depending on DST.
- `scripts/_elexon_helpers.py` — `robust_request` retry wrapper used by `build_base`.
- `scripts/_timeseries_helpers.py` — distribution-shaping helpers for prices (notably `transform_prices_expand_low_windows`).
- `scripts/_plotting_helpers.py`, `scripts/_debug_helpers.py` — utilities, not in the rule graph.
- `scripts/_tokens.py` — **gitignored**; must contain `ENTSOE_API_KEY = '...'`.
- `scripts/calibrate_line_capacities.py` — exports `get_line_grouping`, `insert_flow_constraints`, `tune_line_capacities`, `anchors`, `freeze_battery_commitments`, `freeze_interconnector_commitments`, `safe_solve` (all imported by `solve_network.py`).

## Prerun assets (shipped in `data/prerun/`)

These are outputs of `rules/prerun_rules.smk`. Each rule's `output:` is commented out so they don't re-run by default — to regenerate, uncomment and ensure ≥300 days of `data/base/` exist (most rules enforce this with a lambda check at parse time).

- `prepared_bmus.csv` — BMU master list with locations, carriers, capacities.
- `roc_values.csv` — Renewable Obligation Certificate values per unit.
- `cfd_strike_prices.csv` — strike prices from historical CfD register snapshots (`data/cfd_registers/`).
- `nuclear_marginal_cost.csv` — lowest observed day-ahead price across a year (used as nuclear MC floor).
- `battery_phs_capacities.csv` — estimated battery/PHS power and energy capacities from PN history.
- `meritorder_slope_factors.csv` — supply-curve slope factors per fuel.
- `thermal_costs/{year}-week{week}.csv` — weekly thermal generator MC (rolling 30-day window).
- `balancing_prices/{year}-week{week}.csv` — default balancing prices.
- `flow_constraints_{year}.csv` — annual constraint limits (SSE-SP, SCOTEX, SSHARN, FLOWSTH, SEIMP).
- `load_weights.csv`, `zonal_layout.geojson`, `europe_avg_day_ahead_prices_2030.csv`, `helper_network.nc`.

## Notebooks

`notebooks/` is gitignored (`*.ipynb`) but holds substantial exploratory work and the paper-figure builders. Shared utilities: `notebooks/plotting_constants.py`, `notebooks/data_getter.py`. To rebuild figures, the README workflow is: run the pipeline across all of 2022–2024, `mkdir summaries`, run `docs/gather_all.py`, then notebooks become functional.

## Conventions and gotchas

- **`config.yaml` is mostly cosmetic.** Only `tech_colors`, `nice_names` and `countries_cost_slopes` are consumed. Scenario start/end at the top of the file are not read. Don't add config keys expecting them to be wired up.
- **Time zones.** Always work in `Europe/London`, convert to UTC at the boundary. Settlement periods per day vary (46/48/50) — use `build_sp_register(day)`.
- **HiGHS is the default solver** (`4ecfe4a`). The Snakefile passes the solver param robustly even when missing (`033adea`).
- **European prices are not adjusted** by interconnector flow — a known limitation flagged with a warning (`e439f57`). Listed as a 🦂 improvement in the README (EuroMod approach).
- **`add_backup_generators` in `solve_network.py`** adds high-cost extendable backup generators at every AC bus. It is "provisional" per the most recent commit (`e1c85c8`) — be aware before claiming infeasibilities are real.
- **`fix_year`** in `rules/run.smk` silently maps any `day >= 2026` to its 2025 equivalent for input lookup. Future-network runs depend on this; don't be surprised when 2025 base files are read for a 2030 day. (Pre-rnp-branch boundary was day >= 2025 → 2024.)
- **Modeling scope is transmission-only.** GBPower models the ~65-77% of GB generation that is connected at the transmission level (most >50 MW units; all offshore wind, nuclear, large gas, ICs). Distribution-connected ("embedded") generation — ~90% of solar, ~30% of onshore wind, most pre-2024 battery storage — is **netted into the demand profile** rather than represented as units. This mirrors how NESO itself dispatches the system (NESO has no real-time visibility of embedded resources) and is the canonical methodology in GB locational-pricing studies (FTI/Ofgem 2023, Frontier Economics 2023). The implication for validation: total-system projections (e.g. NESO Clean Power 2030's 47 TWh solar) should be multiplied by the carrier's transmission share (~23% for solar in 2030) before comparing to model outputs. `plotting/validation.py` applies this adjustment automatically via `T_SHARES`.
- **`postprocess.smk` is messy by design** — README marks cleanup as a 🐮 task. Don't be alarmed by overlapping inputs/outputs.
- **`ic` wildcard is effectively constant `flex`** — `static` was removed from the constraint but the wildcard remains for compatibility (🐮 task to drop).
- **Branch `rnp`** (current) is the working branch that extends `main` with future-year modelling, the extracted `calibrate_line_capacities` rule, HiGHS-as-default, and `safe_solve` infeasibility relaxation. The original RNP (Reformed National Pricing) scenarios that gave the branch its name have been removed — see the dedicated section below for what still diverges from `main`.

## Commit style

Short, imperative, lower-case subject. Recent examples:

```
added provisional backup generation
added warning that european prices are not adjusted
added timeseries helpers
set highs as default solver
ensured snakemake does not expect solver param
```

## Working with PyPSA networks in GBPower

GBPower is **single-carrier (electricity)** and **single-day** — none of the sector-coupled bus/carrier hierarchy from PyPSA-Eur applies here. Every settlement period is one snapshot and there are 46/48/50 snapshots per day. Both pre- and post-solve, work with the model via `pypsa.Network` and its `_t` time-series attributes.

### Loading networks

Networks are NetCDF files under `results/{day}/`. Conventions:

| File pattern | Stage | Solved? |
|---|---|---|
| `data/raw/lmp_base.nc` | Empty base network (buses, lines, line types) | n/a |
| `results/{day}/network_{ic}.nc` | After `add_electricity` | no |
| `results/{day}/network_{ic}_s.nc` | After `simplify_network` | no |
| `results/{day}/network_{ic}_s_{layout}.nc` | After `cluster_network` | no |
| `results/{day}/network_{ic}_s_{layout}_fut.nc` | After `prepare_future_network` (future years only) | no |
| `results/{day}/network_{ic}_s_{layout}_solved.nc` | Wholesale solution | yes |
| `results/{day}/network_{ic}_s_{layout}_solved_redispatch.nc` | Redispatch (balancing) solution | yes |

`ic` is effectively always `flex`. `layout` is one of `national`, `zonal`, `nodal`.

```python
import pypsa
from pathlib import Path

RESULTS = Path("results")

def network_path(day, layout, solved=True, redispatch=False, ic="flex"):
    suffix = "_solved_redispatch" if redispatch else ("_solved" if solved else "")
    return RESULTS / day / f"network_{ic}_s_{layout}{suffix}.nc"

n = pypsa.Network(network_path("2024-03-21", "nodal"))
```

Always build paths with a helper — never hardcode full filenames. Check `Path.exists()` before loading.

### Components used (this model)

- **`n.buses`** — AC buses only. The `country` column tags GB vs. interconnected neighbours (`FR`, `BE`, `NL`, `NO`, `DK`, `IE`, `DE`). Locations:
  ```python
  gb_buses = n.buses.index[n.buses.country == "GB"]
  foreign_buses = n.buses.index[n.buses.country != "GB"]
  ```
- **`n.generators`** — every BMU is a generator. Carriers in use: `nuclear`, `biomass`, `fossil`, `onwind`, `offwind`, `solar`, plus `local_market` proxy generators on foreign buses that absorb interconnector flow at the European day-ahead price.
- **`n.storage_units`** — carriers `PHS`, `battery` (two-way), `hydro`, `cascade` (one-way). Charge is negative `p`, discharge positive.
- **`n.links`** — interconnectors (`carrier == "interconnector"`, connecting GB bus to a foreign bus) and DC transmission links inside GB. AC transmission is on `n.lines`.
- **`n.loads`** — single electricity load per bus, weighted by `data/prerun/load_weights.csv`.
- **`n.lines`** — AC transmission. Note `calibrate_line_capacities.py` asserts `n.lines.empty` for some flows because the present zonal/nodal modelling runs in a full DC approximation via `n.links`. AC `n.lines` only carries meaning before clustering/simplification.

Always filter by the `carrier` / `bus` / `country` columns, never by string-matching on the index.

### Snapshots and energy conversion — **critical**

Snapshots are **30-minute settlement periods**. A bare `.sum()` of `n.generators_t.p` (which is in MW) therefore gives MW-of-snapshots, not MWh. The codebase consistently multiplies by `0.5` to convert MW → MWh per period:

```python
energy_mwh = n.generators_t.p[unit] * 0.5     # MWh per snapshot
revenue   = n.generators_t.p[unit] * 0.5 * n.buses_t.marginal_price[bus]
```

`n.snapshot_weightings.generators` is usually `1.0` in this model, so PyPSA-Eur's "weight by snapshot_weightings" pattern reduces to multiplying by `0.5`. `n.statistics` methods are aware of this only if the weightings are set; double-check before trusting them for energy totals.

Snapshots are stored in UTC; convert to `Europe/London` for human-facing reporting via `scripts/_helpers.py:to_date_period`.

### Pre-solve inspection

```python
# Capacity by carrier (GB only)
n.generators[n.generators.bus.isin(gb_buses)].groupby("carrier").p_nom.sum()
n.storage_units.groupby("carrier").p_nom.sum()

# Renewable availability time series (post add_electricity — built from PN data)
n.generators_t.p_max_pu                       # per-unit, snapshots × generator
avail_mw = n.generators_t.p_max_pu.multiply(n.generators.p_nom, axis=1)

# Interconnector inventory
n.links[n.links.carrier == "interconnector"][["bus0", "bus1", "p_nom"]]

# Marginal costs (some are time-varying, others scalar)
n.generators.marginal_cost                    # static
n.generators_t.marginal_cost                  # time-varying overrides (thermal, scaled by wholesale)

# CfD/ROC generators (encoded as negative marginal_cost or 0 — see add_electricity.py:175-178)
roc_gens = n.generators.index[n.generators.marginal_cost < 0]
```

Note `local_market` generators on foreign buses have time-varying `marginal_cost` set from `europe_day_ahead_prices` — they are how interconnectors price imports/exports without a full European model.

### Post-solve outputs

```python
n = pypsa.Network(network_path("2024-03-21", "nodal", solved=True))

# Locational marginal prices — £/MWh, snapshots × buses
lmp = n.buses_t.marginal_price

# GB-average wholesale price (load-weighted across GB buses)
gb_loads_t = n.loads_t.p_set[n.loads.index[n.loads.bus.isin(gb_buses)]]
gb_loads_t.columns = n.loads.loc[gb_loads_t.columns, "bus"]
gb_lmp = n.buses_t.marginal_price[gb_buses]
gb_price = (gb_lmp * gb_loads_t).sum(axis=1) / gb_loads_t.sum(axis=1)

# Generator dispatch (MW, snapshots × generator)
n.generators_t.p

# Interconnector flow (positive = import into GB if bus1 is GB)
ics = n.links.index[n.links.carrier == "interconnector"]
imports_mwh = n.links_t.p0[ics].clip(lower=0).sum() * 0.5
exports_mwh = n.links_t.p1[ics].clip(lower=0).sum() * 0.5

# Storage operation
n.storage_units_t.p              # net (positive = discharging)
n.storage_units_t.p_dispatch     # discharging only
n.storage_units_t.p_store        # charging only
n.storage_units_t.state_of_charge

# Objective (wholesale market clearing cost for this day, £)
n.objective
```

### Computing unit revenues

The canonical pattern lives in `scripts/summarize_bmu_revenues.py`. Wholesale revenue for a generator or storage unit:

```python
def unit_wholesale_revenue(n, comp, unit):
    """comp is 'generators' or 'storage_units'; unit is a name or pd.Index."""
    bus = getattr(n, comp).bus[unit]
    p   = getattr(n, comp + "_t").p[unit]
    mp  = n.buses_t.marginal_price[bus]
    if isinstance(unit, str):
        return p * 0.5 * mp
    mp.columns = unit
    return p * 0.5 * mp                           # £ per snapshot
```

CfD top-up follows `get_cfd_revenue` in the same file: `payment = p × (strike_price − bus_lmp) × 0.5`, gated by a 6-hour negative-price exclusion. ROC revenue is `dispatch_mwh × roc_value`.

### Comparing layouts

The paper's analysis hinges on comparing the same day under different layouts. The standard pattern is to load the matched wholesale + redispatch pair per layout:

```python
def load_layout(day, layout):
    return {
        "wholesale":  pypsa.Network(network_path(day, layout, solved=True, redispatch=False)),
        "redispatch": pypsa.Network(network_path(day, layout, solved=True, redispatch=True)),
    }

national = load_layout("2024-03-21", "national")
zonal    = load_layout("2024-03-21", "zonal")
nodal    = load_layout("2024-03-21", "nodal")
```

Balancing volume is `redispatch.generators_t.p − wholesale.generators_t.p` (see `summarize_bmu_revenues.py:229`).

### `n.statistics` caveats in GBPower

- `n.statistics.market_value()` and similar revenue methods can be misleading for storage (round-trip losses → negative net dispatch). Use the manual `p × 0.5 × marginal_price` pattern instead.
- `bus_carrier="AC"` is essentially the only meaningful filter — there is no sector coupling.

### Anti-patterns specific to this repo

1. **Don't sum dispatch in MW.** Always multiply by `0.5` (30-min snapshots). Forgetting this overstates energy 2× and revenue 2×.
2. **Don't mix `solved` and `solved_redispatch` blindly.** Wholesale clearing vs. redispatch produce different dispatch and LMPs; the comparison *is* the analysis.
3. **Don't assume `n.lines` is populated post-clustering.** Inside clustered zonal/nodal networks, transmission lives on `n.links` (DC approximation).
4. **Don't treat `local_market` generators as real GB units** when reporting GB metrics — filter by `n.buses.country == "GB"` first.
5. **Don't read `bus0` direction for interconnectors as canonical.** Check both `bus0` and `bus1` against `country == "GB"` to decide whether `p0` represents an import or an export.
6. **Don't expect `config.yaml` to drive layout/scenario.** The CLI target path (`results/{day}/...`) is what selects layout, day and post-processing target via Snakemake wildcards.

## What the `rnp` branch adds over `main`

`main` is the baseline used in the published paper: national/zonal/nodal wholesale + redispatch, calibrated against historical days in 2022–2024. `rnp` builds on top of it with the additions below. (The original Reformed National Pricing scenarios — `rnp1/2/3` networks and the `gather_rnp_metrics` rule — that gave the branch its name have been removed; see git history if reviving.)

### 1. Future-year (2025–2030) modelling

`main` only models historical days. `rnp` adds the `future` wildcard (`off | 2025..2029 | 2030`) and a new `prepare_future_network` rule that:

- Adds CfD AR4/AR5/AR6 winners (`data/ar{4,5,6}_results.csv`).
- Adds the UK build pipeline 2025–2030 with coordinates (`data/UK_energy_build__2025_2030__with_coordinates.csv`).
- Adds onshore transmission upgrades (`data/GB_onshore_transmission_additions_2025_2030.csv`).
- Scales GB demand to projected levels (`data/gb_electricity_loads.yaml`).
- Swaps in 2030 European spot prices (`data/prerun/europe_avg_day_ahead_prices_2030.csv`, built by `prerun_scripts/build_2030_marginal_price_data.py`).
- Adds Germany to the modelled European countries.

The new `fix_year` helper in `rules/run.smk` silently maps any `day >= 2025` to its 2024 equivalent for *historical* inputs (bids, base networks, calibration) while the future rule overlays the fleet/transmission/demand on top. Future networks are `network_flex_s_{layout}_fut.nc`.

### 2. Solver / robustness changes

- **HiGHS is the default** (`scripts/solve_network.py` uses `solver_name='highs'`); on `main` the solver is configurable via `config['solver']`. The Snakefile on `rnp` no longer expects a `solver` config key (commit `69ca374`).
- **Calibration is its own rule.** `calibrate_line_capacities` was extracted from `solve_network` into its own Snakemake rule producing a single `results/{day}/calibration_factor_{ic}.yaml`. `solve_network` then reads this YAML rather than recomputing.
- **`safe_solve`** wraps each optimisation with a relaxation loop on the calibration factor when infeasible; it's applied to every layout (national redispatch, nodal, zonal redispatch).
- **Backup generators.** `add_backup_generators(n)` adds expensive (`capital_cost=10000`) extendable generators at every AC bus before solving nodal/zonal-redispatch networks (commit `e1c85c8`, marked "provisional"). Treat resulting "infeasibility avoided" outcomes with care — backup dispatch will look like very expensive generation rather than a hard failure.
- **`ensure_thermal_supply`** is imported from `add_electricity` and used to guarantee thermal coverage in degenerate cases.

### 3. Supporting additions

- **`scripts/_timeseries_helpers.py`** — distribution-shaping helpers for prices (notably `transform_prices_expand_low_windows`), added alongside the future-year work to massage 2030 price series.
- **European prices stay inelastic** — `rnp` adds an explicit warning that interconnector flow does not feed back into European day-ahead prices (commit `e439f57`). Listed in README as the 🦂 EuroMod-style improvement.
- **Marginal price scaling deactivated** in some paths (commit `fa2d7fc`) — be aware that `scale_merit_order(...)` may behave differently here than what the function docstring suggests.
- **Notebooks** specifically for future-year exploration: `notebooks/adding_future_projects.ipynb`. (The RNP-analysis notebooks `battery_rnp.ipynb` and `interconnector_rnp.ipynb` are now stale — the underlying `rnp_metrics/` outputs no longer exist.)

### Translating between branches

- A rule referencing `calibration_factor=...yaml` or any `_fut` network is `rnp`-only.
- The `future` wildcard does not exist on `main`.
- If a script on `rnp` imports from `calibrate_line_capacities` (`get_line_grouping`, `insert_flow_constraints`, `tune_line_capacities`, `anchors`, `freeze_*`, `safe_solve`), on `main` those helpers live inline inside `solve_network.py`.

## Contact / license

MIT (see `LICENSE`). Author: Lukas Franken (lukas.franken@ed.ac.uk).
