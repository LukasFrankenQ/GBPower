# GBPower work plan

This is the working scratchpad for the historic/future boundary shift (2024→2025) and a workflow-trimming audit. Companion to the original detailed plan at `~/.claude/plans/okay-some-time-has-piped-octopus.md`.

Phases A–C of the boundary shift are complete. What's left is Phases E–H. Before doing E we want to look at the rule graph and trim outputs that aren't actually needed for the paper.

---

## 1. Boundary shift — what's done

| Phase | State | Notes |
|---|---|---|
| A. 2025 base-data pull | ✅ done | 362/365 days (99.1%); three days lost to ENTSO-E generation gap (2025-11-11/12/13). Two real script fixes shipped: `import time` in `scripts/build_base.py`; defensive try/except + `SEIMPPR* → SEIMP` alias in `prerun_scripts/build_flow_constraints.py`. |
| B. BMU master curation | ✅ done (partial) | 31 BMUs added across `data/raw/temp_located_bmus.csv`, `data/prerun/prepared_bmus.csv`, `data/bmus_prepared.csv`. Silent drop on 2025 generation: **5.1% → 2.0%** (~13 TWh/y → ~5 TWh/y). 4 above-threshold BMUs still unidentified (HAHAW-1, BRCHS-1, OCHLB-1, BROCW-1) — long-tail can wait. |
| C. CfD register update | ✅ done | 5 quarterly 2025 register XLSX added; `build_cfds.py` patched for the 2025-04 LCCC schema rename (sheet `Register`→`Sheet 1`, snake_case columns) and for missing-ID tolerance. 12 new mapping rows added (NNGAO×2, MOWWO×4, LIMKW, NOKYW×2, CLVHS×2, HAGHW). `cfd_strike_prices.csv` regenerated with 20 snapshot columns × 46 BMUs. |
| D. Code boundary shift | ✅ done in initial session | `fix_year`, prepare_future_network assertions, overlay filters, Snakefile wildcard regex, meritorder baseline year, build_base flow-constraint cutoff. |

---

## 2. Remaining boundary-shift phases

### Phase E — regenerate global prerun assets

Each of these reads from `data/base/` and was last built with only ≤2024 days. With 362 days of 2025 now on disk, regeneration is appropriate. Outputs are commented in `rules/prerun_rules.smk` by convention — uncomment, run, re-comment.

| Rule | Output | Notes |
|---|---|---|
| `build_roc_values` | `data/prerun/roc_values.csv` | Re-check the `end_date='2024-03-01'` boundary in `build_roc_values.py:166` — relates to cascading-hydro behaviour change. May need to update. |
| `build_nuclear_bidding_cost` | `data/prerun/nuclear_marginal_cost.csv` | Lowest-observed day-ahead floor. |
| `build_battery_phs_capacities` | `data/prerun/battery_phs_capacities.csv` | 2025 PN gives much better battery fleet sizing — biggest expected accuracy gain in this phase. |
| `build_meritorder_slope_factors` | `data/prerun/meritorder_slope_factors.csv` | Phase D already shifted baseline year to 2025. |
| `build_thermal_generator_prices` | `data/prerun/thermal_costs/2025-week{18..52}.csv` | 35 missing weeks; weeks 1–17 already on disk. |
| `build_balancing_prices` | `data/prerun/balancing_prices/2025-week{18..52}.csv` | 35 missing weeks; same. |

Optionally also re-run `prepare_bmus` to fold the new Phase B additions into a clean `prepared_bmus.csv` (currently they were appended directly to the output file).

### Phase F — extend future horizon to 2031–2035

Currently the `future` wildcard is `off|202[6-9]|2030`. The plan calls for `2026..2035`. Blocked on data sourcing:

- `data/UK_energy_build__2025_2030__with_coordinates.csv` → extend coverage to 2035 (NESO TEC register, ESO Future Energy Scenarios, NIC pipeline). Rename to `_2026_2035_` or keep filename and just append rows.
- `data/GB_onshore_transmission_additions_2025_2030.csv` → extend to 2035 (Beyond 2030 ESO plans, ASTI, HND).
- `data/prerun/europe_avg_day_ahead_prices_2030.csv` → either keep 2030 prices for 2031–2035 (document as a limitation) or source a TYNDP 2035 scenario.
- `data/gb_electricity_loads.yaml` → extend `{year}_neso` series to 2031–2035 (NESO FES).

Once available, widen Snakefile regex to `off|202[6-9]|203[0-5]` and update relevant assertions.

### Phase G — verification

Smoke-test on representative days after Phase E:

1. Historic 2025: `snakemake … results/2025-03-20/system_cost_summary_flex.csv` — confirm 48 snapshots, year=2025 anchor, GB wholesale price tracks ENTSO-E for that day.
2. Future 2026: `results/2026-03-20/…` — confirm `prepare_future_network` short-circuit goes the right way and AR4 winners absent from overlay (since they're now in the 2025 PN baseline).
3. Wildcard check: `snakemake -n results/2035-06-01/…` (once Phase F unblocks) should plan a DAG.

### Phase H — documentation

Update `CLAUDE.md`:
- Intro line: "2022-2024" → "2022-2025"
- `fix_year` callout
- `rnp` branch section's "Future-year (2025-2030) modelling" subhead

Note carried-over limitations (2031–2035 EU prices reusing 2030 values; partial CfD register coverage if Phase C is later refreshed).

---

## 3. Workflow audit — what the paper actually needs

The paper at `~/Desktop/lmp_paper/main.tex` includes **33 figures**. Most are produced by notebooks under `notebooks/`. Mapping notebook → data source → rule below.

### Figures with confident provenance (22 of 33)

| Figure | Notebook | Reads from |
|---|---|---|
| `annual_unit_revenues.pdf`, `annual_unit_revenues_15.pdf` | `unit_revenues.ipynb` | `summaries/total_unit_revenues_flex.csv` |
| `surplus_changes_30_after_policy.pdf`, `thermal_unit_surplus.pdf` | `unit_revenues.ipynb` | same as above |
| `waterfall_chart.pdf`, `curtailment_seb_projections.pdf`, `socioeconomic_benefits.pdf`, `total_monthly_costs*.pdf` | `get_total_results.ipynb` | `summaries/total_summary_flex.csv` + per-day `system_cost_summary_flex.csv` + `network_flex_s_{layout}_solved.nc` |
| `prefix_{2022,2023,2024}_intercon_flow.pdf` | `compare_intercon_flow.ipynb` | `results/{day}/*.nc`, `data/base/{day}/europe_day_ahead_prices.csv` |
| `money_flow_total.pdf` | `money_flow.ipynb` | `summaries/total_summary_flex.csv`, `summaries/marginal_prices_summary_flex.csv` |
| `revenue_stabilisation.pdf` | `revenue_stabilisation.ipynb` | per-day `system_cost_summary_flex.csv`, `summaries/total_unit_revenues_flex.csv` |
| `compare_wholesale_balancing_three_years.pdf` | `compare_wholesale_balancing.ipynb` | per-day solved `.nc` networks |
| `north_south_split.pdf` | `split_northsouth.ipynb` | per-day solved `.nc` networks |
| `layouts.pdf` | `plot_regions.ipynb` | `data/regions_onshore.geojson`, `data/regions_offshore.geojson`, `data/prerun/zonal_layout.geojson` |
| `merit_order.pdf` | `build_supply_curve.ipynb` (parameterised: `merit_order_{day}_{period}.pdf`) | `data/prerun/prepared_bmus.csv` + per-day base data |
| `model_bmu_carrier_count.pdf`, `storage_capacities.pdf` | `new_data_analysis.ipynb` | `data/prerun/prepared_bmus.csv`, `data/prerun/battery_phs_capacities.csv` |
| `wind_bid_prices.pdf` | `estimate_roc_levels.ipynb` | per-day `data/base/{day}/bids.csv`, `data/prerun/roc_values.csv` |
| `wind_cases_from_2022-01-01_to_2024-12-31.pdf` | `energy_crisis_shift.ipynb` | `summaries/total_summary_flex.csv`, `summaries/marginal_prices_summary_flex.csv` |
| `dispatch_increase_gwh.pdf`, `dispatch_increase_perc.pdf` | `look_loads.ipynb` | `summaries/total_unit_dispatch_flex.csv` |
| `balancing_volume_validation.pdf` | `plot_balancing_volume_comparison.ipynb` | per-day solved+redispatch `.nc` networks |

### Figures whose source isn't obvious (11 of 33)

Either generated outside the notebooks/ directory, hand-drawn, or generated by a parameterised notebook that emits multiple variants:

- `boundaries.pdf`, `capacity.pdf`, `network_layout.pdf`, `network_load.pdf` — likely from one of the geospatial notebooks; need a closer look.
- `monthly_bidding_volume.pdf`, `thermal_unit_schedule_revenues.pdf`, `total_monthly_costs_w_comparison.pdf` — probably variants of figures `get_total_results.ipynb` produces.
- `compare_wholesale_balancing_2025-03.pdf` — parameterised variant of `compare_wholesale_balancing_three_years.pdf`; same notebook.
- `socioeconomic_benefits_with_correlations.pdf` — commented out in the paper.
- `smr_estimate.png` — looks like an external/illustrative graphic.

These don't change the trim picture — none rely on data we'd consider removing.

### Data-source dependency graph (what figures rely on)

```
PER-DAY (results/{day}/):
  system_cost_summary_flex.csv      -> get_total_results, revenue_stabilisation
  network_flex_s_{layout}_solved.nc  -> compare_wholesale_balancing, money_flow, split_northsouth, compare_intercon_flow
  network_flex_s_{layout}_solved_redispatch.nc -> plot_balancing_volume_comparison
  bmu_revenues_flex_{layout}.csv    -> gather_all.py only (aggregation input)

AGGREGATED (summaries/, built by docs/gather_all.py):
  total_unit_revenues_flex.csv       -> unit_revenues, revenue_stabilisation
  total_summary_flex.csv             -> get_total_results, money_flow, energy_crisis_shift
  marginal_prices_summary_flex.csv   -> money_flow, energy_crisis_shift
  total_unit_dispatch_flex.csv       -> look_loads

PRERUN / STATIC:
  prepared_bmus.csv, cfd_strike_prices.csv, roc_values.csv, battery_phs_capacities.csv
  -> read by all paper-figure notebooks

BASE DATA (data/base/{day}/):
  bids.csv -> estimate_roc_levels
  europe_day_ahead_prices.csv -> compare_intercon_flow
```

---

## 4. Trim candidates

This is paper-figure coverage. Items below are flagged as "unused by paper figures".

### Resolved

- **`gather_rnp_metrics` rule and `scripts/gather_rnp_metrics.py`** — removed along with the rest of the RNP workflow. The RNP scenarios (`rnp1/2/3` networks) and their per-day aggregated diagnostics under `rnp_metrics/` are gone; see git history if reviving.

### High-confidence trim (unused by paper figures)

| Item | Where | Justification |
|---|---|---|
| `frontend/{day}/marginal_costs_{ic}.csv`, `frontend/{day}/thermal_dispatch_{ic}.csv` | `summarize_frontend_data` in `postprocess.smk` | Built for an external web frontend; never read by `docs/gather_all.py` or any paper-figure notebook. |
| `results/{day}/gb_total_load_{ic}.csv` | `summarize_bmu_revenues` in `postprocess.smk` | Not aggregated, not read by any paper notebook. |
| `summaries/total_intercon_dispatch_flex.csv` | `docs/gather_all.py` | Aggregated but no notebook consumes it. |

### Medium-confidence

| Item | Where | Justification |
|---|---|---|
| `summaries/total_summary_revenues_{ic}_{layout}.csv`, `total_summary_dispatch_{ic}_{layout}.csv` | `docs/gather_all.py` | Layout-specific summaries; paper notebooks only read the monolithic `total_summary_flex.csv`. |

### Don't trim (load-bearing for paper)

- `system_cost_summary_flex.csv` — direct paper-figure consumer.
- `bmu_revenues_flex_{layout}.csv` — aggregated into `total_unit_revenues_flex.csv` which feeds revenue figures.
- All per-day solved `.nc` networks — multiple notebooks open them directly.
- All `summaries/total_*_flex.csv` family.

---

## 5. Suggested next step

Phase F's data acquisition (build pipeline, transmission, EU prices for 2031–2035) is the longest user-driven blocker; sourcing those could happen in parallel with Phase E (regenerating global prerun assets with 2025 base data).
