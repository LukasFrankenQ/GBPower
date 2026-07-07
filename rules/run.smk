# SPDX-FileCopyrightText: : 2024 Lukas Franken
#
# SPDX-License-Identifier: MIT

rule add_electricity:
    params:
        countries_cost_slopes=config['countries_cost_slopes'],
        ic_slope_factor=config.get('ic_slope_factor', 1.0),
    input:
        network='data/raw/lmp_base.nc',
        roc_values='data/prerun/roc_values.csv',
        bmus=ancient('data/prerun/prepared_bmus.csv'),
        load_weights=ancient('data/prerun/load_weights.csv'),
        cfd_strike_prices=ancient('data/prerun/cfd_strike_prices.csv'),
        nuclear_marginal_price='data/prerun/nuclear_marginal_cost.csv',
        battery_phs_capacities='data/prerun/battery_phs_capacities.csv',
        meritorder_slope_factors="data/prerun/meritorder_slope_factors.csv",
        interconnection_helpers='data/interconnection_helpers.yaml',
        # Future-year days (>= 2026) reuse the 2025-anchor base data via fix_year().
        thermal_generation_costs=lambda wildcards: nearest_weekly("thermal_costs", wildcards.day),
        day_ahead_prices=lambda wc: f"data/base/{fix_year(wc.day)}/day_ahead_prices.csv",
        maximum_export_limits=lambda wc: f"data/base/{fix_year(wc.day)}/maximum_export_limits.csv",
        physical_notifications=lambda wc: f"data/base/{fix_year(wc.day)}/physical_notifications.csv",
        europe_day_ahead_prices=lambda wc: f"data/base/{fix_year(wc.day)}/europe_day_ahead_prices.csv",
        europe_generation=lambda wc: f"data/base/{fix_year(wc.day)}/europe_generation.csv",
        nemo_powerflow=lambda wc: f"data/base/{fix_year(wc.day)}/nemo_powerflow.csv",
    output:
        network="results/{day}/network_{ic}.nc"
    resources:
        mem_mb=4000,
    log:
        "../logs/networks/{day}_{ic}.log",
    conda:
        "../envs/environment.yaml",
    script:
        "../scripts/add_electricity.py"


rule simplify_network:
    input:
        network="results/{day}/network_{ic}.nc",
        regions_onshore="data/regions_onshore.geojson",
        regions_offshore="data/regions_offshore.geojson",
        tech_costs="data/costs_2020.csv",
        interconnection_helpers='data/interconnection_helpers.yaml',
    output:
        network="results/{day}/network_{ic}_s.nc",
        # busmap="results/prenetworks/{day}/busmap_s.csv",
        # connection_costs=RESOURCES + "live_data/{date}_{period}/connection_costs_s.csv",
    resources:
        mem_mb=1500,
    log:
        "../logs/networks/{day}_{ic}_s.log",
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/simplify_network.py"


rule cluster_network:
    input:
        network="results/{day}/network_{ic}_s.nc",
        tech_costs="data/costs_2020.csv",
        target_regions=lambda wildcards: f"data/{wildcards.layout}_zones.geojson" if wildcards.layout in ["national", "fti", "eso"] else [],
        zonal_layout="data/prerun/zonal_layout.geojson",
        regions_onshore="data/regions_onshore_s.geojson",
        regions_offshore="data/regions_offshore_s.geojson",
        interconnection_helpers='data/interconnection_helpers.yaml',
        transmission_boundaries='data/transmission_boundaries.yaml',
    output:
        network="results/{day}/network_{ic}_s_{layout}.nc",
    resources:
        mem_mb=1500,
    log:
        "../logs/networks/{day}_{ic}_s_{layout}.log",  
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/cluster_network.py"


rule calibrate_line_capacities:
    input:
        bmus="data/prerun/prepared_bmus.csv",
        bids="data/base/{day}/bids.csv",
        network_nodal="results/{day}/network_{ic}_s_nodal.nc",
        network_national="results/{day}/network_{ic}_s_national.nc",
        network_zonal="results/{day}/network_{ic}_s_zonal.nc",
        transmission_boundaries='data/transmission_boundaries.yaml',
        boundary_flow_constraints="data/base/{day}/boundary_flow_constraints.csv",
    output:
        calibration_factor="results/{day}/calibration_factor_{ic}.yaml",
    resources:
        mem_mb=1500,
    log:
        "../logs/networks/{day}_{ic}_calibrated.log",  
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/calibrate_line_capacities.py"


def fix_year(day):
    y, m, d = map(int, day.split("-"))
    if y >= 2026:
        y = 2025
    return f"{y:04d}-{m:02d}-{d:02d}"


def nearest_weekly(subdir, day, ext="csv"):
    """Resolve data/prerun/<subdir>/{year}-week{week}.<ext> for `day`, falling back
    to the most recent earlier week that exists when the exact file is missing
    (gaps in shipped weekly assets, e.g. 2025 weeks 47-50). Walks back up to ~10
    weeks, crossing into the previous year if needed."""
    import os
    d = datetime.strptime(fix_year(day), "%Y-%m-%d")
    y, w = d.year, d.isocalendar()[1]
    for _ in range(11):
        cand = f"data/prerun/{subdir}/{y}-week{str(w).zfill(2)}.{ext}"
        if os.path.exists(cand):
            return cand
        w -= 1
        if w < 1:
            y -= 1
            w = 52
    return f"data/prerun/{subdir}/{d.year}-week{str(d.isocalendar()[1]).zfill(2)}.{ext}"


rule prepare_future_network:
    params:
        countries_cost_slopes=config['countries_cost_slopes'],
        ic_slope_factor=config.get('ic_slope_factor', 1.0),
    input:
        # if day wildcard is >=2026-01-01, 2025 data is used for the inputs
        # Use lambda function to check date and modify input day if needed
        calibration_factor=lambda wc: f"results/{fix_year(wc.day)}/calibration_factor_{wc.ic}.yaml",
        network_nodal=lambda wc: f"results/{fix_year(wc.day)}/network_{wc.ic}_s_nodal.nc",
        network_national=lambda wc: f"results/{fix_year(wc.day)}/network_{wc.ic}_s_national.nc",
        network_zonal=lambda wc: f"results/{fix_year(wc.day)}/network_{wc.ic}_s_zonal.nc",
        base_network=lambda wc: f"results/{fix_year(wc.day)}/network_{wc.ic}_s.nc",
        europe_day_ahead_prices=lambda wc: f"data/base/{fix_year(wc.day)}/europe_day_ahead_prices.csv",
        # TODO: source data/gb_2025_capacities.csv (DUKES 2025 / DESNZ statistical release / NESO TEC register)
        # and switch this reference + the 'fleet_2024' key on the consumer side once available.
        fleet_2024="data/gb_2024_capacities.csv",
        ar4_results="data/ar4_results.csv",
        ar5_results="data/ar5_results.csv",
        ar6_results="data/ar6_results.csv",
        ar7_results="data/ar7_results.csv",
        marginal_prices_2030="data/prerun/europe_avg_day_ahead_prices_2030.csv",
        future_system_additions="data/UK_energy_build__2025_2030__with_coordinates.csv",
        future_transmission_additions="data/GB_onshore_transmission_additions_2025_2030.csv",
        cfd_strike_prices="data/prerun/cfd_strike_prices.csv",
        future_electricity_demand="data/gb_electricity_loads.yaml",
        meritorder_slope_factors="data/prerun/meritorder_slope_factors.csv",
        interconnection_helpers='data/interconnection_helpers.yaml',
    output:
        network_nodal="results/{day}/network_{ic}_s_nodal_fut.nc",
        network_national="results/{day}/network_{ic}_s_national_fut.nc",
        network_zonal="results/{day}/network_{ic}_s_zonal_fut.nc",
        cfd_strike_prices="results/{day}/cfd_strike_prices_{ic}_fut.csv",
    resources:
        mem_mb=1500,
    log:
        "../logs/networks/{day}_{ic}_fut.log",  
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/prepare_future_network.py"


rule solve_network:
    input:
        bmus="data/prerun/prepared_bmus.csv",
        bids=lambda wc: f"data/base/{fix_year(wc.day)}/bids.csv",
        network_base=lambda wc: f"results/{fix_year(wc.day)}/network_{wc.ic}_s_nodal.nc",
        network_nodal="results/{day}/network_{ic}_s_nodal_fut.nc",
        network_national="results/{day}/network_{ic}_s_national_fut.nc",
        network_zonal="results/{day}/network_{ic}_s_zonal_fut.nc",
        transmission_boundaries='data/transmission_boundaries.yaml',
        boundary_flow_constraints=lambda wc: f"data/base/{fix_year(wc.day)}/boundary_flow_constraints.csv",
        calibration_factor=lambda wc: f"results/{fix_year(wc.day)}/calibration_factor_{wc.ic}.yaml",
    output:
        network_nodal="results/{day}/network_{ic}_s_nodal_solved.nc",
        network_national="results/{day}/network_{ic}_s_national_solved.nc",
        network_national_redispatch="results/{day}/network_{ic}_s_national_solved_redispatch.nc",
        network_zonal="results/{day}/network_{ic}_s_zonal_solved.nc",
        network_zonal_redispatch="results/{day}/network_{ic}_s_zonal_solved_redispatch.nc",
    resources:
        mem_mb=1500,
    log:
        "../logs/networks/{day}_{ic}_solved.log",  
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/solve_network.py"

