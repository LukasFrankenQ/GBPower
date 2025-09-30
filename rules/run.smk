# SPDX-FileCopyrightText: : 2024 Lukas Franken
#
# SPDX-License-Identifier: MIT

rule add_electricity:
    params:
        countries_cost_slopes=config['countries_cost_slopes'],
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
        thermal_generation_costs=lambda wildcards: 'data/prerun/thermal_costs/{year}-week{week}.csv'.format(
            year=datetime.strptime(wildcards.day, '%Y-%m-%d').year,
            week=str(datetime.strptime(wildcards.day, '%Y-%m-%d').isocalendar()[1]).zfill(2)
        ),
        day_ahead_prices='data/base/{day}/day_ahead_prices.csv',
        maximum_export_limits='data/base/{day}/maximum_export_limits.csv',
        physical_notifications='data/base/{day}/physical_notifications.csv',
        europe_day_ahead_prices='data/base/{day}/europe_day_ahead_prices.csv',
        europe_generation='data/base/{day}/europe_generation.csv',
        nemo_powerflow="data/base/{day}/nemo_powerflow.csv",
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
    params:
        solver=config['solver'],
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


rule prepare_future_network:
    params:
        countries_cost_slopes=config['countries_cost_slopes'],
    input:
        # if day wildcard is >2025-01-01, 2024 data is used for the inputs
        # Use lambda function to check date and modify input day if needed
        calibration_factor=lambda wildcards: "results/{}/calibration_factor_{}.yaml".format(
            wildcards.day if datetime.strptime(wildcards.day, "%Y-%m-%d").date() < datetime.strptime("2025-01-01", "%Y-%m-%d").date()
            else wildcards.day.replace(wildcards.day[:4], "2024"),
            wildcards.ic),
        network_nodal=lambda wildcards: "results/{}/network_{}_s_nodal.nc".format(
            wildcards.day if datetime.strptime(wildcards.day, "%Y-%m-%d").date() < datetime.strptime("2025-01-01", "%Y-%m-%d").date()
            else wildcards.day.replace(wildcards.day[:4], "2024"),
            wildcards.ic),
        network_national=lambda wildcards: "results/{}/network_{}_s_national.nc".format(
            wildcards.day if datetime.strptime(wildcards.day, "%Y-%m-%d").date() < datetime.strptime("2025-01-01", "%Y-%m-%d").date()
            else wildcards.day.replace(wildcards.day[:4], "2024"),
            wildcards.ic),
        network_zonal=lambda wildcards: "results/{}/network_{}_s_zonal.nc".format(
            wildcards.day if datetime.strptime(wildcards.day, "%Y-%m-%d").date() < datetime.strptime("2025-01-01", "%Y-%m-%d").date()
            else wildcards.day.replace(wildcards.day[:4], "2024"),
            wildcards.ic),
        base_network=lambda wildcards: "results/{}/network_{}_s.nc".format(
            wildcards.day if datetime.strptime(wildcards.day, "%Y-%m-%d").date() < datetime.strptime("2025-01-01", "%Y-%m-%d").date()
            else wildcards.day.replace(wildcards.day[:4], "2024"),
            wildcards.ic),
        europe_day_ahead_prices=lambda wildcards: "data/base/{}/europe_day_ahead_prices.csv".format(
            wildcards.day if datetime.strptime(wildcards.day, "%Y-%m-%d").date() < datetime.strptime("2025-01-01", "%Y-%m-%d").date()
            else wildcards.day.replace(wildcards.day[:4], "2024"),
            wildcards.ic),
        fleet_2024="data/gb_2024_capacities.csv",
        ar4_results="data/ar4_results.csv",
        ar5_results="data/ar5_results.csv",
        ar6_results="data/ar6_results.csv",
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
    params:
        solver=config['solver'],
    input:
        bmus="data/prerun/prepared_bmus.csv",
        bids="data/base/{day}/bids.csv",
        network_nodal="results/{day}/network_{ic}_s_nodal.nc",
        network_national="results/{day}/network_{ic}_s_national.nc",
        network_zonal="results/{day}/network_{ic}_s_zonal.nc",
        transmission_boundaries='data/transmission_boundaries.yaml',
        boundary_flow_constraints="data/base/{day}/boundary_flow_constraints.csv",
        calibration_factor=lambda wildcards: "results/{}/calibration_factor_{}.yaml".format(
            wildcards.day if datetime.strptime(wildcards.day, "%Y-%m-%d").date() < datetime.strptime("2025-01-01", "%Y-%m-%d").date()
            else wildcards.day.replace(wildcards.day[:4], "2024"),
            wildcards.ic),
    output:
        network_nodal="results/{day}/network_{ic}_s_f{future}_nodal_solved.nc",
        network_national="results/{day}/network_{ic}_s_f{future}_national_solved.nc",
        network_national_redispatch="results/{day}/network_{ic}_s_f{future}_national_solved_redispatch.nc",
        network_zonal="results/{day}/network_{ic}_s_f{future}_zonal_solved.nc",
        network_zonal_redispatch="results/{day}/network_{ic}_s_f{future}_zonal_solved_redispatch.nc",
        network_rnp1="results/{day}/network_{ic}_s_f{future}_rnp1_solved.nc",
        network_rnp2="results/{day}/network_{ic}_s_f{future}_rnp2_solved.nc",
        network_rnp3="results/{day}/network_{ic}_s_f{future}_rnp3_solved.nc",
    resources:
        mem_mb=1500,
    log:
        "../logs/networks/{day}_{ic}_f{future}_solved.log",  
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/solve_network.py"

