# SPDX-FileCopyrightText: : 2024 Lukas Franken
#
# SPDX-License-Identifier: MIT

rule add_electricity:
    input:
        network='data/raw/lmp_base.nc',
        roc_values='data/prerun/roc_values.csv',
        bmus=ancient('data/prerun/prepared_bmus.csv'),
        load_weights=ancient('data/prerun/load_weights.csv'),
        cfd_strike_prices=ancient('data/prerun/cfd_strike_prices.csv'),
        nuclear_marginal_price='data/prerun/nuclear_marginal_cost.csv',
        battery_phs_capacities='data/prerun/battery_phs_capacities.csv',
        interconnection_helpers='data/interconnection_helpers.yaml',
        thermal_generation_costs=lambda wildcards: 'data/prerun/thermal_costs/{year}-week{week}.csv'.format(
            year=datetime.strptime(wildcards.day, '%Y-%m-%d').year,
            week=str(datetime.strptime(wildcards.day, '%Y-%m-%d').isocalendar()[1]).zfill(2)
        ),
        day_ahead_prices='data/base/{day}/day_ahead_prices.csv',
        maximum_export_limits='data/base/{day}/maximum_export_limits.csv',
        physical_notifications='data/base/{day}/physical_notifications.csv',
        europe_day_ahead_prices='data/base/{day}/europe_day_ahead_prices.csv',
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


rule solve_network:
    input:
        bmus="data/prerun/prepared_bmus.csv",
        bids="data/base/{day}/bids.csv",
        network_nodal="results/{day}/network_{ic}_s_nodal.nc",
        network_national="results/{day}/network_{ic}_s_national.nc",
        network_zonal="results/{day}/network_{ic}_s_zonal.nc",
        transmission_boundaries='data/transmission_boundaries.yaml',
        boundary_flow_constraints="data/base/{day}/boundary_flow_constraints.csv",
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
