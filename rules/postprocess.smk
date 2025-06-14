# SPDX-FileCopyrightText: : 2024 Lukas Franken
#
# SPDX-License-Identifier: MIT


rule summarize_bmu_revenues:
    input:
        bids="data/base/{day}/bids.csv",
        offers="data/base/{day}/offers.csv",
        cfd_strike_prices="data/prerun/cfd_strike_prices.csv",
        roc_values="data/prerun/roc_values.csv",
        network_nodal="results/{day}/network_{ic}_s_nodal_solved.nc",
        network_national="results/{day}/network_{ic}_s_national_solved.nc",
        network_national_redispatch="results/{day}/network_{ic}_s_national_solved_redispatch.nc",
        network_zonal="results/{day}/network_{ic}_s_zonal_solved.nc",
        network_zonal_redispatch="results/{day}/network_{ic}_s_zonal_solved_redispatch.nc",
        default_balancing_prices=lambda wildcards: 'data/prerun/balancing_prices/{year}-week{week}.csv'.format(
            year=datetime.strptime(wildcards.day, '%Y-%m-%d').year,
            week=str(datetime.strptime(wildcards.day, '%Y-%m-%d').isocalendar()[1]).zfill(2)
        ),
    output:
        bmu_revenues_nodal="results/{day}/bmu_revenues_{ic}_nodal.csv",
        bmu_revenues_zonal="results/{day}/bmu_revenues_{ic}_zonal.csv",
        bmu_revenues_national="results/{day}/bmu_revenues_{ic}_national.csv",
        bmu_dispatch_nodal="results/{day}/bmu_dispatch_{ic}_nodal.csv",
        bmu_dispatch_zonal="results/{day}/bmu_dispatch_{ic}_zonal.csv",
        bmu_dispatch_national="results/{day}/bmu_dispatch_{ic}_national.csv",
        bmu_revenues_detailed_national="results/{day}/bmu_revenues_detailed_{ic}_national.csv",
        bmu_revenues_detailed_nodal="results/{day}/bmu_revenues_detailed_{ic}_nodal.csv",
        bmu_revenues_detailed_zonal="results/{day}/bmu_revenues_detailed_{ic}_zonal.csv",
        gb_total_load="results/{day}/gb_total_load_{ic}.csv",
    resources:
        mem_mb=1500,
    log:
        "../logs/system_cost/{day}_{ic}.log",
    conda:
        "../envs/environment.yaml",
    script:
        "../scripts/summarize_bmu_revenues.py"


rule summarize_system_cost:
    input:
        bids="data/base/{day}/bids.csv",
        offers="data/base/{day}/offers.csv",
        cfd_strike_prices="data/prerun/cfd_strike_prices.csv",
        roc_values="data/prerun/roc_values.csv",
        network_nodal="results/{day}/network_{ic}_s_nodal_solved.nc",
        network_national="results/{day}/network_{ic}_s_national_solved.nc",
        network_national_redispatch="results/{day}/network_{ic}_s_national_solved_redispatch.nc",
        network_zonal="results/{day}/network_{ic}_s_zonal_solved.nc",
        network_zonal_redispatch="results/{day}/network_{ic}_s_zonal_solved_redispatch.nc",
        bmu_revenues_nodal="results/{day}/bmu_revenues_{ic}_nodal.csv",
        bmu_revenues_zonal="results/{day}/bmu_revenues_{ic}_zonal.csv",
        bmu_revenues_national="results/{day}/bmu_revenues_{ic}_national.csv",
    output:
        marginal_prices="results/{day}/marginal_prices_{ic}.csv",
        system_cost_summary="results/{day}/system_cost_summary_{ic}.csv",
    resources:
        mem_mb=1500,
    log:
        "../logs/system_cost/{day}_{ic}.log",
    conda:
        "../envs/environment.yaml",
    script:
        "../scripts/summarize_system_cost.py"


rule summarize_frontend_data:
    input:
        bmu_revenues_zonal="results/{day}/bmu_revenues_{ic}_zonal.csv",
        bmu_revenues_national="results/{day}/bmu_revenues_{ic}_national.csv",
        nat_who="results/{day}/network_flex_s_national_solved.nc",
        nat_bal="results/{day}/network_flex_s_national_solved_redispatch.nc",
        zon_who="results/{day}/network_flex_s_zonal_solved.nc",
        zon_bal="results/{day}/network_flex_s_zonal_solved_redispatch.nc",
        roc_values="data/prerun/roc_values.csv",
        cfd_strike_prices="data/prerun/cfd_strike_prices.csv",
        system_cost_summary="results/{day}/system_cost_summary_{ic}.csv",
        gb_shape="data/gb_shape.geojson",
        bids="data/base/{day}/bids.csv",
        offers="data/base/{day}/offers.csv",
        default_balancing_prices=lambda wildcards: 'data/prerun/balancing_prices/{year}-week{week}.csv'.format(
            year=datetime.strptime(wildcards.day, '%Y-%m-%d').year,
            week=str(datetime.strptime(wildcards.day, '%Y-%m-%d').isocalendar()[1]).zfill(2)
        ),
    output:
        frontend_revenues="frontend/{day}/revenues_{ic}.csv",
        frontend_dispatch="frontend/{day}/dispatch_{ic}.csv",
        frontend_dispatch_intercon="frontend/{day}/dispatch_flex_{ic}_intercon.csv",
        frontend_marginal_costs="frontend/{day}/marginal_costs_{ic}.csv",
        frontend_thermal_dispatch="frontend/{day}/thermal_dispatch_{ic}.csv",
    resources:
        mem_mb=1500,
    log:
        "../logs/frontend_data/{day}_{ic}.log",
    conda:
        "../envs/environment.yaml",
    script:
        "../scripts/summarize_frontend_data.py"
