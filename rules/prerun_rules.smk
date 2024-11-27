# SPDX-FileCopyrightText: : 2024 Lukas Franken
#
# SPDX-License-Identifier: MIT


# This file contains rules that are not part of the main workflow because they are
# costly to run. The files they produce are therefore shipped with the repository
# For transparency, the rules are still included in the Snakefile, but, by default,
# are not run.
# To run them, remove the protected() function from the output directive of each rule.

from datetime import datetime, timedelta


rule build_europe_day_ahead_prices:
    run:
        raise ValueError("This rule is not implemented yet.")


rule build_roc_values:
    input:
        # Builds estimated ROC values for whichever days are available.
        # but requires 300 days of data from build_base to be preset to run
        lambda wildcards: (
            lambda glob: (
                lambda files: (
                    files if len(files) >= 300 else
                    (_ for _ in ()).throw(
                        Exception(
                            f"Not enough data downloaded in directory 'data/base/'. "
                            f"Expected at least 300 days of data, found {len(files)}."
                            f"Use the rule 'build_base' to download more data."
                        )
                    )
                )
            )(glob.glob("data/base/*"))
        )(
            __import__('glob')
        ),
        bmu_locations="data/bmus_prepared.csv",
        cfd_strike_prices="resources/cfd_strike_prices.csv",
    output:
        # "data/preprocessed/roc_values.csv"
    resources:
        mem_mb=4000,
    log:
        "../logs/roc_values.log",
    conda:
        "../envs/environment.yaml",
    script:
        "../non_workflow_scripts/build_roc_values.py"


rule build_nuclear_bidding_cost:
    input:
        # Looks over ~a year at day-ahead prices to and
        # takes the lowest observed day-ahead price as the marginal cost 
        # for nuclear generation.
        # This rule requires the rule 'build_base' to be run first.
        lambda wildcards: (
            lambda glob: (
                lambda files: (
                    files if len(files) >= 300 else
                    (_ for _ in ()).throw(
                        Exception(
                            f"Not enough data downloaded in directory 'data/base/'. "
                            f"Expected at least 300 days of data, found {len(files)}."
                            f"Use the rule 'build_base' to download more data."
                        )
                    )
                )
            )(glob.glob("data/base/*"))
        )(
            __import__('glob')
        ),
    output:
        # protected("data/preprocessed/nuclear_marginal_cost.csv")
        # "data/preprocessed/nuclear_marginal_cost.csv"
    resources:
        mem_mb=4000,
    log:
        "../logs/nuclear_bidding_cost.log",
    conda:
        "../envs/environment.yaml",
    script:
        "../non_workflow_scripts/build_nuclear_bidding_cost.py"


rule build_battery_phs_capacities:
    input:
        # Estimates battery power and energy capacities based on a large dataset of
        # historic physical notifications.
        # Requires 300 days of data from build_base to be preset to run
        lambda wildcards: (
            lambda glob: (
                lambda files: (
                    files if len(files) >= 300 else
                    (_ for _ in ()).throw(
                        Exception(
                            f"Not enough data downloaded in directory 'data/base/'. "
                            f"Expected at least 300 days of data, found {len(files)}."
                            f"Use the rule 'build_base' to download more data."
                        )
                    )
                )
            )(glob.glob("data/base/*"))
        )(
            __import__('glob')
        ),
        bmu_locations="data/bmus_prepared.csv",
    output:
        # "data/preprocessed/battery_phs_capacities.csv"
    resources:
        mem_mb=4000,
    log:
        "../logs/build_battery_phs_capacities.log",
    conda:
        "../envs/environment.yaml",
    script:
        "../non_workflow_scripts/build_battery_phs_capacities.py"


def get_input_files(wildcards):
    year = int(wildcards.year)
    week = int(wildcards.week)

    week_start = datetime.strptime(f'{year}-W{week:02d}-1', "%G-W%V-%u")

    dates = [week_start - timedelta(days=i) for i in range(1, 31)]

    first_date_of_2022 = datetime(2022, 1, 1)

    # Check if any date is before 2022-01-01
    if any(date < first_date_of_2022 for date in dates):
        # Use the first 30 days of 2022 instead
        dates = [first_date_of_2022 + timedelta(days=i) for i in range(30)]

    file_paths = (
        [f'data/base/{date.strftime("%Y-%m-%d")}/physical_notifications.csv' for date in dates] +
        [f'data/base/{date.strftime("%Y-%m-%d")}/day_ahead_prices.csv' for date in dates]

    )
    return file_paths


rule build_thermal_generator_prices:
    input:
        get_input_files,
        bmus='data/bmus_prepared.csv',
    output:
        # protected('resources/thermal_costs/{year}-week{week}.csv')
        'resources/thermal_costs/{year}-week{week}.csv'
    resources:
        mem_mb=4000,
    log:
        '../logs/thermal_costs/{year}-{week}.log',
    conda:
        '../envs/environment.yaml',
    script:
        '../non_workflow_scripts/build_thermal_generator_prices.py'


rule build_bus_regions:
    input:
        base_network="data/raw/lmp_base.nc",
        total_shape="data/gb_shape.geojson",
        offshore_shapes="data/offshore_shapes.geojson",
    output:
        regions_onshore=protected("data/regions_onshore.geojson"),
        regions_offshore=protected("data/regions_offshore.geojson"),
    log:
        "../logs/build_bus_regions.log",
    threads: 1
    resources:
        mem_mb=1000,
    conda:
        "../envs/environment.yaml"
    script:
        "../non_workflow_scripts/build_bus_regions.py"


rule build_load_weights:
    input:
        regions_onshore="data/regions_onshore.geojson",
        gsp_regions="data/gsp_geometries.geojson",
        gsp_regions_lookup="data/gsp_gnode_directconnect_region_lookup.csv",
        demandpeaks="data/FES-2021--Leading_the_Way--demandpk-all--gridsupplypoints.csv",
    output:
        load_weights=protected("data/preprocessed/load_weights.csv"),
    log:
        "../logs/build_load_weights.log",
    threads: 1
    resources:
        mem_mb=1000,
    conda:
        "../envs/environment.yaml"
    script:
        "../non_workflow_scripts/build_load_weights.py"


rule prepare_bmus:
    input:
        regions_onshore="data/regions_onshore.geojson",
        regions_offshore="data/regions_offshore.geojson",
        bmus_locs="data/raw/bmunits_loc.csv",
        bmus_raw="data/raw/temp_located_bmus.csv",
    output:
        bmus=protected("data/preprocessed/prepared_bmus.csv"),
    log:
        "../logs/prepare_bmus.log",
    threads: 1
    resources:
        mem_mb=1000,
    conda:
        "../envs/environment.yaml"
    script:
        "../non_workflow_scripts/prepare_bmus.py"


rule build_cfds:
    input:
        "data/cfd_registers/CFD_Register_Dec_9_2021.xlsx",
        "data/cfd_registers/CFD_Register_Jan_21_2022.xlsx",
        "data/cfd_registers/CFD_Register_April_11_2022.xlsx",
        "data/cfd_registers/CFD_Register_July_5_2022.xlsx",
        "data/cfd_registers/CFD_Register_Sep_27_2022.xlsx",
        "data/cfd_registers/CFD_Register_Dec_5_2022.xlsx",
        "data/cfd_registers/CFD_Register_Mar_29_2023.xlsx",
        "data/cfd_registers/CFD_Register_Jun_30_2023.xlsx",
        "data/cfd_registers/CFD_Register_Sep_29_2023.xlsx",
        "data/cfd_registers/CFD_Register_Jan_2_2024.xlsx",
        "data/cfd_registers/CFD_Register_Jan_24_2024.xlsx",
        "data/cfd_registers/CFD_Register_Apr_3_2024.xlsx",
        "data/cfd_registers/CfD_Register_2024-07-03.xlsx",
        "data/cfd_registers/CfD_Register_2024-09-10.xlsx",
        "data/cfd_registers/CfD_Register_2024-09-18.xlsx",
        bmu_locations="data/temp_located_bmus.csv",
        bmu_mappings="data/cfd_registers/cfd_to_bm_unit_mapping.csv",
    output:
        cfd_strike_prices=protected("data/preprocessed/cfd_strike_prices.csv"),
    resources:
        mem_mb=4000,
    log:
        "../logs/cfds.log",
    conda:
        "../envs/environment.yaml",
    script:
        "../scripts/build_cfds.py"


rule build_zonal_layout:
    input:
        eso_zones="data/raw/eso_zones.geojson",
    output:
        zonal_layout=protected("data/preprocessed/zonal_layout.geojson"),
    log:
        "../logs/zonal_layout.log",
    conda:
        "../envs/environment.yaml",
    script:
        "../non_workflow_scripts/build_zonal_layout.py"
