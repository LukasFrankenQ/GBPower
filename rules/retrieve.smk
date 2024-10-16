# SPDX-FileCopyrightText: : 2024 Lukas Franken
#
# SPDX-License-Identifier: MIT


rule build_flow_constraints:
    output:
        flow_constraints="data/flow_constraints_{year}.csv"
    resources:
        mem_mb=4000,
    log:
        "../logs/flow_constraints_{year}.log",
    script:
        "../scripts/build_flow_constraints.py"


rule build_base:
    input:
        europe_day_ahead_prices="data/europe_day_ahead_prices_GBP.csv""",
        flow_constraints=lambda wildcards: f"data/flow_constraints_{wildcards.day[:4]}.csv",
    output:
        date_register="data/base/{day}/settlement_period_register.csv",
        boundary_flow_constraints="data/base/{day}/boundary_flow_constraints.csv",
        day_ahead_prices="data/base/{day}/day_ahead_prices.csv",
        offers="data/base/{day}/offers.csv",
        bids="data/base/{day}/bids.csv",
        physical_notifications="data/base/{day}/physical_notifications.csv",
        maximum_export_limits="data/base/{day}/maximum_export_limits.csv",
        europe_day_ahead_prices="data/base/{day}/europe_day_ahead_prices.csv",
    resources:
        mem_mb=4000,
    log:
        "../logs/base/{day}.log",
    conda:
        "../envs/environment.yaml",
    script:
        "../scripts/build_base.py"


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
        cfd_strike_prices="resources/cfd_strike_prices.csv",
    resources:
        mem_mb=4000,
    log:
        "../logs/cfds.log",
    conda:
        "../envs/environment.yaml",
    script:
        "../scripts/build_cfds.py"