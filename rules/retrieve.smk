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
        day_ahead_prices="data/base/{day}/day_ahead_prices.csv",
        offers="data/base/{day}/offers.csv",
        bids="data/base/{day}/bids.csv",
        physical_notifications="data/base/{day}/physical_notifications.csv",
        maximum_export_limits="data/base/{day}/maximum_export_limits.csv",
        boundary_flow_constraints="data/base/{day}/boundary_flow_constraints.csv",
        europe_day_ahead_prices="data/base/{day}/europe_day_ahead_prices.csv",
    resources:
        mem_mb=4000,
    log:
        "../logs/base/{day}.log",
    conda:
        "../envs/environment.yaml",
    script:
        "../scripts/build_base.py"
