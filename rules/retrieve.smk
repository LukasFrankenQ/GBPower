# SPDX-FileCopyrightText: : 2024 Lukas Franken
#
# SPDX-License-Identifier: MIT


rule build_base:
    output:
        date_register="data/base/{day}/settlement_period_register.csv",
        day_ahead_prices="data/base/{day}/day_ahead_prices.csv",
        offers="data/base/{day}/offers.csv",
        bids="data/base/{day}/bids.csv",
        physical_notifications="data/base/{day}/physical_notifications.csv",
        maximum_export_limits="data/base/{day}/maximum_export_limits.csv",
        boundary_flow_limits="data/base/{day}/boundary_flow_limits.csv",
        interconnector_prices="data/base/{day}/interconnector_prices.csv",
    resources:
        mem_mb=4000,
    log:
        "../logs/base/{day}.log",
    conda:
        "../envs/environment.yaml",
    script:
        "../scripts/build_base.py"

