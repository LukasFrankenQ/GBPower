# SPDX-FileCopyrightText: : 2024 Lukas Franken
#
# SPDX-License-Identifier: MIT

rule prepare_network:
    input:
        roc_values='data/roc_values.csv',
        bmus='data/bmus_prepared.csv',
        cfd_strike_prices='resources/cfd_strike_prices.csv',
        nuclear_marginal_price='data/nuclear_marginal_cost.csv',
        battery_phs_capacities='data/battery_phs_capacities.csv',
        thermal_costs=lambda: wildcards: 'resources/thermal_costs/{year}-week{week}.csv'.format(
            year=datetime.strptime(wildcards.day, '%Y-%m-%d').year,
            week=datetime.strptime(wildcards.day, '%Y-%m-%d').isocalendar()[1]
        ),
        day_ahead_prices='data/base/{day}/day_ahead_prices.csv',
        maximum_export_limits='data/base/{day}/maximum_export_limits.csv',
        physical_notifications='data/base/{day}/physical_notifications.csv',
        europe_day_ahead_prices='data/base/{day}/europe_day_ahead_prices.csv',
    output:
        network="results/prenetworks/{day}/network_nodal.nc"
    resources:
        mem_mb=4000,
    log:
        "../logs/prenetworks/{day}.log",
    conda:
        "../envs/environment.yaml",
    script:
        "../scripts/prepare_network.py"
