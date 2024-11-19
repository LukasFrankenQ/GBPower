# SPDX-FileCopyrightText: : 2024 Lukas Franken
#
# SPDX-License-Identifier: MIT

rule summarize_system_cost:
    input:
        bids="data/base/{day}/bids.csv",
        offers="data/base/{day}/offers.csv",
        network_nodal="results/networks/{day}/network_s_nodal_solved.nc",
        network_national="results/networks/{day}/network_s_national_solved.nc",
        network_national_redispatch="results/networks/{day}/network_s_national_solved_redispatch.nc",
    output:
        balancing_volume="results/{day}/system_cost.csv",
    resources:
        mem_mb=1500,
    log:
        "../logs/system_cost/{day}.log",
    conda:
        "../envs/environment.yaml",
    script:
        "../scripts/summarize_system_cost.py"

