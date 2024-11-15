# SPDX-FileCopyrightText: : 2024 Lukas Franken
#
# SPDX-License-Identifier: MIT

rule summarize_balancing_volume:
    input:
        bids="data/base/{day}/bids.csv",
        offers="data/base/{day}/offers.csv",
        network_nodal="results/networks/{day}/network_s_nodal_solved.nc",
        network_national="results/networks/{day}/network_s_national_solved.nc",
    output:
        balancing_volume="results/{day}/balancing_summary.csv",
    resources:
        mem_mb=1500,
    log:
        "../logs/balancing/{day}.log",
    conda:
        "../envs/environment.yaml",
    script:
        "../scripts/summarize_balancing_volume.py"

