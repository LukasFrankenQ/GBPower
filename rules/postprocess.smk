# SPDX-FileCopyrightText: : 2024 Lukas Franken
#
# SPDX-License-Identifier: MIT

rule summarize_system_cost:
    input:
        bids="data/base/{day}/bids.csv",
        offers="data/base/{day}/offers.csv",
        cfd_strike_prices="data/preprocessed/cfd_strike_prices.csv",
        roc_values="data/preprocessed/roc_values.csv",
        network_nodal="results/{day}/network_{ic}_s_nodal_solved.nc",
        network_national="results/{day}/network_{ic}_s_national_solved.nc",
        network_national_redispatch="results/{day}/network_{ic}_s_national_solved_redispatch.nc",
        network_zonal="results/{day}/network_{ic}_s_zonal_solved.nc",
        network_zonal_redispatch="results/{day}/network_{ic}_s_zonal_solved_redispatch.nc",
    output:
        system_cost_summary="results/{day}/system_cost_summary_{ic}.csv",
    resources:
        mem_mb=1500,
    log:
        "../logs/system_cost/{day}_{ic}.log",
    conda:
        "../envs/environment.yaml",
    script:
        "../scripts/summarize_system_cost.py"
