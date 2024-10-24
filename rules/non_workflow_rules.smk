# SPDX-FileCopyrightText: : 2024 Lukas Franken
#
# SPDX-License-Identifier: MIT


# This file contains rules that are not part of the main workflow because they are
# costly to run. The files they produce are therefore shipped with the repository
# For transparency, the rules are still included in the Snakefile, but, by default,
# are not run.
# To run them, remove the protected() function from the output directive of each rule.


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
        protected("data/roc_values.csv")
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
        protected("data/nuclear_marginal_cost.csv")
    resources:
        mem_mb=4000,
    log:
        "../logs/nuclear_bidding_cost.log",
    conda:
        "../envs/environment.yaml",
    script:
        "../non_workflow_scripts/build_nuclear_bidding_cost.py"
