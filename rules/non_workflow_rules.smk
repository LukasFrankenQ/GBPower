# SPDX-FileCopyrightText: : 2024 Lukas Franken
#
# SPDX-License-Identifier: MIT


rule build_europe_day_ahead_prices:
    run:
        raise ValueError("This rule is not implemented yet.")


rule build_roc_values:
    input:
        # uses whichever days are available.
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
        )
    output:
        protected("data/roc_values.csv")
    resources:
        mem_mb=4000,
    log:
        "../logs/roc_values.log",
    conda:
        "../envs/environment.yaml",
    script:
        "../non_workflow_scripts/build_roc_levels.py"

