from snakemake.remote.HTTP import RemoteProvider as HTTPRemoteProvider
from snakemake.utils import min_version

from scripts._helpers import path_provider

HTTP = HTTPRemoteProvider()


min_version("7.7")



rule build_base:
    params:
        base=config_provider("base"),
    output:
        day_ahead_prices="data/base/day_ahead_prices.csv",
    script:
        "scripts/build_base.py"
    conda:
        "envs/environment.yaml"


