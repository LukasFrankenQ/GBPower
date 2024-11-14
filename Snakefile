# SPDX-FileCopyrightText: : 2017-2024 The PyPSA-Eur Authors, Lukas Franken
#
# SPDX-License-Identifier: MIT

import yaml
from pathlib import Path
from os.path import normpath, exists
from snakemake.utils import min_version
from shutil import copyfile, move, rmtree

from scripts._helpers import path_provider


min_version("8.11")

configfile: "config.yaml"

include: "rules/prerun_rules.smk"
include: "rules/retrieve.smk"
include: "rules/run.smk"
include: "rules/postprocess.smk"
