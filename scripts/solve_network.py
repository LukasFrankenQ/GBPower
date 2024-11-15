# SPDX-FileCopyrightText: : 2024 The PyPSA Authors, Lukas Franken
#
# SPDX-License-Identifier: MIT

import logging

logger = logging.getLogger(__name__)

import pypsa
import pandas as pd

from _helpers import configure_logging

if __name__ == '__main__':

    configure_logging(snakemake)

    calibration_parameters = pd.read_csv(
        snakemake.input['line_calibration'],
        index_col=0,
        ).iloc[:,0]
    
    logger.warning('Currently calibration unaware if tuning lines or links.')

    network = pypsa.Network(snakemake.input['network'])

    network.lines.loc[
        network.lines.index.intersection(calibration_parameters.index), 's_nom'
        ] *= calibration_parameters

    network.links.loc[
        network.links.index.intersection(calibration_parameters.index), 'p_nom'
        ] *= calibration_parameters

    network.export_to_netcdf(snakemake.output['network'])
