# SPDX-FileCopyrightText: : 2024 The PyPSA Authors, Lukas Franken
#
# SPDX-License-Identifier: MIT

import logging

logger = logging.getLogger(__name__)

import pypsa
import pandas as pd

from _helpers import configure_logging, set_nested_attr


if __name__ == '__main__':

    configure_logging(snakemake)

    # calibration_parameters = pd.read_csv(
    #     snakemake.input['line_calibration'],
    #     index_col=0,
    #     ).iloc[:,0]
    
    logger.warning('Currently calibration unaware if tuning lines or links.')

    network_national = pypsa.Network(snakemake.input['network_national'])

    # network.lines.loc[
    #     network.lines.index.intersection(calibration_parameters.index), 's_nom'
        # ] *= calibration_parameters

    # network.links.loc[
        # network.links.index.intersection(calibration_parameters.index), 'p_nom'
    #     ] *= calibration_parameters

    network_nodal = pypsa.Network(snakemake.input['network_nodal'])
    network_nodal.optimize()
    network_nodal.export_to_netcdf(snakemake.output['network_nodal'])

    network_national = pypsa.Network(snakemake.input['network_national'])
    network_national.optimize()
    network_national.export_to_netcdf(snakemake.output['network_national'])

    national_redispatch = pypsa.Network(snakemake.input['network_nodal'])

    print(national_redispatch.loads_t['p_set'].shape)
    for su in network_national.storage_units_t['p'].columns:

        if network_national.storage_units.loc[su, 'carrier'] in ['cascade', 'hydro']:
            continue

        p_set = network_national.storage_units_t['p'][su]

        bus = national_redispatch.storage_units.loc[su, 'bus']
        national_redispatch.remove('StorageUnit', su)

        new_load = national_redispatch.loads_t['p_set'][bus] - p_set

        set_nested_attr(
            national_redispatch,
            f'loads_t.p_set.{bus}',
            new_load
        )

    national_redispatch.optimize()
    national_redispatch.export_to_netcdf(snakemake.output['network_national_redispatch'])  
