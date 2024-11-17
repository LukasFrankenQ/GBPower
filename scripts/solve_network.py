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

    # network.lines.loc[
    #     network.lines.index.intersection(calibration_parameters.index), 's_nom'
        # ] *= calibration_parameters

    # network.links.loc[
        # network.links.index.intersection(calibration_parameters.index), 'p_nom'
    #     ] *= calibration_parameters

    network_national = pypsa.Network(snakemake.input['network_national'])
    network_national.optimize()
    network_national.export_to_netcdf(snakemake.output['network_national'])

    national_redispatch = pypsa.Network(snakemake.input['network_nodal'])
    network_nodal = pypsa.Network(snakemake.input['network_nodal'])


    flow_constraints = pd.read_csv(
        snakemake.input['boundary_flow_constraints'],
        index_col=0,
        parse_dates=True
    )

    boundaries = {
        'SSE-SP': [13161, 6241, 6146, 6145, 6149, 6150],
        'SCOTEX': [14109, 6139, 11758],
        'SSHARN': [11778, 11780, 5225],
        'SWALEX': [11515, 11519],
        'SEIMP': [6121, 12746, 11742],
        'FLOWSTH': [5203, 11528, 11764, 6203, 5207]
    }

    for boundary in flow_constraints.columns:


        limit = flow_constraints[boundary]
        lines = pd.Index(boundaries[boundary], dtype=str)

        try:
            nameplate_capacity = network_nodal.lines.loc[lines, 's_nom'].sum()
        except KeyError:
            nameplate_capacity = network_nodal.links.loc[lines, 'p_nom'].sum()
        
        flow_max_pu = limit / nameplate_capacity

        logger.info(f'Tuning flow constraint for {boundary} by factor {flow_max_pu.mean():.2f}')

        try:
            network_nodal.lines_t.s_max_pu[boundary] = flow_max_pu
            national_redispatch.lines_t.s_max_pu[boundary] = flow_max_pu
        except KeyError:
            print('goood second except!')
            network_nodal.links_t.p_max_pu[boundary] = flow_max_pu
            national_redispatch.links_t.p_max_pu[boundary] = flow_max_pu

    network_nodal.optimize()
    network_nodal.export_to_netcdf(snakemake.output['network_nodal'])

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
