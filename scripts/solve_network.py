# SPDX-FileCopyrightText: : 2024 The PyPSA Authors, Lukas Franken
#
# SPDX-License-Identifier: MIT

import logging

logger = logging.getLogger(__name__)

import pypsa
import pandas as pd

from _helpers import configure_logging, set_nested_attr


def insert_flow_constraints(
    n,
    flow_constraints,
    boundaries,
    calibration_parameter,
    ):

    for boundary in flow_constraints.columns:

        limit = flow_constraints[boundary]
        lines = pd.Index(boundaries[boundary], dtype=str)

        try:
            nameplate_capacity = n.lines.loc[lines, 's_nom'].sum()
        except KeyError:
            nameplate_capacity = n.links.loc[lines, 'p_nom'].sum()
        
        flow_max_pu = limit / nameplate_capacity * calibration_parameter

        logger.info(f'Tuning flow constraint for {boundary} by factor {flow_max_pu.mean():.2f}')

        if lines[0] in n.lines.index:
            for line in lines:
                n.lines_t.s_max_pu[line] = flow_max_pu.values
        else:
            for line in lines:
                n.links_t.p_max_pu[line] = flow_max_pu.values


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

    # national market does not need transmission calibration
    n_national = pypsa.Network(snakemake.input['network_national'])
    n_national.optimize()
    n_national.export_to_netcdf(snakemake.output['network_national'])

    n_national_redispatch = pypsa.Network(snakemake.input['network_nodal'])

    n_zonal = pypsa.Network(snakemake.input['network_zonal'])
    n_zonal_redispatch = pypsa.Network(snakemake.input['network_nodal'])

    n_nodal = pypsa.Network(snakemake.input['network_nodal'])

    flow_constraints = pd.read_csv(
        snakemake.input['boundary_flow_constraints'],
        index_col=0,
        parse_dates=True
    )

    boundaries = {
        'SSE-SP': [13161, 6241, 6146, 6145, 6149, 6150],
        'SCOTEX': [14109, 6139, 11758],
        'SSHARN': [11778, 11780, 5225],
        'SEIMP': [6121, 12746, 11742],
        'FLOWSTH': [5203, 11528, 11764, 6203, 5207]
    }

    calibration_parameter = 0.75

    args = (flow_constraints, boundaries, calibration_parameter)

    insert_flow_constraints(n_nodal, *args)
    insert_flow_constraints(n_national_redispatch, *args)
    insert_flow_constraints(n_zonal, *args)
    insert_flow_constraints(n_zonal_redispatch, *args)

    n_nodal.optimize()
    n_nodal.export_to_netcdf(snakemake.output['network_nodal'])

    n_zonal.optimize()
    n_zonal.export_to_netcdf(snakemake.output['network_nodal'])

    # redispatch calculation (only used to estimate balancing volume)
    # computes the nodal flow after wholesale commitments have been made
    # in the wholesale market. Therefore, battery (and interconnector po-
    # -sitions if ic wildcard == 'flex') positions are inserted into a
    # nodal network layout.

    def insert_battery_positions(n_from, n_to):

        for su in n_from.storage_units_t['p'].columns:

            if n_from.storage_units.loc[su, 'carrier'] in ['cascade', 'hydro']:
                continue

            p_set = n_from.storage_units_t['p'][su]

            bus = n_to.storage_units.loc[su, 'bus']
            n_to.remove('StorageUnit', su)

            new_load = n_to.loads_t['p_set'][bus] - p_set

            set_nested_attr(
                n_to,
                f'loads_t.p_set.{bus}',
                new_load
            )
    
    def insert_interconnector_positions(n_from, n_to):
        
        ic = n_from.links.loc[n_from.links.carrier == 'interconnector'].index
        n_to.links_t.p_set.loc[:, ic] = n_to.links_t.p0.loc[:, ic]


    insert_battery_positions(n_national, n_national_redispatch)
    insert_battery_positions(n_zonal, n_zonal_redispatch)

    if snakemake.wildcards.ic == 'flex':
        insert_interconnector_positions(n_national, n_national_redispatch)
        insert_interconnector_positions(n_zonal, n_zonal_redispatch)

    n_national_redispatch.optimize()
    n_national_redispatch.export_to_netcdf(snakemake.output['network_national_redispatch'])  

    n_zonal_redispatch.optimize()
    n_zonal_redispatch.export_to_netcdf(snakemake.output['network_zonal_redispatch'])  
