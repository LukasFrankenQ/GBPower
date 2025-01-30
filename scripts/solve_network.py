# SPDX-FileCopyrightText: : 2024 The PyPSA Authors, Lukas Franken
#
# SPDX-License-Identifier: MIT

import logging

logger = logging.getLogger(__name__)

import yaml
import pypsa
import numpy as np
import pandas as pd
import networkx as nx

from tabulate import tabulate
from _helpers import configure_logging, set_nested_attr


def insert_flow_constraints(
    n,
    flow_constraints,
    boundaries,
    calibration_parameters,
    groupings,
    model_name=None,
    ):

    used_lines = set()

    if not model_name is None:
        logger.info(f'\nInserting flow constraints for {model_name}:\n')

    for boundary in flow_constraints.columns:

        limit = flow_constraints[boundary]
        lines = pd.Index(boundaries[boundary], dtype=str)

        lines = lines.difference(used_lines)

        try:
            nameplate_capacity = n.lines.loc[lines, 's_nom'].sum()
        except KeyError:
            nameplate_capacity = n.links.loc[lines, 'p_nom'].sum()
        
        flow_max_pu = limit / nameplate_capacity * calibration_parameters[boundary]

        logger.info(f'Tuning flow constraint for {boundary} by factor {flow_max_pu.mean():.2f}')

        if groupings is not None:        
            lines = lines.append(pd.Index(groupings[boundary]))

        assert not any([line in used_lines for line in lines]), 'Line used in multiple boundaries'
        used_lines.update(set(lines))

        lines = pd.Index(set(lines))

        if lines[0] in n.lines.index:
            for line in lines:
                pu = pd.Series(flow_max_pu.values, index=n.snapshots, name=line)

                n.lines_t.s_max_pu = pd.concat([pu, n.lines_t.s_max_pu], axis=1)
                n.lines_t.s_min_pu = pd.concat([pu.mul(-1.), n.lines_t.s_min_pu], axis=1)

        else:
            for line in lines:
                pu = pd.Series(flow_max_pu.values, index=n.snapshots, name=line)

                n.links_t.p_max_pu = pd.concat([pu, n.links_t.p_max_pu], axis=1)
                n.links_t.p_min_pu = pd.concat([pu.mul(-1.), n.links_t.p_min_pu], axis=1)


def freeze_battery_commitments(n_from, n_to):
    '''
    Takes wholesale commitments of batteries and PHS from n_from and inserts them 
    into n_to, i.e. n_to HAS TO operate the respective storage units in the same
    way as n_from.
    '''

    for su in n_from.storage_units_t['p'].columns:

        # if n_from.storage_units.loc[su, 'carrier'] in ['cascade', 'hydro']:
        #     continue

        p_set = n_from.storage_units_t['p'][su]

        bus = n_to.storage_units.loc[su, 'bus']
        n_to.remove('StorageUnit', su)

        new_load = n_to.loads_t['p_set'][bus] - p_set

        set_nested_attr(
            n_to,
            f'loads_t.p_set.{bus}',
            new_load
        )


def freeze_interconnector_commitments(n_from, n_to):
    '''
    Takes wholesale commitments of interconnectors from n_from and inserts them 
    into n_to, i.e. n_to HAS TO operate the respective interconnectors in the same
    way as n_from.
    '''
    
    ic = n_from.links.loc[n_from.links.carrier == 'interconnector'].index
    n_to.links_t.p_set.loc[:, ic] = n_from.links_t.p0.loc[:, ic]


def get_line_grouping(
        buses,                # network buses
        lines,                # network lines
        boundaries,           # dict: boundary_name -> list of line IDs forming that boundary
        anchor_buses          # dict: boundary_name -> list of known buses for that boundary
    ):
    """
    n: The network object with n.buses and n.links
    boundaries: e.g. {"Scotland-England": ["Line1", "Line2"], ...}
    anchor_buses: e.g. {"Scotland-England": "BusN1", ...} for BFS to identify 'north' side
    """

    boundary_assignments = {}
    
    # 1) Build a graph of the entire network
    G = nx.Graph()
    
    # Add all buses as nodes
    for bus_name in buses.index:
        G.add_node(bus_name)

    # Add all lines as edges
    for line_id in lines.index:
        bus0 = lines.loc[line_id, 'bus0']
        bus1 = lines.loc[line_id, 'bus1']
        G.add_edge(bus0, bus1, key=line_id)  # store the line_id in "key" or as an attribute

    # 2) Iterate over boundaries in the given order (north -> south):
    G_tmp = G.copy()
    for boundary_name, boundary_line_ids in boundaries.items():

        # 2a) Temporarily remove boundary lines from the graph
        #     We'll do this by making a *copy* of G and removing those edges
        #     so as not to destroy the original.
        for line_id in boundary_line_ids:
            # Need to find the buses to remove the correct edge
            bus0 = lines.loc[line_id, 'bus0']
            bus1 = lines.loc[line_id, 'bus1']
            if G_tmp.has_edge(bus0, bus1):
                G_tmp.remove_edge(bus0, bus1)
        
    for boundary_name, boundary_line_ids in boundaries.items():

        # 2b) Find which connected component contains the known "anchor bus"
        connected_buses = list()

        for bus in anchor_buses[boundary_name]:
            # BFS (or connected_component) from anchor_bus in G_tmp
            # This set of buses is the "north side" for this boundary
            connected_buses += list(nx.bfs_tree(G_tmp, source=bus))

        # 2c) Find all lines (edges) that have both endpoints in connected.
        #     We want to assign them the fraction for this boundary
        #     -- or possibly override only if they do not already have a factor assigned,
        #        depending on your logic.
        
        boundary_assignments[boundary_name] = []

        for line_id in lines.index:
            b0 = lines.loc[line_id, 'bus0']
            b1 = lines.loc[line_id, 'bus1']

            if b0 in connected_buses and b1 in connected_buses:
                boundary_assignments[boundary_name].append(line_id)
    
    return boundary_assignments


def rstu(n):
    for i in n.storage_units.index:
        n.remove('StorageUnit', i)


def safe_solve(n):

    status = 'not_solved'
    factor = 1

    while status != 'ok':
        logger.info("Solving with factor %s", factor)
        # n.generators.p_nom *= factor
        n.links.loc[n.links.carrier != 'interconnector', 'p_nom'] *= factor
        status, _ = n.optimize()
        factor *= 1.05

        if factor > 10:
            raise Exception('Failed to solve redispatch problem')

    return status, factor


if __name__ == '__main__':

    configure_logging(snakemake)

    logger.warning('Currently calibration unaware if tuning lines or links.')

    model_execution_overview = list()

    # national market does not need transmission calibration
    n_national = pypsa.Network(snakemake.input['network_national'])
    # print('warning: dropping storage units!')
    # rstu(n_national)

    status, _ = n_national.optimize()
    n_national.export_to_netcdf(snakemake.output['network_national'])

    model_execution_overview.append(
        ('national wholesale', status, '-') 
    )

    n_national_redispatch = pypsa.Network(snakemake.input['network_nodal'])
    # rstu(n_national_redispatch)

    n_zonal = pypsa.Network(snakemake.input['network_zonal'])
    # rstu(n_zonal)
    n_zonal_redispatch = pypsa.Network(snakemake.input['network_nodal'])
    # rstu(n_zonal_redispatch)

    n_nodal = pypsa.Network(snakemake.input['network_nodal'])
    # rstu(n_nodal)

    assert n_nodal.lines.empty, 'Current setup is for full DC approximation.'

    flow_constraints = pd.read_csv(
        snakemake.input['boundary_flow_constraints'],
        index_col=0,
        parse_dates=True
    )

    with open(snakemake.input['transmission_boundaries']) as f:
        boundaries = yaml.safe_load(f)

    calibration_parameters = {
        'SSE-SP': 0.8,
        'SCOTEX': 0.6,
        'SSHARN': 0.6,
        'FLOWSTH': 1.,
        'SEIMP': 1.,
    }

    anchors = {
        'SSE-SP': ['6441'],
        'SCOTEX': ['5912'],
        'SSHARN': ['5946'],
        'FLOWSTH': ['6010', '5250'],
        'SEIMP': ['4977'],
    }

    groupings = get_line_grouping(
        n_nodal.buses, 
        n_nodal.links.loc[n_nodal.links.carrier != 'interconnector', :],
        boundaries,
        anchors
        )

    args = (flow_constraints, boundaries, calibration_parameters, groupings)
    # args = (flow_constraints, boundaries, calibration_parameters, None)

    insert_flow_constraints(n_nodal, *args, model_name='nodal wholesale')
    insert_flow_constraints(n_national_redispatch, *args, model_name='national balancing')
    insert_flow_constraints(n_zonal, *args, model_name='zonal wholesale')
    insert_flow_constraints(n_zonal_redispatch, *args, model_name='nodal wholesale')

    status, _ = n_nodal.optimize()
    n_nodal.export_to_netcdf(snakemake.output['network_nodal'])

    model_execution_overview.append(
        ('nodal wholesale', status, '-') 
    )

    status, _ = n_zonal.optimize()
    n_zonal.export_to_netcdf(snakemake.output['network_zonal'])

    model_execution_overview.append(
        ('zonal wholesale', status, '-') 
    )

    # redispatch calculation (only used to estimate balancing volume)
    # computes the nodal flow after commitments have been made in
    # the wholesale market. Therefore, battery (and interconnector po-
    # sitions if ic wildcard == 'flex') positions are inserted into a
    # nodal network layout.

    logger.info('Freezing battery commitments')
    freeze_battery_commitments(n_national, n_national_redispatch)
    freeze_battery_commitments(n_zonal, n_zonal_redispatch)

    if snakemake.wildcards.ic == 'flex':
        logger.info('Freezing interconnector commitments')
        freeze_interconnector_commitments(n_national, n_national_redispatch)
        freeze_interconnector_commitments(n_zonal, n_zonal_redispatch)


    status, factor = safe_solve(n_national_redispatch)
    # n_national_redispatch.optimize()
    n_national_redispatch.export_to_netcdf(snakemake.output['network_national_redispatch'])  

    model_execution_overview.append(
        ('national redispatch', status, str(np.around(factor, decimals=2))) 
    )

    status, factor = safe_solve(n_zonal_redispatch)
    # n_zonal_redispatch.optimize()
    n_zonal_redispatch.export_to_netcdf(snakemake.output['network_zonal_redispatch'])  

    model_execution_overview.append(
        ('zonal redispatch', status, str(np.around(factor, decimals=2))) 
    )

    print('')
    print((title := 'Model Execution Overview'))
    print("-" * len(title) + '\n')
    print(
        tabulate(
            model_execution_overview,
            headers=['Model', 'Status', 'Factor'],
            tablefmt='pretty'
        )
    )
    print('')
