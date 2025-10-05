# SPDX-FileCopyrightText: : 2025 Lukas Franken
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
from summarize_system_cost import get_bidding_volume
from _helpers import configure_logging, set_nested_attr


def freeze_battery_commitments(n_from, n_to):
    '''
    Takes wholesale commitments of batteries and PHS from n_from and inserts them 
    into n_to, i.e. n_to HAS TO operate the respective storage units in the same
    way as n_from.
    '''

    '''
    for su in n_from.storage_units_t['p'].columns:

        # if n_from.storage_units.loc[su, 'carrier'] in ['cascade', 'hydro']:
        #     continue

        p_set = n_from.storage_units_t['p'][su]

        bus = n_to.storage_units.loc[su, 'bus']

        n_to.remove('StorageUnit', su)

        load_name = n_to.loads.index[n_to.loads.bus == bus][0]
        new_load = n_to.loads_t['p_set'][load_name] - p_set

        set_nested_attr(
            n_to,
            f'loads_t.p_set.{bus}',
            new_load
        )
    '''
    units = n_from.storage_units_t['p'].columns
    n_to.storage_units_t.p_set.loc[:, units] = n_from.storage_units_t.p.loc[:, units]


def freeze_interconnector_commitments(n_from, n_to):
    '''
    Takes wholesale commitments of interconnectors from n_from and inserts them 
    into n_to, i.e. n_to HAS TO operate the respective interconnectors in the same
    way as n_from.
    '''

    ic = n_from.links.loc[n_from.links.carrier == 'interconnector'].index
    n_to.links_t.p_set.loc[:, ic] = n_from.links_t.p0.loc[:, ic]


def rstu(n):
    for i in n.storage_units.index:
        n.remove('StorageUnit', i)


def safe_solve(n, factor=1, solver_name='highs'):

    status = 'not_solved'
    hold = n.copy()

    while status != 'ok':
        logger.info(f"\nSolving with factor {factor:.2f}\n")
        n.links = hold.links.copy()

        tune_line_capacities(n, factor)
        status, _ = n.optimize(solver_name=solver_name)
        
        if status != 'ok':
            factor *= 1.02

        if factor > 10:
            raise Exception('Failed to solve redispatch problem')

    return status, np.around(factor, decimals=3)


def insert_flow_constraints(
    n,
    flow_constraints,
    boundaries,
    # calibration_parameters,
    groupings,
    future_boundary_additions=None,
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
        
        # flow_max_pu = limit / nameplate_capacity * calibration_parameters[boundary]
        flow_max_pu = limit / nameplate_capacity

        logger.info(f'Tuning flow constraint for {boundary} by factor {flow_max_pu.mean():.2f}')

        if groupings is not None:
            lines = lines.append(pd.Index(groupings[boundary]))

        duplicate_lines = [line for line in lines if line in used_lines]
        assert not duplicate_lines, f'Line(s) {duplicate_lines} used in multiple boundaries'
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
        
        if future_boundary_additions is not None and boundary in future_boundary_additions:
            for new_line in n.links.index.intersection(future_boundary_additions[boundary]):
                logger.info(f'Adding flow constraint for future line {new_line} to {boundary}')

                pu = pd.Series(flow_max_pu.values, index=n.snapshots, name=new_line)
                n.links_t.p_max_pu = pd.concat([pu, n.links_t.p_max_pu], axis=1)
                n.links_t.p_min_pu = pd.concat([pu.mul(-1.), n.links_t.p_min_pu], axis=1)



def get_line_grouping(
        buses,                # network buses
        lines,                # network lines
        boundaries,           # dict: boundary_name -> list of line IDs forming that boundary
        anchor_buses          # dict: boundary_name -> list of known buses for that boundary
    ):
    """
    Returns the lines of regions neighboring transmission boundaries such that 
    the thermal constraints available for the boundaries themselves are also applied
    to the regions surrounding them.

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


def tune_line_capacities(n, factor):
    '''
    Multiplies line capacities by a factor.
    '''
    assert n.lines.empty, 'Current setup is for full DC approximation.'
    n.links.loc[n.links.carrier != 'interconnector', 'p_nom'] *= factor


# provides the name of one bus within the cluster of buses around transmission boundaries
# In a transmission network where the transmission lines that constitute the boundary
# are removed all buses that are in the same network graph as the anchor buses are assigned
# as the regional interpretation of the boundaries.
# I.e. for instance all lines in Scotland north of SSE-SP have the same thermal constraints
# applied to them as the boundary itself.
anchors = {
    'SSE-SP': ['6441'],
    'SCOTEX': ['5912'],
    'SSHARN': ['5946'],
    'FLOWSTH': ['6010', '5250'],
    'SEIMP': ['4977'],
}


if __name__ == '__main__':

    logger.warning('Relaxation factors for zonal and nodal should start at national redispatch relaxation factor')

    configure_logging(snakemake)

    idx = pd.IndexSlice

    bids = pd.read_csv(snakemake.input['bids'], index_col=[0,1], parse_dates=True)
    bmus = pd.read_csv(snakemake.input['bmus'], index_col=0)

    bmus = bmus.loc[bmus['lat'] != 'distributed']
    bmus['lat'] = bmus['lat'].astype(float)

    bids = bids.loc[idx[:, 'vol'], :].sum()
    bids.index = bids.index.get_level_values(0)

    # select bmus that are likely to curtail due to grid congestion
    renewable_bmus = bmus[
        bmus.carrier.isin(['onwind', 'offwind', 'hydro', 'cascade'])
        ].index
    thermal_bmus = bmus[
        (bmus.carrier.isin(['fossil', 'biomass', 'coal'])) & 
        (bmus['lat'] > 55.3)
    ].index

    bid_counting_units = renewable_bmus.union(thermal_bmus)

    # Get total daily bidding volume for these generators
    daily_volume = bids.loc[
        bids.index.intersection(bid_counting_units)
        ].sum()

    flow_constraints = pd.read_csv(
        snakemake.input['boundary_flow_constraints'],
        index_col=0,
        parse_dates=True
    )

    with open(snakemake.input['transmission_boundaries']) as f:
        boundaries = yaml.safe_load(f)['existing_boundaries']

    logger.warning('Currently calibration unaware if tuning lines or links.')

    # national market does not need transmission calibration
    n_nodal = pypsa.Network(snakemake.input['network_nodal'])
    n_national = pypsa.Network(snakemake.input['network_national'])
    # n_zonal = pypsa.Network(snakemake.input['network_zonal'])

    groupings = get_line_grouping(
        n_nodal.buses, 
        n_nodal.links.loc[n_nodal.links.carrier != 'interconnector', :],
        boundaries,
        anchors
        )

    # args = (flow_constraints, boundaries, calibration_parameters, groupings)
    args = (flow_constraints, boundaries, groupings)

    n_national_redispatch = pypsa.Network(snakemake.input['network_nodal'])
    # n_zonal_redispatch = pypsa.Network(snakemake.input['network_nodal'])

    assert n_nodal.lines.empty, 'Current setup is for full DC approximation.'

    insert_flow_constraints(n_national_redispatch, *args, model_name='national balancing')
    insert_flow_constraints(n_nodal, *args, model_name='nodal wholesale')
    # insert_flow_constraints(n_zonal, *args, model_name='zonal wholesale')
    # insert_flow_constraints(n_zonal_redispatch, *args, model_name='zonal redispatch')

    tolerance = 0.05 # modelled balancing volume can deviate from actual balancing volume by this much

    print('\n\nstarting line calibration based on national model\n\n')
    status, _ = n_national.optimize(solver_name=solver_name)
    # n_national.export_to_netcdf(snakemake.output['network_national'])

    freeze_battery_commitments(n_national, n_national_redispatch)

    if snakemake.wildcards.ic == 'flex':
        logger.info('Freezing interconnector commitments')
        freeze_interconnector_commitments(n_national, n_national_redispatch)

    # the following loop ensures that modelled balancing volume matches actual balancing volume
    tuned = False
    counter = 0

    # Initialize binary search bounds
    left = 0.5  # minimum reasonable scaling factor
    right = 2.0 # maximum reasonable scaling factor

    unsolvable_case = False # special case deals with matching balancing volume is not possible for feasible model

    while not tuned:
        # Try the midpoint of the current range
        if not unsolvable_case:
            line_scaling_factor = (left + right) / 2

        hold_redispatch = n_national_redispatch.copy()
        tuned_line_capacities = tune_line_capacities(hold_redispatch, line_scaling_factor)
        status, _ = hold_redispatch.optimize(solver_name=solver_name)

        if status == 'ok':
            balancing_volume = get_bidding_volume(n_national, hold_redispatch).sum()
            error = abs(balancing_volume - daily_volume)

            if error <= tolerance * daily_volume:
                tuned = True

            elif unsolvable_case:
                tuned = True

            else:
                # Update binary search bounds based on whether we need more or less capacity
                if balancing_volume > daily_volume:
                    # Too much balancing volume - need to increase line capacity
                    left = line_scaling_factor
                else:
                    # Too little balancing volume - need to decrease line capacity  
                    right = line_scaling_factor
        else:
            # If infeasible, need more line capacity
            if not unsolvable_case:
                left = line_scaling_factor
            else:
                line_scaling_factor += 0.02

        counter += 1
        if status == 'ok' and right - left < 0.01:  # Convergence check
            tuned = True
        elif status == 'warning' and right - left < 0.01:
            unsolvable_case = True

        counter += 1
        if counter > 100:
            raise Exception('Failed to tune line capacities')

        if status == 'ok':
            logger.info(f'Received balancing volume {balancing_volume:.2f} with line scaling factor {line_scaling_factor:.2f}')
        else:
            logger.info(f'Model infeasible with line scaling factor {line_scaling_factor:.2f}')

        # solved first such that line relaxation factor can also be applied to the other models
        # status, relaxation_factor = safe_solve(n_national_redispatch)

    print('=============================================================================')
    logger.info(f'Successfully tuned line capacities after {counter} iterations and a line scaling factor of {line_scaling_factor:.2f}')
    logger.info(f'Modelled balancing volume: {balancing_volume*1e-3:.2f} GWh, actual balancing volume: {daily_volume*1e-3:.2f} GWh')
    print('=============================================================================')

    with open(snakemake.output['calibration_factor'], 'w') as f:
        yaml.dump({'calibration_factor': line_scaling_factor}, f)
