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
from _helpers import configure_logging, set_nested_attr
from summarize_system_cost import get_bidding_volume
from solve_network import safe_solve


def insert_flow_constraints(
    n,
    flow_constraints,
    boundaries,
    # calibration_parameters,
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
        
        # flow_max_pu = limit / nameplate_capacity * calibration_parameters[boundary]
        flow_max_pu = limit / nameplate_capacity

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