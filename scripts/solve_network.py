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
from summarize_system_cost import get_bidding_volume
from calibrate_line_capacities import (
    get_line_grouping,
    insert_flow_constraints,
    tune_line_capacities,
    anchors,
    freeze_battery_commitments,
    freeze_interconnector_commitments,
    safe_solve
)


if __name__ == '__main__':

    logger.warning('Relaxation factors for zonal and nodal should start at national redispatch relaxation factor')

    configure_logging(snakemake)

    solver_name = snakemake.params['solver']
    
    flow_constraints = pd.read_csv(
        snakemake.input['boundary_flow_constraints'],
        index_col=0,
        parse_dates=True
    )

    with open(snakemake.input['transmission_boundaries']) as f:
        boundaries = yaml.safe_load(f)['existing_boundaries']
    with open(snakemake.input['transmission_boundaries']) as f:
        future_boundary_additions = yaml.safe_load(f)['future_additions']

    with open(snakemake.input['calibration_factor']) as f:
        calibration_factor = yaml.safe_load(f)['calibration_factor']

    logger.warning('Currently calibration unaware if tuning lines or links.')

    model_execution_overview = list()

    # national market does not need transmission calibration
    n_national = pypsa.Network(snakemake.input['network_national'])
    n_nodal = pypsa.Network(snakemake.input['network_nodal'])
    n_zonal = pypsa.Network(snakemake.input['network_zonal'])

    base_n_nodal = pypsa.Network(snakemake.input['network_base'])

    groupings = get_line_grouping(
        base_n_nodal.buses, 
        base_n_nodal.links.loc[n_nodal.links.carrier != 'interconnector', :],
        boundaries,
        anchors
        )

    args = (flow_constraints, boundaries, groupings)

    n_national_redispatch = pypsa.Network(snakemake.input['network_nodal'])
    n_zonal_redispatch = pypsa.Network(snakemake.input['network_nodal'])

    assert n_nodal.lines.empty, 'Current setup is for full DC approximation.'

    insert_flow_constraints(n_national_redispatch, *args, future_boundary_additions=future_boundary_additions, model_name='national balancing')
    insert_flow_constraints(n_nodal, *args, future_boundary_additions=future_boundary_additions, model_name='nodal wholesale')
    insert_flow_constraints(n_zonal, *args, future_boundary_additions=future_boundary_additions, model_name='zonal wholesale')
    insert_flow_constraints(n_zonal_redispatch, *args, future_boundary_additions=future_boundary_additions, model_name='zonal redispatch')

    # RNP models
    # rnp1: IC have nodal price signal
    # rnp2: batteries have nodal price signal
    # rnp3: both have nodal price signal
    # (no redispatch model needed, balancing volume is determined from the difference
    # between rnpx to n_nodal)

    rnp1 = n_national.copy()
    rnp2 = n_national.copy()
    rnp3 = n_national.copy()

    #################### National market ####################

    print('\n\nstarting national wholesale model\n\n')
    status, _ = n_national.optimize(solver_name=solver_name)
    n_national.export_to_netcdf(snakemake.output['network_national'])

    model_execution_overview.append(
        ('national wholesale', status, '-', '-') 
    )

    freeze_battery_commitments(n_national, n_national_redispatch)

    if snakemake.wildcards.ic == 'flex':
        logger.info('Freezing interconnector commitments')
        freeze_interconnector_commitments(n_national, n_national_redispatch)

    status, _ = n_national_redispatch.optimize(solver_name=solver_name)
    balancing_volume = get_bidding_volume(n_national, n_national_redispatch).sum()
    n_national_redispatch.export_to_netcdf(snakemake.output['network_national_redispatch'])  

    model_execution_overview.append(
        (
            'national redispatch',
            status,
            str(np.around(calibration_factor, decimals=2)),
            f'{balancing_volume*1e-3:.2f}'
        ) 
    )

    #################### Nodal market ####################

    status, relaxation_factor = safe_solve(n_nodal, calibration_factor)

    assert status == 'ok', f'Nodal wholesale model infeasible. Applied relax factor {calibration_factor:.2f}'

    model_execution_overview.append(
        (
            'nodal wholesale',
            status,
            str(np.around(calibration_factor, decimals=2)),
            '0.00'
        ) 
    )

    n_nodal.export_to_netcdf(snakemake.output['network_nodal'])

    #################### RNP models ####################

    freeze_interconnector_commitments(n_nodal, rnp1)

    freeze_battery_commitments(n_nodal, rnp2)

    freeze_interconnector_commitments(n_nodal, rnp3)
    freeze_battery_commitments(n_nodal, rnp3)

    nice_names = {
        'rnp1': 'ICs nodal price signal',
        'rnp2': 'Batteries nodal price signal',
        'rnp3': 'Both nodal price signal',
    }

    for model in ['rnp1', 'rnp2', 'rnp3']:

        n_rnp = globals()[model]
        status, relaxation_factor = safe_solve(n_rnp, calibration_factor)
        # balancing_volume = get_bidding_volume(n_rnp, n_nodal).sum()

        assert status == 'ok', f'{nice_names[model]} model infeasible. Applied relax factor {calibration_factor:.2f}'

        globals()[model].export_to_netcdf(snakemake.output[f'network_{model}'])

        model_execution_overview.append(
            (
                nice_names[model],
                status,
                str(np.around(calibration_factor, decimals=2)),
                '-'
                # f'{balancing_volume*1e-3:.2f}'
            ) 
        )

    #################### Zonal market ####################

    status, relaxation_factor = safe_solve(n_zonal, calibration_factor)

    # status, _ = n_zonal.optimize()

    # assert status == 'ok', f'Zonal wholesale model infeasible. Applied relax factor {relaxation_factor:.2f}'
    assert status == 'ok', f'Zonal wholesale model infeasible. Applied relax factor {calibration_factor:.2f}'

    n_zonal.export_to_netcdf(snakemake.output['network_zonal'])

    model_execution_overview.append(
        (
            'zonal wholesale',
            status,
            str(np.around(calibration_factor, decimals=2)),
            '-'
        ) 
    )

    freeze_battery_commitments(n_zonal, n_zonal_redispatch)
    if snakemake.wildcards.ic == 'flex':
        freeze_interconnector_commitments(n_zonal, n_zonal_redispatch)

    # status, relaxation_factor = safe_solve(n_zonal_redispatch) # old way of doing it
    # relax_line_capacities(n_zonal_redispatch, relaxation_factor) # new way of doing it
    # status, _ = n_zonal_redispatch.optimize(solver_name=solver_name)

    status, relaxation_factor = safe_solve(n_zonal_redispatch, calibration_factor)

    assert status == 'ok', f'Zonal redispatch model infeasible. Applied relax factor {calibration_factor:.2f}'
    n_zonal_redispatch.export_to_netcdf(snakemake.output['network_zonal_redispatch'])  

    balancing_volume = get_bidding_volume(n_zonal, n_zonal_redispatch).sum()
    logger.info(f'Zonal balancing volume: {balancing_volume*1e-3:.2f} GWh')

    model_execution_overview.append(
        (
            'zonal redispatch',
            status,
            str(np.around(calibration_factor, decimals=2)),
            f'{balancing_volume*1e-3:.2f}'
        ) 
    )


    print('')
    print((title := 'Model Execution Overview'))
    print("-" * len(title) + '\n')
    print(
        tabulate(
            model_execution_overview,
            headers=['Model', 'Status', 'Factor', 'Balancing Volume (GWh)'],
            tablefmt='pretty'
        )
    )
    print('')
