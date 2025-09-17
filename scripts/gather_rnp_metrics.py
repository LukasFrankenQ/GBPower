# SPDX-FileCopyrightText: : 2024 The PyPSA Authors, Lukas Franken
#
# SPDX-License-Identifier: MIT

import logging

logger = logging.getLogger(__name__)

import pypsa
import numpy as np
import pandas as pd

from _helpers import configure_logging


def get_energy_balance(n):

    ics = n.links.index[n.links.carrier == 'interconnector']

    nice_balance = pd.Series()
    nice_balance['imports'] = n.links_t.p0[ics].clip(lower=0).sum().mul(0.5e-3).sum()
    nice_balance['exports'] = n.links_t.p1[ics].clip(lower=0).sum().mul(0.5e-3).sum()

    carriers = ['biomass', 'fossil', 'nuclear', 'offwind', 'onwind', 'solar']
    available_carriers = [c for c in carriers if ('Generator', c) in n.statistics.energy_balance().index]
    generation = n.statistics.energy_balance().loc[idx['Generator', available_carriers]].mul(0.5e-3)
    generation.index = generation.index.get_level_values(1)
    nice_balance = pd.concat([nice_balance, generation])

    storage = n.statistics.energy_balance().loc[idx['StorageUnit']].mul(0.5e-3)
    storage.index = storage.index.get_level_values(0)
    nice_balance = pd.concat([nice_balance, storage])

    return nice_balance


if __name__ == '__main__':

    configure_logging(snakemake)

    idx = pd.IndexSlice

    nod = pypsa.Network(snakemake.input['network_nodal'])
    nat = pypsa.Network(snakemake.input['network_national'])
    rnp1 = pypsa.Network(snakemake.input['network_rnp1'])
    rnp2 = pypsa.Network(snakemake.input['network_rnp2'])


    mp = pd.concat([
        n.buses_t.marginal_price for n in [nat, nod, rnp1, rnp2]
    ], axis=1, keys=['national', 'nodal', 'rnp1', 'rnp2'])

    mp.to_csv(snakemake.output['marginal_prices'])

    storage_operation = pd.concat([
        nat.storage_units_t.p,
        nod.storage_units_t.p
    ], axis=1, keys=['national', 'nodal'])

    storage_operation.to_csv(snakemake.output['storage_operation'])

    ebs = pd.concat([
        get_energy_balance(n)
        for n in [nat, nod, rnp1, rnp2]
    ], axis=1, keys=['national', 'nodal', 'rnp1', 'rnp2'])

    ebs.to_csv(snakemake.output['energy_balance'])

    # Get interconnector flows from national and nodal models
    ics = nat.links.index[nat.links.carrier == 'interconnector']

    # Create MultiIndex columns for national and nodal flows
    flows = pd.concat([
        nat.links_t.p0[ics],
        nod.links_t.p0[ics]
    ], axis=1, keys=['national', 'nodal'])

    # Sort column levels to have model type as level 0 and interconnectors as level 1
    flows = flows.reorder_levels([0,1], axis=1)

    flows.to_csv(snakemake.output['interconnector_flows'])
