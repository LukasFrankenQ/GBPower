# SPDX-FileCopyrightText: : 2024 Lukas Franken
#
# SPDX-License-Identifier: MIT

"""
Obtain the battery and PHS capacities from the physical notifications DataFrame. 
For charging and discharging, the maximum and minimum values are taken, respectively.
Energy capacity is assumed to equal the volume of maximum continuous charging.
"""

import logging

logger = logging.getLogger(__name__)

import sys
import pandas as pd
from tqdm import tqdm
from pathlib import Path

sys.path.append(str(Path.cwd() / 'scripts'))
from _helpers import configure_logging


def get_capacities(name, pns):
    """
    Obtain energy + power capacities from the physical notifications DataFrame. 
    For charging and discharging, the maximum and minimum values are taken, respectively.
    Energy capacity is assumed to equal the maximum volume of continuous charging.

    Parameters:
    name (str): The name of the battery unit.
    pns (pd.DataFrame): The physical notifications DataFrame containing power data.

    Returns:
    tuple: A tuple containing the power capacity (MW) and energy capacity (MWh).
    """

    power = pns[[name]].dropna()
    power.index = pd.to_datetime(power.index)

    power['date'] = power.index.date

    daily_charging_energies = []

    for _, group in power.groupby('date'):
        charging = group[name] < 0  # Boolean Series: True where charging

        charging_diff = charging.astype(int).diff()
        start_indices = charging_diff[charging_diff == 1].index
        stop_indices = charging_diff[charging_diff == -1].index

        if charging.iloc[0]:
            start_indices = start_indices.insert(0, charging.index[0])

        if charging.iloc[-1]:
            stop_indices = stop_indices.append(pd.Index([charging.index[-1]]))

        if len(start_indices) > 0 and len(stop_indices) > 0:

            if stop_indices[0] < start_indices[0]:
                stop_indices = stop_indices[1:]
            if len(start_indices) > len(stop_indices):
                start_indices = start_indices[:-1]

            charge_start = start_indices[0]
            charge_stop = stop_indices[0]

            charging_period = group.loc[charge_start:charge_stop]

            energy_charged = (charging_period[name] * 0.5).sum()  # Energy in MWh

            daily_charging_energies.append(-energy_charged)
        else:
            daily_charging_energies.append(0.)

    return (
        power[name].abs().max(), # Power capacity in MW
        max(daily_charging_energies) # Energy capacity in MWh
    )


if __name__ == '__main__':

    configure_logging(snakemake)

    logger.info('Estimating battery and PHS power and energy capacities.')

    pns = []
    for d in tqdm(snakemake.input):

        if 'bmus_prepared.csv' in d:
            continue

        try:
            pns.append(
                pd.read_csv(
                    Path(d) /
                    'physical_notifications.csv',
                    index_col=0,
                    parse_dates=True
                )
            )
        except FileNotFoundError:
            pass
    
    pns = pd.concat(pns).sort_index()

    bmus = pd.read_csv(snakemake.input['bmu_locations'], index_col=0)

    assets = bmus.loc[bmus['carrier'].isin(['battery', 'PHS'])].index

    power_caps = []
    energy_caps = []

    for name in tqdm(assets):
        power_cap, energy_cap = get_capacities(name, pns)

        power_caps.append(power_cap)
        energy_caps.append(energy_cap)

    capacities = pd.DataFrame({
            'power_cap[MW]': power_caps,
            'energy_cap[MWh]': energy_caps,
        },
        index=assets
    )

    capacities.to_csv(snakemake.output[0])
