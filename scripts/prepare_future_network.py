# SPDX-FileCopyrightText: : 2025 Lukas Franken
#
# SPDX-License-Identifier: MIT

# adds infrastructure to the network based on the chosen date and the expected
# rollout between now and 2030

# for generation this is project-level wind based on CfD rounds 4,5,6
# for batteries this is a scaling of existing battery capacity according to expected rollout
# for interconnectors this is unit level addition of Greenlink, NeuConnect, etc.
# for onshore transmission these are individual projects that improve connectivity of highly producing areas
# for offshore transmission these are two north-south HVDC subsea cables 
# for load, this is a scaling of existing load according to expected rollout
# for European electricity prices, this is a scaling of existing prices according to prices anticipated by TYNDP 2024


import logging

logger = logging.getLogger(__name__)

import sys
import yaml
import pypsa
import numpy as np
import pandas as pd
import networkx as nx

from pyproj import Transformer

from _helpers import configure_logging
from _timeseries_helpers import transform_prices_to_target_mean


# Set up transformer from OSGB36 / British National Grid to WGS84 lat/lon
transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)

# Map OS two-letter codes to their 100 km square origins
# Source: Ordnance Survey National Grid
OS_SQUARES = {
    "SV": (0, 0), "SW": (100000, 0), "SX": (200000, 0), "SY": (300000, 0), "SZ": (400000, 0), "TV": (500000, 0),
    "SQ": (0, 100000), "SR": (100000, 100000), "SS": (200000, 100000), "ST": (300000, 100000), "SU": (400000, 100000), "TQ": (500000, 100000), "TR": (600000, 100000),
    "SL": (0, 200000), "SM": (100000, 200000), "SN": (200000, 200000), "SO": (300000, 200000), "SP": (400000, 200000), "TL": (500000, 200000), "TM": (600000, 200000),
    "SF": (0, 300000), "SG": (100000, 300000), "SH": (200000, 300000), "SJ": (300000, 300000), "SK": (400000, 300000), "TF": (500000, 300000), "TG": (600000, 300000),
    "SA": (0, 400000), "SB": (100000, 400000), "SC": (200000, 400000), "SD": (300000, 400000), "SE": (400000, 400000), "TA": (500000, 400000), "TB": (600000, 400000),
    "NV": (0, 500000), "NW": (100000, 500000), "NX": (200000, 500000), "NY": (300000, 500000), "NZ": (400000, 500000), "OV": (500000, 500000),
    "NQ": (0, 600000), "NR": (100000, 600000), "NS": (200000, 600000), "NT": (300000, 600000), "NU": (400000, 600000), "OQ": (500000, 600000), "OR": (600000, 600000),
    "NL": (0, 700000), "NM": (100000, 700000), "NN": (200000, 700000), "NO": (300000, 700000), "NP": (400000, 700000), "OL": (500000, 700000), "OM": (600000, 700000),
    "NF": (0, 800000), "NG": (100000, 800000), "NH": (200000, 800000), "NJ": (300000, 800000), "NK": (400000, 800000), "OF": (500000, 800000), "OG": (600000, 800000),
    "NA": (0, 900000), "NB": (100000, 900000), "NC": (200000, 900000), "ND": (300000, 900000), "NE": (400000, 900000), "OA": (500000, 900000), "OB": (600000, 900000),
    "HV": (0, 1000000), "HW": (100000, 1000000), "HX": (200000, 1000000), "HY": (300000, 1000000), "HZ": (400000, 1000000), "JV": (500000, 1000000)
}

def osgrid_to_lonlat(os_ref: str):
    os_ref = os_ref.strip().upper()
    letters = os_ref[:2]
    digits = os_ref[2:]
    
    if letters not in OS_SQUARES:
        raise ValueError(f"Unknown grid letters: {letters}")
    
    # Half-split digits for easting/northing
    n = len(digits) // 2
    easting_digits = digits[:n]
    northing_digits = digits[n:]
    
    # Scale factor (e.g., 3 digits → 100 m precision)
    scale = 10 ** (5 - n)
    
    easting = OS_SQUARES[letters][0] + int(easting_digits) * scale
    northing = OS_SQUARES[letters][1] + int(northing_digits) * scale
    
    lon, lat = transformer.transform(easting, northing)
    return lat, lon

# Extract lat/lon from OS grid references
def add_lat_lon(df):
    lats = []
    lons = []
    for loc in df['Project Location']:
        try:
            lat, lon = osgrid_to_lonlat(loc)
            lats.append(lat)
            lons.append(lon)
        except (ValueError, TypeError):
            lats.append(None)
            lons.append(None)

    df['Latitude'] = lats
    df['Longitude'] = lons

    return df


def adjust_to_future_year(
    n,
    year,
    current_fleet,
    future_wind_generators,
    future_assets,
    future_onshore_line_upgrades,
    base,
    europe_day_ahead_prices,
    country_coords,
    countries_cost_slopes,
    meritorder_slope_factors,
    ):

    n = n.copy()

    if year == 'off':
        return

    year = int(year)

    assert n.snapshots[0].year == 2025, 'model year has to be 2025 for future simulation'

    print('Warning! Check for changes in fossil, nuclear and biomass fleet')
    print('Warning! Not yet increased demand')

    buslocs = n.buses.loc[n.buses.country == 'GB', ['x', 'y']]

    ##### Onshore transmission lines #####

    for line, row in future_onshore_line_upgrades.iterrows():

        pt_start = row[['lon_start', 'lat_start']].values
        # Calculate distances from point to all bus locations
        distances = np.sqrt(
            (buslocs['x'] - pt_start[0])**2 + 
            (buslocs['y'] - pt_start[1])**2
        )
        bus0 = distances.idxmin()
        
        pt_end = row[['lon_end', 'lat_end']].values
        distances = np.sqrt(
            (buslocs['x'] - pt_end[0])**2 + 
            (buslocs['y'] - pt_end[1])**2
        )
        bus1 = distances.idxmin()

        if bus0 == bus1:
            continue

        local_lines = base.links.index[
            (base.links.bus0 == bus0) |
            (base.links.bus1 == bus1) |
            (base.links.bus0 == bus1) |
            (base.links.bus1 == bus0)
            ]

        nameplate_capacity = base.links.loc[local_lines, 'p_nom'].mean()
        calibrated_capacity = n.links.loc[local_lines, 'p_nom'].mean()

        calibration_factor = calibrated_capacity / nameplate_capacity
        
        cols = n.links_t.p_max_pu.columns
        p_max_pu = n.links_t.p_max_pu.loc[:, cols.intersection(local_lines)]
        p_min_pu = n.links_t.p_min_pu.loc[:, cols.intersection(local_lines)]

        if p_max_pu.empty:
            p_max_pu = 1
            p_min_pu = -1
        else:        
            p_max_pu = p_max_pu.mean(axis=1)
            p_min_pu = p_min_pu.mean(axis=1)

        n.add(
            'Link',
            line,
            bus0=bus0,
            bus1=bus1,
            p_nom=row['p_nom'] * calibration_factor,
            p_max_pu=p_max_pu,
            p_min_pu=p_min_pu,
            carrier='AC',
        )

    # Assets are treated differently; iterating over one-by-one assets first

    # ignored Offshore Permitted Reduction because the model works with physical notifications
    # ignored tidal, geothermal
    
    ##### Wind #####

    current_capacities = {
        'onwind': current_fleet.loc['Onshore wind', 'Installed_Capacity_GW'] * 1000,
        'offwind': current_fleet.loc['Offshore wind', 'Installed_Capacity_GW'] * 1000,
        'solar': current_fleet.loc['Solar PV', 'Installed_Capacity_GW'] * 1000,
    }

    network_capacities = {
        pypsa_carrier: pd.concat((
            n.generators.loc[gen, 'p_nom'] * (
                n.generators_t.p_max_pu[gen]
            if gen in n.generators_t.p_max_pu.columns
        else pd.Series(1, index=n.generators_t.p_max_pu.index)
            )
            for gen in n.generators.index[n.generators.carrier == pypsa_carrier]
        ), axis=1).sum(axis=1)
        for pypsa_carrier in ['onwind', 'offwind', 'solar']
    }

    # Pre-compute normalized fleet-availability profiles (peak = 1.0) per carrier.
    # The old formula `new_capacity = network_capacity × (size / current_capacity)` mis-scales:
    # for wind it over-states (network_capacity.max() > current_capacity), and for solar it
    # under-states by 100-1000× (model's existing solar fleet ~18 MW vs national ~17 GW).
    fleet_profiles = {}
    for pcarrier, nc in network_capacities.items():
        peak = nc.max()
        fleet_profiles[pcarrier] = (nc / peak) if peak > 0 else pd.Series(1.0, index=nc.index)

    for name, row in future_wind_generators.iterrows():

        print(f'Adding {name}')

        if year < row['pessimistic_delivery_year']:
            continue

        if row['pessimistic_delivery_year'] <= 2025:
            continue

        pcarrier = row['pypsa_carrier']
        pt = row[['Longitude', 'Latitude']].values

        # Calculate distances from point to all bus locations
        distances = np.sqrt(
            (buslocs['x'] - pt[0])**2 +
            (buslocs['y'] - pt[1])**2
        )
        closest_bus = distances.idxmin()

        # Direct p_nom from row; p_max_pu from the carrier's normalized fleet profile.
        p_nom = float(row['Size (MW)'])
        p_max_pu = fleet_profiles[pcarrier]

        n.add(
            'Generator',
            name,
            bus=closest_bus,
            carrier=pcarrier,
            p_nom=p_nom,
            p_max_pu=p_max_pu,
            marginal_cost=0,
        )

    ##### Offshore wind from the build pipeline (rows not already covered by AR registers / Phase B PN data)
    # AR registers (handled in the loop above) provide the wind farms that won CfD contracts.
    # The build pipeline CSV additionally contains projects that procured via different routes
    # (older AR rounds, negotiated CfDs, etc.). To avoid double-counting we skip rows whose name
    # is already represented in the AR registers or in the 2025 PN-curated BMU master.
    _ar_names_lc = {str(n).lower() for n in future_wind_generators.index}
    # Hardcoded skiplist for projects already in 2025 PN data (Phase B curation)
    _pn_phase_b_skip = {'moray west', 'neart na gaoithe (nng)'}

    ow_rows = future_assets.loc[
        (future_assets.Year > 2025) &
        (future_assets.Year <= year) &
        (future_assets.Department == 'Offshore wind')
    ]
    current_capacity_ow = current_capacities['offwind']
    network_capacity_ow = network_capacities['offwind']
    for _, row in ow_rows.iterrows():
        bname = row['Asset_Name']
        if pd.isna(row['Latitude']) or pd.isna(row['Longitude']):
            logger.info(f"Skipping build-pipeline offshore wind {bname}: no coordinates")
            continue
        name_lc = str(bname).lower()
        if name_lc in _pn_phase_b_skip:
            continue
        if any(arn in name_lc or name_lc in arn for arn in _ar_names_lc):
            continue
        pt = (row['Longitude'], row['Latitude'])
        distances = np.sqrt((buslocs['x'] - pt[0])**2 + (buslocs['y'] - pt[1])**2)
        closest_bus = distances.idxmin()
        new_capacity = network_capacity_ow * (row['Capacity_MW'] / current_capacity_ow)
        p_nom = new_capacity.max()
        if p_nom <= 0:
            continue
        p_max_pu = new_capacity / p_nom
        n.add(
            'Generator',
            f'{bname} (build_pipeline)',
            bus=closest_bus,
            carrier='offwind',
            p_nom=p_nom,
            p_max_pu=p_max_pu,
            marginal_cost=0,
        )
        logger.info(f"Added build-pipeline offshore wind {bname}: {row['Capacity_MW']:.0f} MW at bus {closest_bus}")

    ##### Renewable + battery capacity expansion to CP30 / Beyond-2030 targets
    # The AR4-7 register additions only reach delivery years 2028-2030. To track NESO
    # Clean Power 2030 + CCC 7CB Beyond 2030 trajectories beyond that, scale each
    # carrier's fleet to its year-target. Total-system CP30/CCC numbers are multiplied
    # by the transmission share (since GBPower only models transmission-connected units).
    #
    # Sources:
    #   - NESO Clean Power 2030 Annex 1 (Dec 2024) — 2030 mid-point targets
    #   - CCC 7th Carbon Budget Balanced Pathway (Feb 2025) — 2040 anchors
    #   - DESNZ/NESO Connections Reform Annex (Apr 2025) — transmission share
    RENEWABLE_TARGETS_MW = {
        'offwind':  {2024: 14_800, 2030: 46_500, 2035: 67_000, 2040: 88_000},
        'onwind':   {2024: 14_200, 2030: 28_000, 2035: 30_000, 2040: 32_000},
        'solar':    {2024: 17_200, 2030: 46_000, 2035: 64_000, 2040: 82_000},
        'battery':  {2024:  4_700, 2030: 25_000, 2035: 30_000, 2040: 35_000},
    }
    T_SHARE = {  # transmission-connected fraction (linear 2024→2030, flat after)
        'offwind':  {2024: 1.00, 2030: 1.00},
        'onwind':   {2024: 0.70, 2030: 0.55},
        'solar':    {2024: 0.10, 2030: 0.23},
        'battery':  {2024: 0.30, 2030: 0.59},
    }

    def _interp(d, yr):
        yrs = sorted(d.keys())
        if yr <= yrs[0]: return d[yrs[0]]
        if yr >= yrs[-1]: return d[yrs[-1]]
        for i in range(len(yrs) - 1):
            if yrs[i] <= yr <= yrs[i+1]:
                f = (yr - yrs[i]) / (yrs[i+1] - yrs[i])
                return d[yrs[i]] + f * (d[yrs[i+1]] - d[yrs[i]])
        return d[yrs[-1]]

    # NOTE: uniform p_nom scaling of wind/solar fleets is disabled.
    # Tried earlier: `n.generators.loc[gens_c, 'p_nom'] *= factor` to hit CP30 targets.
    # Result: for solar especially (only 2 transmission BMUs ~18 MW) the scaling factor
    # was ~590x, concentrating ~10 GW solar at 2 buses with marginal_cost ~ 0.
    # That made the LP feasible but degenerate — all bus prices collapsed to 0.
    # PyPSA's netcdf export then skipped marginal_price (it equals the schema default 0.0),
    # silently breaking downstream postprocessing. Re-enable only with a careful spatial
    # spread (e.g. distribute new capacity across many buses with non-zero marginal_cost).
    # For now the AR4-7 + build-pipeline additions are the only source of new wind/solar.
    for carrier in ('offwind', 'onwind', 'solar'):
        gens_c = n.generators.index[n.generators.carrier == carrier]
        current_T = n.generators.loc[gens_c, 'p_nom'].sum()
        target_total = _interp(RENEWABLE_TARGETS_MW[carrier], year)
        target_T = target_total * _interp(T_SHARE[carrier], year)
        logger.info(f"{carrier} {year}: current {current_T:.0f} MW vs CP30-adjusted target {target_T:.0f} MW (gap {target_T - current_T:+.0f} MW, scaling disabled)")

    # Batteries: same target-based approach (replaces the old build-pipeline plateau)
    target_batt_total = _interp(RENEWABLE_TARGETS_MW['battery'], year)
    target_batt_T = target_batt_total * _interp(T_SHARE['battery'], year)
    batt_idx = n.storage_units.index[n.storage_units.carrier == 'battery']
    current_batt_T = n.storage_units.loc[batt_idx, 'p_nom'].sum()
    if current_batt_T > 0 and target_batt_T > current_batt_T:
        factor = target_batt_T / current_batt_T
        n.storage_units.loc[batt_idx, 'p_nom'] *= factor
        # also scale energy capacity proportionally to keep duration unchanged
        if 'max_hours' in n.storage_units.columns:
            pass  # duration preserved automatically via p_nom × max_hours
        logger.info(f"battery: scaled fleet {current_batt_T:.0f} → {target_batt_T:.0f} MW (factor {factor:.2f})")

    # Hinkley Point C: 2 EPR reactors, 1,630 MW each, commissioning 2030 + 2031 per EDF Feb 2026.
    # Located at the existing Hinkley site (51.21°N, -3.13°W).
    HPC_REACTORS = [
        ('Hinkley Point C-1', 2030, 1630, 51.21, -3.13),
        ('Hinkley Point C-2', 2031, 1630, 51.21, -3.13),
    ]
    for hpc_name, hpc_year, hpc_mw, hpc_lat, hpc_lon in HPC_REACTORS:
        if year >= hpc_year and hpc_name not in n.generators.index:
            distances = np.sqrt((buslocs['x'] - hpc_lon)**2 + (buslocs['y'] - hpc_lat)**2)
            closest_bus = distances.idxmin()
            n.add('Generator', hpc_name, bus=closest_bus, carrier='nuclear',
                  p_nom=hpc_mw, p_max_pu=0.85, marginal_cost=10.0)
            logger.info(f"Added {hpc_name}: {hpc_mw} MW at bus {closest_bus}")


    ##### Offshore transmission lines (all of which are HVDC)

    # line capacities are scaled such that the model achieves realistic constraint management
    # therefore adding the new line at nameplate capacity will overestimate capacity
    # we therefore scale the nameplate capacity by considering the Western HVDC Link (in the model '8009')
    # the factor is its power capacity in the model divided by its nameplate capacity (2,250 MW)

    if '8009' in n.links.index:

        scaling_factor = n.links.loc['8009', 'p_nom'] / 2250

        new_assets = future_assets.loc[
            (future_assets.Year > 2025) &
            (future_assets.Year <= year) &
            (future_assets.Department == 'Transmission')
        ]

        for _, row in new_assets.iterrows():

            print('Adding Transmission Line: ', row.Asset_Name)

            kwargs = {
                'p_nom': row['Capacity_MW'] * scaling_factor,
                'type': 'HVDC underwater cable',
                'carrier': 'DC',
            }

            if '8009' in n.links_t.p_max_pu.columns:
                kwargs['p_max_pu'] = n.links_t.p_max_pu['8009']
                kwargs['p_min_pu'] = n.links_t.p_min_pu['8009']
            
            pt_start = row[['Start_Longitude', 'Start_Latitude']].values
            # Calculate distances from point to all bus locations
            distances = np.sqrt(
                (buslocs['x'] - pt_start[0])**2 + 
                (buslocs['y'] - pt_start[1])**2
            )
            bus0 = distances.idxmin()

            if bus0 == bus1:
                continue
            
            pt_end = row[['End_Longitude', 'End_Latitude']].values
            distances = np.sqrt(
                (buslocs['x'] - pt_end[0])**2 + 
                (buslocs['y'] - pt_end[1])**2
            )
            bus1 = distances.idxmin()

            n.add(
                'Link',
                row.Asset_Name,
                bus0=bus0,
                bus1=bus1,
                **kwargs
            )

    #### Interconnectors

    ic_countries = {
        'NeuConnect': 'Germany',
        'LionLink': 'Netherlands',
        'Greenlink': 'Ireland',
    }

    generator_steps = 20
    ic_subset = future_assets.loc[
        (future_assets.Year > 2025) &
        (future_assets.Year <= year) &
        (future_assets.Department == 'Interconnector')
        ]

    for _, row in ic_subset.iterrows():

        name = row['Asset_Name'].split(' ')[0]
        logger.info(f"Adding interconnector {name}.")
        country = ic_countries[name]

        start = np.array([row.Start_Longitude, row.Start_Latitude])

        distances = np.sqrt(
            (buslocs['x'] - start[0])**2 + 
            (buslocs['y'] - start[1])**2
        )
        bus1 = distances.idxmin()

        if country in n.buses.index:
            other_country_ic = n.links.index[n.links.bus0 == country][0]

            # relative to existing capacity, how much is added by the new interconnector
            country_gen = n.generators.index[n.generators.bus == country]
            country_load = n.loads.index[n.loads.bus == country] 
            country_ic_cap = n.generators.loc[country_gen, 'p_nom'].sum() / 2

            scale_factor = (row['Capacity_MW'] + country_ic_cap) / country_ic_cap

            n.loads.loc[country_load, 'p_set'] *= scale_factor

            n.generators.loc[country_gen, 'p_nom'] *= scale_factor

            mcs = n.generators_t['marginal_cost'].loc[:, country_gen]
            avg_mc = mcs.mean(axis=1)

            n.generators_t['marginal_cost'].loc[:, country_gen] += (mcs.subtract(avg_mc, axis=0)) * scale_factor

            n.add(
                'Link',
                name,
                bus0=country,
                bus1=bus1,
                carrier='interconnector',
                efficiency=0.99,
                p_nom=row['Capacity_MW'],
                p_min_pu=-1.,
                p_max_pu=1.,
                ramp_limit_up=n.links.loc[other_country_ic, 'ramp_limit_up'],
                ramp_limit_down=n.links.loc[other_country_ic, 'ramp_limit_down'],
            )

        else: # this only triggers for NeuConnect, a GB-Germany interconnector

            country_lon = country_coords[country][0]
            country_lat = country_coords[country][1]

            n.add(
                'Bus',
                country,
                carrier='electricity',
                x=country_coords[country][0],
                y=country_coords[country][1],
                country=country,
                )

            m0 = europe_day_ahead_prices.loc[:, country]
            m0.index = n.snapshots
            print('warning! simplified slope conversion to £!')
            s = countries_cost_slopes[country] / 1000 / 1.15 # convert to £/MW

            month = n.snapshots[0].strftime('%Y-%m')
            s *= meritorder_slope_factors.loc[month, 'factor_relative_to_2019']

            ic_p_nom = row['Capacity_MW']
            lower_cost_boundary = - s * ic_p_nom
            upper_cost_boundary = s * ic_p_nom

            cost_steps = np.linspace(
                lower_cost_boundary,
                upper_cost_boundary,
                generator_steps).tolist()

            step_capacity = ic_p_nom / generator_steps * 2

            rru = n.links.loc[n.links.carrier == 'interconnector', 'ramp_limit_up'].mean()
            rrd = n.links.loc[n.links.carrier == 'interconnector', 'ramp_limit_down'].mean()

            for i, cost_step in enumerate(cost_steps):
                n.add(
                    "Generator",
                    country.lower() + '_local_market_' + str(i),
                    bus=country,
                    p_nom=step_capacity,
                    marginal_cost=m0 + cost_step,
                    carrier="local_market",
                )

            n.add(
                "Load",
                country.lower() + '_local_market',
                bus=country,
                p_set=ic_p_nom,
                carrier=country,
                )

            n.add(
                'Link',
                name,
                bus0=country,
                bus1=bus1,
                carrier='interconnector',
                efficiency=0.99,
                p_nom=row['Capacity_MW'],
                p_min_pu=-1,
                p_max_pu=1,
                ramp_limit_up=rru,
                ramp_limit_down=rrd,
            )


    return n



if __name__ == '__main__':

    configure_logging(snakemake)
    logger.warning("\n YET TO ADJUST EUROPE WHOLESALE PRICES FOR FUTURE YEAR\n")

    nodal = pypsa.Network(snakemake.input['network_nodal'])
    zonal = pypsa.Network(snakemake.input['network_zonal'])
    national = pypsa.Network(snakemake.input['network_national'])

    cfd_strike_prices = pd.read_csv(snakemake.input['cfd_strike_prices'], index_col=0, parse_dates=True)

    if int(snakemake.wildcards.day[:4]) < 2026:
        nodal.export_to_netcdf(snakemake.output['network_nodal'])
        zonal.export_to_netcdf(snakemake.output['network_zonal'])
        national.export_to_netcdf(snakemake.output['network_national'])
        cfd_strike_prices.to_csv(snakemake.output['cfd_strike_prices'])
        sys.exit()
    
    with open(snakemake.input['future_electricity_demand'], 'r') as f:
        future_loads = yaml.safe_load(f)

    europe_day_ahead_prices = pd.read_csv(snakemake.input['europe_day_ahead_prices'], index_col=0, parse_dates=True)

    meritorder_slope_factors = pd.read_csv(
        snakemake.input['meritorder_slope_factors'],
        index_col=0
        )
    countries_cost_slopes = snakemake.params['countries_cost_slopes']
    # fixes yaml related issue; NO for Norway translates to False in yaml
    if False in list(countries_cost_slopes.keys()):
        countries_cost_slopes['NO'] = countries_cost_slopes[False]
        del countries_cost_slopes[False]

    # Steepen the foreign supply curves; see add_electricity.py for rationale (concentrated
    # IC injection has a larger local price impact than EuroMod's system-wide net-demand slope).
    ic_slope_factor = snakemake.params['ic_slope_factor']
    countries_cost_slopes = {k: v * ic_slope_factor for k, v in countries_cost_slopes.items()}

    with open(snakemake.input['interconnection_helpers'], 'r') as f:
        interconnection_helpers = yaml.safe_load(f)
        country_coords = interconnection_helpers['country_coords']

    load_scaling_factor = (
        future_loads[f'{snakemake.wildcards.day[:4]}_neso'] /
        future_loads['2025_neso']
    )

    current_fleet = pd.read_csv(
        snakemake.input['fleet_2024'],
        index_col=0
    )

    future_system_additions = pd.read_csv(
        snakemake.input['future_system_additions']
    )

    # preprocess CfD results from AR4, AR5, AR6

    # csv version of the main table from (reasoning for focus on wind is given in the paper)
    # https://assets.publishing.service.gov.uk/media/65b1463d160765000d18f834/contracts-for-difference-cfd-allocation-round-4-results.pdf
    ar4 = pd.read_csv(snakemake.input['ar4_results'], index_col=0)
    # csv version of the main table from
    # https://assets.publishing.service.gov.uk/media/64fa0473fdc5d10014fce820/cfd-ar5-results.pdf
    ar5 = pd.read_csv(snakemake.input['ar5_results'], index_col=0)
    # csv version of the main table from
    # https://assets.publishing.service.gov.uk/media/66d6ad7c6eb664e57141db4b/Contracts_for_Difference_Allocation_Round_6_results.pdf
    ar6 = pd.read_csv(snakemake.input['ar6_results'], index_col=0)
    # AR7 + AR7a (offshore + floating + onshore + solar + tidal); 201 projects.
    ar7 = pd.read_csv(snakemake.input['ar7_results'], index_col=0)

    ar4 = ar4.rename(columns={'Strike Price(£/MWh)': 'Strike Price (£/MWh)', 'Size(MW)': 'Size (MW)'})
    ar5 = ar5.rename(columns={'ProjectLocation': 'Project Location'})

    ar5 = add_lat_lon(ar5)
    ar6 = add_lat_lon(ar6)
    ar7 = add_lat_lon(ar7)

    future_wind_generators = pd.concat((ar4, ar5, ar6, ar7))

    pypsa_carrier_mapper = {
        'Onshore Wind (>5MW)': 'onwind',
        'Offshore Wind': 'offwind',
        'Floating Offshore Wind': 'offwind',
        'Remote Island Wind (RIW)': 'onwind',
        'Solar Photovoltaic': 'solar',
        # 'Tidal Stream' intentionally unmapped: AR7a has 4 projects totalling ~21 MW; negligible.
    }

    future_wind_generators['pypsa_carrier'] = future_wind_generators['Technology Type'].map(pypsa_carrier_mapper)
    # Drop rows with unmapped technology (Tidal Stream etc.)
    n_dropped = future_wind_generators['pypsa_carrier'].isna().sum()
    if n_dropped:
        logger.info(f"Dropping {n_dropped} CfD projects with unmapped technology (e.g. Tidal Stream).")
    future_wind_generators = future_wind_generators[future_wind_generators['pypsa_carrier'].notna()]

    future_wind_generators['optimistic_delivery_year'] = future_wind_generators['Delivery Year'].str.split('/').str[0].astype(int)
    future_wind_generators['pessimistic_delivery_year'] = future_wind_generators['Delivery Year'].str.split('/').str[1].astype(int) + 2000

    future_wind_generators.loc['Viking Wind Farm AR5', ['Longitude', 'Latitude']] = (-1.394, 60.264)

    for unit in future_wind_generators.index:
        cfd_strike_prices.loc[unit, :] = future_wind_generators.loc[unit, 'Strike Price (£/MWh)']

    base_n = pypsa.Network(
        snakemake.input['base_network']
    )

    future_onshore_line_upgrades = pd.read_csv(
        snakemake.input['future_transmission_additions'],
        index_col=0
    )

    for layout in ['nodal', 'zonal', 'national']:

        future_n = globals()[f'{layout}']

        gb_loads = future_n.loads.loc[future_n.loads.carrier == 'electricity'].index
        future_n.loads_t.p_set.loc[:, gb_loads] *= load_scaling_factor

        future_n = adjust_to_future_year(
            future_n,
            snakemake.wildcards.day[:4],
            current_fleet,
            future_wind_generators,
            future_system_additions,
            future_onshore_line_upgrades,
            base_n,
            europe_day_ahead_prices,
            country_coords,
            countries_cost_slopes,
            meritorder_slope_factors,
        )

        future_n.export_to_netcdf(snakemake.output[f'network_{layout}'])
    
    cfd_strike_prices.to_csv(snakemake.output['cfd_strike_prices'])
