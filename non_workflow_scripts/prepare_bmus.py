# SPDX-FileCopyrightText: : 2024 Lukas Franken
#
# SPDX-License-Identifier: MIT

"""
Script to create the file 'data/preprocessed/prepared_bmus.csv'.

Inserting some manual fixed to clean and enhanced data
on Balancing Mechanism Units

"""

import logging

logger = logging.getLogger(__name__)

import sys
import pandas as pd
import geopandas as gpd
from pathlib import Path

sys.path.append(str(Path.cwd() / 'scripts'))
from _helpers import configure_logging


manual_battery_locs = {
    'RSCRB-1': ('distributed', 'distributed'),
    'PNYCB-1': (51.709, -3.564),
    'ARNKB-2': (51.484, 0.344),
    'AG-HEL0DN': (56.166, -3.204),
    'WBURB-41': (53.364, -0.538),
    'WBURB-43': (53.364, -0.538),
    'CRSSB-1': (53.481, -2.959),
    'AG-HEL0CP': (56.324, -2.995),
    'AG-HLIM01': (56.941, -2.255),
    'AG-JSTK02': (51.231, -0.334),
    'NEWTB-1': (53.138, -1.327),
    'FFSE01': (55.016, -1.473),
    'BROAB-1': (51.413, 0.33),
    'BHOLB-1': (50.721, -1.998),
    'ZEN02A': (52.07, 0.591),
    'AG-GEDF01': (53.197, -2.254),
    'AG-GEDF02': (53.197, -2.254),
    'AG-HLIM02': (50.852, -1.194),
    'JAMBB-1': (56.341, -3.294),
    'BFSE01': (52.585, -1.22),
    'DOLLB-1': (51.6, 0.56),
    'AG-HLIM03': (51.616, -1.99),
    'AG-MSTK01': (53.791, -0.405),
    'PILLB-1': (53.796, -0.407),
    'PILLB-2': (53.796, -0.407),
    'SKELB-1': (53.553, -2.775),
    'BURWB-1': (52.279, 0.316),
    'AG-HEL00G': (53.787, -2.643),
    'AG-HLIM04': (51.598, -1.494),
    'AG-HLIM01': (51.598, -1.494),
    'KEMB-1': (51.364, 0.737),
    'AG-MFLX02': (53.398, -1.247),
    'BUSTB-1': (52.549, -2.02),
    'AG-HSTK02': (51.615, -1.962),
    'COVNB-1': (52.409, -1.508),
    'AG-HSTK01': (51.615, -1.962),
    'CLAYB-1': (51.552, 0.298),
    'CLAYB-2': (51.552, 0.298),
    'AG-ASTK05': (52.197, 0.126),
    'TOLLB-1': (51.735, -0.245),
    'ILMEB-1': (51.741, -0.892),
    'ARNKB-1': (52.61, -2.),
    'LITRB-1': (56.104, -3.33),
    'THMRB-1': (51.766, -0.894),
    'CUPAB-1': (56.316, -3.013),
    'AG-PFLX01': (56.412, -3.423),
    'AG-ZEN03J': (51.302, 0.477),
    'NFSE02': (55.907, -3.588),
    'FARNB-1': (51.21, -0.791),
    'AG-LLIM01': (51.516, -2.652),
    'BARNB-1': (53.558, -1.457),
}

onwind_farms = [
    'TMNCW-1',
    'BLARW-1',
    'BRDUW-1',
    'GRGRW-1',
    'CREAW-1',
    'CUMHW-1',
    'DALQW-1',
    'FAARW-1',
    'FAARW-2',
    'GLNKW-1',
    'KENNW-1',
    'KLGLW-1',
    'KYPEW-1',
    'SAKNW-1',
    'PNYCW-1',
    'SOKYW-1',
    ]

offwind_farms = [
    'BLLA-1',
    'BLLA-2',
]

carrier_mapper = {
    'offwind': ['offwind', 'floating wind'],
    'onwind': ['onwind'],
    'biomass': ['biomass'],
    'fossil': ['gas', 'gas-fired', 'gas turbine', 'CCGT', 'powerstation', 'CHP', 'coal', 'oil'],
    'demand_flex': ['supply', 'natural gas processing'],
    'PHS': ['PHS'],
    'battery': ['battery'],
    'solar': ['solar', 'solar power station', 'PV'],
    'nuclear': ['nuclear'],
    'hydro': ['cascade', 'dam', 'hydro', 'hydro-scheme'],
    'load': ['submarine power cable', 'steel mill'],
    'interconnector': ['interconnector', 'HVDC submarine'],
}

if __name__ == '__main__':

    configure_logging(snakemake)

    logger.info('Preparing BMUs, cleaning and enhancing data.')

    bmus = pd.read_csv(
            snakemake.input["bmus_raw"],
            index_col=0
        ).dropna()

    old_bmus = pd.read_csv(
            snakemake.input["bmus_locs"],
            index_col=0
        )
    supply = old_bmus.loc[old_bmus.carrier == 'supply'].index

    bmus = pd.concat([bmus] + [
        pd.DataFrame({
                'lon': ['distributed'] * len(supply),
                'lat': ['distributed'] * len(supply),
                'capacity': [0.] * len(supply),
                'carrier': ['supply'] * len(supply),
                }, index=supply)
    ])

    bmus.loc[bmus.carrier == 'supply']
    # bmus = bmus.loc[bmus['lon'] != 0.]

    bmus.loc[onwind_farms, 'carrier'] = 'onwind'
    bmus.loc[offwind_farms, 'carrier'] = 'offwind'

    for clean_name, old_carriers in carrier_mapper.items():
        bmus.loc[bmus['carrier'].isin(old_carriers), 'carrier'] = clean_name

    dflex_index = bmus.loc[bmus.carrier == 'demand_flex'].index
    bmus.loc[dflex_index, ['lat', 'lon']] = 'distributed'

    bmus = bmus[~bmus.index.duplicated(keep='first')]

    hydro = bmus.loc[bmus.carrier == 'hydro'].index

    cas = hydro[hydro.str.contains('CAS')].to_list() + ['LCSMH-1']
    non_cas = hydro[~hydro.str.contains('CAS')].to_list()

    bmus.loc[cas, 'carrier'] = 'cascade'

    missing_batteries = bmus.loc[(bmus.carrier == 'battery') & (bmus['lon'] == 0.)].index

    for name, loc in manual_battery_locs.items():
        bmus.loc[name, ['lat', 'lon']] = loc

    bmus.loc['RSCRB-1', 'carrier'] = 'demand_flex'

    clean = bmus
    clean.loc[clean.carrier == 'demand_flex']

    bmus.at['SOLUU-1', 'lon'] = str(float(bmus.at['SOLUU-1', 'lon']) + 0.01)

    bmus = bmus.loc[bmus['lon'] != 0]

    # assign network nodes
    regions_onshore = gpd.read_file(snakemake.input["regions_onshore"])
    regions_offshore = gpd.read_file(snakemake.input["regions_offshore"])

    mapper = bmus.loc[bmus['lon'] != 'distributed']

    bmu_mapper = gpd.GeoDataFrame(
            mapper[['carrier', 'lon', 'lat']],
            geometry=gpd.points_from_xy(x=mapper['lon'], y=mapper['lat']),
            index=mapper.index,
            crs="EPSG:4326"
            )

    joined_onshore = bmu_mapper.sjoin(regions_onshore)
    joined_offshore = bmu_mapper.sjoin(regions_offshore)

    joined = pd.concat((
        joined_onshore,
        joined_offshore,
        ))

    joined = (
        joined
        [~joined.index.duplicated(keep='first')]
        .rename(columns={'index_right': 'bus'})
        [['carrier', 'lon', 'lat', 'bus']]
    )

    joined = pd.concat((
        joined,
        pd.concat((
            (b := bmus.loc[bmus['lon'] == 'distributed']),
            pd.DataFrame({'bus': ["distributed"] * len(b)}, index=b.index)
        ), axis=1)
    ))

    joined = joined[['carrier', 'bus', 'lon', 'lat']]

    assert joined.index.is_unique, 'Index is not unique'
    assert joined.isna().sum().sum() == 0, 'There are missing values'

    joined.to_csv(snakemake.output[0])
