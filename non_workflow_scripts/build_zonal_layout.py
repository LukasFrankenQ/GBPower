# SPDX-FileCopyrightText: : 2024 Lukas Franken
#
# SPDX-License-Identifier: MIT

"""
Groups the 20 zones from the eso zonal layout into 6 regions.
The original 20 zones are sourced from 
https://www.neso.energy/publications/electricity-ten-year-statement-etys
These regions are chosen such that boundaries between zones are
reflective of the boundaries on which nationalGrid makes
boundary flow and limits available for most time steps.

These are (from north to south)
- SSE-SP
- SCOTEX
- SSHARN
- FLOWSTH
- SEIMP

"""

import logging 

logger = logging.getLogger()

import sys
import geopandas as gpd
from pathlib import Path
from shapely.ops import unary_union

sys.path.append(str(Path.cwd() / 'scripts'))
from _helpers import configure_logging


def remove_overlaps(geoseries):
    """
    Adjusts a GeoSeries of polygons to ensure they do not overlap.
    
    Parameters:
    - geoseries: A GeoSeries containing polygon geometries.
    
    Returns:
    - A new GeoSeries with non-overlapping polygons.
    """
    non_overlapping_polygons = []
    assigned_areas = None

    for polygon in geoseries:
        if not polygon.is_valid:
            polygon = polygon.buffer(0)

        if assigned_areas is None:
            new_polygon = polygon
            assigned_areas = polygon
        else:
            new_polygon = polygon.difference(assigned_areas)

            assigned_areas = unary_union([assigned_areas, polygon])

        non_overlapping_polygons.append(new_polygon)

    return gpd.GeoSeries(non_overlapping_polygons, index=geoseries.index, crs=geoseries.crs)


if __name__ == '__main__':

    configure_logging(snakemake)

    logger.info('Building zonal layout')

    zone_grouper = {
        'Northern Scotland': ['GB0 Z1_1', 'GB0 Z1_2', 'GB0 Z1_3', 'GB0 Z1_4', 'GB0 Z2'],
        'Scotland Central Belt': ['GB0 Z3', 'GB0 Z4', 'GB0 Z5', 'GB0 Z6'],
        'Northern England': ['GB0 Z7'],
        'Midlands': ['GB0 Z8', 'GB0 Z9', 'GB0 Z10', 'GB0 Z11'],
        'Southern Belt': ['GB0 Z12', 'GB0 Z14', 'GB0 Z13'],
        'Southern England': ['GB0 Z17', 'GB0 Z16', 'GB0 Z15'],
    }

    eso_zones = gpd.read_file(snakemake.input['eso_zones'])

    new_zones = gpd.GeoDataFrame(index=[], geometry=[], crs=eso_zones.crs)

    for key, zones in zone_grouper.items():
        new_zones.loc[key, 'geometry'] = eso_zones.set_index('name').loc[zones].union_all()

    # Remove weird numerical artifacts and ensure no overlaps
    new_zones.loc[['Southern Belt']] = new_zones.loc[['Southern Belt']].buffer(0.01)
    new_zones['geometry'] = remove_overlaps(new_zones['geometry'])

    new_zones.index.name = 'name'
    new_zones.to_file(snakemake.output['zonal_layout'])
