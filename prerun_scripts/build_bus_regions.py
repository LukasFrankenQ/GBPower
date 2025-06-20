# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2017-2024 The PyPSA-Eur Authors, Lukas Franken
#
# SPDX-License-Identifier: MIT
"""
Creates Voronoi shapes for each bus representing both onshore and offshore
regions.

Relevant Settings
-----------------

.. code:: yaml

    countries:

.. seealso::
    Documentation of the configuration file ``config/config.yaml`` at
    :ref:`toplevel_cf`

Inputs
------

- ``resources/country_shapes.geojson``: confer :ref:`shapes`
- ``resources/offshore_shapes.geojson``: confer :ref:`shapes`
- ``networks/base.nc``: confer :ref:`base`

Outputs
-------

- ``resources/regions_onshore.geojson``:

    .. image:: img/regions_onshore.png
        :scale: 33 %

- ``resources/regions_offshore.geojson``:

    .. image:: img/regions_offshore.png
        :scale: 33 %

Description
-----------
"""

import logging

import sys
import pypsa
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from scipy.spatial import Voronoi
from shapely.geometry import Polygon

sys.path.append(str(Path.cwd() / 'scripts'))
from _helpers import REGION_COLS, configure_logging

logger = logging.getLogger(__name__)


def voronoi_partition_pts(points, outline):
    """
    Compute the polygons of a voronoi partition of `points` within the polygon
    `outline`. Taken from
    https://github.com/FRESNA/vresutils/blob/master/vresutils/graph.py.

    Attributes
    ----------
    points : Nx2 - ndarray[dtype=float]
    outline : Polygon
    Returns
    -------
    polygons : N - ndarray[dtype=Polygon|MultiPolygon]
    """
    points = np.asarray(points)

    if len(points) == 1:
        polygons = [outline]
    else:
        xmin, ymin = np.amin(points, axis=0)
        xmax, ymax = np.amax(points, axis=0)
        xspan = xmax - xmin
        yspan = ymax - ymin

        # to avoid any network positions outside all Voronoi cells, append
        # the corners of a rectangle framing these points
        vor = Voronoi(
            np.vstack(
                (
                    points,
                    [
                        [xmin - 3.0 * xspan, ymin - 3.0 * yspan],
                        [xmin - 3.0 * xspan, ymax + 3.0 * yspan],
                        [xmax + 3.0 * xspan, ymin - 3.0 * yspan],
                        [xmax + 3.0 * xspan, ymax + 3.0 * yspan],
                    ],
                )
            )
        )

        polygons = []
        for i in range(len(points)):
            poly = Polygon(vor.vertices[vor.regions[vor.point_region[i]]])

            if not poly.is_valid:
                poly = poly.buffer(0)

            with np.errstate(invalid="ignore"):
                poly = poly.intersection(outline)

            polygons.append(poly)

    return polygons


if __name__ == "__main__":

    configure_logging(snakemake)

    countries = ["GB"]

    n = pypsa.Network(snakemake.input.base_network)

    country_shapes = gpd.read_file(snakemake.input.total_shape).set_index("name")[
        "geometry"
    ]
    offshore_shapes = gpd.read_file(snakemake.input.offshore_shapes)
    offshore_shapes = offshore_shapes.reindex(columns=REGION_COLS).set_index("name")[
        "geometry"
    ]

    onshore_regions = []
    offshore_regions = []

    for country in countries:
        c_b = n.buses.country == country

        onshore_shape = country_shapes[country]
        # onshore_locs = n.buses.loc[c_b & n.buses.substation_lv, ["x", "y"]]
        onshore_locs = n.buses.loc[c_b, ["x", "y"]]
        onshore_regions.append(
            gpd.GeoDataFrame(
                {
                    "name": onshore_locs.index,
                    "x": onshore_locs["x"],
                    "y": onshore_locs["y"],
                    "geometry": voronoi_partition_pts(
                        onshore_locs.values, onshore_shape
                    ),
                    "country": country,
                }
            )
        )

        if country not in offshore_shapes.index:
            continue
        offshore_shape = offshore_shapes[country]
        # offshore_locs = n.buses.loc[c_b & n.buses.substation_off, ["x", "y"]]
        offshore_locs = n.buses.loc[c_b, ["x", "y"]]
        offshore_regions_c = gpd.GeoDataFrame(
            {
                "name": offshore_locs.index,
                "x": offshore_locs["x"],
                "y": offshore_locs["y"],
                "geometry": voronoi_partition_pts(offshore_locs.values, offshore_shape),
                "country": country,
            }
        )
        offshore_regions_c = offshore_regions_c.loc[offshore_regions_c.area > 1e-2]
        offshore_regions.append(offshore_regions_c)

    pd.concat(onshore_regions, ignore_index=True).to_file(
        snakemake.output.regions_onshore
    )
    if offshore_regions:
        pd.concat(offshore_regions, ignore_index=True).to_file(
            snakemake.output.regions_offshore
        )
    else:
        offshore_shapes.to_frame().to_file(snakemake.output.regions_offshore)