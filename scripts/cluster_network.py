# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2017-2024 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT

# coding: utf-8
"""
Creates networks clustered to ``{cluster}`` number of zones with aggregated
buses, generators and transmission corridors.

Relevant Settings
-----------------

.. code:: yaml

    clustering:
      cluster_network:
      aggregation_strategies:
      focus_weights:

    solving:
        solver:
            name:

    lines:
        length_factor:

.. seealso::
    Documentation of the configuration file ``config/config.yaml`` at
    :ref:`toplevel_cf`, :ref:`renewable_cf`, :ref:`solving_cf`, :ref:`lines_cf`

Inputs
------

- ``resources/regions_onshore_elec_s{simpl}.geojson``: confer :ref:`simplify`
- ``resources/regions_offshore_elec_s{simpl}.geojson``: confer :ref:`simplify`
- ``resources/busmap_elec_s{simpl}.csv``: confer :ref:`simplify`
- ``networks/elec_s{simpl}.nc``: confer :ref:`simplify`
- ``data/custom_busmap_elec_s{simpl}_{clusters}.csv``: optional input

Outputs
-------

- ``resources/regions_onshore_elec_s{simpl}_{clusters}.geojson``:

    .. image:: img/regions_onshore_elec_s_X.png
        :scale: 33 %

- ``resources/regions_offshore_elec_s{simpl}_{clusters}.geojson``:

    .. image:: img/regions_offshore_elec_s_X.png
        :scale: 33 %

- ``resources/busmap_elec_s{simpl}_{clusters}.csv``: Mapping of buses from ``networks/elec_s{simpl}.nc`` to ``networks/elec_s{simpl}_{clusters}.nc``;
- ``resources/linemap_elec_s{simpl}_{clusters}.csv``: Mapping of lines from ``networks/elec_s{simpl}.nc`` to ``networks/elec_s{simpl}_{clusters}.nc``;
- ``networks/elec_s{simpl}_{clusters}.nc``:

    .. image:: img/elec_s_X.png
        :scale: 40  %

Description
-----------

.. note::

    **Why is clustering used both in** ``simplify_network`` **and** ``cluster_network`` **?**

        Consider for example a network ``networks/elec_s100_50.nc`` in which
        ``simplify_network`` clusters the network to 100 buses and in a second
        step ``cluster_network``` reduces it down to 50 buses.

        In preliminary tests, it turns out, that the principal effect of
        changing spatial resolution is actually only partially due to the
        transmission network. It is more important to differentiate between
        wind generators with higher capacity factors from those with lower
        capacity factors, i.e. to have a higher spatial resolution in the
        renewable generation than in the number of buses.

        The two-step clustering allows to study this effect by looking at
        networks like ``networks/elec_s100_50m.nc``. Note the additional
        ``m`` in the ``{cluster}`` wildcard. So in the example network
        there are still up to 100 different wind generators.

        In combination these two features allow you to study the spatial
        resolution of the transmission network separately from the
        spatial resolution of renewable generators.

    **Is it possible to run the model without the** ``simplify_network`` **rule?**

        No, the network clustering methods in the PyPSA module
        `pypsa.clustering.spatial <https://github.com/PyPSA/PyPSA/blob/master/pypsa/clustering/spatial.py>`_
        do not work reliably with multiple voltage levels and transformers.

.. tip::
    The rule :mod:`cluster_networks` runs
    for all ``scenario`` s in the configuration file
    the rule :mod:`cluster_network`.

Exemplary unsolved network clustered to 512 nodes:

.. image:: img/elec_s_512.png
    :scale: 40  %
    :align: center

Exemplary unsolved network clustered to 256 nodes:

.. image:: img/elec_s_256.png
    :scale: 40  %
    :align: center

Exemplary unsolved network clustered to 128 nodes:

.. image:: img/elec_s_128.png
    :scale: 40  %
    :align: center

Exemplary unsolved network clustered to 37 nodes:

.. image:: img/elec_s_37.png
    :scale: 40  %
    :align: center
"""


import logging

import sys
import yaml
import pypsa
import linopy
import numpy as np
import pandas as pd
import geopandas as gpd

from functools import reduce

from pypsa.clustering.spatial import get_clustering_from_busmap
from _helpers import configure_logging, update_p_nom_max, load_costs

logger = logging.getLogger(__name__)


def normed(x):
    return (x / x.sum()).fillna(0.0)


def get_feature_for_hac(n, buses_i=None, feature=None):
    if buses_i is None:
        buses_i = n.buses.index

    if feature is None:
        feature = "solar+onwind-time"

    carriers = feature.split("-")[0].split("+")
    if "offwind" in carriers:
        carriers.remove("offwind")
        carriers = np.append(
            carriers, n.generators.carrier.filter(like="offwind").unique()
        )

    if feature.split("-")[1] == "cap":
        feature_data = pd.DataFrame(index=buses_i, columns=carriers)
        for carrier in carriers:
            gen_i = n.generators.query("carrier == @carrier").index
            attach = (
                n.generators_t.p_max_pu[gen_i]
                .mean()
                .rename(index=n.generators.loc[gen_i].bus)
            )
            feature_data[carrier] = attach

    if feature.split("-")[1] == "time":
        feature_data = pd.DataFrame(columns=buses_i)
        for carrier in carriers:
            gen_i = n.generators.query("carrier == @carrier").index
            attach = n.generators_t.p_max_pu[gen_i].rename(
                columns=n.generators.loc[gen_i].bus
            )
            feature_data = pd.concat([feature_data, attach], axis=0)[buses_i]

        feature_data = feature_data.T
        # timestamp raises error in sklearn >= v1.2:
        feature_data.columns = feature_data.columns.astype(str)

    feature_data = feature_data.fillna(0)

    return feature_data


def distribute_clusters(n, n_clusters, focus_weights=None, solver_name="scip"):
    """
    Determine the number of clusters per country.
    """
    L = (
        n.loads_t.p_set.mean()
        .groupby(n.loads.bus)
        .sum()
        .groupby([n.buses.country, n.buses.sub_network])
        .sum()
        .pipe(normed)
    )

    N = n.buses.groupby(["country", "sub_network"]).size()

    assert (
        n_clusters >= len(N) and n_clusters <= N.sum()
    ), f"Number of clusters must be {len(N)} <= n_clusters <= {N.sum()} for this selection of countries."

    if isinstance(focus_weights, dict):
        total_focus = sum(list(focus_weights.values()))

        assert (
            total_focus <= 1.0
        ), "The sum of focus weights must be less than or equal to 1."

        for country, weight in focus_weights.items():
            L[country] = weight / len(L[country])

        remainder = [
            c not in focus_weights.keys() for c in L.index.get_level_values("country")
        ]
        L[remainder] = L.loc[remainder].pipe(normed) * (1 - total_focus)

        logger.warning("Using custom focus weights for determining number of clusters.")

    assert np.isclose(
        L.sum(), 1.0, rtol=1e-3
    ), f"Country weights L must sum up to 1.0 when distributing clusters. Is {L.sum()}."

    m = linopy.Model()
    clusters = m.add_variables(
        lower=1, upper=N, coords=[L.index], name="n", integer=True
    )
    m.add_constraints(clusters.sum() == n_clusters, name="tot")
    # leave out constant in objective (L * n_clusters) ** 2
    m.objective = (clusters * clusters - 2 * clusters * L * n_clusters).sum()
    if solver_name == "gurobi":
        logging.getLogger("gurobipy").propagate = False
    elif solver_name not in ["scip", "cplex"]:
        logger.info(
            f"The configured solver `{solver_name}` does not support quadratic objectives. Falling back to `scip`."
        )
        solver_name = "scip"
    m.solve(solver_name=solver_name)
    return m.solution["n"].to_series().astype(int)


def busmap_for_n_clusters(
    n,
    n_clusters,
    solver_name,
    focus_weights=None,
    algorithm="kmeans",
    feature=None,
    **algorithm_kwds,
):
    if algorithm == "kmeans":
        algorithm_kwds.setdefault("n_init", 1000)
        algorithm_kwds.setdefault("max_iter", 30000)
        algorithm_kwds.setdefault("tol", 1e-6)
        algorithm_kwds.setdefault("random_state", 0)

    def fix_country_assignment_for_hac(n):
        from scipy.sparse import csgraph

        # overwrite country of nodes that are disconnected from their country-topology
        for country in n.buses.country.unique():
            m = n[n.buses.country == country].copy()

            _, labels = csgraph.connected_components(
                m.adjacency_matrix(), directed=False
            )

            component = pd.Series(labels, index=m.buses.index)
            component_sizes = component.value_counts()

            if len(component_sizes) > 1:
                disconnected_bus = component[
                    component == component_sizes.index[-1]
                ].index[0]

                neighbor_bus = n.lines.query(
                    "bus0 == @disconnected_bus or bus1 == @disconnected_bus"
                ).iloc[0][["bus0", "bus1"]]
                new_country = list(set(n.buses.loc[neighbor_bus].country) - {country})[
                    0
                ]

                logger.info(
                    f"overwriting country `{country}` of bus `{disconnected_bus}` "
                    f"to new country `{new_country}`, because it is disconnected "
                    "from its initial inter-country transmission grid."
                )
                n.buses.at[disconnected_bus, "country"] = new_country
        return n


    if algorithm == "hac":
        feature = get_feature_for_hac(n, buses_i=n.buses.index, feature=feature)
        n = fix_country_assignment_for_hac(n)

    if (algorithm != "hac") and (feature is not None):
        logger.warning(
            f"Keyword argument feature is only valid for algorithm `hac`. "
            f"Given feature `{feature}` will be ignored."
        )

    n.determine_network_topology()

    n_clusters = distribute_clusters(
        n, n_clusters, focus_weights=focus_weights, solver_name=solver_name
    )


def clustering_for_n_clusters(
    n,
    n_clusters,
    custom_busmap=False,
    aggregate_carriers=None,
    line_length_factor=1.25,
    aggregation_strategies=dict(),
    solver_name="scip",
    algorithm="hac",
    feature=None,
    extended_link_costs=0,
    focus_weights=None,
):
    if not isinstance(custom_busmap, pd.Series):
        busmap = busmap_for_n_clusters(
            n, n_clusters, solver_name, focus_weights, algorithm, feature
        )
    else:
        busmap = custom_busmap

    line_strategies = aggregation_strategies.get("lines", dict())
    generator_strategies = aggregation_strategies.get("generators", dict())
    one_port_strategies = aggregation_strategies.get("one_ports", dict())

    clustering = get_clustering_from_busmap(
        n,
        busmap,
        aggregate_generators_weighted=True,
        aggregate_generators_carriers=aggregate_carriers,
        aggregate_one_ports=["Load"],
        line_length_factor=line_length_factor,
        line_strategies=line_strategies,
        generator_strategies=generator_strategies,
        one_port_strategies=one_port_strategies,
        scale_link_capital_costs=False,
    )

    if not n.links.empty:
        nc = clustering.network
        nc.links["underwater_fraction"] = (
            n.links.eval("underwater_fraction * length").div(nc.links.length).dropna()
        )
        nc.links["capital_cost"] = nc.links["capital_cost"].add(
            (nc.links.length - n.links.length)
            .clip(lower=0)
            .mul(extended_link_costs)
            .dropna(),
            fill_value=0,
        )

    return clustering


def cluster_regions(busmaps, input=None, output=None):
    busmap = reduce(lambda x, y: x.map(y), busmaps[1:], busmaps[0])

    for which in ("regions_onshore", "regions_offshore"):
        regions = gpd.read_file(getattr(input, which))
        regions = regions.reindex(columns=["name", "geometry"]).set_index("name")
        regions_c = regions.dissolve(busmap)
        regions_c.index.name = "name"
        regions_c = regions_c.reset_index()
        
        # if not which == "regions_onshore":
        # regions_c.to_file(getattr(output, which), driver="GeoJSON")


def make_busmap(n, zones):
    df = gpd.GeoDataFrame(
        index=n.buses.index,
        geometry=gpd.points_from_xy(n.buses.x, n.buses.y),
        crs="EPSG:4326",
        )

    # return gpd.sjoin(df, zones, how="left").rename(columns={"index_right": 0})[0]
    return df.sjoin(zones, how="left").rename(columns={"name": 0}).dropna()[0]


if __name__ == "__main__":

    configure_logging(snakemake)

    solver_name = 'highs'

    n = pypsa.Network(snakemake.input.network)

    params = {
        'cluster_network': {
            'algorithm': 'kmeans',
            'feature': 'solar+onwind-time',
            'exclude_carriers': list(set(
                n.generators.carrier.unique().tolist() +
                n.storage_units.carrier.unique().tolist() +
                n.links.carrier.unique().tolist()
                )),
            'consider_efficiency_classes': False,
            },
        'aggregation_strategies': {},
        'renewable_carriers': [
            'solar',
            'onwind',
            'offwind',
            'hydro',
            'cascade',
            'PHS',
            ],
        'conventional_carriers': [
            'nuclear',
            'fossil',
            ],
        'max_hours': {},
        'length_factor': 1.25,
        'costs': {
            "year": 2030,
            "version": "v0.8.0",
            "rooftop_share": 0.14,  # based on the potentials, assuming (0.1 kW/m2 and 10 m2/person)
            "social_discountrate": 0.02,
            "fill_values": {
                "FOM": 0,
                "VOM": 0,
                "efficiency": 1,
                "fuel": 0,
                "investment": 0,
                "lifetime": 25,
                "CO2 intensity": 0,
                "discount rate": 0.07
            },
            "marginal_cost": {
            },
            "emission_prices": {
                "enable": False,
                "co2": 0.0,
                "co2_monthly_prices": False
            }
        }
    }

    if snakemake.wildcards.layout == "nodal":
        n.export_to_netcdf(snakemake.output["network"])
        sys.exit()

    elif snakemake.wildcards.layout == "zonal":
        zonal_layout = gpd.read_file(snakemake.input.zonal_layout)

        buses = gpd.GeoDataFrame(
            n.buses,
            geometry=gpd.points_from_xy(n.buses.x, n.buses.y),
            crs=zonal_layout.crs
            ).sjoin(zonal_layout)

        with open(snakemake.input['transmission_boundaries']) as f:
            boundaries = yaml.safe_load(f)

        boundary_lines = reduce(lambda x, y: x + y, list(boundaries.values()))

        for link, row in n.links.iterrows():
            if row.carrier != 'AC':
                continue

            cond1 = buses.loc[row.bus0, 'name'] == buses.loc[row.bus1, 'name']
            cond2 = not link in boundary_lines

            if cond1 and cond2:
                n.links.loc[link, 'p_nom'] = np.inf

        n.export_to_netcdf(snakemake.output["network"])
        sys.exit()


    # remove integer outputs for compatibility with PyPSA v0.26.0
    n.generators.drop("n_mod", axis=1, inplace=True, errors="ignore")

    exclude_carriers = params['cluster_network']["exclude_carriers"]
    aggregate_carriers = set(n.generators.carrier) - set(exclude_carriers)
    conventional_carriers = set(params['conventional_carriers'])

    target_regions = gpd.read_file(snakemake.input.target_regions).set_index("name")[["geometry"]]
    custom_busmap = make_busmap(n, target_regions)

    custom_busmap.dropna(inplace=True)

    with open(snakemake.input['interconnection_helpers'], 'r') as f:
        country_names = list(yaml.safe_load(f)['country_coords'])

    network_countries = n.buses.index.intersection(country_names)
    custom_busmap.drop(
        custom_busmap.index.intersection(network_countries),
        inplace=True
        )

    custom_busmap = pd.concat((
        custom_busmap,
        pd.Series(
            network_countries,
            index=network_countries,
        )
    ))

    n_clusters = custom_busmap.nunique()

    if params['cluster_network'].get("consider_efficiency_classes", False):
        carriers = []
        for c in aggregate_carriers:
            gens = n.generators.query("carrier == @c")
            low = gens.efficiency.quantile(0.10)
            high = gens.efficiency.quantile(0.90)
            if low >= high:
                carriers += [c]
            else:
                labels = ["low", "medium", "high"]
                suffix = pd.cut(
                    gens.efficiency, bins=[0, low, high, 1], labels=labels
                ).astype(str)
                carriers += [f"{c} {label} efficiency" for label in labels]
                n.generators.update(
                    {"carrier": gens.carrier + " " + suffix + " efficiency"}
                )
        aggregate_carriers = carriers

    if n_clusters == len(n.buses):
        # Fast-path if no clustering is necessary
        busmap = n.buses.index.to_series()
        linemap = n.lines.index.to_series()
        clustering = pypsa.clustering.spatial.Clustering(
            n, busmap, linemap, linemap, pd.Series(dtype="O")
        )
    else:
        Nyears = n.snapshot_weightings.objective.sum() / 8760

        hvac_overhead_cost = load_costs(
            snakemake.input.tech_costs,
            params['costs'],
            params['max_hours'],
            Nyears,
        ).at["HVAC overhead", "capital_cost"]

        clustering = clustering_for_n_clusters(
            n,
            n_clusters,
            custom_busmap,
            aggregate_carriers,
            params['length_factor'],
            params['aggregation_strategies'],
            solver_name,
            params['cluster_network']["algorithm"],
            params['cluster_network']["feature"],
            hvac_overhead_cost,
        )

    update_p_nom_max(clustering.network)

    if params['cluster_network'].get("consider_efficiency_classes"):
        labels = [f" {label} efficiency" for label in ["low", "medium", "high"]]
        nc = clustering.network
        nc.generators["carrier"] = nc.generators.carrier.replace(labels, "", regex=True)

    clustering.network.meta = dict(
        snakemake.config, **dict(wildcards=dict(snakemake.wildcards))
    )
    clustering.network.export_to_netcdf(snakemake.output.network)