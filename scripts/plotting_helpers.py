import sys
import yaml
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from collections import OrderedDict
from matplotlib.patches import Patch


script_path = Path(__file__).resolve()
sys.path.append(str(script_path.parent.parent))

from data.interconnection_helpers import interconnection_countries

with open(script_path.parent.parent / 'config.yaml', 'r') as f:

    config = yaml.safe_load(f)
    tech_colors = config['tech_colors']
    nice_names = config['nice_names']


def plot_merit_order(n, period):
    """
    Plots a merit order curve for electricity supply based on the provided dataframe.

    Parameters:
    - df: pandas DataFrame with columns 'marginal_cost', 'p_nom', and 'carrier'.
    - period: Settlement period for which to plot the merit order.

    Returns:
    - A matplotlib plot displaying the merit order.
    """

    ints_marginal_cost = []
    ints_marginal_cap = []

    for ic, country in interconnection_countries.items():
        ints_marginal_cost.append(
            n.generators_t.marginal_cost[country.lower() + '_local_market'].iloc[period]
            )
        ints_marginal_cap.append(n.links.p_nom.loc[ic])

    df = pd.concat((
        n.generators[['marginal_cost', 'p_nom', 'carrier']],
        n.storage_units[['marginal_cost', 'p_nom', 'carrier']],
        pd.DataFrame({
            "marginal_cost": ints_marginal_cost,
            'p_nom': ints_marginal_cap,
            'carrier': ['interconnector'] * len(ints_marginal_cost)
            }, index=list(interconnection_countries))
    ))

    df = df.loc[df.carrier.isin(list(tech_colors))]

    intermittents = n.generators_t.p_max_pu.columns.intersection(df.index)
    df.loc[intermittents, "p_nom"] *= n.generators_t.p_max_pu[intermittents].iloc[period]

    df_sorted = df.sort_values(by=['marginal_cost', 'carrier']).reset_index(drop=True)
    df_sorted['p_nom'] = df_sorted['p_nom'] / 1e3  # Convert capacity to GW

    df_sorted['cum_capacity'] = df_sorted['p_nom'].cumsum()
    df_sorted['x_start'] = df_sorted['cum_capacity'] - df_sorted['p_nom']

    df_sorted['color'] = df_sorted['carrier'].map(tech_colors).fillna('grey')

    _, ax = plt.subplots(figsize=(8, 5))

    df_sorted.loc[df_sorted['marginal_cost'] == 0., 'marginal_cost'] += 6

    bars = ax.bar(
        x=df_sorted['x_start'],
        height=df_sorted['marginal_cost'],
        width=df_sorted['p_nom'],
        align='edge',
        color=df_sorted['color'],
        edgecolor='none',
        alpha=0.8,
    )

    for bar in bars:
        x_left = bar.get_x()
        x_right = x_left + bar.get_width()
        y_bottom = bar.get_y()
        y_top = y_bottom + bar.get_height()

        ax.plot([x_left, x_right], [y_top, y_top], color='black', linewidth=1)

        ax.plot([x_left, x_right], [y_bottom, y_bottom], color='black', linewidth=1)

    ax.set_xlabel('Available Generation Capacity (GW)')
    ax.set_ylabel('Marginal Price (£/MWh)')

    y_min = df_sorted['marginal_cost'].min() - 5
    y_max = df_sorted['marginal_cost'].max() + 5
    ax.set_ylim(y_min, y_max)

    carriers = df_sorted['carrier'].unique()
    handles = [
        Patch(facecolor=tech_colors.get(carrier, 'grey'), label=nice_names.get(carrier, carrier))
        for carrier in carriers
    ]
    
    # any time of day that is firmly within the day even during daylight saving time
    date = n.snapshots[10].strftime('%Y-%m-%d')

    ax.legend(handles=handles, title=f'{date} Period {period}', ncol=2)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()


def generation_mix_to_ax(ax, totals, tech_colors, nice_names):

    area_kwargs = dict(
        alpha=0.8,
        linewidth=0.5,
        edgecolor='black',
    )

    ax.stackplot(
        totals.columns,
        totals.clip(lower=0),
        labels=[nice_names[tech] for tech in totals.index],
        colors=[tech_colors[tech] for tech in totals.index],
        **area_kwargs
        )

    ax.stackplot(
        totals.columns,
        totals.clip(upper=0),
        labels=[nice_names[tech] for tech in totals.index],
        colors=[tech_colors[tech] for tech in totals.index],
        **area_kwargs
        )

    def get_unique_handles_labels(ax):
        handles, labels = ax.get_legend_handles_labels()
        unique = OrderedDict()
        for handle, label in zip(handles, labels):
            if label not in unique:
                unique[label] = handle
        return list(unique.values()), list(unique.keys())

    # Clean up the legend
    handles, labels = get_unique_handles_labels(ax)
    ax.legend(handles, labels)
    ax.set_xlim(totals.columns.min(), totals.columns.max())

    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('Generation [GW]')
