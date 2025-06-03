import numpy as np

#### gif parameters

gif_length = 20 # seconds
fps = 20

#### plot sizes

width_unit = 6
height_unit = 4

##### plotting constants
policy_colors = {
    'national': 'red',
    'no_ftr_zonal': 'rebeccapurple', # policy 1 in the paper
    'zonal': 'lime', # policy 2 in the paper
    'zonal_with_policy': 'deepskyblue', # policy 3 in the paper
}
# zonal_with_policy_color = 'magenta'
# zonal_with_policy_color = 'darkorange'


nice_names = {
    'wholesale': 'Wholesale Market',
    'wholesale buying': 'Electricity Procurement',
    'roc_payments': 'Renewable Obligation Certificates',
    'cfd_payments': 'Contracts for Differences',
    'congestion_rent': 'Intra-GB Congestion Rents',
    'offer_cost': 'Redispatch - Offers',
    'bid_cost': 'Redispatch - Bids',
}

color_dict = {
    'wholesale': '#F78C6B',
    'wholesale selling': '#F78C6B',
    'wholesale buying': '#c3763c',
    'roc_payments': '#EF476F',
    'cfd_payments': '#06D6A0',
    'congestion_rent': '#FFD166',
    'offer_cost': '#073B4C',
    'bid_cost': '#118AB2',
}

carrier_colors = {
    "onwind": "#7ac677",
    "offwind": "#6895dd",
    "hydro": "#0079c1",
    "biomass": "#dbc263",
    "fossil": "#f6986b",
    "nuclear": 'orange',
    "imports": "#dd75b0",
    "cascade": "#0079c1",
    "PHS": "#46caf0",
    "solar": "#f9d002",
    "battery": "darkred",
    "interconnector": "plum",
}

nice_carrier_names = {
    "onwind": "Onshore Wind",
    "offwind": "Offshore Wind",
    "hydro": "Hydro",
    "biomass": "Biomass",
    "fossil": "Fossil",
    "nuclear": "Nuclear",
    "imports": "Imports",
    "cascade": "Cascade",
    "PHS": "Pumped Hydro Storage",
    "solar": "Solar",
    "battery": "Battery",
    "interconnector": "Interconnector",
}

def stack_to_ax(df, ax, text_y_offset=0.2):

    s = df.sum()

    x_loc = '{}-{}'.format(df.index[0][0], df.index[0][1])

    pos = s.loc[s > 0]
    neg = s.loc[s < 0]

    pos_quants = [q for q in list(nice_names) if q in pos.index]
    neg_quants = [q for q in list(nice_names) if q in neg.index]

    pos = pos.loc[pos_quants]
    pos_bottom = [0] + pos.cumsum().tolist()[:-1]

    for (name, value), b in zip(pos.items(), pos_bottom):

        ax.bar(
            x_loc,
            bottom=b,
            height=value,
            label=nice_names[name],
            color=color_dict[name]
            )

    neg = neg.loc[neg_quants]
    neg_bottom = [0] + neg.cumsum().tolist()[:-1]

    for (name, value), b in zip(neg.items(), neg_bottom):
        # assert len(neg) == 1
        ax.bar(
            x_loc,
            bottom=0,
            height=value,
            label=nice_names[name],
            color=color_dict[name]
            )

    ax.text(
        x_loc, pos.sum() + text_y_offset,
        '{}'.format(np.around(s.sum(), decimals=1)),
        ha='center',
    )

    ax.scatter([x_loc], [s.sum()], color='w', zorder=10, edgecolor='k', s=50)


def plot_stacked_horizontal_bar(series, ax, right_tick=True):
    """
    Plot a horizontal stacked bar chart with negative values to the left and positive values to the right.
    
    Parameters:
    -----------
    series : pd.Series
        Series containing values to plot, with negative and positive values.
    ax : matplotlib.axes.Axes
        The axes to plot on.
    """
    # Separate positive and negative values
    pos_series = series[series > 0].copy()
    neg_series = series[series < 0].copy()
    
    # Sort values for better visualization
    pos_series = pos_series.sort_values(ascending=False)
    neg_series = neg_series.sort_values(ascending=True)
    
    # Calculate cumulative sums for stacking
    # pos_cumsum = pos_series.cumsum()
    # neg_cumsum = neg_series.cumsum()
    
    # Calculate the total negative value (for x-axis alignment)
    total_neg = neg_series.sum()
    
    # Plot negative values (to the left)
    left = total_neg
    for idx, val in neg_series.items():
        width = abs(val)
        left += width
        ax.barh(0, -width, left=left, height=0.5, label=idx, color=color_dict.get(idx, 'gray'))
    
    # Plot positive values (to the right)
    left = 0
    for idx, val in pos_series.items():
        ax.barh(0, val, left=left, height=0.5, label=idx, color=color_dict.get(idx, 'gray'))
        left += val
    
    # Set x-ticks with one tick at the boundary between negative and positive

    ax.scatter([total_neg + pos_series.sum()], [0], color='white', edgecolor='black', s=100, marker='o')

    # ax.set_xticks([0, total_neg, total_neg + pos_series.sum()])
    # ax.set_xticklabels(['0', f'{total_neg:.1f}', f'{total_neg + pos_series.sum():.1f}'])
    # Remove y-ticks and set labels
    xticks = ax.get_xticks()
    xticklabels = ax.get_xticklabels()

    # Add the key points to the x-ticks without creating text objects
    more_ticks = [0, total_neg]
    if right_tick:
        more_ticks.append(total_neg + pos_series.sum())
    xticks = np.unique(np.concatenate([xticks, more_ticks]))
    xticklabels = [f'{x:.1f}' for x in xticks]

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    ax.set_yticks([])
    ax.set_xlabel('(£bn)')
    # ax.axvline(x=total_neg, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlim(total_neg, pos_series.sum())
    ax.set_ylim(-0.25, 0.25)

    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    
    return ax