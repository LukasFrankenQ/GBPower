"""
Auto-converted from notebooks/money_flow.ipynb.
Edits should target this file directly; the .ipynb is the source-of-truth for exploratory work only.
"""

from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent

# --- cell 0 ---
import sys
import pypsa
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from plotting_constants import nice_names, color_dict, stack_to_ax

sys.path.append(str(REPO_ROOT / 'scripts'))
from _helpers import classify_north_south

idx = pd.IndexSlice

# --- cell 1 ---
# day = '2022-01-03'
day = 'total'

# --- cell 2 ---
if day != 'total':
    cost_summary = pd.read_csv(
        REPO_ROOT / 'results' / f'{day}' / 'system_cost_summary_flex.csv', 
        index_col=[0,1],
        parse_dates=[0]
        )
else:
    cost_summary = pd.read_csv(
        REPO_ROOT / 'summaries' / 'total_summary_flex.csv', 
        index_col=[0,1],
        parse_dates=[0]
        )

index = cost_summary.index.get_level_values(0).unique()

cost_summary.head()

# --- cell 3 ---
# rev_path = str(REPO_ROOT / 'summaries' / 'total_summary_revenues_flex_{}.csv')
if day != 'total':
    rev_path = str(REPO_ROOT / 'results' / f'{day}' / 'bmu_revenues_flex_{}.csv')
    unit_scaling = 1e-6
else:
    rev_path = str(REPO_ROOT / 'summaries' / 'total_summary_revenues_flex_{}.csv')
    unit_scaling = 1e-9

rev_national = pd.read_csv(rev_path.format('national'), index_col=0, header=[0,1,2], parse_dates=True).loc[index].sum().mul(unit_scaling)
rev_zonal = pd.read_csv(rev_path.format('zonal'), index_col=0, header=[0,1,2], parse_dates=True).loc[index].sum().mul(unit_scaling)

# --- cell 4 ---
rev_national = pd.concat((
    rev_national.loc[idx[:, ['wind', 'disp', 'hydro', 'storage', 'nuclear'], :]],
    pd.Series(rev_national.loc[idx[:, 'intercon', :]].sum(), index=pd.MultiIndex.from_tuples([('total', 'interconnector', 'wholesale')]))
))
rev_zonal = pd.concat((
    rev_zonal.loc[idx[:, ['wind', 'disp', 'hydro', 'storage', 'nuclear'], :]],
    pd.Series(rev_zonal.loc[idx[:, 'intercon', :]].sum(), index=pd.MultiIndex.from_tuples([('total', 'interconnector', 'wholesale')]))
))

# --- cell 5 ---
if day != 'total':
    mp = pd.read_csv(
        REPO_ROOT / 'results' / f'{day}' / 'marginal_prices_flex.csv',
        index_col=0,
        parse_dates=[0],
        header=[0,1]
        )
else:
    mp = pd.read_csv(
        REPO_ROOT / 'summaries' / 'marginal_prices_summary_flex.csv',
        index_col=0,
        parse_dates=[0],
        header=[0,1]
        )

mp = mp.loc[index]

# --- cell 6 ---
idx = pd.IndexSlice
buses = mp.loc[:,idx['zonal',:]].columns.get_level_values(1)

# --- cell 7 ---
weights = pd.read_csv(REPO_ROOT / 'data' / 'prerun' / 'load_weights.csv', index_col=0)

weights = weights.loc[buses.astype(int)]
weights /= weights.sum()

# --- cell 8 ---
if day != 'total':
    load = pd.read_csv(
        REPO_ROOT / 'results' / f'{day}' / 'gb_total_load_flex.csv',
        index_col=0,
        parse_dates=[0]
    ).rename(columns={'0': 'load'})
else:
    load = pd.read_csv(
        REPO_ROOT / 'summaries' / 'total_gb_load.csv',
        index_col=0,
        parse_dates=[0]
    ).rename(columns={'0': 'load'})

load = pd.DataFrame(
    np.outer(load.values, weights.values),
    index=load.index,
    columns=weights.index.astype(str)
    ).loc[index]

load.head()

# --- cell 9 ---
network = pypsa.Network(
    # REPO_ROOT / 'results' / f'{day}' / 'network_flex_s_nodal.nc'
    REPO_ROOT / 'results' / '2024-03-24' / 'network_flex_s_nodal.nc'
)

buses = network.buses[['x', 'y']]
buses.loc[:, ['region']] = buses.apply(lambda x: classify_north_south(x['x'], x['y']), axis=1)

# --- cell 10 ---
north_loads = load.loc[:, buses.loc[buses.region == 'north'].index]
south_loads = load.loc[:, buses.loc[buses.region == 'south'].index.intersection(load.columns)]

print(north_loads.shape)
print(south_loads.shape)

# --- cell 11 ---
import warnings
warnings.filterwarnings('ignore')

def get_wholesale_costs(mp, loads, mode):

    idx = pd.IndexSlice

    if mode == 'national':
        return np.inner(
            loads.sum(axis=1).values,
            mp.loc[:,idx['national', 'GB']].values
            ) * 0.5
        
    elif mode == 'zonal':
        total = 0
        for col in loads.columns:
            total += np.inner(
                loads.loc[:,col].values,
                mp.loc[:,idx['zonal', col]].values
                )
        return total * 0.5


costs = pd.Series(0, # is in units M£
    index=pd.MultiIndex.from_product([
        ['wholesale'],
        ['north', 'south'],
        ['national', 'zonal']
        ]
    ).append(
        pd.MultiIndex.from_product([
            ['roc', 'cfd', 'bids', 'offers', 'cfd'],
            ['socialised'],
            ['national', 'zonal']
            ]
        )
    )
)

costs.loc[idx['wholesale', 'north', 'national']] = get_wholesale_costs(mp, north_loads, 'national') * unit_scaling
costs.loc[idx['wholesale', 'south', 'national']] = get_wholesale_costs(mp, south_loads, 'national') * unit_scaling
costs.loc[idx['wholesale', 'north', 'zonal']] = get_wholesale_costs(mp, north_loads, 'zonal') * unit_scaling
costs.loc[idx['wholesale', 'south', 'zonal']] = get_wholesale_costs(mp, south_loads, 'zonal') * unit_scaling

costs.loc[idx['roc', 'socialised', 'national']] = cost_summary.loc[idx[:, 'roc_payments'], 'national'].sum() * 1e6 * unit_scaling
costs.loc[idx['cfd', 'socialised', 'national']] = cost_summary.loc[idx[:, 'cfd_payments'], 'national'].sum() * 1e6 * unit_scaling
costs.loc[idx['bids', 'socialised', 'national']] = cost_summary.loc[idx[:, 'bid_cost'], 'national'].sum() * 1e6 * unit_scaling
costs.loc[idx['offers', 'socialised', 'national']] = cost_summary.loc[idx[:, 'offer_cost'], 'national'].sum() * 1e6 * unit_scaling
costs.loc[idx['congestion_rent', 'socialised', 'national']] = cost_summary.loc[idx[:, 'congestion_rent'], 'national'].sum() * 1e6 * unit_scaling

costs.loc[idx['roc', 'socialised', 'zonal']] = cost_summary.loc[idx[:, 'roc_payments'], 'zonal'].sum() * 1e6 * unit_scaling
costs.loc[idx['cfd', 'socialised', 'zonal']] = cost_summary.loc[idx[:, 'cfd_payments'], 'zonal'].sum() * 1e6 * unit_scaling
costs.loc[idx['bids', 'socialised', 'zonal']] = cost_summary.loc[idx[:, 'bid_cost'], 'zonal'].sum() * 1e6 * unit_scaling
costs.loc[idx['offers', 'socialised', 'zonal']] = cost_summary.loc[idx[:, 'offer_cost'], 'zonal'].sum() * 1e6 * unit_scaling
costs.loc[idx['congestion_rent', 'socialised', 'zonal']] = cost_summary.loc[idx[:, 'congestion_rent'], 'zonal'].sum() * 1e6 * unit_scaling

costs

# --- cell 12 ---
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon

def create_curved_polygon(start, end, height, width, resolution=30):

    x0, y0 = start
    x1, y1 = end

    left_radius = (height - y0) / 2
    right_radius = (height - y1) / 2

    # --- Left Half-Circle Arc ---
    left_center = (x0, y0 + left_radius)
    angles_left = np.linspace(-np.pi/2, np.pi/2, resolution)
    left_arc = [
        (left_center[0] - left_radius * np.cos(theta), left_center[1] + left_radius * np.sin(theta))
        for theta in angles_left
    ]

    # --- Horizontal Segment ---
    horizontal_xs = np.linspace(x0, x1, resolution)
    horizontal_segment = [(x, height) for x in horizontal_xs]

    # --- Right Half-Circle Arc ---
    right_center = (x1, y1 + right_radius)
    angles_right = np.linspace(np.pi/2, -np.pi/2, resolution)
    right_arc = [
        (right_center[0] + right_radius * np.cos(theta), right_center[1] + right_radius * np.sin(theta))
        for theta in angles_right
    ]

    # --- Combine the Upper Curve ---
    upper_curve = left_arc + horizontal_segment + right_arc

    # Create a LineString from the upper curve and then buffer it to get the full polygon.
    line = LineString(upper_curve)
    polygon = line.buffer(width/2)

    left_removal = Polygon([
        (x0, y0 + width/2),
        (x0, y0 - width/2),
        (x0 + width/2, y0 - width/2),
        (x0 + width/2, y0 + width/2),
    ])
    right_removal = Polygon([
        (x1, y1 + width/2),
        (x1, y1 - width/2),
        (x1 - width/2, y1 - width/2),
        (x1 - width/2, y1 + width/2),
    ])

    return polygon.difference(right_removal).difference(left_removal)

# --- cell 13 ---
def get_positive_cost_total(costs, mode):
    part1 = costs.loc[idx['wholesale', 'north', mode]].sum()
    part2 = costs.loc[idx['wholesale', 'south', mode]].sum()

    ss = costs.loc[idx[:, 'socialised', mode]]
    part3 = ss.loc[ss > 0].sum()

    return part1 + part2 + part3

# --- cell 14 ---
def get_consumer_box_limits(costs, mode, gap):

    # box1 = costs.loc[idx['wholesale', 'north', mode]].sum()
    # box2 = costs.loc[idx['wholesale', 'south', mode]].sum()
    box1 = costs.loc[idx['wholesale', 'north', mode]].sum() + costs.loc[idx['wholesale', 'south', mode]].sum()

    ss = costs.loc[idx[:, 'socialised', mode]]
    box2 = ss.loc[ss > 0].sum()

    total = box1 + box2

    box1 = box1 / total
    box2 = box2 / total

    box1_upper = 1
    box1_lower = box1_upper - box1

    box2_upper = box1_lower - gap
    box2_lower = box2_upper - box2

    if mode == 'zonal':
        box2_lower = box2_lower - abs(costs.loc[idx['congestion_rent', :, mode]].sum()) / total

    return (
        (box1_upper, box1_lower),
        (box2_upper, box2_lower),
    )


gap = 0.02
box_width = 0.03
left_end = 0.
right_end = 1.

# --- cell 15 ---
get_consumer_box_limits(costs, 'zonal', gap)

# --- cell 16 ---
def get_generator_box_limits(revs, gap):

    box_heights = {}

    revs = revs.copy() / revs.sum()

    for region, carrier in zip(revs.index.get_level_values(0), revs.index.get_level_values(1)):

        box_heights[(region, carrier)] = revs.loc[idx[region, carrier, :]].sum()

    box_upper = {}
    box_lower = {}

    # Sort by region first, then carrier
    sorted_items = sorted(box_heights.keys(), key=lambda x: (x[0], x[1]))

    current_upper = 1
    for item in sorted_items:
        box_upper[item] = current_upper
        box_lower[item] = current_upper - box_heights[item]
        current_upper = box_lower[item] - gap

    # Handle interconnector separately since it's in 'total' region
    box_upper[('total', 'interconnector')] = current_upper
    box_lower[('total', 'interconnector')] = current_upper - box_heights[('total', 'interconnector')]

    return box_lower, box_upper

# --- cell 17 ---
# Helper function to draw curved flow paths
def bezier_path(start, end, turning_ratio, num_samples=100):
    """
    Creates a smooth cubic Bézier curve as a NumPy array of points,
    suitable for use in Sankey diagrams.

    Parameters:
        start (tuple or list): (x, y) coordinates for the starting point.
        end (tuple or list): (x, y) coordinates for the ending point.
        turning_ratio (float): A value between 0 and 1 that defines the x-position
                               of the turning point. A value of 0 means the turning
                               point is at the start's x-coordinate, while 1 is at the end's.
        num_samples (int): The number of points to sample along the curve.

    Returns:
        numpy.ndarray: An array of shape (num_samples, 2) with the (x, y) coordinates of the curve.
    """
    # Convert start and end to NumPy arrays
    start = np.array(start, dtype=float)
    end = np.array(end, dtype=float)
    
    # Compute the common x-coordinate for both control points based on turning_ratio
    cp_x = start[0] + turning_ratio * (end[0] - start[0])
    
    # Define control points
    cp1 = np.array([cp_x, start[1]])
    cp2 = np.array([cp_x, end[1]])
    
    # Create an array of parameter values t in [0, 1]
    t = np.linspace(0, 1, num_samples).reshape(-1, 1)  # shape (num_samples, 1)
    one_minus_t = 1 - t

    # Evaluate the cubic Bézier curve at each t using the formula:
    # B(t) = (1-t)^3 * P0 + 3*(1-t)^2*t * P1 + 3*(1-t)*t^2 * P2 + t^3 * P3
    curve = (one_minus_t**3) * start + \
            3 * (one_minus_t**2) * t * cp1 + \
            3 * one_minus_t * (t**2) * cp2 + \
            (t**3) * end

    return curve[:,0], curve[:,1]

# --- cell 18 ---
class FlowStart:

    def __init__(self, upper_bound, lower_bound):
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.current_upper = upper_bound

    def __str__(self):
        return f"FlowStart(upper_bound={self.upper_bound:.4f}, lower_bound={self.lower_bound:.4f}, current_upper={self.current_upper:.4f})"

    def add_flow(self, flow):
        return_value = self.current_upper
        self.current_upper -= flow
        return return_value
    
    def __repr__(self):
        return self.__str__()

# --- cell 19 ---
def reset_flowstarts(rev_lowers, rev_uppers):
    flowstarts = {}

    # for block_name in ['con_n', 'con_s', 'soc']:
    for block_name in ['con', 'soc']:
        block = globals()[block_name]
        flowstarts[block_name] = FlowStart(
            *block
            )

    for (name, upper), (name2, lower) in zip(rev_uppers.items(), rev_lowers.items()):
        assert name == name2
        flowstarts[name] = FlowStart(
            upper,
            lower
            )
    
    return flowstarts

# --- cell 20 ---
## make unit in per year

rev_national = rev_national / 3
rev_zonal = rev_zonal / 3
costs = costs / 3

# --- cell 21 ---
def prepare_plot_data(mode):

    con, soc = get_consumer_box_limits(costs, mode, gap)

    rev = globals()[f'rev_{mode}']
    rev = rev.sort_index()

    rev_lowers, rev_uppers = get_generator_box_limits(rev, gap)
    rev_flows = rev.copy() / rev.abs().sum()

    return con, soc, rev_lowers, rev_uppers, rev_flows, rev.sum()

# --- cell 22 ---
def get_largest_total():

    return max(
        globals()[f'rev_national'].sum(),
        globals()[f'rev_zonal'].sum(),
    )
    
get_largest_total()

# --- cell 23 ---
nice_asset_names = {
    ('north','disp'): 'Thermal North',
    ('south','disp'): 'Thermal South',
    ('north','wind'): 'Wind North',
    ('south','wind'): 'Wind South',
    ('north','hydro'): 'Hydro North',
    ('south','hydro'): 'Hydro South',
    ('north','nuclear'): 'Nuclear North',
    ('south','nuclear'): 'Nuclear South',
    ('north','storage'): 'Storage North',
    ('south','storage'): 'Storage South',
    ('total','interconnector'): 'Interconnectors',
    'savings': 'Consumer\nSavings',
    'congestion_rent': 'Congestion\nRent',
}

# --- cell 24 ---
def plot_stacked_bars(ax, upper_series, lower_series, bar_height=0.3, gap=0.1):
    """
    Plot two stacked horizontal barplots from two series.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    upper_series : pd.Series
        Series with values and colors for the upper barplot. 
        Index is used for labels. Values should be tuples of (value, color).
    lower_series : pd.Series
        Series with values and colors for the lower barplot.
        Index is used for labels. Values should be tuples of (value, color).
    bar_height : float, optional
        Height of each bar. Default is 0.3.
    gap : float, optional
        Gap between the two barplots. Default is 0.1.
    """
    # Calculate positions for the bars
    upper_y = bar_height + gap
    lower_y = 0
    
    bar_kwargs = {
        'edgecolor': 'k',
        'linewidth': 0.5,
    }

    text_threshold = 0.033

    # Plot upper barplot
    left = 0
    for i, (name, (value, color)) in enumerate(upper_series.items()):
        ax.barh(upper_y, value, height=bar_height, left=left, color=color, **bar_kwargs)
        # Add value text in the middle of the segment
        text_x = left + value/2

        if name == ('south', 'wind'):
            print_name = 'Wind\nSouth'
        elif name == ('south', 'nuclear'):
            print_name = 'Nuclear\nSouth'
        elif name == ('total', 'interconnector'):
            print_name = 'Intercons'
        else:
            print_name = nice_asset_names[name]

        if value > text_threshold:
            ax.text(text_x, upper_y, f"{value:.2f}", ha='center', va='center', fontsize=8)
            ax.text(text_x, upper_y + bar_height/2 + 0.05, print_name, ha='center', va='bottom', fontsize=8)

        left += value
    
    # Plot lower barplot
    left = 0
    for i, (name, (value, color)) in enumerate(lower_series.items()):

        if name == ('north', 'hydro'):
            print_name = 'Hydro\nNorth'
        elif name == ('north', 'disp'):
            print_name = 'Thermal\nNorth'
        else:
            print_name = nice_asset_names[name]

        ax.barh(lower_y, value, height=bar_height, left=left, color=color, **bar_kwargs)
        # Add value text in the middle of the segment
        text_x = left + value/2

        if value > text_threshold:
            ax.text(text_x, lower_y, f"{value:.2f}", ha='center', va='center', fontsize=8)
            ax.text(text_x, lower_y - bar_height/2 - 0.05, print_name, ha='center', va='top', fontsize=8)

        left += value

    # Set y-axis limits to accommodate both bars and labels
    ax.set_ylim(-0.5, upper_y + 0.5)
    ax.set_yticks([])

    # ax.plot([0,0], [-bar_height/2, upper_y + gap], color='k', linewidth=2)

    # Remove y-axis line
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    return ax

# --- cell 25 ---
color_mapper = {
    'roc': color_dict['roc_payments'],
    'cfd': color_dict['cfd_payments'],
    'bids': color_dict['bid_cost'],
    'offers': color_dict['offer_cost'],
    'wholesale': color_dict['wholesale'],
    'congestion_rent': color_dict['congestion_rent']
}

# --- cell 26 ---
special_y_shifts = {
    ('north', 'disp'): 0.025,
    ('north', 'hydro'): 0.0125,
    ('north', 'nuclear'): 0.0,
    ('north', 'storage'): -0.02,
    ('north', 'wind'): -0.02,
    ('south', 'hydro'): 0.01,
    ('south', 'storage'): -0.01,
}

# --- cell 27 ---
fig, axs = plt.subplots(1, 2, figsize=(15, 9))

alpha = 0.5
rectangle_kwargs = {
    'color': 'k',
    'alpha': 0.7,
    'edgecolor': 'k',
}

nice_consumer_names = {
    # 'con_n': 'Consumers\nNorth',
    # 'con_s': 'Consumers\nSouth',
    'con': 'Wholesale\nCosts',
    'soc': 'Socialised\nCosts',
}

text_padding = 0.01

# def add_flow(ax, left_upper, right_upper, box_width, volume, tp=0.5):
def add_flow(
        ax,
        left_upper,
        right_upper,
        flow_vol,
        total,
        tp=0.5,
        color='r',
        text_threshold=0.02,
        end=None,
        fill_kwargs={},
        ):

    start = (box_width, left_upper - flow_vol/2)

    if end is None:
        end = (1-box_width, right_upper - flow_vol/2)

    x, y = bezier_path(start, end, tp)

    prevent_overlap = 0.002
    ax.fill_between(x, y-flow_vol/2 + prevent_overlap, y+flow_vol/2 - prevent_overlap, alpha=alpha, color=color, **fill_kwargs)

    if flow_vol > text_threshold:
        ax.text(
            x[-1] - text_padding,
            y[-1],
            f'{flow_vol*total:.2f} bn£/a',
            ha='right',
            fontsize=8
            )

for mode, ax in zip(['national', 'zonal'], axs):

    con, soc, rev_lowers, rev_uppers, rev_flows, rev_total = prepare_plot_data(mode)

    # weight_north = (con_n[0] - con_n[1]) / (con_n[0] - con_n[1] + con_s[0] - con_s[1])
    # weight_south = (con_s[0] - con_s[1]) / (con_n[0] - con_n[1] + con_s[0] - con_s[1])

    flowstarts = reset_flowstarts(rev_lowers, rev_uppers)

    for (region, carrier), flow in rev_flows.loc[idx[:, :, 'wholesale']].items():
        # north_flow = flow * weight_north
        # south_flow = flow * weight_south

        add_flow(
            ax,
            flowstarts['con'].add_flow(flow),
            flowstarts[(region, carrier)].add_flow(flow),
            flow,
            rev_total,
            color=color_mapper['wholesale']
            )

        # add_flow(
        #     ax,
        #     flowstarts['con_s'].add_flow(south_flow),
        #     flowstarts[(region, carrier)].add_flow(south_flow),
        #     south_flow,
        #     rev_total,
        #     color=color_mapper['wholesale']
        #     )

    for factor, tp in zip(['roc', 'cfd', 'bids', 'offers'], [0.5, 0.5, 0.5, 0.5]):

        for (region, carrier), flow in rev_flows.loc[idx[:, :, factor]].items():

            if flow == 0:
                continue

            add_flow(
                ax,
                flowstarts['soc'].add_flow(flow),
                flowstarts[(region, carrier)].add_flow(flow),
                flow,
                rev_total,
                color=color_mapper[factor],
                tp=tp,
                )
    
    if mode == 'zonal':
        congestion_rent_flow = abs(costs.loc[idx['congestion_rent', :, 'zonal']].sum()) / rev_total

        add_flow(
            ax,
            flowstarts['soc'].add_flow(congestion_rent_flow),
            1,
            congestion_rent_flow,
            rev_total,
            color=color_mapper['congestion_rent'],
            end=(0.5, flowstarts['soc'].current_upper - 0.1),
            fill_kwargs={'edgecolor': 'k'}
            )

    # for cost_party in ['con_n', 'con_s', 'soc']:
    for cost_party in ['con', 'soc']:

        ax.add_patch(plt.Rectangle((
            left_end,
            flowstarts[cost_party].current_upper),
            box_width,
            flowstarts[cost_party].upper_bound - flowstarts[cost_party].current_upper,
            **rectangle_kwargs
            ))
        ax.text(
            - text_padding,
            flowstarts[cost_party].current_upper + (flowstarts[cost_party].upper_bound - flowstarts[cost_party].current_upper)/2,
            # flowstarts[cost_party].current_upper,
            nice_consumer_names[cost_party],
            ha='right',
            va='center',
            rotation=0,
            fontsize=9,
            # weight='bold',
            )

    if mode == 'zonal':
        ss = costs.loc[costs.index.get_level_values(0) != 'congestion_rent']

        cost_total = get_positive_cost_total(costs, 'zonal')

        # saving_value = ss.loc[idx[:,:,'national']].sum() - ss.loc[idx[:,:,'zonal']].sum()
        print('warning current setting savings value to align with socioeconomic benefit analyis')
        saving_value = 0.91

        saving_extent = saving_value / cost_total
        savings_upper = flowstarts[cost_party].current_upper - 2*gap
        savings_lower = savings_upper - saving_extent

        ax.add_patch(plt.Rectangle((
            left_end,
            savings_lower),
            box_width,
            savings_upper - savings_lower,
            **rectangle_kwargs
            ))
        
        ax.text(
            left_end - text_padding,
            savings_lower + (savings_upper - savings_lower)/2,
            'Consumer\nSavings',
            ha='right',
            va='center',
            fontsize=9,
            # weight='bold'
            )
        
        ax.text(
            left_end - text_padding,
            savings_lower + (savings_upper - savings_lower)/2 - 0.038,
            f'Total: {saving_value:.2f} bn£/a',
            ha='right',
            va='center',
            fontsize=8,
            weight='bold',
        )

    asset_revenues = pd.Series(rev_uppers).sub(pd.Series(rev_lowers)).mul(rev_total)

    for lower, upper, name in zip(rev_lowers.values(), rev_uppers.values(), rev_lowers.keys()):

        min_height = 0.002
        height = max(upper-lower, min_height)

        ax.add_patch(plt.Rectangle(
            (right_end - box_width, lower),
            box_width,
            height,
            **rectangle_kwargs
            ))

        ax.text(
            right_end + box_width + text_padding,
            lower + (upper-lower)/2 + special_y_shifts.get(name, 0),
            nice_asset_names[name],
            ha='left',
            va='center',
            fontsize=9,
            #weight='bold'
            )

        total_revenue = asset_revenues.loc[name]
        summary_message = f'\nTotal: {total_revenue:.2f} bn£/a'

        if mode == 'zonal':
            va = 'bottom'
        else:
            va = 'center'

        if mode == 'zonal':
            x_shift = 0.22
            y_shift = 0.0
        else:
            x_shift = 0.0
            y_shift = - 0.011

        # total revenue
        ax.text(
            right_end + box_width + text_padding + x_shift,
            lower + (upper-lower)/2 + special_y_shifts.get(name, 0) + y_shift,
            summary_message,
            ha='left',
            va=va,
            fontsize=8,
            weight='bold',
            )

        if mode == 'zonal':
            revenue_diff = total_revenue - rev_national.loc[name].sum()
            percent_diff = revenue_diff / rev_national.loc[name].sum()

            sign = '+' if revenue_diff > 0 else '-'

            text_color = 'limegreen' if revenue_diff > 0 else 'orangered'
            diff_message = f'{sign}{abs(revenue_diff):.3f} bn£/a ({sign}{abs(percent_diff):.2%})'

            # revenues difference to national
            ax.text(
                right_end + box_width + text_padding + x_shift,
                lower + (upper-lower)/2 + special_y_shifts.get(name, 0),
                diff_message,
                ha='left',
                va='top',
                fontsize=8,
                color=text_color,
                weight='bold'
                )

    ax.axis('off')
    ax.text(
        0.5,
        1.07,
        mode.capitalize(),
        ha='center',
        va='center',
        fontsize=10,
        weight='bold'
        )

    if mode == 'zonal':

        rev_diff = (rev_zonal - rev_national).groupby(level=[0,1]).sum().sort_values() / rev_total
        for (region, carrier), height in rev_diff.items():

            if height > 0:
                ax.add_patch(plt.Rectangle(
                    (right_end, flowstarts[region, carrier].upper_bound - height),
                    box_width,
                    height,
                    facecolor='limegreen',
                    alpha=0.9,
                    edgecolor='forestgreen'
                    ))
            else:
                ax.add_patch(plt.Rectangle(
                    (right_end, flowstarts[region, carrier].current_upper),
                    box_width,
                    abs(height),
                    facecolor='orangered',
                    alpha=0.9,
                    edgecolor='darkred'
                    ))
    
# Create a new axis below the existing ones
# Get the figure and the position of the existing axes
fig = ax.figure
ax0_pos = axs[0].get_position()
ax1_pos = axs[1].get_position()

# Calculate the position for the new axis
# It should span the combined width of axs[0] and axs[1]
left = ax0_pos.x0
right = ax1_pos.x1
width = right - left - 0.2

# Position it below the existing axes with some padding
bottom = min(ax0_pos.y0, ax1_pos.y0) - 0.15  # Adjust this value for desired spacing
height = 0.12  # Adjust height as needed

# Create the new axis
# new_ax = fig.add_axes([left - 0.08, bottom, width, height])

# You can customize the new axis here
# new_ax.axis('off')  # Turn off axis if you just want to use it for annotations

savings = costs.loc[costs.index.get_level_values(0) != 'congestion_rent'].groupby(level=2).sum()
savings = savings.loc['national'] - savings.loc['zonal']

diff = (rev_zonal - rev_national).groupby(level=[0,1]).sum()

gains = diff.loc[diff > 0]
losses = diff.loc[diff < 0].abs()

gains = gains.apply(lambda x: (x, 'limegreen'))
losses = losses.apply(lambda x: (x, 'orangered'))

gains = pd.concat([
    gains,
    pd.Series(
        [(costs.loc[idx['congestion_rent', :, 'zonal']].abs().sum(), color_dict['congestion_rent']),
        (savings, 'mediumorchid')],
        index=['congestion_rent', 'savings'],
        name=('total', 'congestion_rent'))
    ])

# plot_stacked_bars(new_ax, gains, losses)
# new_ax.set_xlabel('National - Zonal Reallocation (bn£/a)', fontsize=9)
# new_ax.tick_params(axis='x', labelsize=8)

ax0_pos = axs[0].get_position()
ax1_pos = axs[1].get_position()

# Calculate new positions with padding
padding = 0.05  # Adjust this value to increase/decrease padding
new_ax0_width = ax0_pos.width * 0.95  # Slightly reduce width to make room
new_ax1_width = ax1_pos.width * 0.95

# Set new positions
axs[0].set_position([ax0_pos.x0, ax0_pos.y0, new_ax0_width, ax0_pos.height])
axs[1].set_position([ax1_pos.x0 + padding, ax1_pos.y0, new_ax1_width, ax1_pos.height])

# Update the position of the new axis to match the new positions of axs[0] and axs[1]
ax0_pos = axs[0].get_position()
ax1_pos = axs[1].get_position()
left = ax0_pos.x0
right = ax1_pos.x1
width = right - left
# new_ax.set_position([left + 0.0, bottom, width, height])

handles = []
labels = []

nice_names = {
    'wholesale': 'Wholesale Market',
    'wholesale buying': 'Electricity Procurement',
    'roc_payments': 'Renewable Obligation Certificates',
    'cfd_payments': 'Contracts for Differences',
    'congestion_rent': 'Intra-GB Congestion Rents',
    'offer_cost': 'Redispatch - Offers',
    'bid_cost': 'Redispatch - Bids',
}

for key, value in color_dict.items():
    if key in ['wholesale buying', 'wholesale selling']:
        continue

    handles.append(plt.Rectangle((0, 0), 1, 1, color=value))
    labels.append(nice_names.get(key, key))

# handles.append(plt.Line2D([0], [0], color='w', linestyle='-'))
# labels.append('')
handles.append(plt.Rectangle((0, 0), 1, 1, color='limegreen'))
labels.append('Revenue Increase')
handles.append(plt.Rectangle((0, 0), 1, 1, color='orangered'))
labels.append('Revenue Decrease')

axs[-1].legend(
    handles,
    labels,
    # loc='center left',
    bbox_to_anchor=(0.75, -0.),
    fontsize=8,
    frameon=False,
    ncol=4,
    )

axs[0].text(
    1.3,
    1.07,
    '2022-01-01 - 2024-12-31',
    ha='center',
    va='center',
    fontsize=10,
    weight='bold',
    )

axs[0].text(
    -0.1,
    1.08,
    'a',
    ha='center',
    va='center',
    fontsize=12,
    weight='bold',
    )

axs[1].text(
    0.1,
    1.08,
    'b',
    ha='center',
    va='center',
    fontsize=12,
    weight='bold',
    )

'''
axs[0].text(
    0.,
    -0.3,
    'c',
    ha='center',
    va='center',
    fontsize=12,
    weight='bold',
    )
'''

# plt.tight_layout()
plt.savefig(f'money_flow_{day}.pdf', bbox_inches='tight')
plt.show()

# --- cell 28 ---
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon

def create_curved_polygon(start, end, height, width, resolution=30):
    """
    Creates a polygon that starts on the left, arcs upward to a horizontal segment,
    then arcs downward to an endpoint on the right. The resulting curve is then 
    buffered to yield a full polygon.
    
    Parameters:
        start (tuple): (x, y) coordinate for the start point (left end).
        end (tuple):   (x, y) coordinate for the end point (right end).
                       (For simplicity, we assume start and end share the same y value.)
        buffer_distance (float): The distance to “thicken” the curve.
        resolution (int): Number of points used to sample each arc segment.
    
    Returns:
        A Shapely Polygon representing the buffered curve.
    """
    x0, y0 = start
    x1, y1 = end

    left_radius = (height - y0) / 2
    right_radius = (height - y1) / 2

    # --- Left Half-Circle Arc ---
    left_center = (x0, y0 + left_radius)
    angles_left = np.linspace(-np.pi/2, np.pi/2, resolution)
    left_arc = [
        (left_center[0] - left_radius * np.cos(theta), left_center[1] + left_radius * np.sin(theta))
        for theta in angles_left
    ]

    # --- Horizontal Segment ---
    horizontal_xs = np.linspace(x0, x1, resolution)
    horizontal_segment = [(x, height) for x in horizontal_xs]

    # --- Right Half-Circle Arc ---
    right_center = (x1, y1 + right_radius)
    angles_right = np.linspace(np.pi/2, -np.pi/2, resolution)
    right_arc = [
        (right_center[0] + right_radius * np.cos(theta), right_center[1] + right_radius * np.sin(theta))
        for theta in angles_right
    ]

    # --- Combine the Upper Curve ---
    upper_curve = left_arc + horizontal_segment + right_arc

    # Create a LineString from the upper curve and then buffer it to get the full polygon.
    line = LineString(upper_curve)
    polygon = line.buffer(width/2)

    left_removal = Polygon([
        (x0, y0 + width/2),
        (x0, y0 - width/2),
        (x0 + width/2, y0 - width/2),
        (x0 + width/2, y0 + width/2),
    ])

    polygon = polygon.difference(left_removal)

    right_removal = Polygon([
        (x1, y1 + width/2),
        (x1, y1 - width/2),
        (x1 - width/2, y1 - width/2),
        (x1 - width/2, y1 + width/2),
    ])

    polygon = polygon.difference(right_removal)

    return polygon

# Example usage:
if __name__ == '__main__':
    start = (1, 3)
    end = (10, 2)
    buffer_distance = 0.5  # How much to buffer (i.e. the “thickness” around the curve)
    height = 4
    
    poly = create_curved_polygon(start, end, height, buffer_distance, resolution=30)
    
    # Plotting the buffered polygon:
    xs, ys = poly.exterior.xy
    plt.figure(figsize=(8, 4))
    plt.fill(xs, ys, alpha=0.5, edgecolor='k')
    plt.title('Buffered Curved Polygon')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.show()
