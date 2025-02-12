import numpy as np

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
    "interconnector": "magenta",
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
