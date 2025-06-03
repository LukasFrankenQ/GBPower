import pandas as pd
import numpy as np
from pathlib import Path

idx = pd.IndexSlice
bmus_alldata = pd.read_csv(Path.cwd().parent / 'data' / 'prerun' / 'prepared_bmus.csv', index_col=0)

def get_all_bmus():
    bmus = bmus_alldata.copy()
    return bmus.loc[bmus.lon != 'distributed']


def get_units(carrier):
    if isinstance(carrier, list):
        return bmus_alldata.loc[bmus_alldata.carrier.isin(carrier)].index
    else:
        units = bmus_alldata.loc[bmus_alldata.carrier.str.contains(carrier)].index

    return units


def get_consumer_cost(fn):

    cc = pd.read_csv(
        fn,
        index_col=[0,1],
        parse_dates=True
        )

    return cc.loc[~(cc.index.get_level_values(1) == 'balancing_volume')]


def get_model_balancing_volume(fn):

    bv = pd.read_csv(
        fn,
        index_col=[0,1],
        parse_dates=True
        )
    return bv.loc[bv.index.get_level_values(1) == 'balancing_volume']


def get_interconnector_names():
    return [
        'IFA1',
        'Moyle',
        'BritNed',
        'IFA2',
        'EastWest',
        'Viking',
        'ElecLink',
        'NSL',
        'Nemo',
        ]


def get_revenue_data(fn):

    rev = pd.read_csv(
        fn,
        index_col=[0,1,2],
        parse_dates=True
        )

    idx = pd.IndexSlice
    keepers = ['bid_cost', 'cfd', 'offer_cost', 'roc', 'wholesale']

    rev = rev.loc[idx[:,:, keepers], :].sort_index().replace(np.nan, 0)
    return rev.loc[~rev.index.get_level_values(0).isin(get_interconnector_names())]


def get_dispatch_data(fn):

    disp = pd.read_csv(
        fn,
        index_col=[0,1,2],
        parse_dates=True
        )
    
    return disp.loc[~disp.index.get_level_values(0).isin(get_interconnector_names())]


def get_unit_schedule(data, unit, layout, mode, time_slice):
    """Get wholesale schedule for a unit in either national or zonal market.

    Args:
        data (pd.DataFrame): Schedule data with 3-level index
        unit (list or str): Unit name (index level 0)
        layout (str): Either 'national' or 'zonal' (index level 1)
        mode (str): Either 'wholesale' or 'redispatch' (index level 2)
        time_slice (slice): Time period to extract
    
    Returns:
        pd.Series: Wholesale schedule for unit in specified market
    """
    if not isinstance(data.columns, pd.DatetimeIndex):
        data.columns = pd.to_datetime(data.columns)

    who = data.loc[(unit, layout, 'wholesale'), time_slice.intersection(data.columns)].sum()
    if mode == 'wholesale':
        return who
    else:
        return (
            data.loc[
                (unit, layout, 'redispatch'),
                time_slice.intersection(data.columns)
                ].sum()
                - who
        )


def get_unit_revenue(data, unit, layout, time_slice):

    if not isinstance(data.columns, pd.DatetimeIndex):
        data.columns = pd.to_datetime(data.columns)

    return (
        data
        .loc[
            idx[unit, layout, :],
            time_slice.intersection(data.columns)
            ]
        .sum().sum()
    )


def get_thermal_dispatch_data(fn):
    """
    Returns half-hourly data for thermal units only (in MWh).
    """
    return pd.read_csv(
        fn,
        index_col=[0],
        parse_dates=[0],
        header=[0,1]
        ).replace(np.nan, 0) * 0.5


def get_marginal_costs_data(fn):
    """
    Returns half-hourly data for thermal units only (in £/MWh).
    """
    return pd.read_csv(
        fn,
        index_col=[0],
        parse_dates=[0],
        )


def get_wholesale_expenses(dispatch, marginal_cost, unit, layout, time_slice):
    """Get expenses for a single unit in a given time period.
    
    Args:
        unit (str): The unit ID
        layout (str): The layout of the unit
        time_slice (pd.DatetimeIndex): Time period to get expenses for
        
    Returns:
        pd.Series: Expenses time series for the unit
    """
    idx = pd.IndexSlice

    avail = time_slice.intersection(marginal_cost.index).intersection(dispatch.index)
    
    assert unit in dispatch.columns.get_level_values(1) and unit in marginal_cost.columns, f'Method for thermal units only'

    expenses = pd.Series(
        dispatch.loc[avail, idx[layout, unit]] * marginal_cost.loc[avail, unit],
        index=avail,
        dtype=float,
        name=unit
    ).replace(np.nan, 0)

    return expenses.sum()


def get_balancing_expenses(
        daily_dispatch,
        hh_dispatch,
        marginal_cost,
        balancing_markup,
        unit,
        layout,
        time_slice
        ):
    """Get expenses for a single unit in a given time period.
    
    Args:
        unit (str): The unit ID
        layout (str): The layout of the unit
        time_slice (pd.DatetimeIndex): Time period to get expenses for
        balancing_markup (float): Additional cost per MWh for balancing
        
    Returns:
        pd.Series: Expenses time series for the unit including wholesale and balancing costs
    """

    idx = pd.IndexSlice

    avail = time_slice.intersection(marginal_cost.index).intersection(hh_dispatch.index)
    assert unit in hh_dispatch.columns.get_level_values(1) and unit in marginal_cost.columns, f'Method for thermal units only'

    # Calculate weighted average marginal cost
    basic_running_cost = (
        (hh_dispatch.loc[avail, idx[layout, unit]] * marginal_cost.loc[avail, unit]).sum() / 
        hh_dispatch.loc[avail, idx[layout, unit]].sum()
    )

    balancing_volume = get_unit_schedule(daily_dispatch, unit, layout, 'redispatch', time_slice)
    return balancing_volume * (basic_running_cost + balancing_markup)


def get_europe_prices(base_path):
    days = pd.date_range(start='2022-01-01', end='2024-12-31', freq='D').strftime('%Y-%m-%d')

    prices = []
    for day in days:
        try:
            prices.append(
                pd.read_csv(
                    base_path / day / 'europe_day_ahead_prices.csv',
                    index_col=0,
                    parse_dates=True,
                )
            )
        except FileNotFoundError:
            print(f'No prices for {day}')

    europe_prices = pd.concat(prices)
    europe_prices.index = pd.to_datetime(europe_prices.index).tz_localize(None)
    
    return europe_prices


def get_interconnector_dispatch(fn):
    return pd.read_csv(fn, index_col=[0], parse_dates=True, header=[0,1])


def get_gas_prices():
    fp = pd.read_excel(
        Path.cwd() / 'systemaveragepriceofgasdataset130225.xlsx',
        sheet_name=3,
        parse_dates=True,
        index_col=0,
        skiprows=1
        )

    df = (
        fp.iloc[4:]
        .rename(
        columns={
            'Unnamed: 1': 'day_price',
            'Unnamed: 2': 'before_week_average',
        })
        .replace('[x]', np.nan)
        ['day_price']
        )
    df.index.name = 'date'
    df.index = pd.to_datetime(df.index)
    return df * 10


def get_marginal_prices(fn):
    return pd.read_csv(fn, index_col=[0], parse_dates=True, header=[0,1])


def get_ic_congestion_rent(
        ic_flow,
        marginal_prices_gb,
        eu_prices,
        network,
        layout,
        time_slice,
        ):

        idx = pd.IndexSlice

        if not isinstance(ic_flow.index, pd.DatetimeIndex):
            ic_flow.index = pd.to_datetime(ic_flow.index)

        inter = ic_flow.index.intersection(time_slice).intersection(eu_prices.index)

        ics = ic_flow.columns.get_level_values(0).unique()
        ic_flow = ic_flow.loc[inter, idx[:, layout]].replace(np.nan, 0.)
        ic_flow.columns = ic_flow.columns.droplevel(1)

        if layout == 'zonal':
            zonal_intercon_buses = dict(network.links.loc[ics, 'bus1'])
        else:            
            zonal_intercon_buses = {}

        intercon_countries = network.links.loc[ics, 'bus0']

        congestion_rent = 0

        for ic in ics:
            # ic_flow_series = ic_flow.loc[:, ic]
            gb_bus = zonal_intercon_buses.get(ic, 'GB')
            country = intercon_countries[ic]

            gb_price = marginal_prices_gb.loc[inter, idx[layout, gb_bus]]
            country_price = eu_prices.loc[inter, country]

            congestion_rent += 0.5 * ((gb_price - country_price) * ic_flow.loc[inter, ic]).abs().sum()

        return congestion_rent


def get_export_revenues(exports, gb_prices, time_slice, layout, network):

    exports = exports.replace(np.nan, 0.)
    assert (exports >= 0).all().all(), 'Export values cannot be negative'

    if not isinstance(exports.index, pd.DatetimeIndex):
        exports.index = pd.to_datetime(exports.index)

    inter = time_slice.intersection(gb_prices.index).intersection(exports.index)
    assert len(inter) > 0, 'No intersection between export values, prices, and time slice'

    if layout == 'zonal':
        buses = dict(network.links.loc[exports.columns.get_level_values(0).unique(), 'bus1'])
    else:            
        buses = {}

    revenues = 0

    for col in exports.columns.get_level_values(0).unique():
        bus = buses.get(col, 'GB')

        revenues += exports.loc[inter, idx[col, layout]].mul(gb_prices.loc[inter, idx[layout, bus]]).sum()
    
    return revenues


def get_import_cost(imports, eu_prices, time_slice, layout, network):

    imports = imports.replace(np.nan, 0.)
    assert (imports >= 0).all().all(), 'Import values cannot be negative'

    if not isinstance(imports.index, pd.DatetimeIndex):
        imports.index = pd.to_datetime(imports.index)

    inter = time_slice.intersection(eu_prices.index).intersection(imports.index)
    assert len(inter) > 0, 'No intersection between export values, prices, and time slice'

    intercon_countries = network.links.loc[imports.columns.get_level_values(0).unique(), 'bus0']

    cost = 0

    for col in imports.columns.get_level_values(0).unique():
        country = intercon_countries[col]

        cost += imports.loc[inter, idx[col, layout]].mul(eu_prices.loc[inter, country]).sum()
    
    return cost

