# maps interconnectors to their BMU names
interconnection_mapper = {
    'BritNed': ['IBG'],
    'IFA1': ['IFD', 'IFG'],
    'IFA2': ['I2D', 'I2G', 'ING'],
    'EastWest': ['IID'],
    'Moyle': ['IMD'],
    'Viking': ['IVD', 'IVG'],
    'ElecLink': ['IED', 'IEG', 'ILD', 'ILG'],
    'NSL': ['ISD', 'ISG'],
}

# source: https://www.ofgem.gov.uk/energy-policy-and-regulation/policy-and-regulatory-programmes/interconnectors
interconnection_capacities = {
    'IFA1': 2000, # MW
    'Moyle': 500, # MW
    'BritNed': 1000, # MW
    'IFA2': 1000, # MW
    'EastWest': 500, # MW (also called EWIC)
    'Viking': 1400, # MW,
    'ElecLink': 1000, # MW
    'NSL': 1400, # MW
    'Nemo': 1000, # MW
}

interconnection_countries = {
    'IFA1': 'France',
    'Moyle': 'Ireland',
    'BritNed': 'Netherlands',
    'IFA2': 'France',
    'EastWest': 'Ireland',
    'Viking': 'Denmark',
    'ElecLink': 'France',
    'NSL': 'Norway',
    'Nemo': 'Belgium',
}
