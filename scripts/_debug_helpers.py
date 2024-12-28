'''Helpers for quick bug locating'''

def remove_interconnectors(n):
    '''helper function for debugging'''

    for l in n.links.index[n.links.carrier == 'interconnector']:
        n.remove("Link", l)
    
    for g in n.generators.index[n.generators.carrier == 'local_market']:
        n.remove("Generator", g)
    
    for l in n.loads.index[n.loads.carrier != 'electricity']:
        n.remove("Load", l)


def remove_storage_units(n):
    '''helper function for debugging'''

    for s in n.storage_units.index:
        n.remove("StorageUnit", s)