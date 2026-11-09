"""One registered coefficient vector, fixed for the entire scratch run."""
A = .08975162861857286
B = .3336832108335229
COEFFICIENTS = {
    'position': 1., 'shape': .25,
    'penetration': A, 'module_load_mean': A, 'module_max_excess10': A,
    'wirelength': .5*B, 'compactness': .5*B,
    'boundary_contribution': 2*B, 'grouping_contribution': 2*B,
    'mib_contribution': 2*B,
}

def configuration():
    return {'coefficients': dict(COEFFICIENTS), 'dynamic_weights': False,
            'aggregation': 'within layout as registered, then arithmetic mean over layouts',
            'soft_normalization': 'shared public nb+ng+nm, preserving E90/E94 soft aggregation',
            'source': 'E90/E94 fixed F objective expanded into ten coefficient-controlled terms'}
