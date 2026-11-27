"""Registered coefficients; component expressions are unchanged E90 copies."""
COEFFICIENTS = {
    'C': {'overlap': .08975162861857286, 'task': .3336832108335229},
    'G': {'overlap': .17950325723714572, 'task': .3336832108335229},
    # Fixed old-F diagnostic only; never a second selection proxy for G.
    'F': {'overlap': .08975162861857286, 'task': .3336832108335229},
}

def configuration(phase):
    return {'phase': phase, 'reference': 1., **COEFFICIENTS[phase],
            'shape': .25, 'aggregation': 'per-case then arithmetic mean',
            'selection': 'same phase total for backward/monitors/best/LR/early stop'}
