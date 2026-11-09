"""Registered coefficients; component expressions are unchanged E90 copies."""
COEFFICIENTS = {
    'P': {'overlap': .029917209539524287, 'task': .11122773694450763},
    'F': {'overlap': .08975162861857286, 'task': .3336832108335229},
}

def configuration(phase):
    return {'phase': phase, 'reference': 1., **COEFFICIENTS[phase],
            'shape': .25, 'aggregation': 'per-case then arithmetic mean',
            'selection': 'same phase total for backward/monitors/best/LR/early stop'}
