"""Exact E90/E93 C Loss; no auxiliary weight change in this LR experiment."""
COEFFICIENTS = {
    'F': {'overlap': .08975162861857286, 'task': .3336832108335229},
}

def configuration(phase):
    return {'phase': phase, 'reference': 1., **COEFFICIENTS[phase],
            'shape': .25, 'aggregation': 'per-case then arithmetic mean',
            'selection': 'same fixed F total for backward/monitors/best/LR/early stop'}
