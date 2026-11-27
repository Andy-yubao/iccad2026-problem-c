"""Exact E55 pure reference objective for empirical fitting diagnostics."""
def configuration(arm):
    if not arm.startswith('reference'):
        raise ValueError(arm)
    return {'weights': {'center_local': 1., 'shape': .25}, 'count_factor': False}
