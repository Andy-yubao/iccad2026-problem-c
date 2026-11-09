from .losses import WEIGHTS


def configuration(arm):
    if arm.startswith('reference'):
        from .reference_config import configuration as reference
        return {'revision': 'E57.reference.v1', **reference(arm),
                'selection': 'same fixed objective on unchanged dev_monitor500'}
    if arm not in ('geometry96', 'geometry160'):
        raise ValueError(arm)
    return {'revision': 'E57.E42_fixed_geometry.v1', 'weights': dict(WEIGHTS),
            'count_factor': False, 'selection': 'same fixed G on unchanged dev_monitor500',
            'schedule': 'full fixed weights; no ramp or occupancy selection proxy'}
