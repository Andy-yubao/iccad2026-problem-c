"""Fixed E42 center/shape definitions and weights; all other objectives absent."""
def configuration(arm):
    if arm != 'reference_position_shape':
        raise ValueError(arm)
    return {'revision': 'E55.reference_position_shape.v1',
            'weights': {'center_local': 1., 'shape': .25},
            'count_factor': False, 'center_smooth_l1_beta': .1,
            'shape': 'squared raw log(w/h) error',
            'normalization': 'center divided by sqrt(public area); variable-dimension masks; per-layout means then equal batch mean',
            'selection': 'same fixed total; dev_monitor500 arithmetic mean',
            'disabled': ['relative', 'contact', 'overlap', 'overlap_area', 'boundary',
                         'mib', 'bbox', 'compactness', 'void', 'crowding',
                         'module_overlap', 'clearance', 'wire_guard'],
            'schedule': 'constant from first batch; no ramp, no tolerance band'}
