"""Same twelve effective block applications; sharing is the only paired difference."""
ARMS = {name: {'width': 96, 'layers': 12, 'rectangle': True, 'ffn_dropout': .05,
               'shared_blocks': blocks, 'correction_scale_xy': .5,
               'correction_scale_shape': .5, 'minimum_xy_scale': .1}
        for name, blocks in (('unshared_residual', 12), ('shared_residual', 3))}


def model_config(arm):
    return dict(ARMS[arm])
