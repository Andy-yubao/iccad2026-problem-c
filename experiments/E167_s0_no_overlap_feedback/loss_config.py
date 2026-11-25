ARM = 'e167_s0'
BLOCK_SCALE = .25
EXIT_WEIGHTS = {4: .10}
OVERLAP_SCALE = {0: .25, 4: 0., 8: 1., 12: 1.}


def configuration(arm):
    return dict(arm=arm, exit_weights={str(k): v for k, v in EXIT_WEIGHTS.items()},
                overlap_scale={str(k): v for k, v in OVERLAP_SCALE.items()},
                group_block_scale=BLOCK_SCALE, reference_scale=1.)
