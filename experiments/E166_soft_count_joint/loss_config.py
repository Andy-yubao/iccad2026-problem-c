ARM = 'e166_backend'
PROTECTED_EPOCHS = 20
L0_LR_SCALE = .1
BLOCK_SCALE = .25
L0_AUX_WEIGHT = .05
EXIT_WEIGHTS = {4: .1, 8: .1, 12: .8}
OVERLAP_SCALE = {0: .25, 4: 0., 8: 1., 12: 1.}
REFERENCE_SCALE = {0: 1., 4: 1., 8: 1., 12: .25}
SOFT_COUNT_SCALE = {8: .01, 12: .02}
SOFT_COUNT_TAU = .01


def configuration(arm):
    return dict(arm=arm, exit_weights={str(k): v for k, v in EXIT_WEIGHTS.items()},
                overlap_scale={str(k): v for k, v in OVERLAP_SCALE.items()},
                l0_aux_weight=L0_AUX_WEIGHT, group_block_scale=BLOCK_SCALE,
                reference_scale={str(k): v for k, v in REFERENCE_SCALE.items()},
                soft_count_scale={str(k): v for k, v in SOFT_COUNT_SCALE.items()},
                soft_count_tau=SOFT_COUNT_TAU, protected_backend_epochs=PROTECTED_EPOCHS,
                l0_lr_scale=L0_LR_SCALE)
