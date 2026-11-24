ARMS = ('e165_l0', 'e165_backend')
PROTECTED_EPOCHS = 20
L0_LR_SCALE = .1
BLOCK_SCALE = .25
L0_AUX_WEIGHT = .05
EXIT_WEIGHTS = {4: .1, 8: .1, 12: .8}
OVERLAP_SCALE = {0: .25, 4: 0., 8: 1., 12: 1.}


def configuration(arm):
    return dict(arm=arm, exit_weights={str(k): v for k, v in EXIT_WEIGHTS.items()},
                overlap_scale={str(k): v for k, v in OVERLAP_SCALE.items()},
                l0_aux_weight=L0_AUX_WEIGHT, group_block_scale=BLOCK_SCALE,
                reference_scale=1., protected_backend_epochs=PROTECTED_EPOCHS,
                l0_lr_scale=L0_LR_SCALE)
