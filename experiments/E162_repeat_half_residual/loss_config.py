"""Fixed independent stage objectives; all quantities refer to stage exits."""
ARMS=('e161_l0','e161_s0','e161_s1','e161_s2')
REFERENCE=(1.,.75,.5,.5)
OVERLAP=(.25,.25,1.,1.)
BLOCK=.25

def configuration(arm):
    k=ARMS.index(arm)
    return dict(version='E161.no_mib.exit_blockage.v1',arm=arm,stage=k,reference=REFERENCE[k],
        overlap_scale=OVERLAP[k],group_block_scale=BLOCK,block_aggregation='original MST edge capacity mean; worst obstacle per edge; free-contact-window gate',
        predict_all=True,geometry_feedback=False,intermediate_layouts=False,
        selection='same stage total as training/monitor/best/LR/earlystop')
