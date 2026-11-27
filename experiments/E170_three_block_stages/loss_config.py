"""Loss and learning-rate parameters frozen by the E170 plan."""
import json
from pathlib import Path

PLAN = json.loads(Path(__file__).with_name('plan.json').read_text(encoding='utf-8'))
TRAINING_ARMS = (*PLAN['stages'], 'joint')
PREFIX_LR_SCALE = PLAN['prefix_lr_scale']
BLOCK_SCALE = PLAN['group_block_scale']
OVERLAP_SCALE = PLAN['overlap_scale']
REFERENCE_SCALE = PLAN['reference_scale']
SOFT_COUNT_SCALE = PLAN['soft_count_scale']
SOFT_COUNT_TAU = PLAN['soft_count_tau']


def exit_weights(stage, joint):
    if not joint:
        return {stage: PLAN['protected_exit_weight']}
    return {**dict.fromkeys(range(stage), PLAN['joint_prefix_weight'] / stage),
            stage: PLAN['joint_current_exit_weight']}


def configuration(stage, joint):
    return dict(stage=stage, joint=joint, exit_weights={str(i): w for i, w in exit_weights(stage, joint).items()},
                overlap_scale=OVERLAP_SCALE[:stage + 1], reference_scale=REFERENCE_SCALE[:stage + 1],
                soft_count_scale=SOFT_COUNT_SCALE[:stage + 1], group_block_scale=BLOCK_SCALE,
                soft_count_tau=SOFT_COUNT_TAU, prefix_lr_scale=PREFIX_LR_SCALE)
