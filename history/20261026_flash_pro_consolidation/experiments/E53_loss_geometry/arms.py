"""Single unified-Loss arm on the unchanged E42 AB architecture."""
ARMS = {name: {'width': 96, 'layers': 12, 'rectangle': True, 'ffn_dropout': .05}
        for name in ('geometry','tolerant','clearance','wire_guard','anchor_ablation','count_ablation')}


def model_config(arm):
    return dict(ARMS[arm])
