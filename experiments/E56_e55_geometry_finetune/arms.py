"""Unchanged E55 architecture; one E42-geometry post-training arm."""
ARMS = {'e42_geometry': {'width': 96, 'layers': 12, 'rectangle': True, 'ffn_dropout': .05}}


def model_config(arm):
    return dict(ARMS[arm])
