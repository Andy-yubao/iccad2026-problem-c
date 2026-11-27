"""Prespecified capacity study. Architecture candidates await capacity results."""
ARMS = {
    'base96x4': {'width': 96, 'layers': 4},
    'deep96x8': {'width': 96, 'layers': 8},
    'deep96x12': {'width': 96, 'layers': 12},
    'wide160x4': {'width': 160, 'layers': 4},
}


def model_config(arm):
    return dict(ARMS[arm])
