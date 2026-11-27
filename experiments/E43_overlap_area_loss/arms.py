"""One area-loss arm on the frozen E42 AB configuration."""
ARMS = {'rectangle_area': {'width': 96, 'layers': 12, 'rectangle': True, 'ffn_dropout': .05}}


def model_config(arm):
    return dict(ARMS[arm])
