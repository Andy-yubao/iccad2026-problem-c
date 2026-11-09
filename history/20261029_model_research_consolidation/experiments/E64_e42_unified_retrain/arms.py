"""One scratch baseline, exactly the frozen E42 public-input architecture."""
ARMS = {'rectangle_ab': {'width': 96, 'layers': 12, 'rectangle': True, 'ffn_dropout': .05}}


def model_config(arm):
    return dict(ARMS[arm])
