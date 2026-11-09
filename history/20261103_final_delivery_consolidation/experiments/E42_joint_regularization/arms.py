"""Exactly one joint AB experiment; no augmentation or isolated A/B arms."""
ARMS = {'rectangle_ab': {'width': 96, 'layers': 12, 'rectangle': True, 'ffn_dropout': .05}}

def model_config(arm):
    return dict(ARMS[arm])
