"""Paired E42 post-training arms; unchanged public-input architecture."""
ARMS = {name: {'width': 96, 'layers': 12, 'rectangle': True, 'ffn_dropout': .05}
        for name in ('control', 'weight075', 'softband005')}


def model_config(arm):
    return dict(ARMS[arm])
