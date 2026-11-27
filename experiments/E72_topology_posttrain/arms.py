"""Paired E42 post-training arms; unchanged public-input architecture."""
ARMS = {name: {'width': 192, 'layers': 12, 'rectangle': True, 'ffn_dropout': .05}
        for name in ('topology',)}


def model_config(arm):
    return dict(ARMS[arm])
