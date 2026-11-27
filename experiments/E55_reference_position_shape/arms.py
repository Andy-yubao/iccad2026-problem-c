"""One scratch-trained reference-position/shape control."""
ARMS = {'reference_position_shape': {'width': 96, 'layers': 12,
        'rectangle': True, 'ffn_dropout': .05}}


def model_config(arm):
    return dict(ARMS[arm])
