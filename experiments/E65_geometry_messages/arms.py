"""Paired additive message-value experiment, fixed original E42 trunk."""
ARMS = {name: {'width': 96, 'layers': 12, 'rectangle': True, 'ffn_dropout': .05,
               'message_mode': mode, 'message_width': 24, 'message_scale': .1}
        for name, mode in (('geometry_message', 'geometry'), ('feature_message', 'features'), ('baseline', 'none'))}


def model_config(arm):
    return dict(ARMS[arm])
