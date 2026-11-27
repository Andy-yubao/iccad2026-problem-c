"""Same token network and initialization; only the new branch's group incidence differs."""
ARMS = {name: {'width': 96, 'layers': 12, 'rectangle': True, 'ffn_dropout': .05,
               'pool_mode': mode, 'token_width': 24, 'token_scale': .1}
        for name, mode in (('group_tokens', 'groups'), ('global_tokens', 'global'))}


def model_config(arm):
    return dict(ARMS[arm])
