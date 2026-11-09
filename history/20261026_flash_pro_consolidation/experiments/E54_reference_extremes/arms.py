"""Paired weights-only branches; identical frozen architecture."""
ARMS = {name: {'width':96, 'layers':12, 'rectangle':True, 'ffn_dropout':.05}
        for name in ('no_reference','reference_only')}

def model_config(arm):
    return dict(ARMS[arm])
