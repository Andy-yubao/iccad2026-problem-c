"""Matched-parameter continuation versus coupled spatial correction."""
COMMON=dict(width=192,layers=12,rectangle=True,ffn_dropout=.05,corrector_width=192,message_width=64,max_correction=.15)
ARMS={name:dict(COMMON,cooperative=(name=='cooperative')) for name in ('control','cooperative')}
def model_config(arm):return dict(ARMS[arm])
