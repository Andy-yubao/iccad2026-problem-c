"""Architecture and trainable mask stay identical between Loss arms."""
ARMS={name:dict(width=192,layers=12,rectangle=True,ffn_dropout=.05) for name in ('simple','legacy')}
def model_config(arm):return dict(ARMS[arm])
