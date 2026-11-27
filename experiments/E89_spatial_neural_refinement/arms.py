ARMS={name:dict(width=192,layers=12,rectangle=True,ffn_dropout=.05,spatial_mode=name) for name in ("spatial","pooled")}
def model_config(arm):return dict(ARMS[arm])
