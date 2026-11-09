ARMS={'large_original160':dict(width=160,layers=12,rectangle=True,ffn_dropout=.05,graph_prenorm=False)}
def model_config(arm):return dict(ARMS[arm])
def weight_decay(arm):
    if arm not in ARMS:raise ValueError(arm)
    return .0003
