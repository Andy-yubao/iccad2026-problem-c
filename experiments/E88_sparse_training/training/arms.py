"""One fixed-weight ablation; identical E87 model and trainable mask."""
ARMS={"simple":dict(width=192,layers=12,rectangle=True,ffn_dropout=.05)}
def model_config(arm):return dict(ARMS[arm])
