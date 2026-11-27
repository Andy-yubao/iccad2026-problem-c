"""Exact E74 architecture; fresh all-trainable parameters."""
ARMS={"simple":dict(width=192,layers=12,rectangle=True,ffn_dropout=.05)}
def model_config(arm):return dict(ARMS[arm])
