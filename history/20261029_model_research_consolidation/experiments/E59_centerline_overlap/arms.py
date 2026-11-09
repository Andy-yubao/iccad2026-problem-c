ARMS={
 'original96':dict(width=96,layers=12,rectangle=True,ffn_dropout=.05,graph_prenorm=False),
 'centerline96':dict(width=96,layers=12,rectangle=True,ffn_dropout=.05,graph_prenorm=False),
 'repair160':dict(width=160,layers=12,rectangle=True,ffn_dropout=.05,graph_prenorm=True),
 'large_original160':dict(width=160,layers=12,rectangle=True,ffn_dropout=.05,graph_prenorm=True),
 'large_centerline160':dict(width=160,layers=12,rectangle=True,ffn_dropout=.05,graph_prenorm=True),
}
def model_config(arm):return dict(ARMS[arm])
def weight_decay(arm):return 3e-4
