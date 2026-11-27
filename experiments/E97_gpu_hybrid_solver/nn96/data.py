"""Public interface only; no dataset loader or private cross-experiment import."""
LEGACY=('features','adjacency','mib','group','mask','area','known','specs','constraints','scale','valid')
EXTRA=('port_xy','port_valid','p2b_weights','b2b_weights')
PUBLIC=(*LEGACY,*EXTRA)
def placer_inputs(g):return {k:g[k] for k in LEGACY}
