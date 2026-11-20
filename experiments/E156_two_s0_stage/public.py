"""Public-only MIB and structural-role contract for every stage."""
BASE=('features','adjacency','mib','group','mask','area','known','specs','constraints','scale','valid')
EXTRA=('role_features','port_features','port_valid','role_hops','role_edges','role_group')
PUBLIC=BASE+EXTRA
def inputs_only(batch):return {key:batch[key] for key in PUBLIC}
