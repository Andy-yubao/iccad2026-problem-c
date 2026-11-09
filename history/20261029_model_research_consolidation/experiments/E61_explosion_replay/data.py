"""Fixed-padding CPU batching; only a minibatch is transferred to the GPU."""
from .cache import load_rows
from .batches import collate as batch_collate
def collate(rows,device):return batch_collate([r['graph'] for r in rows],list(range(len(rows))),device,max_nodes=120),None
def register_resident(pack):raise RuntimeError('E59 uses staged batches, no whole-dataset GPU pack')
