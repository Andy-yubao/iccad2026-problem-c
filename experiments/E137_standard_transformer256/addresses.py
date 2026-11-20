"""Independent deterministic public address renaming; no labels or dropout RNG."""
import numpy as np
import torch

DOMAIN = 20261114137
DIAGNOSTIC_EPOCH = 1000000


def address_rows(counts, indices, seed=0, epoch=None, max_nodes=120):
    result = np.zeros((len(counts), max_nodes), dtype=np.int64)
    for row, (count, index) in enumerate(zip(counts, indices)):
        count, index = int(count), int(index)
        if not 0 < count <= max_nodes <= 128 or index < 0:
            raise ValueError('Invalid public module count/address membership')
        result[row, :count] = (np.arange(count) if epoch is None else
            np.random.default_rng(np.random.SeedSequence([DOMAIN, int(seed), int(epoch), index])).permutation(128)[:count])
    return torch.from_numpy(result)


def attach_addresses(batch, rows, indices, seed=0, epoch=None):
    ids = [int(i) for i in indices]
    counts = [rows[i]['graph']['n'] for i in ids]
    batch['address'] = address_rows(counts, ids, seed, epoch, batch['valid'].shape[1]).to(batch['valid'].device)
    return batch
