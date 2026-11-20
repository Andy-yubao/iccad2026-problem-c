"""Reproducible address renaming independent of dropout and axis-swap RNG."""
import numpy as np
import torch

DOMAIN = 20261112131


def address_rows(counts, indices, seed=0, epoch=None, max_nodes=120):
    if len(counts) != len(indices):
        raise ValueError('Address membership mismatch')
    output = np.tile(np.arange(max_nodes, dtype=np.int64), (len(indices), 1))
    for row, (n, index) in enumerate(zip(counts, indices)):
        n, index = int(n), int(index)
        if not 0 < n <= max_nodes <= 128 or index < 0:
            raise ValueError('Invalid node count/address member')
        if epoch is not None:
            rng = np.random.default_rng(np.random.SeedSequence([DOMAIN, int(seed), int(epoch), index]))
            output[row, :n] = rng.permutation(n)
    return torch.from_numpy(output)


def attach_addresses(batch, rows, indices, seed=0, epoch=None):
    counts = [rows[int(i)]['graph']['n'] for i in indices]
    address = address_rows(counts, indices, seed, epoch, batch['valid'].shape[1])
    batch['address'] = address.to(batch['valid'].device, non_blocking=False)
    return batch
