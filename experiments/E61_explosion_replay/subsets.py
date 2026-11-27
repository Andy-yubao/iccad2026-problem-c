def indices(size,seed):
    if size!=64000 or seed!=0:raise ValueError('E61 authorizes one64k/seed0 large model')
    return list(range(64000))
