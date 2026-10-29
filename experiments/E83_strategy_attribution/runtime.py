"""Deterministic inference runtime without any optimizer/training entrypoint."""
import random
import numpy as np
import torch
def seed_all(seed,threads=2):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
