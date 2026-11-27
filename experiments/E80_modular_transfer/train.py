"""Frozen diagnostic utilities only; no training entry point or optimizer."""
import random,time
import numpy as np
import torch
from .data import collate
from .model import decode
from .batches import inputs_only
from .loss import per_case
from .diagnostics import geometry_checks,distribution
def seed_all(seed, threads=4):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)

def selection(values):
    # Exactly the scalar optimized by backward(), with fixed full weights.
    return values['total']

@torch.no_grad()
def monitor_losses(model, rows, device, arm, epoch, batch_size, *, should_stop=lambda: False, start_epoch=6, return_records=False):
    model.eval(); sums = {}; hard_legal = 0; inference_seconds = 0.; diagnostics = []
    for start in range(0, len(rows), batch_size):
        if should_stop():
            return None
        selected = [rows[i] for i in range(start, min(start+batch_size, len(rows)))]
        if str(device).startswith('cuda'): torch.cuda.synchronize(device)
        started = time.perf_counter()
        graph, targets = collate(selected, device)
        state = model(inputs_only(graph))
        positions = (decode(state, graph)*graph['scale'][:, None, None]).cpu().double().numpy()
        if str(device).startswith('cuda'): torch.cuda.synchronize(device)
        inference_seconds += time.perf_counter()-started
        per_layout = per_case(state, graph, arm)
        values = {key: value.mean() for key, value in per_layout.items()}
        diagnostic_keys = ('common_original_G', 'topology_regression', 'topology_flip_fraction', 'source_true_intersection', 'center_local', 'center_objective', 'shape', 'relative_local', 'overlap', 'contact', 'boundary', 'mib', 'bbox', 'collision_topology', 'true_intersection', 'box_expansion', 'topology_unconditional', 'total')
        loss_records = torch.stack([per_layout[k] for k in diagnostic_keys], -1).cpu().tolist()
        for row, predicted, row_losses in zip(selected, positions, loss_records):
            _, checks, diagnostic = geometry_checks(predicted, row['case'].solver_inputs)
            diagnostic.update({'loss_'+k: value for k, value in zip(diagnostic_keys, row_losses)})
            hard_legal += int(checks['is_feasible'])
            diagnostics.append({'member': list(row['member']), **diagnostic})
        keys=list(values)
        scalars=torch.stack([values[k].detach() for k in keys]).cpu().tolist()
        for key, value in zip(keys,scalars): sums[key] = sums.get(key, 0.) + value*len(selected)
    result = {key: value/len(rows) for key, value in sums.items()}
    result.update(count=len(rows), selection_proxy=selection(result), raw_hard_legal=hard_legal,
                  raw_hard_legal_rate=hard_legal/len(rows),
                  batched_inference_seconds_per_case=inference_seconds/len(rows))
    result['draft_diagnostics'] = distribution(diagnostics, [k for k in diagnostics[0] if k != 'member'])
    if return_records:
        result['diagnostic_records'] = diagnostics
    return result
