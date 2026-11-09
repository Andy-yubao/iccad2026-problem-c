"""Frozen eval-only forwards for missing case/module diagnostics; no optimizer exists."""
import gc
import math
import numpy as np
import torch
import torch.nn.functional as F
from .io import ROOT, HERE, read, save, sha
from .data import load_rows, collate
from .batches import inputs_only
from .base_model import GeometryModel as E42Model, decode
from .model import GeometryModel as E65Model
from .losses import components, WEIGHTS

RUNS = {
    'E42': ROOT/'experiments/E42_joint_regularization/runs/20261023_rectangle_w96d12_ab_seed0',
    'E64': ROOT/'experiments/E64_e42_unified_retrain/runs/20261027_e42_fixedG_scratch_seed0',
    'E65_geometry': ROOT/'experiments/E65_geometry_messages/runs/20261027_geometry_message_seed0',
    'E65_feature': ROOT/'experiments/E65_geometry_messages/runs/20261027_feature_message_seed0',
    'E66_unshared': ROOT/'experiments/E66_shared_refinement/runs/20261028_unshared_residual_seed0',
    'E66_shared': ROOT/'experiments/E66_shared_refinement/runs/20261028_shared_residual_seed0',
    'E67_group': ROOT/'experiments/E67_multiscale_tokens/runs/20261028_group_tokens_seed0',
}


def key(member):
    return member[0]+':'+str(member[1])


def raw_overlap(positions, inputs):
    p = positions.astype(float).copy()
    c, specs = np.asarray(inputs[5]), np.asarray(inputs[6])
    locked = c[:, :2].any(1)
    p[locked, 2:] = specs[locked, 2:]
    p[c[:, 1] != 0] = specs[c[:, 1] != 0]
    lo = np.maximum(p[:, None, :2], p[None, :, :2])
    hi = np.minimum((p[:, :2]+p[:, 2:])[:, None], (p[:, :2]+p[:, 2:])[None])
    inter = np.maximum(hi-lo, 0).prod(-1)
    return float(np.triu(inter, 1).sum()/np.asarray(inputs[1]).sum())


def module_diagnostics(state, graph):
    valid = graph['valid']; mask = graph['mask'][..., 0].bool() & valid
    local = graph['area'].clamp_min(1e-12).sqrt()
    diff = state[..., :2]-graph['target'][..., :2]
    rho = F.smooth_l1_loss(state[..., :2]/local[..., None], graph['target'][..., :2]/local[..., None],
                          reduction='none', beta=.1).mean(-1)
    arrays = [x.detach().cpu().numpy() for x in (diff, rho, local, mask, valid)]
    result = []
    for difference, center_loss, size, mutable, present in zip(*arrays):
        count = int(present.sum()); difference = difference[:count]; size = size[:count]
        mutable = mutable[:count]; center_loss = center_loss[:count]*mutable
        delta = np.linalg.norm(difference, axis=-1)
        total = center_loss.sum()
        mean_shift = difference[mutable].mean(0) if mutable.any() else np.zeros(2)
        raw_mse = float((difference[mutable]**2).sum(-1).mean()) if mutable.any() else 0.
        # Diagnostic normalization only; no new model candidate is generated or scored.
        corrected = (difference-mean_shift)/np.maximum(size[:, None], 1e-12)
        abs_error = abs(corrected)
        corrected_rho = np.where(abs_error < .1, .5*corrected**2/.1, abs_error-.05).mean(-1)
        row = {'mean_center_error_global': float(delta[mutable].mean()) if mutable.any() else 0.,
               'mean_center_error_local': float((delta/size)[mutable].mean()) if mutable.any() else 0.,
               'mean_shift_global': float(np.linalg.norm(mean_shift)),
               'translation_fraction_center_mse': float(np.dot(mean_shift, mean_shift)/max(raw_mse, 1e-12)),
               'center_loss_after_remove_mean_shift': float(corrected_rho[mutable].mean()) if mutable.any() else 0.,
               'top10pct_modules_center_loss_share': float(np.sort(center_loss)[-max(1, math.ceil(.1*count)):].sum()/max(total, 1e-12))}
        for label, indices in zip(('small', 'middle', 'large'), np.array_split(np.argsort(size, kind='stable'), 3)):
            chosen = indices[mutable[indices]]
            row[label+'_center_loss_share'] = float(center_loss[indices].sum()/max(total, 1e-12))
            row[label+'_center_error_global'] = float(delta[chosen].mean()) if len(chosen) else 0.
            row[label+'_center_error_local'] = float((delta[chosen]/size[chosen]).mean()) if len(chosen) else 0.
        result.append(row)
    return result


@torch.no_grad()
def infer(name, rows, check_deadline, *, validate_only=False):
    checkpoint = RUNS[name]/'checkpoints/best.pt'
    digest = sha(checkpoint)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cls = E42Model if name == 'E42' else E65Model
    model = cls(**state['model_config']).cuda().eval()
    model.load_state_dict(state['model'], strict=True)
    del state
    stored = {key(row['member']): row for row in read(RUNS['E65_geometry']/'full_dev.json')['records']}
    results = []; reference = []; largest_stored_error = 0.
    for start in range(0, len(rows), 32):
        check_deadline()
        selected = rows[start:start+32]
        graph, _ = collate(selected, 'cuda')
        prediction = model(inputs_only(graph))
        values = components(prediction, graph)
        summaries = module_diagnostics(prediction, graph)
        positions = (decode(prediction, graph)*graph['scale'][:, None, None]).cpu().numpy()
        term_matrix = torch.stack([values[k] for k in WEIGHTS]+[values['geometry']], -1).cpu().numpy()
        if name == 'E42':
            # Counterfactual reference-anchored decode, not an optimum or lower bound.
            anchor = components(graph['target'], graph)
            anchor_matrix = torch.stack([anchor[k] for k in WEIGHTS]+[anchor['geometry']], -1).cpu().numpy()
        for j, row in enumerate(selected):
            terms = dict(zip([*WEIGHTS, 'total'], map(float, term_matrix[j])))
            assert abs(sum(WEIGHTS[k]*terms[k] for k in WEIGHTS)-terms['total']) < 1e-5
            if name == 'E65_geometry':
                old = stored[key(row['member'])]
                largest_stored_error = max(largest_stored_error, max(abs(terms[k]-old['loss_'+k]) for k in [*WEIGHTS, 'total']))
            results.append({'member': list(row['member']), **{'loss_'+k: v for k, v in terms.items()},
                            'overlap_area_fraction': raw_overlap(positions[j, :row['graph']['n']], row['case'].solver_inputs),
                            **summaries[j]})
            if name == 'E42':
                reference.append({'member': list(row['member']), **dict(zip([*WEIGHTS, 'total'], map(float, anchor_matrix[j])))})
        if not validate_only and (len(results) % 320 == 0 or len(results) == len(rows)):
            print('Frozen diagnostic forward', name, len(results), '/', len(rows), flush=True)
    if name == 'E65_geometry':
        assert largest_stored_error < 2e-5, largest_stored_error
    if not validate_only:
        expected = read(RUNS[name]/'full_dev.json')['losses']['geometry']
        actual = float(np.mean([row['loss_total'] for row in results]))
        assert abs(expected-actual) < 1e-5, (name, expected, actual)
    assert sha(checkpoint) == digest
    del model, graph, prediction, values
    gc.collect(); torch.cuda.empty_cache()
    return results, reference, {'checkpoint_sha256': digest, 'stored_per_case_max_abs_error': largest_stored_error,
                                 'count': len(rows), 'optimizer_created': False, 'training_mode': False}


def payload(row):
    inputs = tuple(int(x) if i == 0 else np.asarray(x) for i, x in enumerate(row['case'].solver_inputs))
    reference = row['graph']['reference'].numpy()*float(row['graph']['scale'])
    return list(row['member']), inputs, reference, row['graph']['features'].numpy()
