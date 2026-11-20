"""Read-only audit of frozen monitors and saved dev500 geometry; no inference."""
import json
import pathlib
import types
import torch
import torch.nn.functional as F
from experiments.E153_mib_role_topology import simple_loss
from experiments.E153_mib_role_topology.base_loss import A, B, masked_channels_mean
from experiments.E153_mib_role_topology.base_resident import ResidentRows
from experiments.E153_mib_role_topology.data import load_rows
from experiments.E153_mib_role_topology.group_tree import minimum_group_tree
from experiments.E153_mib_role_topology.loss import layer_loss

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[3]
EVAL = HERE.parent / 'e153_backend_seed0/evaluation_mib_dev500'
PUBLIC = types.FunctionType(simple_loss.public_components.__code__,
    dict(simple_loss.public_components.__globals__, decode=lambda p, g: p, _BITS={}))
TERMS = ('overlap_weighted', 'wire_compact_weighted', 'grouping_weighted',
         'boundary_weighted', 'public_shape_weighted',
         'public_preplaced_position_weighted', 'mib_weighted', 'group_bottleneck_weighted')


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def coordinate_loss(p, g):
    """L12 geometry terms, reference=0, drift excluded; preserve actual XYWH."""
    out = PUBLIC(p, {k: g[k] for k in simple_loss.PUBLIC})
    valid = g['valid']
    c = g['constraints']
    spec = g['specs']
    fixed = valid & ((c[..., 0] != 0) | (c[..., 1] != 0))
    pre = valid & (c[..., 1] != 0)
    wh = p[..., 2:].clamp_min(1e-12)
    safe_wh = torch.where(fixed[..., None], wh, torch.ones_like(wh))
    spec_wh = torch.where(fixed[..., None], spec[..., 2:], torch.ones_like(wh)).clamp_min(1e-12)
    shape = masked_channels_mean(F.smooth_l1_loss(safe_wh.log(), spec_wh.log(),
        reduction='none', beta=.1), fixed)
    scale = torch.where(pre[..., None], spec[..., 2:], torch.ones_like(wh)).clamp_min(1e-12).prod(-1).sqrt()
    center = p[..., :2] + wh / 2
    target_center = torch.where(pre[..., None], spec[..., :2] + spec[..., 2:] / 2, torch.zeros_like(center))
    position = masked_channels_mean(F.smooth_l1_loss(center / scale[..., None],
        target_center / scale[..., None], reduction='none', beta=.1), pre)
    same = (g['mib'] > 0) & valid[:, :, None] & valid[:, None, :]
    # Equal-area training: .5 * sum(log-WH differences squared) == .25 * log-ratio difference squared.
    difference = .5 * (wh.log()[:, :, None] - wh.log()[:, None, :]).square().sum(-1)
    peers = same.sum(-1)
    node = (difference * same).sum(-1) / peers.clamp_min(1)
    mib = (node * (peers > 0)).sum(-1) / (peers > 0).sum(-1).clamp_min(1)
    local = g['area'].clamp_min(1e-12).sqrt()
    costs = simple_loss.contact_cost(p, (local[:, :, None] + local[:, None, :]) / 2)
    _, tree = minimum_group_tree(costs, g['group'], valid, return_tree=True)
    result = dict(overlap_weighted=A*out['overlap_total'],
        wire_compact_weighted=.5*B*(out['wirelength']+out['compactness']),
        grouping_weighted=2*B*out['grouping_contribution'],
        boundary_weighted=2*B*out['boundary_contribution'],
        public_shape_weighted=shape, public_preplaced_position_weighted=position,
        mib_weighted=2*B*mib, group_bottleneck_weighted=2*B*(costs*tree).amax((1, 2)))
    result['total_without_reference_or_drift'] = sum(result.values())
    return result


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cases = [read(p) for p in sorted((EVAL/'cases').glob('*.json'))]
    assert len(cases) == 500
    rows = load_rows('dev_monitor500')
    assert all(list(r['member']) == c['member'] for r, c in zip(rows, cases))
    pack = ResidentRows(rows, 'cpu', max_nodes=120)
    values = {'draft': [], 'final': []}
    equivalence = 0.
    with torch.inference_mode():
        for start in range(0, 500, 20):
            ids = list(range(start, min(start+20, 500)))
            g = pack.batch(ids)
            for stage, key in [('draft', 'draft_positions'), ('final', 'positions')]:
                p = torch.zeros(len(ids), 120, 4)
                for j, i in enumerate(ids):
                    p[j, :cases[i]['n']] = torch.tensor(cases[i][key]) / g['scale'][j]
                output = coordinate_loss(p, g)
                for j, i in enumerate(ids):
                    values[stage].append(dict(index=i, **{k:float(v[j]) for k,v in output.items()}))
                if stage == 'draft':
                    state = torch.cat((p[..., :2]+p[..., 2:]/2,
                        (p[..., 2].clamp_min(1e-12)/p[..., 3].clamp_min(1e-12)).log()[..., None]), -1)
                    registered = layer_loss(state, g, 0., True)
                    for k in TERMS:
                        equivalence = max(equivalence, float((output[k]-registered[k]).abs().max()))
    summary = {stage: {k:sum(v[k] for v in records)/500 for k in records[0] if k!='index'}
               for stage, records in values.items()}
    initial = {}
    for exp, run in [('E149_role_initial','e149_role_initial_seed0'),
                     ('E153_mib_role_topology','e153_initial_seed0')]:
        p = ROOT/'experiments'/exp/'runs'/run
        progress = read(p/'progress.json')
        best = next(h for h in progress['history'] if h['epoch']==progress['best_epoch'])
        initial[exp] = dict(best_epoch=progress['best_epoch'], dev=best['dev_monitor'])
    bad = [dict(index=c['index'], draft_ratio=1+c['raw_metrics']['area_gap'],
        final_ratio=1+c['metrics']['area_gap'], Q=c['metrics']['cost'],
        grouping_violations=c['metrics']['grouping_violations'],
        rescue_seeds=c['work']['rescue_seeds'],
        projection_infeasible=any(t.get('status')=='PrimalInfeasible' for t in c['trace']))
        for c in cases if c['metrics']['area_gap']>.2]
    bad.sort(key=lambda c:c['final_ratio'], reverse=True)
    result = dict(scope='Actual saved XYWH; L12 geometry terms, no reference or L8-to-L12 drift; no decoder reprojection. Not aggregate training Loss.',
        draft_reencoding_max_component_error=equivalence, initial_best=initial,
        coordinate_means=summary, area_over_1p2=bad, per_case=values,
        final_result=read(EVAL/'results.json'), test_read=False, inference_calls=0, postprocessing_calls=0)
    (HERE/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('per_case','initial_best')},indent=2))


if __name__ == '__main__':
    main()
