"""Trace five fixed training cases through unchanged E98 best71, without repair."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from shapely.geometry import box
from shapely.ops import unary_union

from datasets.reference_layouts import ROOT, ReferenceLayouts, registry
from ..batches import sample, collate, inputs_only
from ..generalization_baseline.model import GeometryModel, decode
from visualization.layer_trace import write_gallery, render_case, render_curves, render_overview

CHECKPOINT = ROOT / 'experiments/E98_loss_weight_learning/runs/20261104_generalization96_seed0/checkpoints/best.pt'
OUT = ROOT / 'experiments/E98_loss_weight_learning/runs/20261106_layer_diagnosis5'
GALLERY = ROOT / 'visualization/E98_layer_diagnosis/20261106_best71_train5'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def geometry(rects, reference, scale, constraints):
    p = np.asarray(rects, dtype=np.float64)
    areas = p[:, 2:].prod(1)
    union = unary_union([box(x, y, x + w, y + h) for x, y, w, h in p]).area
    high = p[:, :2] + p[:, 2:]
    bbox = float(np.prod(high.max(0) - p[:, :2].min(0)))
    centers = p[:, :2] + p[:, 2:] / 2
    target_centers = reference[:, :2] + reference[:, 2:] / 2
    free = np.asarray(constraints)[:, 1] == 0
    delta = centers[free] - target_centers[free]
    low = np.maximum(p[:, None, :2], p[None, :, :2])
    size = np.maximum(np.minimum(high[:, None], high[None]) - low, 0)
    inter = size.prod(-1)
    np.fill_diagonal(inter, 0)
    return dict(repeated_coverage_fraction=float(max(0, areas.sum() - union) / areas.sum()),
                pair_intersection_over_area=float(inter.sum() / 2 / areas.sum()),
                bbox_over_area=bbox / float(areas.sum()),
                blank_fraction=float(max(0, bbox - union) / max(bbox, 1e-12)),
                center_rmse_to_reference=float(np.sqrt(np.mean(np.sum(delta ** 2, axis=1))) / scale) if free.any() else 0.,
                overlapping_pairs=int((inter > 1e-8).sum() // 2))


def capture(model, graph):
    states, hidden, feedback, branches = [], [], [], [{} for _ in model.layers]
    handles = []
    def head_hook(module, args, result):
        hidden.append(args[0].detach().cpu().numpy()[0].copy())
        states.append(result.detach().clone())
    def feedback_hook(module, args, result):
        feedback.append(result.detach().cpu().numpy()[0].copy())
    handles += [model.head.register_forward_hook(head_hook),
                model.state_embed.register_forward_hook(feedback_hook)]
    for index, layer in enumerate(model.layers):
        for name, module in [('attention', layer.output), ('local', layer.local), ('ffn', layer.ff[2])]:
            def hook(module, args, result, i=index, key=name):
                branches[i][key] = result.detach().cpu().numpy()[0].copy()
            handles.append(module.register_forward_hook(hook))
    try:
        output = model(graph)
    finally:
        for handle in handles:
            handle.remove()
    if len(states) != 13 or not torch.equal(states[-1], output):
        raise AssertionError('Expected initial readout plus 12 true layer readouts')
    # On the same five cases only: prove extraction did not change the final output.
    plain = model(graph)
    if not torch.equal(output, plain):
        raise AssertionError('Hooks changed inference')
    return states, hidden, feedback, branches


def main():
    if OUT.exists() or GALLERY.exists():
        raise FileExistsError('Existing diagnostic evidence is not overwritten')
    OUT.mkdir(parents=True)
    GALLERY.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    saved = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    config = saved['model_config']
    model = GeometryModel(**config).eval()
    model.load_state_dict(saved['model'], strict=True)
    parameter_before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    reg = registry()
    records = reg['sets']['train']['records']
    indices = []
    for target in (21, 45, 70, 95, 120):
        indices.append(min((i for i in range(len(records)) if i not in indices),
                           key=lambda i: (abs(records[i]['n'] - target), i)))
    source_names = ['experiments/E98_loss_weight_learning/generalization_baseline/model.py',
                    'experiments/E98_loss_weight_learning/features.py',
                    'experiments/E98_loss_weight_learning/batches.py',
                    'datasets/reference_layouts.py', 'datasets/worksets.py',
                    'visualization/layer_trace.py',
                    'experiments/E98_loss_weight_learning/layer_diagnosis/run.py']
    frozen = dict(authorization='User explicitly requested intermediate-layout inspection on a few examples before proposing architecture changes.',
                  created_at=datetime.now(timezone.utc).isoformat(), checkpoint=str(CHECKPOINT.relative_to(ROOT)),
                  checkpoint_sha256=sha(CHECKPOINT), config=config, best_epoch=saved['progress']['best_epoch'],
                  inference='CPU FP32 eval/no_grad, batch1, hooks only; training used GPU FP32; not a latency/quality evaluation',
                  selection='First closest-N train member for targets 21,45,70,95,120; chosen using metadata before viewing outputs',
                  indices=indices, members=[records[i] for i in indices],
                  registry_sha256=reg['manifest_sha256'],
                  historical_input='Original registered E98 MIB mean-area transform retained for source fidelity. Not a new no-MIB baseline; no MIB penalty, Q or postprocessing computed.',
                  supervision='Free reference is used only after inference for visual/diagnostic comparison; never passed to model',
                  training=False, test_payload_read=False, final_output_bitwise_check='Hooked and unhooked forward compared on the same five cases',
                  sources={name: sha(ROOT / name) for name in source_names})
    write(OUT / 'frozen.json', frozen)
    dataset = ReferenceLayouts('train')
    cases, arrays = [], {}
    with torch.no_grad():
        for case_number, index in enumerate(indices):
            item = dataset[index]
            case = item['case']
            reference = np.asarray(item['positions'], dtype=np.float64)
            prepared = sample(case.solver_inputs, torch.as_tensor(reference).float())
            batch = collate([prepared], [0], 'cpu')
            graph = inputs_only(batch)
            states, hidden, feedback, branches = capture(model, graph)
            scale = float(graph['scale'][0])
            constraints = np.asarray(case.solver_inputs[5])
            free = constraints[:, 1] == 0
            shaped = constraints[:, :2].any(1) == 0
            stages = []
            previous_centers = previous_state = None
            for depth, state_tensor in enumerate(states):
                state = state_tensor[0].numpy()
                rects = decode(state_tensor, graph)[0].numpy().astype(np.float64) * scale
                centers = rects[:, :2] + rects[:, 2:] / 2
                metrics = geometry(rects, reference, scale, constraints)
                metrics.update(hidden_rms=rms(hidden[depth]),
                               raw_ratio_clamp_fraction=float((np.abs(state[shaped, 2]) >= 5).mean()) if shaped.any() else 0.,
                               free_center_at_origin_fraction=float((rects[free, :2] <= 1e-10).mean()) if free.any() else 0.,
                               decoded_center_shift_mean=float(np.linalg.norm(centers[free] / scale - state[free, :2], axis=1).mean()) if free.any() else 0.)
                if depth:
                    metrics.update(center_step_mean=float(np.linalg.norm((centers - previous_centers)[free], axis=1).mean() / scale),
                                   raw_center_step_mean=float(np.linalg.norm((state - previous_state)[free, :2], axis=1).mean()),
                                   hidden_change_relative=rms(hidden[depth] - hidden[depth - 1]) / max(rms(hidden[depth - 1]), 1e-12),
                                   feedback_rms=rms(feedback[depth - 1]),
                                   feedback_over_hidden=rms(feedback[depth - 1]) / max(rms(hidden[depth - 1]), 1e-12),
                                   **{key + '_rms': rms(value) for key, value in branches[depth - 1].items()})
                else:
                    metrics.update(center_step_mean=0., raw_center_step_mean=0.)
                stages.append(dict(depth=depth, positions=rects.tolist(), metrics=metrics))
                previous_centers, previous_state = centers, state
            all_rects = np.concatenate([reference] + [np.asarray(s['positions']) for s in stages])
            low = all_rects[:, :2].min(0)
            high = (all_rects[:, :2] + all_rects[:, 2:]).max(0)
            span = max(high - low)
            center = (low + high) / 2
            limits = [float(center[0] - .55 * span), float(center[0] + .55 * span),
                      float(center[1] - .55 * span), float(center[1] + .55 * span)]
            value = dict(id=f'case{case_number + 1:02d}', train_index=index, n=len(reference),
                         member=item['member'], scale=scale, constraints=constraints.tolist(),
                         reference=reference.tolist(), axis_limits=limits, stages=stages)
            cases.append(value)
            arrays[f'case{case_number + 1}_hidden'] = np.stack(hidden)
            arrays[f'case{case_number + 1}_raw_state'] = np.stack([s[0].numpy() for s in states])
            arrays[f'case{case_number + 1}_feedback'] = np.stack(feedback)
            render_case(value, GALLERY / f'{value["id"]}_stages.png')
            render_curves(value, GALLERY / f'{value["id"]}_curves.png')
            print(f"Traced {value['id']}: N={value['n']}, all 13 readouts; unchanged final output", flush=True)
    if not all(torch.equal(parameter_before[k], v) for k, v in model.state_dict().items()):
        raise AssertionError('Model parameters changed')
    if sha(CHECKPOINT) != frozen['checkpoint_sha256']:
        raise AssertionError('Checkpoint changed')
    np.savez_compressed(OUT / 'hidden_traces.npz', **arrays)
    result = dict(identity=frozen, cases=cases,
                  caveats=['Five training cases, not generalization evidence.',
                           'Only final layer was supervised by the training loss. Intermediate shared-head readouts were not trained as independent early exits.',
                           'Clamped decoder can change center/shape interpretation; raw state and clamp rates are retained.',
                           'Observational traces cannot determine the causal benefit of removing position feedback or reallocating depth.'])
    write(OUT / 'trace.json', result)
    write_gallery(result, GALLERY)
    render_overview(cases, GALLERY / 'overview.png')
    write(OUT / 'verification.json', dict(status='completed', case_count=5, head_calls_per_case=13,
          all_final_outputs_equal=True, parameters_unchanged=True, checkpoint_unchanged=True,
          training=False, test_payload_read=False, postprocessing_calls=0, Q_calls=0,
          gallery=str(GALLERY.relative_to(ROOT))))
    print('Completed:', GALLERY / 'index.html', flush=True)


if __name__ == '__main__':
    main()
