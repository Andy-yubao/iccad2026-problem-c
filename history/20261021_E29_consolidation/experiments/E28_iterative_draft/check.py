"""Five unique cases: three exposed train/dev rows and two terminal counterexamples."""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from .io import ROOT, HERE, save, checkpoint, sources, snapshot
from .data import SOURCE_CACHE, source_manifest, augment, collate, contact_tree, public_row
from .model import ARMS, DraftModel, inputs_only, trainable_state
from .prepare import SUPPLEMENT_KEYS
from .train import seed_all, make_model, run as train_run
from .loss import loss, wire
from .base_model import decode
from .policy import PlateauController, LR_LEVELS
from .final_evaluate import predict, freeze_once, summary
from datasets.reference_layouts import ReferenceLayouts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tag', default='20261021_smoke5_final_seed0')
    args = ap.parse_args()
    output = (HERE / 'runs' / args.tag).resolve()
    if not output.is_relative_to(HERE / 'runs'):
        raise ValueError('Check output must stay in E28/runs')
    output.mkdir(parents=True, exist_ok=False)
    seed_all(0); begin = time.perf_counter(); device = 'cuda' if torch.cuda.is_available() else 'cpu'
    original = {}
    for role in ('train', 'dev'):
        manifest = source_manifest(role)
        value = torch.load(SOURCE_CACHE / role / manifest['parts'][0]['name'], map_location='cpu', weights_only=False)
        original[role] = value['rows'][:2 if role == 'train' else 1]
    rows = [augment(r) for r in original['train'] + original['dev']]
    prototype = make_model('recurrent_full', device).eval()
    with torch.no_grad():
        graph = collate(rows, device)
        initial = prototype.initial(inputs_only(graph)).cpu()
    for row, draft in zip(rows, initial):
        row['graph']['draft'] = draft[:row['graph']['n']].clone()
        row['graph']['contact_tree'] = contact_tree(row, row['graph']['draft'])
    counts = {}; checks = {}
    for arm in ARMS:
        seed_all(0); model = make_model(arm, device).eval(); graph = collate(rows, device)
        with torch.no_grad():
            predicted = model(inputs_only(graph))
        counts[arm] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if not torch.equal(predicted, graph['draft']):
            raise AssertionError('Zero-output refiner must preserve frozen draft exactly')
    assert len(set(counts.values())) == 1
    checks['matched_trainable_parameters'] = counts
    checks['zero_head_preserves_draft_all_arms'] = True
    graph = collate(rows, device); model = make_model('recurrent_full', device).eval()
    poisoned = {k: v.clone() for k, v in graph.items()}
    poisoned['target'].fill_(1000); poisoned['reference'].fill_(-1000)
    with torch.no_grad():
        assert torch.equal(model(inputs_only(graph)), model(inputs_only(poisoned)))
    try:
        model(graph)
        raise AssertionError('Network accepted supervision keys')
    except ValueError:
        pass
    for role in ('local_test', 'test', 'tail_test'):
        try:
            ReferenceLayouts(role)
            raise AssertionError('Sealed test role accepted')
        except PermissionError:
            pass
    checks['input_isolation_and_test_rejection'] = True
    rebuilt_public = public_row(rows[0]['case'].solver_inputs, ('unrelated_file_identity', 999))
    assert torch.equal(rebuilt_public['graph']['latent'], rows[0]['graph']['latent'])
    checks['random_keys_require_only_public_inputs'] = True
    graph = collate(rows, device)
    state = graph['draft']
    with torch.no_grad():
        model = make_model('single_full', device).eval()
        boxes = decode(state, graph); center = boxes[..., :2] + boxes[..., 2:] / 2
        delta = graph['pin_xy'][:, None] - center[:, :, None]
        dense = (model.pin_encoder(torch.cat((delta, delta.abs()), -1)) * graph['pin_weights'][..., None]).sum(2)
        dense = dense / graph['pin_weights'].sum(-1, keepdim=True).clamp_min(1e-8)
        sparse = model.terminal_message(graph, state)
    assert torch.allclose(dense, sparse, atol=2e-6, rtol=1e-5)
    checks['sparse_terminal_matches_dense'] = {'max_difference': float((dense - sparse).abs().max()),
        'terminal_slots': int(graph['pin_xy'].shape[1]), 'max_connected_slots': int(graph['pin_indices'].shape[-1])}
    # Exercise the actual supplemental serialization format without a full data stage.
    supplement_path = output / 'supplement_roundtrip.pt'
    checkpoint(supplement_path, {'rows': [{'member': list(row['member']),
        'graph': {key: row['graph'][key] for key in SUPPLEMENT_KEYS}} for row in rows]})
    saved = torch.load(supplement_path, map_location='cpu', weights_only=False)['rows']
    rebuilt = [{**raw, 'graph': {**raw['graph'], **extra['graph']}}
               for raw, extra in zip(original['train'] + original['dev'], saved)]
    rebuilt_graph = collate(rebuilt, device)
    assert all(torch.equal(graph[key], rebuilt_graph[key]) for key in graph)
    supplement_path.unlink()
    checks['supplement_serialization_roundtrip'] = True
    # Same centroid and total weight, different terminal distribution and L1 cost.
    def toy(pins):
        return (1, torch.ones(1), torch.empty(0, 3),
                torch.tensor([[0., 0., 1.], [1., 0., 1.], [2., 0., 1.]]),
                torch.tensor(pins), torch.zeros(1, 5), torch.full((1, 4), -1.))
    toy_a = public_row(toy([[0., 0.], [0., 0.], [3., 0.]]), ('synthetic', 0))
    toy_b = public_row(toy([[1., 0.], [1., 0.], [1., 0.]]), ('synthetic', 1))
    ga, gb = collate([toy_a], device), collate([toy_b], device)
    state = torch.tensor([[[.5, .5, 0.]]], device=device)
    full = make_model('single_full', device).eval(); pooled = make_model('single_pooled', device).eval()
    with torch.no_grad():
        pooled_difference = float((pooled.terminal_message(ga, state) - pooled.terminal_message(gb, state)).abs().max())
        full_difference = float((full.terminal_message(ga, state) - full.terminal_message(gb, state)).abs().max())
        costs = [float(wire(decode(state, g), g)[0]) for g in (ga, gb)]
    assert pooled_difference < 1e-6 and full_difference > 1e-6 and abs(costs[0] - costs[1]) > .1
    checks['terminal_counterexample'] = {'pooled_message_max_difference': pooled_difference,
        'full_message_max_difference': full_difference, 'wire_costs': costs,
        'scope': 'Representation can distinguish a lost-information counterexample; not learned quality evidence'}
    # One real batch32 gradient/update, repeating two authorized rows explicitly.
    seed_all(0); model = make_model('recurrent_full', device).train()
    batch = [rows[i % 2] for i in range(32)]; graph = collate(batch, device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    started = time.perf_counter()
    values = loss(model(inputs_only(graph)), graph)
    assert all(torch.isfinite(v) for v in values.values())
    values['total'].backward()
    assert all(p.grad is None for p in model.base.parameters())
    norm = float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True))
    assert norm > 0
    optimizer.step()
    if device == 'cuda':
        torch.cuda.synchronize()
    step_seconds = time.perf_counter() - started
    checks['batch32_update'] = {'unique_training_rows': 2, 'repeated_rows': 32, 'gradient_norm': norm,
        'seconds': step_seconds, 'loss_before_update': float(values['total'].detach()),
        'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else None,
        'frozen_base_has_no_gradients': True}
    warm_steps = []
    for _ in range(2):
        if device == 'cuda': torch.cuda.synchronize()
        started = time.perf_counter(); optimizer.zero_grad(set_to_none=True)
        loss(model(inputs_only(graph)), graph)['total'].backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
        optimizer.step()
        if device == 'cuda': torch.cuda.synchronize()
        warm_steps.append(time.perf_counter() - started)
    checks['batch32_update']['additional_warm_step_seconds'] = warm_steps
    positions, baseline, seconds = predict(model.eval(), rows[2], device)
    assert np.isfinite(positions).all()
    from scoring.evaluation import evaluate_case
    scored = evaluate_case(rows[2]['case'], positions, seconds)
    real_graph = collate([rows[2]], device)
    with torch.no_grad():
        real_state = model(inputs_only(real_graph))
        physical_wire = float(wire(decode(real_state, real_graph), real_graph)[0] * real_graph['scale'][0])
    assert abs(physical_wire - scored['metrics']['hpwl_total']) / max(1., scored['metrics']['hpwl_total']) < 1e-5
    checks['online_raw_official_score'] = {'metrics': scored['metrics'],
        'seconds': seconds, 'public_wire_matches_official_relative_tolerance': 1e-5,
        'note': 'One implementation row after three repeated batches; no quality claim'}
    fake_records = [{'metrics': {'is_feasible': True, 'cost': 1.02}, 'block_count': 50, 'solve_seconds': .01},
                    {'metrics': {'is_feasible': False, 'cost': 10.}, 'block_count': 50, 'solve_seconds': .01}]
    assert abs(summary(fake_records)['weighted_Q_all_cases'] - 5.51) < 1e-10
    checks['infeasible_Q10_retained_in_summary'] = True
    # Constant metric checks the protection, three exact drops, and two lowest epochs.
    param = torch.nn.Parameter(torch.ones(1)); opt = torch.optim.AdamW([param], lr=LR_LEVELS[0])
    controller = PlateauController(opt); transitions = []; first_stop = None
    for epoch in range(1, 40):
        old = controller.drops; controller.step(1., epoch)
        if controller.drops != old:
            transitions.append([epoch, opt.param_groups[0]['lr']])
        if controller.should_stop():
            first_stop = epoch; break
    assert transitions == [[18, .00025], [21, .0001], [24, .00005]] and first_stop == 26
    checks['lr_protection_and_lowest_two_epochs'] = {'transitions': transitions, 'first_stop_epoch': first_stop}
    # Real checkpoint round trip; both paths see exactly the same two small rows.
    def arguments(name, resume=False):
        return argparse.Namespace(arm='recurrent_full', output=output / name, device=device, seed=0,
                                  batch_size=32, max_hours=.08, max_epochs=3, resume=resume)
    check_rows = (rows[:2], rows[2:])
    train_run(arguments('continuous'), check_rows=check_rows, stop_after_steps=2)
    train_run(arguments('resumed'), check_rows=check_rows, stop_after_steps=1)
    first = torch.load(output / 'resumed/checkpoints/latest.pt', map_location='cpu', weights_only=False)
    train_run(arguments('resumed', True), check_rows=check_rows, stop_after_steps=2)
    a = torch.load(output / 'continuous/checkpoints/latest.pt', map_location='cpu', weights_only=False)
    b = torch.load(output / 'resumed/checkpoints/latest.pt', map_location='cpu', weights_only=False)
    assert all(torch.equal(a['model'][k], b['model'][k]) for k in a['model'])
    assert a['scheduler'] == b['scheduler'] and a['progress']['step'] == b['progress']['step'] == 2
    assert b['elapsed_seconds'] > first['elapsed_seconds']
    checks['checkpoint_resume'] = {'parameter_identical': True, 'scheduler_identical': True,
                                   'steps': 2, 'budget_preserved': True}
    freeze_once(output / 'freeze_check', {'candidate': 'A'})
    try:
        freeze_once(output / 'freeze_check', {'candidate': 'B'}, True)
        raise AssertionError('Changed final candidate accepted')
    except ValueError:
        pass
    checks['frozen_final_rejects_changed_candidate'] = True
    result = {'status': 'passed', 'unique_real_cases': 3, 'synthetic_cases': 2,
              'members': [list(r['member']) for r in rows], 'device': device,
              'elapsed_seconds': time.perf_counter() - begin, 'checks': checks, 'sources': sources(),
              'formal_training_started': False, 'test_read': False, 'postprocessing_calls': 0,
              'quality_claim': 'Implementation and counterexample only; no validation improvement established'}
    assert result['elapsed_seconds'] < 20 * 60
    save(output / 'results.json', result)
    snapshot(output / 'source_snapshot.zip', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('sources', 'checks')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
