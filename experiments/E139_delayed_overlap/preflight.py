"""Bounded preparation on five cached training members; no formal training."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import ast
import hashlib
import json
import time
import traceback
import torch
from . import train
from .io import HERE, ROOT, save, sha, sources
from .model import GeometryModel
from .data import load_rows, cache_identity
from .resident import ResidentRows
from .batches import inputs_only
from .loss import per_case, base_per_case, supervised_per_case, TOTAL_COMPONENTS
from .loss_config import ARMS, layer_weights, NO_OVERLAP_LAYERS
from .cuda_step import CapturedStep

PARENT = ROOT/'experiments/E134_supervision_interval'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def inherited_engine():
    provenance = read(HERE/'provenance.json')
    gate = read(PARENT/'preflight.json')
    if sha(PARENT/'preflight.json') != provenance['parent_preflight_sha256']:
        raise ValueError('Parent preflight changed')
    if gate['status'] != 'passed' or not gate['ctrl_c_resume_verified']:
        raise ValueError('Missing parent continuation proof')
    if not all(all(row['bitwise_equal'].values()) for row in gate['resume_checks']):
        raise ValueError('Parent resume equality failed')
    if not gate['queue_signal']['explicit_resume_advances'] or not gate['queue_signal']['no_next_stage_after_pause']:
        raise ValueError('Missing parent queue signal proof')
    if not gate['evaluation_signal']['first_committed_file_unchanged']:
        raise ValueError('Missing parent evaluation prefix proof')
    checked = []
    for name, record in provenance['copied_originals'].items():
        parent = ROOT/record['parent']
        if sha(parent) != record['sha256'] or gate['sources'].get(record['parent']) != record['sha256']:
            raise ValueError('Parent runtime drift since verified preflight: '+name)
        expected = parent.read_text(encoding='utf-8')
        for old, new in provenance['identity_mapping'].items():
            expected = expected.replace(old, new)
        if name == 'evaluation/run.py':
            expected = expected.replace("'# E134 '", "'# E139 '")
        if name == 'run_queue.py':
            expected = expected.replace('Foreground four-cell queue.', 'Foreground single-arm queue.')
            expected = expected.replace('All four cells and original dev500 postprocessing completed.',
                                        'E139 training and original dev500 postprocessing completed.')
            expected = expected.replace('# E134 四组结果', '# E139 延后重叠监督结果')
            expected = expected.replace('只比较本轮四组，不以历史E119 Q≈1.02作为可靠标杆。',
                                        '复用E134对照另行比较；不以历史E119 Q≈1.02作为可靠标杆。')
        if (HERE/name).read_text(encoding='utf-8') != expected:
            raise ValueError('Unexpected continuation/numerical source change: '+name)
        checked.append(name)
    return dict(mode='inherited E134 proof, all copied runtime bytes/text verified after identity/report substitution',
        parent_preflight_sha256=provenance['parent_preflight_sha256'], checked_files=checked,
        training_resume_verified=True, queue_resume_verified=True, evaluation_prefix_verified=True,
        new_interruption_simulations=0)


def cache_coverage(plan):
    from datasets.random_reference import RandomReference
    identity = cache_identity(plan)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if key != plan['required_cache_key']:
        raise ValueError('Shared cache identity changed')
    base = ROOT/'datasets/static_tensor_cache_v1'/key
    if read(base/'identity.json') != identity:
        raise ValueError('Cache producer mismatch')
    report = {}
    for role in ('train', 'dev_monitor500'):
        dataset = RandomReference(role)
        blocks = {}
        for index in dataset.indices:
            blocks.setdefault(index//256, []).append(index)
        for block, indices in blocks.items():
            path = base/dataset.role/f'{block:04d}.pt'
            entry = read(path.with_suffix('.json'))
            if not path.is_file() or entry['identity'] != key:
                raise ValueError('Missing compatible cache block')
            if any(entry['members'][str(i)] != dataset.records[i]['member'] for i in indices):
                raise ValueError('Cache member mismatch')
        report[role] = dict(members=len(dataset.indices), chunks=len(blocks), metadata_only=True)
    return dict(key=key, roles=report)


def objective_checks(output, batch, arm):
    from experiments.E134_supervision_interval.loss import base_per_case as parent_base
    weights = layer_weights(arm)
    states = output.detach().clone().requires_grad_(True)
    actual = per_case(states, batch, arm)
    expected = 0
    for layer, weight in weights.items():
        full = parent_base(states[layer-1], batch)
        own = supervised_per_case(states[layer-1], batch, layer)
        current_full = base_per_case(states[layer-1], batch)
        for name in full:
            torch.testing.assert_close(current_full[name], full[name], rtol=0, atol=0)
        if layer in NO_OVERLAP_LAYERS:
            retained = sum(full[name] for name in TOTAL_COMPONENTS if name != 'overlap_weighted')
            torch.testing.assert_close(own['total'], retained, rtol=0, atol=0)
            assert torch.equal(own['overlap_weighted'], torch.zeros_like(own['overlap_weighted']))
        else:
            retained = full['total']
            torch.testing.assert_close(own['total'], full['total'], rtol=0, atol=0)
        for name in TOTAL_COMPONENTS:
            if name != 'overlap_weighted':
                torch.testing.assert_close(own[name], full[name], rtol=0, atol=0)
        expected = expected+weight*retained
    torch.testing.assert_close(actual['total'], expected, rtol=0, atol=0)
    actual_grad = torch.autograd.grad(actual['total'].sum(), states)[0]
    expected_grad = torch.autograd.grad(expected.sum(), states)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-6, atol=1e-6)
    for layer in (1, 3, 4, 5, 7, 9, 11):
        assert torch.count_nonzero(actual_grad[layer-1]) == 0
    torch.testing.assert_close(actual['final_total'], parent_base(states[11], batch)['total'], rtol=0, atol=0)
    # Force a heavily overlapping layout so omission cannot pass on zero collisions.
    collision = torch.zeros_like(states[0])
    collision[..., 0] = torch.arange(collision.shape[1], device=collision.device)*.001
    full = parent_base(collision, batch)
    assert bool((full['overlap_weighted'] > 0).all())
    for layer in NO_OVERLAP_LAYERS:
        own = supervised_per_case(collision, batch, layer)
        torch.testing.assert_close(full['total']-own['total'], full['overlap_weighted'], rtol=1e-5, atol=1e-5)
        for name in ('penetration', 'module_load_mean', 'module_max_excess10', 'overlap_total'):
            torch.testing.assert_close(own[name], full[name], rtol=0, atol=0)
    return dict(exact_parent_full_loss=True, early_retained_terms_exact=True,
        aggregate_value_exact=True, aggregate_gradient_verified=True,
        unsupervised_readout_direct_gradients_zero=True, late_full_loss_exact=True,
        synthetic_nonzero_collision_omission_verified=True, raw_collision_diagnostics_retained=True)


def main():
    started = time.perf_counter()
    result = dict(status='preparing', unique_train_cases=5, formal_training_started=False,
        formal_optimizer_updates=0, dev_payloads=0, test_payloads=0, postprocessing_calls=0,
        Q_calls=0, ctrl_c_resume_verified=False)
    try:
        for path in HERE.rglob('*.py'):
            ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        plan = read(HERE/'plan.json')
        auth = read(HERE/'authorization.json')
        if auth['training_enabled'] or auth['allowed_arms'] or (HERE/'runs').exists():
            raise ValueError('Formal run during preparation')
        result['continuation_proof'] = inherited_engine()
        result['cache'] = cache_coverage(plan)
        for control in plan['comparison_controls']:
            if sha(ROOT/control['result']) != control['results_sha256']:
                raise ValueError('Existing control changed')
        arm = ARMS[0]
        weights = layer_weights(arm)
        if weights != {2: .05, 6: .05, 8: .05, 10: .05, 12: .8}:
            raise ValueError('Unexpected user layer/weight schedule')
        if {str(k): v for k, v in weights.items()} != plan['arms'][arm]['layer_weights']:
            raise ValueError('Plan/runtime weight mismatch')
        train.seed_all(0)
        model = GeometryModel(**train.MODEL)
        digest = hashlib.sha256(b''.join(t.contiguous().numpy().tobytes() for t in model.state_dict().values())).hexdigest()
        parent_gate = read(PARENT/'preflight.json')
        previous = next(r for r in parent_gate['arms'] if r['arm'] == 'e119_stride3')
        if digest != previous['initial_state_sha256']:
            raise ValueError('Scratch initialization changed')
        if sum(p.numel() for p in model.parameters()) != 712275:
            raise ValueError('Parameter count changed')
        rows = load_rows('train', plan['preflight_train_indices'])
        if len({tuple(row['member']) for row in rows}) != 5:
            raise ValueError('Wrong short-check members')
        result['members'] = [list(row['member']) for row in rows]
        pack = ResidentRows(rows, 'cuda', max_nodes=120)
        small = pack.batch(range(5))
        public = inputs_only(small)
        if set(public) & {'reference', 'target', 'pin_coordinates'}:
            raise ValueError('Labels leaked into model input')
        if not (public['mask'][public['valid']] == 1).all() or not (public['constraints'][..., 2] == 0).all():
            raise ValueError('Prediction/MIB boundary changed')
        model.cuda().eval()
        with torch.inference_mode():
            output = model(public)
        if output.shape[0] != 12:
            raise ValueError('Original S1..S12 readouts changed')
        result['objective_checks'] = objective_checks(output, small, arm)
        model.train()
        optimizer = train.optimizer_for(model)
        before = {k: t.detach().clone() for k, t in model.state_dict().items()}
        cap = CapturedStep(model, optimizer, pack.batch([i % 5 for i in range(32)]), arm)
        if any(not torch.equal(t, before[k]) for k, t in model.state_dict().items()):
            raise ValueError('Capture changed parameters')
        values, norm = cap.replay()
        torch.cuda.synchronize()
        if cap.projection.steps != 1 or torch.equal(model.head[0].weight, before['head.0.weight']):
            raise ValueError('Disposable guarded update failed')
        for block in (model.initial_blocks[0], model.layers[0], model.layers[-1]):
            if not block.qkv.weight.grad.abs().sum() > 0:
                raise ValueError('Missing front/backbone gradient')
        if read(HERE/'authorization.json')['training_enabled'] or (HERE/'runs').exists():
            raise ValueError('Preparation activated formal training')
        result.update(status='passed', ctrl_c_resume_verified=True, sources=sources(),
            arm=arm, weights=weights, initial_state_sha256=digest, parameters=712275,
            disposable_optimizer_updates=1, disposable_loss=float(values['total']),
            gradient_norm=float(norm), training_model_not_reused=True)
        save(HERE/'preparation_complete.json', dict(status='ready_not_started', training_enabled=False,
            formal_training_started=False, formal_runs_exist=False, source_identity=sources(),
            training_command='python -u -B -m experiments.E139_delayed_overlap.run_queue --start'))
    except BaseException as error:
        result.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        result['elapsed_seconds'] = time.perf_counter()-started
        save(HERE/'preflight.json', result)
        print(json.dumps({k: v for k, v in result.items() if k not in ('sources', 'members', 'cache', 'continuation_proof', 'traceback')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
