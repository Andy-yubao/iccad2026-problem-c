"""Bounded implementation check: two train members, one dev member, no sealed data."""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from .io import HERE, ROOT, sha, save
from .arms import ARMS
from .cache import verify
from .data import load_rows, collate
from .batches import inputs_only
from .model import GeometryModel, decode
from .loss import losses
from .train import seed_all, PlateauController, run, parser, monitor_losses
from .inference import predict, graph as public_graph
from .freeze import BASELINE, frozen_checkpoint


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tag', default='20261022_implementation_check3')
    args = ap.parse_args()
    folder = HERE/'runs'/args.tag
    if folder.exists():
        raise ValueError('Use a new tag; do not overwrite an implementation check')
    start = time.perf_counter(); seed_all(0)
    device = 'cuda'
    if not torch.cuda.is_available():
        raise RuntimeError('This check must exercise the actual CUDA training device')
    report = {'test_read': False, 'implementation_check': True}
    for name in ('loss.py', 'losses.py', 'occupancy.py', 'batches.py'):
        source = ROOT/'history/20261020_E25_consolidation/experiments/E17_e10_geometry_v5'/name
        if sha(HERE/name) != sha(source):
            raise AssertionError('E17 Loss/batching changed: '+name)
    report['loss_and_batching_byte_identical_to_E17'] = True
    verify('train'); verify('dev')
    report['shared_cache_manifest_verified'] = True
    for role in ('local_test', 'test', 'tail_test'):
        try:
            load_rows(role, [0])
        except (ValueError, PermissionError):
            pass
        else:
            raise AssertionError('Sealed role accepted: '+role)
    train = load_rows('train', [0, 1], use_cache=False)
    dev = load_rows('dev_monitor500', [0], use_cache=False)
    rows = train+dev
    report['members'] = {'train': [list(r['member']) for r in train], 'dev': [list(r['member']) for r in dev]}
    # Repetition exercises the specified batch32 without accessing 32 unique members.
    batch, _ = collate(train*16, device)
    public = inputs_only(batch)
    report['arms'] = {}
    for arm, config in ARMS.items():
        seed_all(0); model = GeometryModel(**config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        torch.cuda.synchronize(); began = time.perf_counter()
        state = model(public)
        value = losses(state, batch, None, arm, 2., include_quality=False)
        value['total'].backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step(); torch.cuda.synchronize()
        if not all(torch.isfinite(v).all() for v in model.parameters()):
            raise AssertionError('Nonfinite parameters')
        contaminated = dict(batch, reference=torch.randn_like(batch['reference']), target=torch.randn_like(batch['target']))
        model.eval()
        with torch.no_grad():
            clean = model(inputs_only(batch)); polluted = model(inputs_only(contaminated))
        if not torch.equal(clean, polluted):
            raise AssertionError('Reference geometry leaked into network input')
        try:
            model(batch)
        except ValueError:
            pass
        else:
            raise AssertionError('Network accepted supervision keys')
        report['arms'][arm] = {'parameters': sum(p.numel() for p in model.parameters()),
            'batch32_update_seconds': time.perf_counter()-began, 'finite_gradient_norm': float(gradient),
            'loss': float(value['total'].detach()), 'label_isolation': True}
        del model, optimizer, state, clean, polluted, value
    # A plateau must traverse all four tiers and train/validate twice at the lowest tier.
    parameter = torch.nn.Parameter(torch.ones(())); optimizer = torch.optim.SGD([parameter], lr=1e-3)
    controller = PlateauController(optimizer); levels = []
    for epoch in range(1, 27):
        controller.step(1., epoch); levels.append(optimizer.param_groups[0]['lr'])
        if epoch < 26 and controller.should_stop():
            raise AssertionError('Premature early stop')
    if not controller.should_stop() or controller.lowest_completed_epochs != 2 or controller.drops != 3:
        raise AssertionError('Four-tier scheduler failed')
    copied = PlateauController(optimizer); copied.load_state_dict(controller.state_dict())
    if copied.state_dict() != controller.state_dict():
        raise AssertionError('Scheduler restore differs')
    report['four_tier_schedule_and_two_lowest_epochs'] = True
    # Save/restore must reproduce uninterrupted parameter updates and preserve budget.
    def training_args(path, resume=False):
        command = ['--arm', 'base96x4', '--output', str(path), '--device', device,
                   '--max-hours', '.05', '--max-epochs', '5']
        if resume:
            command.append('--resume')
        return parser().parse_args(command)
    continuous = folder/'continuous'; interrupted = folder/'interrupted'
    run(training_args(continuous), implementation_check=True, stop_after_steps=2, check_rows=(train*16, dev))
    run(training_args(interrupted), implementation_check=True, stop_after_steps=1, check_rows=(train*16, dev))
    first = torch.load(interrupted/'checkpoints/latest.pt', map_location='cpu', weights_only=False)
    run(training_args(interrupted, True), implementation_check=True, stop_after_steps=2, check_rows=(train*16, dev))
    resumed = torch.load(interrupted/'checkpoints/latest.pt', map_location='cpu', weights_only=False)
    uninterrupted = torch.load(continuous/'checkpoints/latest.pt', map_location='cpu', weights_only=False)
    if resumed['elapsed_seconds'] <= first['elapsed_seconds']:
        raise AssertionError('Cumulative budget reset')
    delta = max(float((resumed['model'][k]-v).abs().max()) for k, v in uninterrupted['model'].items())
    if delta != 0. or resumed['progress']['step'] != 2:
        raise AssertionError('Interrupted training differs: '+str(delta))
    report['exact_two_step_resume'] = {'max_parameter_delta': delta, 'budget_preserved': True}
    # Frozen-best guard rejects check weights as formal models and rejects uncompleted training.
    try:
        frozen_checkpoint(interrupted)
    except ValueError:
        pass
    else:
        raise AssertionError('Implementation checkpoint became a formal candidate')
    baseline_state = torch.load(BASELINE, map_location='cpu', weights_only=False)
    baseline = GeometryModel(**baseline_state['model_config']).eval(); baseline.load_state_dict(baseline_state['model'])
    for row in rows:
        direct, _ = predict(baseline, row['case'].solver_inputs)
        g, _ = collate([row], 'cpu')
        with torch.no_grad():
            cached = (decode(baseline(inputs_only(g)), g)[0]*g['scale'][0]).numpy()
        c, spec = np.asarray(row['case'].solver_inputs[5]), np.asarray(row['case'].solver_inputs[6])
        cached[c[:, :2].any(1), 2:] = spec[c[:, :2].any(1), 2:]
        cached[c[:, 1] != 0] = spec[c[:, 1] != 0]
        if not np.allclose(cached, direct, atol=1e-5, rtol=1e-6):
            raise AssertionError('Public direct inference differs from prepared-row inference')
    report['online_feature_and_decode_parity'] = True
    # Exercise the copied full solver on the same single dev member, with no new sample access.
    from .canonical_native import warmup
    from .final import evaluate_row
    warmup()
    record = evaluate_row(0, dev[0], baseline)
    if 'error' in record or not record['metrics']['is_feasible']:
        raise AssertionError('Direct terminal integration failed: '+str(record.get('error', record['metrics'])))
    report['one_dev_terminal_integration'] = {'member': list(dev[0]['member']), 'hard_legal': True,
        'Q': record['metrics']['cost'], 'seconds': record['seconds']}
    report['monitor'] = monitor_losses(baseline, dev, 'cpu', 'base96x4', 2., 32)
    report['seconds'] = time.perf_counter()-start
    report['cuda_peak_allocated_GB'] = torch.cuda.max_memory_allocated()/1e9
    report['torch_threads'] = torch.get_num_threads()
    report['torch_interop_threads'] = torch.get_num_interop_threads()
    save(folder/'results.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
