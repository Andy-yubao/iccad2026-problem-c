"""Three-member architecture check; CPU preparation avoids concurrent GPU training."""
import argparse
import time
import torch
import numpy as np
from .io import HERE, ROOT, save, sha
from .data import load_rows, collate
from .batches import inputs_only
from .model import GeometryModel
from .loss import losses
from .arms import ARMS


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    ap.add_argument('--tag', default='20261022_structure_check3_cpu')
    args = ap.parse_args(); started = time.perf_counter()
    destination = HERE/'runs'/args.tag/'results.json'
    if destination.exists():
        raise ValueError('Use a new tag; preserve earlier evidence')
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    train_rows = load_rows('train', [0, 1], use_cache=False)
    dev_rows = load_rows('dev_monitor500', [0], use_cache=False)
    rows = train_rows+dev_rows
    gradient_rows = train_rows*16 if args.device == 'cuda' else train_rows
    graph, _ = collate(gradient_rows, args.device); public = inputs_only(graph)
    validation_graph, _ = collate(dev_rows, args.device)
    if args.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    original = HERE/'reference_model.py'
    if sha(original) != sha(ROOT/'experiments/E25_integrated_baseline/model.py'):
        raise AssertionError('Reference code changed')
    from .reference_model import GeometryModel as ReferenceModel
    torch.manual_seed(0); reference = ReferenceModel(**ARMS['geometry_control']).to(args.device)
    reference_state = reference.state_dict()
    with torch.no_grad():
        initial = reference(public)
    report = {'test_read': False, 'implementation_check': True, 'device': args.device,
              'members': [list(row['member']) for row in rows], 'arms': {},
              'gradient_members': [list(row['member']) for row in train_rows],
              'validation_members': [list(row['member']) for row in dev_rows],
              'gradient_batch_size': len(gradient_rows), 'deterministic_algorithms': True}
    for arm, config in ARMS.items():
        torch.manual_seed(0); model = GeometryModel(**config).to(args.device)
        shared_delta = 0.
        for key, value in reference_state.items():
            candidate = model.state_dict()[key]
            if candidate.shape != value.shape:
                candidate = candidate[:, :7]
            shared_delta = max(shared_delta, float((candidate-value).abs().max()))
        if shared_delta != 0:
            raise AssertionError('Common initialization changed: '+arm)
        state = model(public)
        delta = float((state-initial).abs().max())
        if not torch.allclose(state, initial, atol=1e-5, rtol=1e-5):
            raise AssertionError('Neutral extension changed initial prediction: '+str(delta))
        value = losses(state, graph, None, arm, 2., include_quality=False)
        value['total'].backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        if arm != 'geometry_control':
            extra_norm = sum(float(block.bias[0].weight.grad[:, 7:].abs().sum()) for block in model.layers)
            if extra_norm == 0:
                raise AssertionError('New rectangle channels cannot learn')
            if arm == 'rectangle_no_gap' and any(torch.count_nonzero(block.bias[0].weight.grad[:, 11:15]) for block in model.layers):
                raise AssertionError('Gap ablation has nonzero gap gradients')
        else:
            extra_norm = None
        model.eval()
        with torch.no_grad():
            validation_state = model(inputs_only(validation_graph))
            validation_values = losses(validation_state, validation_graph, None, arm, 2., include_quality=True)
        if validation_state.requires_grad or any(v.requires_grad for v in validation_values.values()):
            raise AssertionError('Validation must remain outside autograd')
        if not all(torch.isfinite(v).all() for v in validation_values.values()):
            raise AssertionError('Nonfinite validation check')
        report['arms'][arm] = {'model_config': config, 'parameters': sum(p.numel() for p in model.parameters()),
            'common_initial_parameter_delta': shared_delta, 'initial_output_delta': delta,
            'finite_gradient_norm': float(gradient), 'extra_channel_gradient_l1': extra_norm,
            'validation_autograd_disabled': True}
    if args.device == 'cuda':
        torch.cuda.synchronize()
        report['cuda_peak_allocated_GB'] = torch.cuda.max_memory_allocated()/1e9
    report['seconds'] = time.perf_counter()-started
    report['training'] = 'No formal training; E38 remains the prioritized GPU task'
    save(destination, report); print(report, flush=True)


if __name__ == '__main__':
    main()
