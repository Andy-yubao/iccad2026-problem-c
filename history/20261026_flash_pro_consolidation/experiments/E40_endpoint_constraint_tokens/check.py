"""Three actual members plus two same-centroid endpoint counterexamples."""
import argparse
import time
import numpy as np
import torch
from .io import HERE, ROOT, save, sha
from .data import load_rows, collate
from .public import inputs_only, EXTRA_KEYS
from .model import GeometryModel, decode
from .reference_model import GeometryModel as PlainModel
from .loss import losses
from .arms import ARMS
from .inference import graph as online_graph


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    ap.add_argument('--tag', default='20261023_structure_check5_cpu')
    args = ap.parse_args(); started = time.perf_counter()
    destination = HERE/'runs'/args.tag/'results.json'
    if destination.exists():
        raise ValueError('Use a fresh check tag')
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    if sha(HERE/'reference_model.py') != sha(ROOT/'experiments/E25_integrated_baseline/model.py'):
        raise AssertionError('Reference model copy changed')
    train_rows = load_rows('train', [0, 1], use_cache=False)
    dev_rows = load_rows('dev_monitor500', [0], use_cache=False)
    rows = train_rows+dev_rows
    gradient_rows = train_rows*16 if args.device == 'cuda' else train_rows
    graph, _ = collate(gradient_rows, args.device); public = inputs_only(graph)
    validation_graph, _ = collate(dev_rows, args.device)
    if args.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    report = {'test_read': False, 'implementation_check': True, 'device': args.device,
              'members': [list(row['member']) for row in rows], 'arms': {},
              'gradient_members': [list(row['member']) for row in train_rows],
              'validation_members': [list(row['member']) for row in dev_rows],
              'gradient_batch_size': len(gradient_rows), 'deterministic_algorithms': True}
    torch.manual_seed(0); original = PlainModel(**ARMS['geometry_control']).to(args.device)
    with torch.no_grad():
        neutral = original({k: v for k, v in public.items() if k not in EXTRA_KEYS})
    for arm in ('geometry_control', 'full_tokens', 'pooled_tokens'):
        torch.manual_seed(0); model = GeometryModel(**ARMS[arm]).to(args.device)
        with torch.no_grad():
            initial = model(public)
        delta = float((initial-neutral).abs().max())
        if delta > 1e-6:
            raise AssertionError('Neutral token branches changed initial output')
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            value = losses(model(public), graph, None, arm, 2., include_quality=False)
            value['total'].backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
        if arm != 'geometry_control':
            branch_gradient = sum(float(p.grad.abs().sum()) for layer in model.interactions
                                  for name, p in layer.named_parameters() if 'gate' not in name and p.grad is not None)
            if branch_gradient == 0:
                raise AssertionError('Token branches did not start learning after gate update')
        else:
            branch_gradient = None
        polluted = dict(graph, reference=torch.randn_like(graph['reference']), target=torch.randn_like(graph['target']))
        with torch.no_grad():
            if not torch.equal(model(public), model(inputs_only(polluted))):
                raise AssertionError('Supervision leaked into token inputs')
        model.eval()
        with torch.no_grad():
            validation_state = model(inputs_only(validation_graph))
            validation_values = losses(validation_state, validation_graph, None, arm, 2., include_quality=True)
        if validation_state.requires_grad or any(v.requires_grad for v in validation_values.values()):
            raise AssertionError('Validation must remain outside autograd')
        if not all(torch.isfinite(v).all() for v in validation_values.values()):
            raise AssertionError('Nonfinite validation check')
        report['arms'][arm] = {'parameters': sum(p.numel() for p in model.parameters()),
            'neutral_output_delta': delta, 'finite_gradient_norm': float(norm),
            'learned_branch_gradient_l1': branch_gradient, 'validation_autograd_disabled': True}
    report['parameter_control_parameters'] = sum(p.numel() for p in GeometryModel(**ARMS['parameter_control']).parameters())
    # Exact sparse endpoint evaluation must preserve the same weighted nonlinear message.
    # Keep the deliberately dense reference small and on CPU; GPU checks exercise
    # the production sparse aggregation with the actual deterministic batch32 path.
    diagnostic_graph, _ = collate(rows, 'cpu')
    diagnostic_public = inputs_only(diagnostic_graph)
    torch.manual_seed(0); sparse_model = GeometryModel(**ARMS['full_tokens'])
    torch.manual_seed(0); diagnostic_original = PlainModel(**ARMS['geometry_control'])
    layer = sparse_model.interactions[0]
    with torch.no_grad():
        state = diagnostic_original({k: v for k, v in diagnostic_public.items() if k not in EXTRA_KEYS})
        rectangles = decode(state, diagnostic_graph); center = rectangles[..., :2]+rectangles[..., 2:]/2
        local = diagnostic_graph['area'].clamp_min(1e-12).sqrt()
        delta = (diagnostic_graph['terminal_xy'][:, None]-center[:, :, None])/local[..., None, None]
        dense = (layer.pin_encoder(torch.cat((delta, delta.abs()), -1))*diagnostic_graph['terminal_weight'][..., None]).sum(2)
        bi, ni, pi = diagnostic_graph['terminal_edge_index'].unbind(1)
        delta = (diagnostic_graph['terminal_xy'][bi, pi]-center[bi, ni])/local[bi, ni, None]
        encoded = layer.pin_encoder(torch.cat((delta, delta.abs()), -1))
        b, n, w = dense.shape
        sparse = dense.new_zeros((b*n, w)).index_add(0, bi*n+ni,
            encoded*diagnostic_graph['terminal_edge_weight'][:, None]).reshape(b, n, w)
        if not torch.allclose(dense, sparse, atol=1e-5, rtol=1e-6):
            raise AssertionError('Sparse endpoint message differs from dense definition')
        report['sparse_dense_message_max_difference'] = float((dense-sparse).abs().max())
    if report['arms']['full_tokens']['parameters'] != report['arms']['pooled_tokens']['parameters']:
        raise AssertionError('Endpoint information control changed parameter count')
    # Two distributions share exactly the same weighted centroid, edge count and total weight.
    inputs = []
    for pins in (np.array([[0., 0.], [2., 0.]]), np.array([[1., 0.], [1., 0.]])):
        inputs.append((2, np.ones(2), np.empty((0, 3)), np.array([[0., 0., 1.], [1., 0., 1.]]),
                       pins, np.zeros((2, 5), dtype=int), np.full((2, 4), -1.)))
    first, second = [online_graph(value) for value in inputs]
    if not torch.equal(first['features'], second['features']):
        raise AssertionError('Counterexample must preserve original network features')
    torch.manual_seed(0); full = GeometryModel(**ARMS['full_tokens']).eval()
    for layer in full.interactions:
        layer.pin_gate.data.fill_(.3)
    pooled = GeometryModel(**ARMS['pooled_tokens']).eval(); pooled.load_state_dict(full.state_dict())
    with torch.no_grad():
        full_difference = float((full(first)-full(second)).abs().max())
        pooled_difference = float((pooled(first)-pooled(second)).abs().max())
    if full_difference <= 1e-6 or pooled_difference > 1e-6:
        raise AssertionError('Full endpoint representation failed the same-centroid counterexample')
    report['same_centroid_counterexample'] = {'full_output_difference': full_difference,
        'pooled_output_difference': pooled_difference, 'identical_parameters_and_original_features': True}
    # The online and prepared paths must build precisely the same extra public tensors.
    for row in rows:
        batch, _ = collate([row], 'cpu'); online = online_graph(row['case'].solver_inputs)
        for key in EXTRA_KEYS:
            if not torch.equal(batch[key], online[key]):
                raise AssertionError('Online public-token construction differs: '+key)
    report['online_token_parity'] = True
    report['active_pin_counts'] = diagnostic_graph['terminal_valid'].sum(1).tolist()
    report['constraint_token_counts'] = diagnostic_graph['constraint_valid'].sum(1).tolist()
    report['sparse_dense_reference_device'] = 'cpu'
    if args.device == 'cuda':
        torch.cuda.synchronize()
        report['cuda_peak_allocated_GB'] = torch.cuda.max_memory_allocated()/1e9
    report['seconds'] = time.perf_counter()-started
    save(destination, report); print(report, flush=True)


if __name__ == '__main__':
    main()
