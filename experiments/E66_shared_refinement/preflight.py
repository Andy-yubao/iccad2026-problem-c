"""Same five train cases: fixed G, additive bounds, shared unroll, Adam and exact resume."""
import gc
import time
import torch
from .io import HERE, ROOT, save, sha, sources
from .initialization import metadata, verify_protected
from .arms import model_config
from .model import GeometryModel
from .batches import inputs_only
from .resident import ResidentRows
from .data import load_rows
from .loss import losses
from .loss_reference import components as original_components, WEIGHTS
from .train import seed_all, parser, run, selection
from .cuda_step import CapturedStep

ARMS = ('unshared_residual', 'shared_residual')


def same(a, b, path='state'):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b), path
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for key in a:
            same(a[key], b[key], path+'/'+str(key))
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b), path
        for index, (x, y) in enumerate(zip(a, b)):
            same(x, y, path+'/'+str(index))
    else:
        assert a == b, path


def clear():
    from . import data
    data._resident.clear()
    gc.collect()
    torch.cuda.empty_cache()


def check_arm(arm, batch):
    seed_all(0)
    model = GeometryModel(**model_config(arm)).cuda().eval()
    pack = ResidentRows(batch, 'cuda', 120)
    graph = pack.batch(list(range(32)))
    public = inputs_only(graph)
    calls = [0]*len(model.layers)
    handles = []
    for index, block in enumerate(model.layers):
        def count_call(module, inputs, output, index=index):
            calls[index] += 1
        handles.append(block.register_forward_hook(count_call))
    with torch.no_grad():
        initial = model.head(model.input(public['features'])*public['valid'][..., None])
        actual_state = model(public)
    for handle in handles:
        handle.remove()
    assert torch.equal(initial, actual_state), 'Zero correction must preserve initial coarse head'
    assert sum(calls) == 12 and all(c == 12//len(model.layers) for c in calls)
    with torch.no_grad():
        huge = torch.full_like(initial, 100.)
        updated = model.apply_correction(initial, huge, public)
        expected_delta = torch.cat((.5*public['area'].clamp_min(1e-12).sqrt().clamp_min(.1)[...,None].expand(-1,-1,2),
                                    torch.full_like(initial[...,2:], .5)), -1)*public['mask']*public['valid'][...,None]
        torch.testing.assert_close(updated-initial, expected_delta, rtol=1e-5, atol=1e-6)
        assert torch.equal((updated-initial)*(1-public['mask']), torch.zeros_like(initial))
        model.correction_head[-1].bias.fill_(.2)
        shifted = model(public)
        expected = initial.clone()
        for _ in range(12):
            expected = model.apply_correction(expected, torch.full_like(initial,.2), public)
        torch.testing.assert_close(shifted, expected, rtol=1e-6, atol=1e-6)
        model.correction_head[-1].bias.zero_()
    try:
        model({**public, 'reference': graph['reference']})
    except ValueError:
        pass
    else:
        raise AssertionError('Network must reject supervision inputs')
    state = actual_state.detach().requires_grad_()
    original_values = original_components(state, graph)
    target = sum(WEIGHTS[key]*original_values[key].mean() for key in WEIGHTS)
    values = losses(state, graph, None, arm, 0)
    torch.testing.assert_close(values['total'], target, rtol=1e-6, atol=1e-6)
    gradient = torch.autograd.grad(values['total'], state, retain_graph=True)[0]
    old_gradient = torch.autograd.grad(target, state)[0]
    torch.testing.assert_close(gradient, old_gradient, rtol=1e-5, atol=1e-6)
    fixed = losses(state, graph, None, arm, 200, include_quality=False)
    assert torch.equal(values['total'], fixed['total']) and selection(values) is values['total']
    # Extra padding is ignored: state and masks only use the original five layouts.
    cropped = {key: (value[:, :119, :119] if key in ('adjacency', 'mib', 'group') else
                     value[:, :119] if value.ndim >= 2 else value) for key, value in graph.items()}
    cropped_total = losses(state[:, :119], cropped, None, arm)['total']
    torch.testing.assert_close(values['total'], cropped_total, rtol=1e-6, atol=1e-6)
    scratch = {key: value.clone() for key, value in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=3e-4)
    captured = CapturedStep(model, optimizer, graph, arm)
    assert not optimizer.state
    assert all(torch.equal(value, scratch[key]) for key, value in model.state_dict().items())
    captured.inputs['target'].fill_(float('nan'))
    try:
        captured.replay()
    except RuntimeError as error:
        assert 'Nonfinite' in str(error)
    else:
        raise AssertionError('Nonfinite objective must block Adam')
    assert not optimizer.state
    captured.inputs['target'].copy_(graph['target'])
    captured.replay()
    reference_step = {key: value.clone() for key, value in model.state_dict().items()}
    correction_changed = any(not torch.equal(value, scratch[key]) for key, value in reference_step.items() if key.startswith('correction_head.'))
    assert correction_changed
    model.load_state_dict(scratch)
    eager = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=3e-4)
    eager.zero_grad(set_to_none=True)
    losses(model(public), graph, None, arm)['total'].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True, foreach=True)
    eager.step()
    delta = max(float((value-reference_step[key]).abs().max()) for key, value in model.state_dict().items())
    assert delta < 1e-6, delta
    result = {'parameters': sum(value.numel() for value in model.parameters()),
              'neutral_final_output_equals_initial_coarse_head': True, 'effective_applications': sum(calls), 'unique_blocks': len(model.layers), 'bounded_masked_additive_update_verified': True,
              'loss_abs_delta': float((values['total']-target).abs().detach()),
              'output_gradient_max_abs_delta': float((gradient-old_gradient).abs().max()),
              'capture_eager_Adam_max_abs_delta': delta,
              'correction_head_updated': correction_changed, 'public_only_contract': True,
              'nonfinite_prevents_Adam': True, 'fixed_Loss_and_padding': True,
              'peak_allocated_GiB': torch.cuda.max_memory_allocated()/2**30}
    return result


def main():
    started = time.perf_counter()
    seed_all(0)
    protected = verify_protected()
    metadata()
    assert sha(HERE/'loss_reference.py') == sha(ROOT/'experiments/E42_joint_regularization/losses.py')
    indices = [1683, 10027, 5474, 9793, 1814]
    rows = load_rows('train', indices, use_cache=False)
    batch = [rows[index % 5] for index in range(32)]
    checks = {}
    for arm in ARMS:
        checks[arm] = check_arm(arm, batch)
        clear()
        out = HERE/f'runs/20261028_preflight_{arm}_resumed'
        args = parser().parse_args(['--arm', arm, '--output', str(out), '--max-hours', '.1', '--max-epochs', '2'])
        run(args, implementation_check=True, check_rows=(batch, batch), stop_after_steps=1)
        first = torch.load(out/'checkpoints/latest.pt', map_location='cpu', weights_only=False)
        clear()
        args.resume = True
        run(args, implementation_check=True, check_rows=(batch, batch), stop_after_steps=2)
        resumed = torch.load(out/'checkpoints/latest.pt', map_location='cpu', weights_only=False)
        clear()
        args.resume = False
        args.output = HERE/f'runs/20261028_preflight_{arm}_uninterrupted'
        run(args, implementation_check=True, check_rows=(batch, batch), stop_after_steps=2)
        uninterrupted = torch.load(args.output/'checkpoints/latest.pt', map_location='cpu', weights_only=False)
        assert first['progress']['step'] == 1 and resumed['progress']['step'] == uninterrupted['progress']['step'] == 2
        for key in ('model', 'optimizer', 'scheduler'):
            same(resumed[key], uninterrupted[key], key)
        for key in ('torch', 'cuda'):
            same(resumed['rng'][key], uninterrupted['rng'][key], 'rng/'+key)
        assert resumed['elapsed_seconds'] > first['elapsed_seconds']
        assert first['identity'] == resumed['identity'] == uninterrupted['identity']
        assert all(torch.isfinite(value).all() for value in resumed['model'].values())
        checks[arm]['actual_Dropout_Adam_resume_bit_exact'] = True
        del first, resumed, uninterrupted
        clear()
    assert checks['unshared_residual']['parameters'] > checks['shared_residual']['parameters']
    assert verify_protected() == protected
    save(HERE/'runs/20261028_preflight/results.json', {'status': 'passed', 'unique_train_cases': 5,
         'indices': indices, 'padded_nodes': 120, 'arms': checks, 'formal_training_started': False,
         'sources': sources(), 'protected_checkpoints_unchanged': protected,
         'test_read': False, 'seconds': time.perf_counter()-started})
    print('E66 residual/shared arms: five-case Loss/bounds/unroll/Adam/resume preflight passed', flush=True)


if __name__ == '__main__':
    main()
