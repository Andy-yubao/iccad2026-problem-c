"""Same five cases: fixed G, membership pool, shape-only MIB, Adam and exact resume."""
import gc
import time
import torch
from .io import HERE, ROOT, save, sha, sources
from .initialization import metadata, verify_protected
from .arms import model_config
from .model import GeometryModel
from .base_model import GeometryModel as OriginalModel
from .batches import inputs_only
from .resident import ResidentRows
from .data import load_rows
from .loss import losses
from .loss_reference import components as original_components, WEIGHTS
from .train import seed_all, parser, run, selection
from .cuda_step import CapturedStep

ARMS = ('group_tokens', 'global_tokens')


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
    original = OriginalModel(width=96, layers=12, rectangle=True, ffn_dropout=.05).cuda().eval()
    original.load_state_dict({key: value for key, value in model.state_dict().items() if not key.startswith('scales.')})
    public = inputs_only(graph)
    with torch.no_grad():
        expected_state = original(public)
        actual_state = model(public)
    assert torch.equal(expected_state, actual_state), 'Neutral branch must preserve E64 forward exactly'
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
    assignments = model.assignments(public)
    for assignment, active, column in ((assignments[0],assignments[1],3),(assignments[2],assignments[3],2)):
        eligible = (public['constraints'][...,column]>0)&public['valid']
        torch.testing.assert_close(assignment.sum(-1), eligible.float(), rtol=1e-6, atol=1e-6)
        assert torch.equal(active, assignment.sum(1)>0)
    before = model.scales[0].shape_inputs(state.detach(), public)
    changed_xy = state.detach().clone();changed_xy[...,:2] += 7
    assert torch.equal(before, model.scales[0].shape_inputs(changed_xy, public))
    old_mode = model.pool_mode
    with torch.no_grad():
        for exchange in model.scales:
            exchange.group_output.weight.fill_(.001)
            exchange.shape_output.weight.fill_(.001)
        a = model(public)
        model.pool_mode = 'global' if old_mode == 'groups' else 'groups'
        b = model(public)
        sensitivity = float((a-b).abs().max())
        assert sensitivity > 1e-8
        model.pool_mode = old_mode
        for exchange in model.scales:
            exchange.group_output.weight.zero_()
            exchange.shape_output.weight.zero_()
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
    branch_changed = any(not torch.equal(value, scratch[key]) for key, value in reference_step.items() if key.startswith('scales.'))
    assert branch_changed
    model.load_state_dict(scratch)
    eager = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=3e-4)
    eager.zero_grad(set_to_none=True)
    losses(model(public), graph, None, arm)['total'].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True, foreach=True)
    eager.step()
    delta = max(float((value-reference_step[key]).abs().max()) for key, value in model.state_dict().items())
    assert delta < 1e-6, delta
    result = {'parameters': sum(value.numel() for value in model.parameters()),
              'neutral_initial_forward_bit_exact_to_E64': True,
              'loss_abs_delta': float((values['total']-target).abs().detach()),
              'output_gradient_max_abs_delta': float((gradient-old_gradient).abs().max()),
              'membership_sensitivity': sensitivity, 'MIB_input_independent_of_centers': True, 'membership_pool_masks_conserved': True, 'capture_eager_Adam_max_abs_delta': delta,
              'new_branch_updated': branch_changed, 'public_only_contract': True,
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
    assert checks['group_tokens']['parameters'] == checks['global_tokens']['parameters']
    assert verify_protected() == protected
    save(HERE/'runs/20261028_preflight/results.json', {'status': 'passed', 'unique_train_cases': 5,
         'indices': indices, 'padded_nodes': 120, 'arms': checks, 'formal_training_started': False,
         'sources': sources(), 'protected_checkpoints_unchanged': protected,
         'test_read': False, 'seconds': time.perf_counter()-started})
    print('E67 token membership arms: five-case Loss/pooling/MIB/Adam/resume preflight passed', flush=True)


if __name__ == '__main__':
    main()
