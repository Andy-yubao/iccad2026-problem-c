"""Five train cases only: original Loss/gradient, padding, Adam and exact resume."""
import gc
import time
import torch
from .io import HERE, ROOT, save, sha, sources
from .initialization import metadata, verify_protected
from .arms import model_config
from .model import GeometryModel
from .batches import inputs_only, collate as padded_collate
from .data import load_rows, collate
from .loss import losses, per_case
from .loss_reference import components as original_components, WEIGHTS
from .train import seed_all, parser, run, selection, PlateauController, LR_LEVELS
from .cuda_step import CapturedStep


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


def main():
    started = time.perf_counter()
    seed_all(0)
    protected = verify_protected()
    initial = metadata()
    assert not initial['checkpoint_loaded']
    assert sha(HERE/'loss_reference.py') == sha(ROOT/'experiments/E42_joint_regularization/losses.py')
    ids = [1683, 10027, 5474, 9793, 1814]
    rows = load_rows('train', ids, use_cache=False)
    batch = [rows[i % 5] for i in range(32)]
    graph, _ = collate(batch, 'cuda')
    model = GeometryModel(**model_config('rectangle_ab')).cuda().eval()
    scratch = {key: value.clone() for key, value in model.state_dict().items()}
    state = model(inputs_only(graph)).detach().requires_grad_()
    original = original_components(state, graph)
    expected = sum(WEIGHTS[key]*original[key].mean() for key in WEIGHTS)
    values = losses(state, graph, None, 'rectangle_ab', 0)
    torch.testing.assert_close(values['total'], expected, rtol=1e-6, atol=1e-6)
    grad = torch.autograd.grad(values['total'], state, retain_graph=True)[0]
    old_grad = torch.autograd.grad(expected, state)[0]
    torch.testing.assert_close(grad, old_grad, rtol=1e-5, atol=1e-6)
    assert torch.isfinite(grad).all()
    forward_delta = float((values['total']-expected).abs().detach())
    gradient_delta = float((grad-old_grad).abs().max())
    later = losses(state, graph, None, 'rectangle_ab', 200, include_quality=False)
    assert all(torch.equal(values[key], later[key]) for key in values)
    assert selection(values) is values['total']
    per = per_case(state, graph, 'rectangle_ab')
    assert torch.equal(per['total'].mean(), values['total'])
    # Compare each layout alone with its padded instance, using identical output states.
    for index, row in enumerate(rows):
        single = padded_collate([row['graph']], [0], 'cuda')
        count = row['graph']['n']
        one = per_case(state[index:index+1, :count], single, 'rectangle_ab')
        for key in WEIGHTS:
            torch.testing.assert_close(one[key][0], per[key][index], rtol=1e-5, atol=2e-6)
    # Real Adam step on captured/eager eval paths must agree, not just forward scalars.
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=3e-4)
    captured = CapturedStep(model, opt, graph, 'rectangle_ab')
    assert not opt.state
    assert all(torch.equal(value, scratch[key]) for key, value in model.state_dict().items())
    captured.inputs['target'].fill_(float('nan'))
    try:
        captured.replay()
    except RuntimeError as error:
        assert 'Nonfinite' in str(error)
    else:
        raise AssertionError('Nonfinite objective must prevent Adam update')
    assert not opt.state
    assert all(torch.equal(value, scratch[key]) for key, value in model.state_dict().items())
    captured.inputs['target'].copy_(graph['target'])
    captured.replay()
    captured_parameters = {key: value.clone() for key, value in model.state_dict().items()}
    model.load_state_dict(scratch)
    eager_opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=3e-4)
    eager_opt.zero_grad(set_to_none=True)
    losses(model(inputs_only(graph)), graph, None, 'rectangle_ab')['total'].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True, foreach=True)
    eager_opt.step()
    update_delta = max(float((value-captured_parameters[key]).abs().max()) for key, value in model.state_dict().items())
    assert update_delta < 1e-6, update_delta
    assert any(not torch.equal(value, scratch[key]) for key, value in captured_parameters.items())
    # Verify plateau protection and two full epochs at the lowest tier.
    controller = PlateauController(eager_opt)
    for epoch in range(15):
        controller.step(1., epoch)
    assert controller.drops == 0 and controller.early_bad == 0
    stopped_epoch = None
    for epoch in range(15, 30):
        controller.step(1., epoch)
        if controller.should_stop():
            stopped_epoch = epoch
            break
    assert stopped_epoch == 26 and controller.drops == 3
    assert controller.lowest_completed_epochs == 2
    assert eager_opt.param_groups[0]['lr'] == LR_LEVELS[-1]
    del state, original, expected, values, later, grad, old_grad, per, one, model, graph
    del captured, opt, eager_opt, scratch, captured_parameters
    clear()
    out = HERE/'runs/20261027_preflight_resumed'
    args = parser().parse_args(['--output', str(out), '--max-hours', '.1', '--max-epochs', '2'])
    run(args, implementation_check=True, check_rows=(batch, batch), stop_after_steps=1)
    first = torch.load(out/'checkpoints/latest.pt', map_location='cpu', weights_only=False)
    clear()
    args.resume = True
    run(args, implementation_check=True, check_rows=(batch, batch), stop_after_steps=2)
    resumed = torch.load(out/'checkpoints/latest.pt', map_location='cpu', weights_only=False)
    clear()
    args.resume = False
    args.output = HERE/'runs/20261027_preflight_uninterrupted'
    run(args, implementation_check=True, check_rows=(batch, batch), stop_after_steps=2)
    uninterrupted = torch.load(args.output/'checkpoints/latest.pt', map_location='cpu', weights_only=False)
    assert first['progress']['step'] == 1
    assert resumed['progress']['step'] == uninterrupted['progress']['step'] == 2
    for key in ('model', 'optimizer', 'scheduler'):
        same(resumed[key], uninterrupted[key], key)
    for key in ('torch', 'cuda'):
        same(resumed['rng'][key], uninterrupted['rng'][key], 'rng/'+key)
    assert resumed['elapsed_seconds'] > first['elapsed_seconds']
    assert first['identity'] == resumed['identity'] == uninterrupted['identity']
    assert all(int(value['step']) == 2 for value in resumed['optimizer']['state'].values())
    assert all(torch.isfinite(value).all() for value in resumed['model'].values())
    assert verify_protected() == protected
    save(HERE/'runs/20261027_preflight/results.json', {
        'status': 'passed', 'unique_train_cases': 5, 'indices': ids, 'initialization': initial,
        'E42_loss_abs_delta': forward_delta, 'E42_output_gradient_max_abs_delta': gradient_delta,
        'captured_eager_actual_Adam_parameter_max_abs_delta': update_delta,
        'fixed_loss_all_epochs_and_selection': True, 'per_layout_padding_invariant': True,
        'capture_no_weight_or_optimizer_advance': True,
        'injected_nonfinite_blocks_Adam_update': True,
        'actual_dropout_Adam_resume_bit_exact': True, 'optimizer_steps': [1, 2],
        'budget_preserved': True, 'four_tier_plateau_and_early_stop_verified': True,
        'protected_checkpoints_unchanged': protected, 'sources': sources(),
        'formal_training_started': False, 'test_read': False, 'seconds': time.perf_counter()-started})
    print('E64 five-case scratch/unified-Loss/Adam/resume preflight passed', flush=True)


if __name__ == '__main__':
    main()
