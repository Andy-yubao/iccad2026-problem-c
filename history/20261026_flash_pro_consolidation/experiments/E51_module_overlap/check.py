"""Preparation only: synthetic mechanisms and five train/dev rows, zero updates."""
import ast
import json
from pathlib import Path
import time
import numpy as np
import torch
from .io import HERE, ROOT, save, sha, sources
from .module_loss import from_rectangles, definition, components as module_components
from .loss import losses
from .arms import model_config
from .batches import inputs_only
from .data import load_rows, collate
from .diagnostics import module_diagnostics
from .model import GeometryModel, decode
from .pipeline import control_identity, CONTROL, commands
from .train import seed_all, gradient_norm, PlateauController, selection


def synthetic():
    def calculate(p, valid=None, mutable=None):
        p = torch.as_tensor(p, dtype=torch.double)
        if p.ndim == 2: p = p[None]
        a = p[..., 2:].prod(-1)
        v = torch.ones(p.shape[:2], dtype=torch.bool) if valid is None else valid
        m = v if mutable is None else mutable
        return from_rectangles(p, a, v, m)
    separated = [[0, 0, 10, 10], [10, 0, 10, 10]]
    assert calculate(separated)['module_overlap_counted'].item() == 0
    one = [[20, 20, 10, 10], [28, 20, 10, 10], [100, 20, 10, 10]]
    two = [[20, 20, 10, 10], [29, 20, 10, 10], [11, 20, 10, 10]]
    x, y = calculate(one), calculate(two)
    assert torch.allclose(x['module_overlap_base'], y['module_overlap_base'])
    assert y['module_overlap_counted'] > x['module_overlap_counted']
    p = torch.tensor([one], dtype=torch.double)
    assert torch.allclose(calculate(p*7)['module_overlap_counted'], x['module_overlap_counted'], atol=1e-12)
    assert torch.allclose(calculate(p[:, [2, 0, 1]])['module_overlap_counted'], x['module_overlap_counted'])
    padded = torch.cat((p, torch.tensor([[[20, 20, 200, 200]]], dtype=torch.double)), 1)
    assert torch.allclose(calculate(padded, torch.tensor([[True, True, True, False]]))['module_overlap_counted'], x['module_overlap_counted'])
    assert calculate(one, mutable=torch.zeros((1, 3), dtype=torch.bool))['module_overlap_counted'].item() == 0
    # A free-preplaced pair retains both nodes' load and a gradient on the free endpoint.
    assert calculate(one, mutable=torch.tensor([[False, True, False]]))['module_overlap_counted'] > 0
    triple = calculate([[0, 0, 10, 10]]*3)
    assert abs(triple['module_overlap_base'].item()-1) < 1e-12
    # New neighbor appears while an existing overlap remains: no count jump.
    def edge(delta):
        return calculate([[20, 20, 10, 10], [28, 20, 10, 10], [10+delta, 20, 10, 10]])['module_overlap_counted'].item()
    diffs = [abs(edge(e)-edge(-e)) for e in (1e-3, 1e-5, 1e-7)]
    assert diffs[2] < diffs[1] < diffs[0] and diffs[2] < 1e-7
    # Finite difference away from contact and min/max ties, independent area inputs.
    rect = torch.tensor([[[0., 0., 10., 10.], [8., 1., 10., 10.]]], dtype=torch.double, requires_grad=True)
    area = torch.full((1, 2), 100., dtype=torch.double); valid = torch.ones((1, 2), dtype=torch.bool)
    value = from_rectangles(rect, area, valid, valid)['module_overlap_counted'].sum()
    grad, = torch.autograd.grad(value, rect)
    eps = 1e-5; high = rect.detach().clone(); low = rect.detach().clone()
    high[0, 1, 0] += eps; low[0, 1, 0] -= eps
    numerical = (from_rectangles(high, area, valid, valid)['module_overlap_counted'] -
                 from_rectangles(low, area, valid, valid)['module_overlap_counted'])/(2*eps)
    assert torch.allclose(grad[0, 1, 0], numerical[0], atol=1e-8, rtol=1e-6)
    assert (grad[0, 1, :2].abs().sum() > 0)
    return {'passed': True, 'checks': ['zero at contact', 'same load/more neighbors -> larger loss',
        'scale and permutation invariance', 'padding and immutable masks', 'free-preplaced collision',
        'multi-coverage pair multiplicity', 'contact continuity', 'autograd vs finite difference'],
        'same_base': float(x['module_overlap_base']), 'one_neighbor_loss': float(x['module_overlap_counted']),
        'two_neighbor_loss': float(y['module_overlap_counted']), 'contact_differences': diffs}


def main():
    started = time.perf_counter(); seed_all(0)
    synthetics = synthetic()
    parsed = 0
    for path in HERE.rglob('*.py'):
        if 'runs' not in path.relative_to(HERE).parts:
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path)); parsed += 1
    controls = control_identity()
    for name in ('model.py', 'features.py', 'batches.py', 'data.py', 'solver.py'):
        assert sha(HERE/name) == sha(ROOT/'experiments/E42_joint_regularization'/name), name
    # Metadata/cache identity only; payload reads use the five explicit rows below.
    from .cache import verify
    verify('train'); verify('dev')
    rows = load_rows('train', [0, 1, 2], use_cache=False) + load_rows('dev_monitor500', [0, 1], use_cache=False)
    graph, _ = collate(rows, 'cpu')
    state = torch.load(CONTROL/'checkpoints/best.pt', map_location='cpu', weights_only=False)
    model = GeometryModel(**model_config('main')).eval(); model.load_state_dict(state['model'])
    pred = model(inputs_only(graph))
    main_values = losses(pred, graph, None, 'main', 2., include_quality=True)
    training = losses(pred, graph, None, 'main', 2., include_quality=False)
    assert torch.allclose(main_values['total'], training['total'])
    assert main_values['module_overlap_counted'] >= main_values['module_overlap_base']
    assert selection(main_values) is main_values['total']
    assert 'repair' not in main_values and 'occupancy' not in main_values
    assert torch.allclose(main_values['total'], main_values['geometry']+.25*main_values['module_overlap_counted'])
    zero = losses(pred, graph, None, 'main', 0., include_quality=False)
    half = losses(pred, graph, None, 'main', 1., include_quality=False)
    assert torch.equal(zero['total'], main_values['total'])
    assert torch.equal(half['total'], main_values['total'])
    with torch.no_grad():
        evaluation = losses(pred.detach(), graph, None, 'main', 100., include_quality=True)
    assert torch.equal(training['total'].detach(), evaluation['total'])
    assert torch.isfinite(torch.stack(list(main_values.values()))).all()
    params = list(model.parameters())
    gradients = {key: gradient_norm(main_values[key], params) for key in ('base_objective', 'weighted_module_objective', 'overlap', 'total')}
    assert all(np.isfinite(v) and v > 0 for v in gradients.values())
    rectangles = decode(pred, graph).detach().numpy(); extra = module_components(pred, graph)
    for i, row in enumerate(rows):
        n = row['case'].block_count
        diag = module_diagnostics(rectangles[i, :n], graph['area'][i, :n].numpy(), graph['mask'][i, :n, 0].bool().numpy())
        assert np.isclose(diag['module_load_mean'], float(extra['module_load_mean'][i].detach()), rtol=1e-5, atol=1e-7)
    scratch = GeometryModel(**model_config('main')).eval()
    scratch_values = losses(scratch(inputs_only(graph)), graph, None, 'main', 2., include_quality=False)
    assert torch.isfinite(torch.stack(list(scratch_values.values()))).all()
    cuda_check = {'available': torch.cuda.is_available(), 'checked': False}
    if torch.cuda.is_available():
        scratch = scratch.to('cuda'); cuda_graph = {k: v.to('cuda') for k, v in graph.items()}
        cuda_values = losses(scratch(inputs_only(cuda_graph)), cuda_graph, None, 'main', 2., include_quality=False)
        cuda_values['total'].backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in scratch.parameters())
        cuda_check.update(checked=True, finite_backward=True, optimizer_updates=0)
    # Scheduling only; no training updates or actual datasets involved.
    parameter = torch.nn.Parameter(torch.tensor(0.)); optimizer = torch.optim.AdamW([parameter])
    scheduler = PlateauController(optimizer)
    for epoch in range(15, 27): scheduler.step(1., epoch)
    assert scheduler.drops == 3 and scheduler.should_stop() and scheduler.lowest_validated
    for arm in ('main',):
        assert '--resume' not in commands(arm)['train']
    folder = HERE/'runs/20261025_preparation_unified_v2'
    result = {'status': 'passed', 'seconds': time.perf_counter()-started, 'parsed_python_files': parsed,
              'synthetic': synthetics, 'actual_case_count': 5, 'members': [list(r['member']) for r in rows],
              'roles': ['train']*3+['dev_monitor500']*2, 'frozen_E42_gradient_norms': gradients,
              'added_to_base_gradient_norm_ratio': gradients['weighted_module_objective']/gradients['base_objective'],
              'cuda_check': cuda_check, 'loss_definition': definition(), 'control': controls,
              'unified_loss_checks': ['same total with/without gradients', 'same total at epochs 0/1/2/100',
                  'selection is total', 'no occupancy/repair alternate criterion', 'fixed geometry + .25 module counted'],
              'model_parameters': sum(p.numel() for p in model.parameters()),
              'optimizer_updates': 0, 'solver_calls': 0, 'formal_training_started': False,
              'test_read': False, 'sources': sources()}
    save(folder/'results.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('sources', 'control', 'members')}, indent=2))


if __name__ == '__main__': main()
