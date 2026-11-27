"""Five train members only: loss semantics, actual CUDA steps/resume, one flash smoke."""
import copy
import json
import time
import torch
from .io import HERE, ROOT, save, sha
from .arms import model_config
from .model import GeometryModel
from .batches import inputs_only
from .data import load_rows, collate
from .loss import losses
from .train import seed_all, parser, run


def main():
    started = time.perf_counter(); seed_all(0)
    ids = [50, 36, 22, 62, 45]
    rows = load_rows('train', ids, use_cache=False)
    batch = [rows[i % 5] for i in range(32)]
    graph, _ = collate(batch, 'cuda')
    model = GeometryModel(**model_config('reference_position_shape')).cuda().eval()
    state = model(inputs_only(graph)).detach().requires_grad_()
    def loss(g, epoch=0, quality=False):
        return losses(state, g, None, 'reference_position_shape', epoch, include_quality=quality)['total']
    a = loss(graph); b = loss(graph, 200, True)
    assert torch.equal(a, b)
    minimal = {k: graph[k] for k in ('target', 'mask', 'area')}
    assert torch.equal(a, loss(minimal))
    exact = losses(graph['target'], graph, None, 'reference_position_shape', 0)['total']
    assert float(exact) == 0.
    grad = torch.autograd.grad(a, state)[0]
    assert torch.isfinite(grad).all()
    assert torch.equal(grad[graph['mask'] == 0], torch.zeros_like(grad[graph['mask'] == 0]))
    changed = dict(minimal); changed['target'] = graph['target'] + graph['mask']*.3
    assert not torch.equal(a, loss(changed))
    # Literal original E42 center/shape arithmetic, without evaluating any other term.
    import torch.nn.functional as F
    local = graph['area'].clamp_min(1e-12).sqrt(); mask = graph['mask']
    center = (F.smooth_l1_loss(state[..., :2]/local[..., None], graph['target'][..., :2]/local[..., None], reduction='none', beta=.1)*mask[..., :2]).sum((1, 2))/mask[..., :2].sum((1, 2)).clamp_min(1)
    shape = ((state[..., 2]-graph['target'][..., 2]).square()*mask[..., 2]).sum(1)/mask[..., 2].sum(1).clamp_min(1)
    assert torch.equal(a, (center+.25*shape).mean())
    del model, state, grad, graph, a, b, center, shape
    torch.cuda.empty_cache()
    output = HERE/'runs/20261026_preflight_train_resume'
    args = parser().parse_args(['--output', str(output), '--max-hours', '.1', '--max-epochs', '2'])
    run(args, implementation_check=True, check_rows=(batch, batch), stop_after_steps=1)
    one = torch.load(output/'checkpoints/latest.pt', map_location='cpu', weights_only=False)
    args.resume = True
    run(args, implementation_check=True, check_rows=(batch, batch), stop_after_steps=2)
    two = torch.load(output/'checkpoints/latest.pt', map_location='cpu', weights_only=False)
    assert one['progress']['step'] == 1 and two['progress']['step'] == 2
    assert two['elapsed_seconds'] > one['elapsed_seconds']
    assert two['identity'] == one['identity'] and two['identity']['implementation_check']
    assert any(not torch.equal(one['model'][key], two['model'][key]) for key in one['model'])
    assert all(torch.isfinite(v).all() for v in two['model'].values())
    from .final import evaluate_row
    model = GeometryModel(**two['model_config']).eval(); model.load_state_dict(two['model'])
    smoke = evaluate_row(0, rows[0], model)
    assert 'error' not in smoke and smoke['metrics']['is_feasible'], smoke.get('error')
    assert sha(HERE/'solver.py') == sha(ROOT/'experiments/E42_joint_regularization/solver.py')
    save(HERE/'runs/20261026_preflight/results.json', {
        'status': 'passed', 'unique_train_cases': 5, 'indices': ids,
        'fixed_loss_train_monitor_equal': True, 'only_target_mask_area_read_by_loss': True,
        'exact_reference_loss_zero': True, 'fixed_and_padding_loss_gradients_zero': True,
        'center_shape_match_original_E42': True, 'cuda_graph_actual_steps': 2,
        'resume_preserved_identity_optimizer_and_budget': True,
        'one_train_case_flash_hard_legal': smoke['metrics']['is_feasible'],
        'flash_solver_matches_E42_sha256': sha(HERE/'solver.py'),
        'formal_training_started': False, 'test_read': False,
        'seconds': time.perf_counter()-started})
    print('Five-member preflight passed', flush=True)


if __name__ == '__main__':
    main()
