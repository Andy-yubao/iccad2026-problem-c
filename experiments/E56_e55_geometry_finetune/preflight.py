"""Five train members: parity with independently copied E42 Loss and real resume."""
import time
import torch
from .io import HERE, ROOT, save, sha
from .initialization import load, metadata, verify_protected
from .arms import model_config
from .model import GeometryModel
from .batches import inputs_only
from .data import load_rows, collate
from .loss import losses
from .loss_reference import components as original_components, WEIGHTS
from .train import seed_all, parser, run
from .cuda_step import CapturedStep

ARM = 'e42_geometry'


def main():
    started = time.perf_counter(); seed_all(0)
    protected = verify_protected()
    assert sha(HERE/'loss_reference.py') == sha(ROOT/'experiments/E42_joint_regularization/losses.py')
    ids = [50, 36, 22, 62, 45]
    rows = load_rows('train', ids, use_cache=False); batch = [rows[i % 5] for i in range(32)]
    graph, _ = collate(batch, 'cuda')
    model = GeometryModel(**model_config(ARM)).cuda().eval()
    source = load(model, model_config(ARM))
    assert all(torch.equal(value.cpu(), source['model'][key]) for key, value in model.state_dict().items())
    output = model(inputs_only(graph)).detach().requires_grad_()
    current = losses(output, graph, None, ARM, 0, include_quality=False)
    later = losses(output, graph, None, ARM, 200, include_quality=True)
    assert torch.equal(current['total'], later['total'])
    reference = original_components(output, graph)
    for key in WEIGHTS:
        torch.testing.assert_close(current[key], reference[key].mean(), rtol=1e-6, atol=1e-7)
    original_total = sum(WEIGHTS[k]*reference[k].mean() for k in WEIGHTS)
    current_grad = torch.autograd.grad(current['total'], output, retain_graph=True)[0]
    original_grad = torch.autograd.grad(original_total, output)[0]
    torch.testing.assert_close(current_grad, original_grad, rtol=1e-5, atol=1e-6)
    forward_delta = float((current['total']-original_total).detach().abs())
    gradient_delta = float((current_grad-original_grad).abs().max())
    expected = float(current['total'].detach())
    del output, current, later, reference, original_total, current_grad, original_grad, model
    # CUDA capture must preserve weights and reproduce the same eval-mode objective.
    model = GeometryModel(**model_config(ARM)).cuda().eval(); load(model, model_config(ARM))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0., weight_decay=3e-4)
    captured = CapturedStep(model, optimizer, graph, ARM); captured.replay()
    assert abs(float(captured.values['total'])-expected) < 1e-6
    assert all(torch.equal(value.cpu(), source['model'][key]) for key, value in model.state_dict().items())
    del model, optimizer, captured, graph
    torch.cuda.empty_cache()
    out = HERE/'runs/20261026_preflight_train_resume'
    args = parser().parse_args(['--output', str(out), '--max-hours', '.1', '--max-epochs', '2'])
    run(args, implementation_check=True, check_rows=(batch, batch), stop_after_steps=1)
    one = torch.load(out/'checkpoints/latest.pt', map_location='cpu', weights_only=False)
    args.resume = True
    run(args, implementation_check=True, check_rows=(batch, batch), stop_after_steps=2)
    two = torch.load(out/'checkpoints/latest.pt', map_location='cpu', weights_only=False)
    assert one['progress']['step'] == 1 and two['progress']['step'] == 2
    assert one['identity'] == two['identity'] and two['identity']['implementation_check']
    assert two['elapsed_seconds'] > one['elapsed_seconds']
    assert all(int(v['step']) == 1 for v in one['optimizer']['state'].values())
    assert all(int(v['step']) == 2 for v in two['optimizer']['state'].values())
    assert any(not torch.equal(source['model'][key], one['model'][key]) for key in source['model'])
    assert all(torch.isfinite(v).all() for v in two['model'].values())
    from .final import evaluate_row
    model = GeometryModel(**two['model_config']).eval(); model.load_state_dict(two['model'])
    smoke = evaluate_row(0, rows[0], model)
    assert 'error' not in smoke and smoke['metrics']['is_feasible'], smoke.get('error')
    assert verify_protected() == protected
    save(HERE/'runs/20261026_preflight/results.json', {
        'status': 'passed', 'unique_train_cases': 5, 'indices': ids,
        'initialization': metadata(), 'all_initial_parameters_match_E55_best40': True,
        'E42_eight_terms_total_abs_delta': forward_delta,
        'E42_output_gradient_max_abs_delta': gradient_delta,
        'loss_fixed_across_epochs_and_monitoring': True,
        'cuda_graph_capture_preserves_initial_parameters': True,
        'actual_training_steps': 2, 'optimizer_fresh_then_resumed_steps': [1, 2],
        'resume_identity_and_budget_preserved': True,
        'one_train_case_flash_hard_legal': True,
        'protected_checkpoints_unchanged': protected,
        'formal_training_started': False, 'test_read': False,
        'seconds': time.perf_counter()-started})
    print('Five-member E42 Loss parity and E55 post-training/resume preflight passed', flush=True)


if __name__ == '__main__':
    main()
