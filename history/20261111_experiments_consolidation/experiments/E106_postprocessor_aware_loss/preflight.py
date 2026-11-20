"""Five unique fit cases,25 reusable labels, forward/gradient checks; zero updates."""
import time
import numpy as np
import torch
from .common import OWNER, RUN, read, save, sources, sha, setup
from .cache import load_rows
from .bank import initialize, evaluate
from .data import load_case, batch_cases
from .model import PostprocessorCritic
from .objective import fitting_loss, role_exp_mean
from .train import ensure_training_authorized
from .batches import decode


def main():
    setup()
    if (OWNER/'preflight.json').exists():
        raise RuntimeError('Preflight already recorded; no repetitive automatic tests')
    started = time.perf_counter()
    panel = read(OWNER/'panel.json')
    assert panel['total_problems']==5000 and panel['total_drafts']==25000 and panel['drafts_per_problem']==5
    assert len(set(tuple(v['member']) for v in panel['cases']))==5000
    assert panel['split_counts']==dict(fit=3000,validation=1000,check=1000)
    candidates = sorted([r for r in panel['cases'] if r['split']=='fit'], key=lambda r:r['n'])
    chosen = [candidates[i] for i in (0, 749, 1499, 2249, 2999)]
    current = sources()
    identity = dict(sources=current, panel_sha256=sha(OWNER/'panel.json'))
    initialize(identity)
    raw = load_rows('train', [r['train_index'] for r in chosen])
    for spec, row in zip(chosen, raw):
        evaluate(spec['index'], supplied_row=row)
    records = [load_case(r['index']) for r in chosen]
    batch = batch_cases(records)
    state = batch['states'].detach().requires_grad_(True)
    model = PostprocessorCritic(**read(OWNER/'plan.json')['model']).eval()
    prediction = model(batch['graph'], state, group_size=5)
    # Temporary fixed scales check the computation, NOT calibration from five cases.
    temporary = dict(Q_scale=1., auxiliary_scale=[1.,1.,1.], slope_scale=1.)
    values = fitting_loss(prediction, dict(batch, states=state), temporary, role_exp_mean('fit'))
    gradient = torch.autograd.grad(values['total'], [state, *model.parameters()], allow_unused=False)
    assert all(torch.isfinite(v).all() for v in gradient)
    active = batch['graph']['mask']*batch['graph']['valid'][...,None]
    assert (gradient[0]*(1-active)).abs().max()==0
    assert (gradient[0]*active).abs().max()>0
    decoded = decode(state.detach(), batch['graph'])
    locked = ~batch['graph']['mask'][...,2].bool() & batch['graph']['valid']
    pre = ~batch['graph']['mask'][...,0].bool() & batch['graph']['valid']
    assert torch.equal(decoded[...,2:][locked], batch['graph']['specs'][...,2:][locked])
    assert torch.equal(decoded[pre], batch['graph']['specs'][pre])
    try:
        model(dict(batch['graph'], target=state), state)
        raise AssertionError('Label leakage guard did not reject target')
    except ValueError:
        pass
    # Identical related inputs must share masks even in dropout training mode.
    graph0 = {k:v[:5].clone() for k,v in batch['graph'].items()}
    identical = state.detach()[:1].repeat(5,1,1)
    model.train()
    with torch.no_grad():
        tied = model(graph0, identical, group_size=5)['Q']
    assert float((tied-tied[0]).abs().max())<1e-6
    blocked = False
    try:
        ensure_training_authorized()
    except PermissionError:
        blocked = True
    assert blocked
    parameter_count = sum(p.numel() for p in model.parameters())
    assert 1_700_000 < parameter_count < 2_400_000
    assert all(len(r['metrics'])==5 and all(v['is_feasible'] for v in r['metrics']) for r in records)
    save(OWNER/'preflight.json', dict(status='passed', sources=current, unique_train_cases=5,
                                      case_indices=[r['index'] for r in chosen], reusable_Q_labels=25,
                                      actual_parameters=parameter_count, model_config=read(OWNER/'plan.json')['model'],
                                      forward_and_input_parameter_gradients_finite=True, locked_gradient_zero=True,
                                      fixed_position_and_shape_exact_in_decoder=True, tied_dropout_passed=True,
                                      input_label_guard_passed=True, optimizer_updates=0, training_entry_blocked=True,
                                      official_dev_test_payloads=0, internal_check_cases=0,
                                      elapsed_seconds=time.perf_counter()-started, test_read=False))
    print('PASSED:5 fit cases,25 reusable Q labels;',parameter_count,'parameters;0 optimizer updates;training BLOCKED.',flush=True)


if __name__ == '__main__':
    main()
