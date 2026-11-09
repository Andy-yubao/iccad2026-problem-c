"""At most five frozen train members; actual updates and exact restore."""
import ast
import copy
import gc
import json
import time
import numpy as np
import torch
from .io import HERE, ROOT, save, sha, sources, snapshot, checkpoint
from .model import GeometryModel
from .data import load_rows
from .resident import ResidentRows
from .batches import inputs_only, decode
from .loss import per_case
from .loss_config import COEFFICIENTS
from .schedule import PlateauController, LR
from .step_control import GraphStepControl
from .train import seed_all, optimizer_for, rng_state, restore_rng, build_capture, MODEL, fresh_progress

def same(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a.cpu(), b.cpu())
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            same(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            same(x, y)
    else:
        assert a == b

def verify():
    result = json.loads((HERE/'runs/preflight/results.json').read_text(encoding='utf-8'))
    if result['status'] != 'passed' or result['sources'] != sources():
        raise ValueError('Current-source passing preflight required')
    return result

def scheduler_checks():
    p = torch.nn.Parameter(torch.zeros(()))
    opt = torch.optim.AdamW([p], lr=LR[0])
    ctrl = PlateauController(opt)
    trace = []
    for epoch in range(27):
        ctrl.step(1., epoch)
        trace.append([epoch, opt.param_groups[0]['lr'], ctrl.drops, ctrl.should_stop()])
        assert ctrl.should_stop() == (epoch == 26)
        if epoch < 18:
            assert opt.param_groups[0]['lr'] == LR[0]
        if epoch in (14, 19, 24):
            old = copy.deepcopy(ctrl.state_dict())
            ctrl = PlateauController(opt)
            ctrl.load_state_dict(old)
            same(ctrl.state_dict(), old)
    return dict(trace=trace, drop_epochs=[18, 21, 24], earliest_stop_epoch=26)

def main():
    begin = time.perf_counter()
    folder = HERE/'runs/preflight'
    folder.mkdir(parents=True, exist_ok=True)
    result = dict(status='error', test_read=False, postprocessing_calls=0)
    try:
        for p in HERE.glob('*.py'):
            ast.parse(p.read_text(encoding='utf-8'), filename=str(p))
        result['schedule'] = scheduler_checks()
        plan = json.loads((HERE/'plan.json').read_text(encoding='utf-8'))
        ids = plan['preflight_train_indices']
        assert len(ids) == len(set(ids)) == 5
        seed_all(0)
        rows = load_rows('train', ids)
        raw = load_rows('train', ids, use_cache=False)
        for a, b in zip(rows, raw):
            assert a['member'] == b['member']
            for key, value in a['graph'].items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(value, b['graph'][key], rtol=0, atol=0)
        result['raw_cache_identity'] = True
        from .dataset_view import ReferenceLayouts
        assert len(ReferenceLayouts('train')) == 16000
        for role in ('test', 'local_test', 'tail_test'):
            try:
                ReferenceLayouts(role)
            except PermissionError:
                pass
            else:
                raise AssertionError('Sealed payload role accepted')
        pack = ResidentRows(rows, 'cuda', max_nodes=120)
        batch = pack.batch([i % 5 for i in range(32)])
        seed_all(0)
        model = GeometryModel(**MODEL).cuda().eval()
        assert sum(p.numel() for p in model.parameters()) == 1471539
        assert all(p.requires_grad for p in model.parameters())
        try:
            model(batch)
        except ValueError:
            pass
        else:
            raise AssertionError('Network accepted labels/full loss ports')
        with torch.no_grad():
            state = model(inputs_only(batch))
        state = state.detach().requires_grad_()
        values = per_case(state, batch)
        torch.testing.assert_close(values['total'], values['legacy_total'], rtol=3e-6, atol=3e-6)
        grad = torch.autograd.grad(values['total'].mean(), state, retain_graph=True)[0]
        legacy_grad = torch.autograd.grad(values['legacy_total'].mean(), state)[0]
        torch.testing.assert_close(grad, legacy_grad, rtol=3e-5, atol=3e-6)
        assert len(COEFFICIENTS) == 10
        result['ten_terms'] = dict(coefficients=COEFFICIENTS, initial_loss=float(values['total'].mean().detach()),
                                 max_recomposition_error=float((values['total']-values['legacy_total']).abs().max().detach()))
        out = decode(state.detach(), batch)
        locked = ~batch['mask'][..., 2].bool()
        pre = ~batch['mask'][..., 0].bool()
        torch.testing.assert_close(out[..., 2:][locked], batch['specs'][..., 2:][locked], rtol=0, atol=0)
        torch.testing.assert_close(out[pre], batch['specs'][pre], rtol=0, atol=0)
        relative = (out[..., 2]*out[..., 3]/batch['area'].clamp_min(1e-12)-1).abs()
        assert float(relative[batch['valid'] & ~locked].max()) < 1e-5
        from .data import collate
        short, _ = collate([rows[i % 5] for i in range(32)], 'cuda')
        with torch.no_grad():
            short_values = per_case(model(inputs_only(short)), short)
        for key in values:
            torch.testing.assert_close(short_values[key], values[key].detach(), rtol=5e-5, atol=5e-5)
        result['input_decode_padding'] = True
        del model, values, state, grad, legacy_grad, out, short_values
        gc.collect()
        torch.cuda.empty_cache()

        # Eager/captured arithmetic with dropout disabled; real dropout is covered below.
        cfg = {**MODEL, 'ffn_dropout': 0.}
        seed_all(0)
        left = GeometryModel(**cfg).cuda().train()
        right = GeometryModel(**cfg).cuda().train()
        right.load_state_dict(left.state_dict())
        lo, ro = optimizer_for(left), optimizer_for(right)
        cap = build_capture(right, ro, pack)
        proj = GraphStepControl(left)
        lo.zero_grad(set_to_none=False)
        lv = per_case(left(inputs_only(batch)), batch)
        lv['total'].mean().backward()
        ln = torch.nn.utils.clip_grad_norm_(left.parameters(), 1., foreach=True)
        proj.before()
        lo.step()
        proj.after()
        rv, rn = cap.replay(batch)
        for k, v in left.state_dict().items():
            torch.testing.assert_close(v, right.state_dict()[k], rtol=2e-5, atol=2e-6)
        for lstate, rstate in zip(lo.state.values(), ro.state.values()):
            for k in lstate:
                torch.testing.assert_close(lstate[k], rstate[k], rtol=3e-5, atol=3e-6)
        result['eager_captured_actual_update'] = dict(loss_difference=abs(float(lv['total'].mean().detach())-float(rv['total'])),
                                                    gradient_norm_difference=abs(float(ln)-float(rn)))
        del left, right, lo, ro, cap, proj, lv, rv
        gc.collect()
        torch.cuda.empty_cache()

        def trajectory(resume=False):
            seed_all(0)
            model = GeometryModel(**MODEL).cuda()
            opt = optimizer_for(model)
            scheduler = PlateauController(opt)
            progress = fresh_progress(5)
            progress['order'] = np.random.default_rng(0).permutation(5)
            initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            cap = build_capture(model, opt, pack)
            before = rng_state()
            from .train import monitor
            preserved = monitor(model, rows)
            same(before, rng_state())
            scheduler.step(preserved['total'], 0)
            durations = []
            for i in range(4):
                model.train()
                torch.cuda.synchronize()
                tick = time.perf_counter()
                cap.replay(batch)
                torch.cuda.synchronize()
                durations.append(time.perf_counter()-tick)
                progress['step'] += 1
                progress['cursor'] = (progress['cursor']+32) % 5
                if resume and i == 1:
                    checkpoint(folder/'restore_check.pt', dict(model=model.state_dict(), optimizer=opt.state_dict(),
                         scheduler=scheduler.state_dict(), progress=copy.deepcopy(progress), rng=rng_state(),
                         projection=cap.projection.summary(), elapsed_seconds=12.5))
                    saved = torch.load(folder/'restore_check.pt', map_location='cpu', weights_only=False)
                    del cap, model, opt
                    gc.collect()
                    torch.cuda.empty_cache()
                    model = GeometryModel(**MODEL).cuda()
                    model.load_state_dict(saved['model'])
                    opt = optimizer_for(model)
                    opt.load_state_dict(saved['optimizer'])
                    scheduler = PlateauController(opt)
                    scheduler.load_state_dict(saved['scheduler'])
                    progress = saved['progress']
                    restore_rng(saved['rng'])
                    cap = build_capture(model, opt, pack, saved['projection'])
                    same(saved['rng'], rng_state())
                    assert saved['elapsed_seconds'] == 12.5
            for i in range(12):
                key = f'layers.{i}.local.weight'
                assert not torch.equal(initial[key], model.state_dict()[key].cpu()), key
                assert torch.isfinite(model.state_dict()[key]).all()
            final = dict(model={k: v.cpu().clone() for k, v in model.state_dict().items()},
                 optimizer=copy.deepcopy(opt.state_dict()), scheduler=scheduler.state_dict(),
                 progress=progress, rng=rng_state(), projection=cap.projection.summary())
            assert final['projection']['steps'] == 4
            # Failure guards must not change accepted weights, optimizer, or projection.
            if resume:
                old_model = copy.deepcopy(model.state_dict())
                old_opt = copy.deepcopy(opt.state_dict())
                old_projection = cap.projection.summary()
                bad = {k: v.clone() for k, v in batch.items()}
                bad['target'][0, :, 2] = 100.
                for corrupt in ('per_case', 'nan'):
                    if corrupt == 'nan':
                        bad['target'].fill_(float('nan'))
                    try:
                        cap.replay(bad)
                    except RuntimeError:
                        pass
                    else:
                        raise AssertionError('Invalid update accepted')
                    same(old_model, model.state_dict())
                    same(old_opt, opt.state_dict())
                    same(old_projection, cap.projection.summary())
                result['guards'] = True
            del model, opt, cap
            gc.collect()
            torch.cuda.empty_cache()
            return final, durations

        uninterrupted, times = trajectory()
        resumed, _ = trajectory(True)
        same(uninterrupted, resumed)
        result['exact_resume'] = 'four real dropout/Adam updates equal 2+save/rebuild+2; model, optimizer, RNG, projection, order/cursor and schedule bit-exact'
        result['step_microbenchmark_seconds'] = times
        result['status'] = 'passed'
        result['sources'] = sources()
        result['train_indices'] = ids
        result['parameters'] = 1471539
        snapshot(folder/'source_snapshot.zip', dict(sources=result['sources']))
    except BaseException as exc:
        result['error'] = repr(exc)
        raise
    finally:
        result['elapsed_seconds'] = time.perf_counter()-begin
        save(folder/'results.json', result)
        print(json.dumps({k: v for k, v in result.items() if k != 'sources'}, ensure_ascii=False, indent=2), flush=True)

if __name__ == '__main__':
    main()
