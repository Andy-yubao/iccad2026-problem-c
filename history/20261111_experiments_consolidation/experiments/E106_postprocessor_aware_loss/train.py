"""Prepared critic trainer. Explicit current user prohibition blocks entry."""
import argparse
import random
import time
import numpy as np
import torch
from .common import OWNER, RUN, read, save, checkpoint, sha, sources, setup
from .model import PostprocessorCritic
from .data import load_case, batch_cases
from .objective import fitting_loss, role_exp_mean


def ensure_training_authorized():
    auth = read(OWNER/'authorization.json')
    if auth.get('critic_training') is not True:
        raise PermissionError('E106 training is disabled: user requested implementation WITHOUT starting training')
    if auth.get('plan_sha256') != sha(OWNER/'plan.json') or not auth.get('user_instruction'):
        raise PermissionError('A later explicit user authorization must identify this fixed plan')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    # Guard before GPU/model allocation, optimizer construction or training IO.
    ensure_training_authorized()
    setup()
    random.seed(0)
    plan = read(OWNER/'plan.json')
    bank = read(RUN/'bank_results.json')
    current = sources()
    assert bank['status'] == 'complete' and bank['identity']['sources'] == current
    out = RUN/'critic'
    out.mkdir(parents=True, exist_ok=True)
    if (out/'results.json').exists():
        raise RuntimeError('Completed critic is immutable')
    identity = dict(sources=current, bank_sha256=sha(RUN/'bank_results.json'),
                    normalization_sha256=sha(RUN/'normalization.json'))
    if (out/'config.json').exists():
        if not args.resume or read(out/'config.json')['identity'] != identity:
            raise RuntimeError('Existing training requires explicit same-identity resume')
    else:
        if args.resume:
            raise RuntimeError('No previous training to resume')
        save(out/'config.json', dict(identity=identity, plan=plan))
    panel = read(OWNER/'panel.json')['cases']
    rows = {role: [load_case(r['index']) for r in panel if r['split'] == role]
            for role in ('fit', 'validation')}
    normalization = read(RUN/'normalization.json')
    device = 'cuda'
    model = PostprocessorCritic(**plan['model']).to(device)
    policy = plan['proposed_training']
    optimizer = torch.optim.AdamW(model.parameters(), lr=policy['lr_tiers'][0], weight_decay=policy['weight_decay'])
    progress = dict(epoch=0, step=0, order=None, cursor=0, pending_monitor=True,
                    best=None, best_epoch=None, tier=0, plateau_best=None, bad=0,
                    elapsed_seconds=0., history=[])
    def rng_state():
        return dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all(),
                    numpy=np.random.get_state(), python=random.getstate())
    def restore_rng(state):
        torch.set_rng_state(state['torch'])
        torch.cuda.set_rng_state_all(state['cuda'])
        np.random.set_state(state['numpy'])
        random.setstate(state['python'])
    if args.resume:
        old = torch.load(out/'latest.pt', map_location=device, weights_only=False)
        assert old['identity'] == identity
        model.load_state_dict(old['model'])
        optimizer.load_state_dict(old['optimizer'])
        progress = old['progress']
        # RNG tensors must live on CPU for set_rng_state even after CUDA map loading.
        old['rng']['torch'] = old['rng']['torch'].cpu()
        old['rng']['cuda'] = [v.cpu() for v in old['rng']['cuda']]
        restore_rng(old['rng'])
    started, prior = time.perf_counter(), progress['elapsed_seconds']
    exp_means = {role: role_exp_mean(role) for role in ('fit', 'validation', 'check')}
    batch_size = policy['problems_per_batch']
    def write(name):
        progress['elapsed_seconds'] = prior + time.perf_counter()-started
        checkpoint(out/name, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                  progress=progress, rng=rng_state(), identity=identity,
                                  model_config=plan['model']))
    def stopped():
        return (OWNER/'TRAIN_STOP').exists() or (out/'STOP').exists()
    def monitor(role, data):
        model.eval()
        total = dict(total=0., value=0., directions=0., auxiliary=0.)
        with torch.no_grad():
            for start in range(0, len(data), batch_size):
                batch = batch_cases(data[start:start+batch_size], device)
                values = fitting_loss(model(batch['graph'], batch['states'], group_size=5), batch,
                                      normalization, exp_means[role])
                for key in total:
                    total[key] += float(values[key])*len(data[start:start+batch_size])
        return {key: value/len(data) for key, value in total.items()}
    try:
        while True:
            if stopped():
                write('latest.pt')
                save(out/'state.json', dict(status='user_stop', epoch=progress['epoch'], step=progress['step']))
                return
            if progress['pending_monitor']:
                tr = monitor('fit', rows['fit'][:500])
                dv = monitor('validation', rows['validation'])
                value = dv['total']
                if not np.isfinite(value):
                    raise FloatingPointError('Nonfinite monitoring Loss')
                improved = progress['best'] is None or value < progress['best']-1e-5
                if improved:
                    progress['best'], progress['best_epoch'] = value, progress['epoch']
                meaningful = progress['plateau_best'] is None or value < progress['plateau_best']-max(1e-5,abs(progress['plateau_best'])*.001)
                if meaningful:
                    progress['plateau_best'], progress['bad'] = value, 0
                else:
                    progress['bad'] += 1
                patience = policy['lowest_tier_patience'] if progress['tier']==2 else policy['plateau_patience']
                early_stop = progress['bad'] >= patience and progress['tier']==2
                if progress['bad'] >= patience and not early_stop:
                    progress['tier'] += 1
                    progress['bad'], progress['plateau_best'] = 0, value
                    for group in optimizer.param_groups:
                        group['lr'] = policy['lr_tiers'][progress['tier']]
                progress['pending_monitor'] = False
                progress['history'].append(dict(epoch=progress['epoch'], step=progress['step'], train=tr, validation=dv,
                                                lr=optimizer.param_groups[0]['lr']))
                if improved:
                    write('best.pt')
                write('latest.pt')
                save(out/'state.json', dict(status='running', epoch=progress['epoch'], step=progress['step'],
                                             best_epoch=progress['best_epoch'], validation_Loss=value,
                                             lr=optimizer.param_groups[0]['lr']))
                print('CRITIC', progress['epoch'], progress['step'], 'validation', value, flush=True)
                if early_stop:
                    frozen = torch.load(out/'best.pt', map_location=device, weights_only=False)
                    model.load_state_dict(frozen['model'])
                    check_rows = [load_case(r['index']) for r in panel if r['split']=='check']
                    result = dict(status='early_stop', best_epoch=progress['best_epoch'], best_validation_Loss=progress['best'],
                                  epochs=progress['epoch'], updates=progress['step'], check=monitor('check', check_rows),
                                  checkpoint_sha256=sha(out/'best.pt'), automatic_layout_training=False, test_read=False)
                    save(out/'results.json', result)
                    save(out/'state.json', dict(result, status='complete'))
                    return
            if progress['order'] is None:
                progress['order'] = torch.randperm(len(rows['fit'])).tolist()
                progress['cursor'] = 0
            model.train()
            while progress['cursor'] < len(rows['fit']):
                if stopped():
                    break
                ids = progress['order'][progress['cursor']:progress['cursor']+batch_size]
                batch = batch_cases([rows['fit'][i] for i in ids], device)
                optimizer.zero_grad(set_to_none=True)
                values = fitting_loss(model(batch['graph'], batch['states'], group_size=5), batch,
                                      normalization, exp_means['fit'])
                values['total'].backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), policy['clip_norm'])
                if not bool(torch.isfinite(norm) & torch.isfinite(values['total']) & (norm<1e4) & (values['total']<1e6)):
                    raise FloatingPointError('Critic finite/Loss/gradient guard rejected BEFORE AdamW')
                optimizer.step()
                progress['cursor'] += len(ids)
                progress['step'] += 1
                if progress['step'] % 100 == 0:
                    write('latest.pt')
            if stopped():
                continue
            progress['epoch'] += 1
            progress['order'], progress['cursor'], progress['pending_monitor'] = None, 0, True
    except BaseException as exc:
        save(out/'error.json', dict(error=repr(exc), epoch=progress['epoch'], step=progress['step']))
        raise


if __name__ == '__main__':
    main()
