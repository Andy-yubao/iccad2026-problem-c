"""Bounded multi-round training and train-only paired postprocessor feedback."""
import argparse
import copy
import gc
import json
import signal
import threading
import time
import traceback
import zipfile
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from ..io import ROOT, HERE, sha, save, checkpoint, exclusive
from ..train import rng_state, restore_rng
from ..model import GeometryModel
from ..data import load_rows, collate, register_resident
from ..dataset_view import ReferenceLayouts
from ..resident import ResidentRows
from ..batches import inputs_only
from ..weight_learning.engine import CapturedStep, per_case
from ..qualification import evaluate as evaluator
from datasets.reference_layouts import public_hard_reference
from scoring.evaluation import evaluate_case
from .common import OWNER, RUN, PLAN_PATH, MODEL, read, sources, coefficients, q_summary, PhaseSchedule

class Halt(Exception):
    pass

@torch.no_grad()
def monitor(model, rows, co, check):
    model.eval()
    blocks = []
    for start in range(0,len(rows),32):
        check()
        batch,_ = collate(rows[start:start+32], 'cuda')
        out = model(inputs_only(batch))
        values = per_case(out,batch,co)
        if not bool(torch.isfinite(out).all() & torch.isfinite(torch.stack(list(values.values()))).all() & (values['total'].amax()<1000)):
            raise RuntimeError('Monitoring numerical guard rejected')
        keys = list(values)
        blocks.append(torch.stack([values[k] for k in keys],-1).cpu().double().numpy())
    a = np.concatenate(blocks)
    return {**{k:float(a[:,i].mean()) for i,k in enumerate(keys)}, 'count':len(rows)}

def make_model(state, pack, co, resume=False):
    model = GeometryModel(**MODEL).cuda()
    model.load_state_dict(state['model'],strict=True)
    opt = torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.0003)
    schedule = PhaseSchedule()
    if resume:
        opt.load_state_dict(copy.deepcopy(state['optimizer']))
        schedule.load_state_dict(state['scheduler'])
    restore_rng(state['rng'])
    model.train()
    cap = CapturedStep(model,opt,pack.batch([i%len(pack.rows) for i in range(32)]),co,state['projection'])
    return model,opt,cap,schedule

def train_branch(folder, parent_path, co, round_spec, arm, pack, tr_monitor, dev, identity, check, beat):
    latest = folder/'checkpoints/latest.pt'
    best_path = folder/'checkpoints/best.pt'
    parent_sha = sha(parent_path)
    prior_elapsed = 0.
    if latest.exists():
        state = torch.load(latest,map_location='cpu',weights_only=False)
        assert state['identity']==identity and state['coefficients']==co and state['parent_sha256']==parent_sha
        if state['status']=='error':
            raise RuntimeError('Error phase needs diagnosis; no automatic retry')
        if state['finished']:
            return best_path, read(folder/'results.json')
        p = state['progress']
        prior_elapsed = state['phase_elapsed_seconds']
        resumed = True
    else:
        state = torch.load(parent_path,map_location='cpu',weights_only=False)
        p = dict(epoch=0,cursor=0,step=0,order=None,pending_monitor=True,best=None,best_epoch=None,
                 seen=np.zeros(len(pack.rows),dtype=bool),history=[])
        resumed = False
        save(folder/'registration.json',dict(parent=parent_path.relative_to(ROOT).as_posix(),
             parent_sha256=parent_sha,coefficients=co,optimizer='fresh AdamW including control',
             scheduler_source='fixed own-Loss dev500',sampler_seed=round_spec['sampler_seed'],test_read=False))
    torch.set_num_threads(4)
    model,opt,cap,schedule = make_model(state,pack,co,resumed)
    del state
    started = time.perf_counter()
    reason = 'running'
    def store(status, finished=False, best=False):
        value = dict(model=model.state_dict(),model_config=MODEL,optimizer=opt.state_dict(),
             rng=rng_state(),projection=cap.projection.summary(),scheduler=schedule.state_dict(),
             progress=p,coefficients=co,identity=identity,parent_sha256=parent_sha,
             phase_elapsed_seconds=prior_elapsed+time.perf_counter()-started,
             status=status,finished=finished,optimizer_reset_at_phase_start=True,test_read=False)
        checkpoint(latest,value)
        if best:
            checkpoint(best_path,value)
        save(folder/'progress.json',dict(status=status,finished=finished,epoch=p['epoch'],step=p['step'],
             cursor=p['cursor'],seen_count=int(p['seen'].sum()),best=p['best'],best_epoch=p['best_epoch'],
             lr=schedule.lr,scheduler=schedule.state_dict(),history=p['history'],
             phase_elapsed_seconds=value['phase_elapsed_seconds'],test_read=False))
        beat(stage='training',round=round_spec['round'],arm=arm,epoch=p['epoch'],step=p['step'],
             lr=schedule.lr,seen_count=int(p['seen'].sum()))
    try:
        store('running')
        while True:
            check()
            if p['pending_monitor']:
                preserved = rng_state()
                dv = monitor(model,dev,co,check)
                tr = monitor(model,tr_monitor,co,check)
                restore_rng(preserved)
                improved = p['best'] is None or dv['total'] < p['best']-1e-5
                if improved:
                    p['best'],p['best_epoch'] = dv['total'],p['epoch']
                early = schedule.step(dv['total'],p['epoch'])
                for group in opt.param_groups:
                    group['lr'] = schedule.lr
                p['history'].append(dict(epoch=p['epoch'],step=p['step'],train_monitor=tr,
                     dev_monitor=dv,lr_next=schedule.lr,scheduler=schedule.state_dict()))
                p['pending_monitor'] = False
                store('running',best=improved)
                print('TRAIN',round_spec['round'],arm,'epoch',p['epoch'],'dev',dv['total'],
                      'train',tr['total'],'lr',schedule.lr,flush=True)
                if early or p['epoch']>=10:
                    reason = 'early_stop' if early else 'maximum_epochs'
                    break
            if p['order'] is None:
                p['order'] = np.random.default_rng(round_spec['sampler_seed']+p['epoch']).permutation(len(pack.rows))
                p['cursor'] = 0
            model.train()
            interval_start,interval_step = time.perf_counter(),p['step']
            while p['cursor'] < len(pack.rows):
                check()
                ids = p['order'][p['cursor']:p['cursor']+32]
                pack.batch(ids,out=cap.inputs)
                cap.replay()
                p['step'] += 1
                p['cursor'] += len(ids)
                p['seen'][ids] = True
                if p['step']%100==0:
                    span = time.perf_counter()-interval_start
                    throughput = 32*(p['step']-interval_step)/span
                    store('running')
                    print('UPDATE',round_spec['round'],arm,p['step'],'samples/s',throughput,flush=True)
                    interval_start,interval_step = time.perf_counter(),p['step']
            p['epoch'] += 1
            p['cursor'],p['order'],p['pending_monitor'] = 0,None,True
            store('monitor_pending')
        summary = dict(status=reason,epochs=p['epoch'],updates=p['step'],best_epoch=p['best_epoch'],
             best_dev_loss=p['best'],best_checkpoint=best_path.relative_to(ROOT).as_posix(),
             best_sha256=sha(best_path),phase_elapsed_seconds=prior_elapsed+time.perf_counter()-started,
             coefficient_sum=sum(co.values()),test_read=False)
        save(folder/'results.json',summary)
        store(reason,finished=True)
        return best_path,summary
    except Halt:
        store('user_stop')
        raise
    except BaseException:
        store('error')
        raise
    finally:
        del model,opt,cap
        gc.collect()
        torch.cuda.empty_cache()

@torch.inference_mode()
def evaluate_panel(predictor, checkpoint_path, rows, folder, check, beat, stage, **fields):
    from ..qualification.metrics import module_overlap
    digest = sha(checkpoint_path)
    state = torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    predictor.model.load_state_dict(state['model'],strict=True)
    predictor.model.eval()
    del state
    records = []
    with threadpool_limits(limits=1):
        for i,row in enumerate(rows):
            check()
            path = folder/f'{i:04d}.json'
            if path.exists():
                record = read(path)
                assert record['checkpoint_sha256']==digest and record['member']==list(row['member'])
            else:
                draft,final,stats,hard,wall,draft_time = evaluator.request(predictor,row['case'].solver_inputs)
                scored = evaluate_case(row['case'],final,wall)
                record = dict(member=list(row['member']),n=int(row['case'].block_count),
                     metrics=scored['metrics'],independent_hard_checks=hard,
                     complete_seconds=wall,draft_seconds=draft_time,repair_seconds=stats['seconds'],
                     raw_overlap=module_overlap(draft),draft_positions=draft.tolist(),positions=final.tolist(),
                     fallback=stats['row_retained'],trace=stats['trace'],checkpoint_sha256=digest,test_read=False)
                save(path,record)
            if not record['independent_hard_checks']['is_feasible'] or not record['metrics']['is_feasible']:
                raise RuntimeError('Hard legality failed; evidence preserved, no automatic retry')
            records.append(record)
            if (i+1)%10==0:
                beat(stage=stage,completed=i+1,total=len(rows),**fields)
    result = q_summary(records)
    save(folder/'results.json',result)
    print('SCORE',stage,fields,json.dumps(result),flush=True)
    return result

def choose(scores, plan):
    accepted = [a for a in ('plus','minus') if all(
        scores[a][r]['hard_legal']==100 and
        scores[a][r]['mixed_Q'] < scores['control'][r]['mixed_Q']-plan['minimum_absolute_gain_per_repeat']
        for r in range(2))]
    if not accepted:
        return 'control',accepted
    means = {a:float(np.mean([s['mixed_Q'] for s in scores[a]])) for a in accepted}
    if len(accepted)==2 and abs(means['plus']-means['minus'])<=1e-12:
        return 'control',accepted
    return min(accepted,key=lambda a:means[a]),accepted

def gallery(rows, checkpoint_path, check):
    from visualization.layouts import plot_case,geometry_details
    folder = RUN/'terminal/gallery'
    folder.mkdir(exist_ok=True)
    for i,row in enumerate(rows[:100]):
        check()
        record = read(RUN/f'terminal/dev500/{i:04d}.json')
        constraints = np.asarray(row['case'].solver_inputs[5]).tolist()
        reference = (row['graph']['reference']*row['graph']['scale']).numpy().tolist()
        variants = {k:dict(positions=p,soft_details=geometry_details(p,constraints)) for k,p in
                    [('reference',reference),('raw',record['draft_positions']),('final',record['positions'])]}
        image = folder/f'{i:03d}.png'
        if not image.exists():
            plot_case(dict(case_id=f'dev500:{i}',panel='E98 dynamic mixedQ frozen final',constraints=constraints,
                 variants=variants),image,stages=(('reference','Reference'),('raw','L'),('final','Full repair')),
                 show_metrics=False)
    (folder/'index.html').write_text('<!doctype html><meta charset="utf-8"><style>img{width:100%}</style>'+''.join(
        f'<p>{i}</p><img loading="lazy" src="{i:03d}.png">' for i in range(100)),encoding='utf-8')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    plan = read(PLAN_PATH)
    current_sources = sources()
    gate = read(RUN/'preflight.json')
    assert gate['status']=='passed' and gate['sources']==current_sources
    assert sha(ROOT/plan['initial_checkpoint'])==plan['initial_sha256']
    identity = dict(plan_sha256=sha(PLAN_PATH),sources=current_sources)
    if args.resume:
        assert read(RUN/'config.json')['identity']==identity
        if (RUN/'error.json').exists() or (RUN/'results.json').exists():
            raise RuntimeError('Completed/error run is not a normal resume')
        prior = read(RUN/'state.json')['elapsed_seconds']
    else:
        if (RUN/'config.json').exists():
            raise RuntimeError('Run already started; refusing duplicate')
        prior = 0.
    with exclusive(RUN/'training.lock'):
        if not args.resume:
            save(RUN/'config.json',dict(identity=identity,plan=plan))
            with zipfile.ZipFile(RUN/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
                for name in current_sources:
                    archive.write(ROOT/name,name)
        started = time.perf_counter()
        stopped = [False]
        previous_signal = signal.signal(signal.SIGINT,lambda *_:stopped.__setitem__(0,True))
        hb = dict(status='running',stage='initializing',test_read=False)
        closed = threading.Event()
        lock = threading.RLock()
        def check():
            if stopped[0] or (RUN/'STOP').exists() or (OWNER/'STOP').exists():
                raise Halt('user_stop')
        def beat(**fields):
            import os
            with lock:
                hb.update(fields)
                save(RUN/'state.json',dict(**hb,pid=os.getpid(),elapsed_seconds=prior+time.perf_counter()-started,
                     updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
        def heartbeats():
            while not closed.wait(5):
                beat()
        thread = threading.Thread(target=heartbeats,daemon=True)
        thread.start()
        try:
            check()
            evaluator.setup()
            beat(stage='loading_train16k_and_dev500')
            rows = load_rows('train')
            assert [list(r['member']) for r in rows]==[r['member'] for r in ReferenceLayouts('train').records]
            dev = load_rows('dev_monitor500')
            assert [list(r['member']) for r in dev]==[r['member'] for r in ReferenceLayouts('dev_monitor500').records]
            pack = ResidentRows(rows,'cuda',max_nodes=120,max_pins=146)
            register_resident(pack)
            lookup = {tuple(r['member']):i for i,r in enumerate(rows)}
            tr_monitor = [rows[lookup[tuple(r['member'])]] for r in ReferenceLayouts('train_monitor500').records]
            parent_path,center = ROOT/plan['initial_checkpoint'],plan['initial_coefficients']
            summaries = []
            for spec in plan['rounds']:
                check()
                folder = RUN/f"rounds/r{spec['round']:02d}_{spec['name']}"
                if (folder/'selected.json').exists():
                    selected = read(folder/'selected.json')
                    assert sha(ROOT/selected['checkpoint'])==selected['checkpoint_sha256']
                    parent_path,center = ROOT/selected['checkpoint'],selected['coefficients']
                    summaries.append(read(folder/'results.json'))
                    continue
                co = {'control':dict(center),
                      'plus':coefficients(center,spec['direction'],1,plan['log_perturbation']),
                      'minus':coefficients(center,spec['direction'],-1,plan['log_perturbation'])}
                save(folder/'registration.json',dict(round=spec,center=center,coefficients=co,
                     parent=parent_path.relative_to(ROOT).as_posix(),parent_sha256=sha(parent_path),test_read=False))
                paths,training = {},{}
                for arm in plan['branch_arms']:
                    check()
                    paths[arm],training[arm] = train_branch(folder/'branches'/arm,parent_path,co[arm],spec,arm,
                         pack,tr_monitor,dev,identity,check,beat)
                feedback_rows = [rows[j] for j in spec['feedback_train_indices']]
                assert len(feedback_rows)==100 and len({tuple(r['member']) for r in feedback_rows})==100
                assert [list(r['member']) for r in feedback_rows]==spec['feedback_members']
                for row in feedback_rows:
                    case = row['case']
                    case.reference_positions = public_hard_reference(case.solver_inputs[1],case.solver_inputs[5],case.solver_inputs[6])
                torch.set_num_threads(1)
                with torch.inference_mode(),threadpool_limits(limits=1):
                    evaluator.warmup()
                    predictor = evaluator.Predictor(feedback_rows[0]['case'].solver_inputs)
                scores = {a:[] for a in plan['branch_arms']}
                for repetition in range(2):
                    order = plan['branch_arms'] if repetition==0 else plan['branch_arms'][::-1]
                    for arm in order:
                        scores[arm].append((repetition,evaluate_panel(predictor,paths[arm],feedback_rows,
                             folder/f'feedback/{arm}/repeat{repetition}',check,beat,'train_feedback',
                             round=spec['round'],arm=arm,repeat=repetition)))
                scores = {a:[v for _,v in sorted(vals)] for a,vals in scores.items()}
                winner,accepted = choose(scores,plan)
                result = dict(round=spec['round'],name=spec['name'],direction=spec['direction'],
                     training=training,train_feedback=scores,selected_arm=winner,accepted_candidates=accepted,
                     feedback_unique_cases=100,feedback_pool_cases=1000,test_read=False)
                save(folder/'results.json',result)
                selected = dict(round=spec['round'],selected_arm=winner,coefficients=co[winner],
                     checkpoint=paths[winner].relative_to(ROOT).as_posix(),checkpoint_sha256=sha(paths[winner]),
                     selection_source='train100/1000 mixed_Q only; dev used for own-Loss epoch selection, not Q',
                     test_read=False)
                save(folder/'selected.json',selected)
                parent_path,center = paths[winner],co[winner]
                summaries.append(result)
                save(RUN/'rounds_summary.json',dict(completed=len(summaries),rounds=summaries,test_read=False))
                del predictor
                gc.collect()
                torch.cuda.empty_cache()
            frozen = dict(checkpoint=parent_path.relative_to(ROOT).as_posix(),checkpoint_sha256=sha(parent_path),
                 coefficients=center,completed_rounds=len(summaries),plan_sha256=sha(PLAN_PATH),test_read=False)
            if (RUN/'frozen.json').exists():
                assert read(RUN/'frozen.json')==frozen
            else:
                save(RUN/'frozen.json',frozen)
            check()
            torch.set_num_threads(1)
            for row in dev:
                case = row['case']
                case.reference_positions = public_hard_reference(case.solver_inputs[1],case.solver_inputs[5],case.solver_inputs[6])
            with torch.inference_mode(),threadpool_limits(limits=1):
                evaluator.warmup()
                predictor = evaluator.Predictor(dev[0]['case'].solver_inputs)
            terminal = evaluate_panel(predictor,parent_path,dev,RUN/'terminal/dev500',check,beat,'frozen_terminal_dev')
            gallery(dev,parent_path,check)
            result = dict(status='complete',frozen=frozen,terminal_dev500=terminal,
                 total_branch_updates=sum(s['updates'] for r in summaries for s in r['training'].values()),
                 elapsed_seconds=prior+time.perf_counter()-started,test_read=False,
                 interpretation='adaptive dev-Loss monitored, single-seed bounded research; terminal dev Q cannot change frozen selection; unequal actual branch compute; not contest score')
            save(RUN/'results.json',result)
            (RUN/'report.md').write_text('# E98多轮混合Q调权\n\n'+json.dumps(result,ensure_ascii=False,indent=2)+
                 '\n\n旧pilot结果未用于初始化或方向选择。后处理接纳仅用训练100/1000；验证Loss用于阶段内best/LR/早停。测试封存。\n',encoding='utf-8')
            beat(status='complete',stage='complete')
        except Halt:
            beat(status='user_stop',stage='stopped_no_next_stage')
        except BaseException as exc:
            save(RUN/'error.json',dict(error=repr(exc),traceback=traceback.format_exc(),test_read=False))
            beat(status='error',stage='failed_no_retry')
            traceback.print_exc()
            raise
        finally:
            closed.set()
            thread.join(timeout=6)
            signal.signal(signal.SIGINT,previous_signal)

if __name__ == '__main__':
    main()
