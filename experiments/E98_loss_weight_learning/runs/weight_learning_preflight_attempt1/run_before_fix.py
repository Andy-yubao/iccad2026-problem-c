"""One matched round of coefficient direct search using train-only solver feedback."""
import argparse,copy,gc,json,os,signal,threading,time,traceback,zipfile
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from ..io import ROOT,HERE,save,checkpoint,sha,exclusive
from ..train import seed_all,rng_state,restore_rng,optimizer_for,MODEL
from ..model import GeometryModel
from ..batches import inputs_only
from ..data import load_rows,collate,register_resident
from ..resident import ResidentRows
from ..dataset_view import ReferenceLayouts
from ..qualification import evaluate as evaluator
from scoring.evaluation import evaluate_case
from scoring.official import official_evaluator
from datasets.reference_layouts import public_hard_reference,registry
from .engine import CapturedStep,per_case

OWNER=Path(__file__).resolve().parent
PLAN_PATH=OWNER/'plan.json'
PLAN=json.loads(PLAN_PATH.read_text(encoding='utf-8'))
RUN=HERE/PLAN['run']

class Halt(Exception):pass

def sources():
    paths=list(OWNER.glob('*.py'))+[PLAN_PATH,OWNER/'README.md']
    paths += [HERE/p for p in ('model.py','features.py','batches.py','data.py','resident.py','ports.py','cache.py','io.py','simple_loss.py','loss.py','loss_config.py','group_tree.py','captured_projection.py','step_control.py','train.py')]
    result={p.relative_to(ROOT).as_posix():sha(p) for p in paths}
    result.update(evaluator.sources())
    for p in (ROOT/'datasets').glob('*.py'):result[p.relative_to(ROOT).as_posix()]=sha(p)
    for p in ('protocol/paths.json','protocol/reference_workset_v5.json','protocol/reference_workset_v6.json','protocol/split_v1.json','datasets/expanded_reference/__init__.py'):
        result[p]=sha(ROOT/p)
    return result

def verify_parent():
    assert sha(ROOT/PLAN['parent_checkpoint'])==PLAN['parent_sha256']
    assert sha(HERE/'qualification/plan.json')==PLAN['source_baseline_evaluation_plan_sha256']
    saved=torch.load(ROOT/PLAN['parent_checkpoint'],map_location='cpu',weights_only=False)
    assert saved['progress']['step']==29000 and saved['progress']['epoch']==58
    assert saved['model_config']==MODEL
    for name,digest in saved['identity']['sources'].items():
        assert sha(ROOT/name)==digest,('Baseline source drift',name)
    return saved

def verify_preflight():
    value=json.loads((OWNER/'preflight.json').read_text(encoding='utf-8'))
    assert value['status']=='passed' and value['sources']==sources()

def restore_branch(state,pack,coefficients):
    model=GeometryModel(**MODEL).cuda()
    model.load_state_dict(state['model'],strict=True)
    opt=optimizer_for(model);opt.load_state_dict(state['optimizer'])
    for group in opt.param_groups:group['lr']=PLAN['learning_rate']
    restore_rng(state['rng']);model.train()
    cap=CapturedStep(model,opt,pack.batch([i%len(pack.rows) for i in range(32)]),coefficients,state['projection'])
    return model,opt,cap

@torch.no_grad()
def monitor(model,rows,coefficients,halt=lambda:None):
    model.eval();blocks=[]
    for start in range(0,len(rows),32):
        halt();batch,_=collate(rows[start:start+32],'cuda')
        out=model(inputs_only(batch));values=per_case(out,batch,coefficients)
        if not bool(torch.isfinite(out).all() & torch.isfinite(torch.stack(list(values.values()))).all() & (values['total'].amax()<1000)):
            raise RuntimeError('Weighted monitoring guard failed')
        keys=list(values);blocks.append(torch.stack([values[k] for k in keys],-1).cpu().double().numpy())
    a=np.concatenate(blocks)
    return {**{k:float(a[:,j].mean()) for j,k in enumerate(keys)},'count':len(rows),
            'per_case_total':a[:,keys.index('total')].tolist()}

def train_branch(arm,parent,pack,monitor_rows,identity,check,beat):
    folder=RUN/'branches'/arm;latest=folder/'checkpoints/latest.pt'
    coefficients=PLAN['coefficients'][arm]
    if latest.exists():
        saved=torch.load(latest,map_location='cpu',weights_only=False)
        assert saved['identity']==identity and saved['coefficients']==coefficients
        if saved['status']=='error':raise RuntimeError('Error branch needs diagnosis')
        if saved['finished']:return latest
        progress=saved['progress']
    else:
        saved=parent
        progress=dict(epoch=0,cursor=0,order=None,step=0,pending_monitor=True,best=None,best_epoch=None,
                      seen=np.zeros(16000,dtype=bool),history=[],sampler_epoch_offset=58)
        save(folder/'registration.json',dict(arm=arm,coefficients=coefficients,parent_sha256=PLAN['parent_sha256'],
             updates=2000,sample_schedule='complete permutations epoch58..61',learning_rate=5e-5,
             inherited='model/Adam age29000/Dropout RNG/projection',test_read=False))
    torch.set_num_threads(4)
    model,opt,cap=restore_branch(saved,pack,coefficients)
    def store(status,finished=False,best=False):
        state=dict(model=model.state_dict(),model_config=MODEL,optimizer=opt.state_dict(),
             rng=rng_state(),projection=cap.projection.summary(),progress=progress,
             coefficients=coefficients,identity=identity,status=status,finished=finished,
             scheduler=dict(mode='fixed LR and fixed2000 budget',lr=5e-5),
             parent_sha256=PLAN['parent_sha256'],total_optimizer_step=29000+progress['step'])
        checkpoint(latest,state)
        if best:checkpoint(folder/'checkpoints/best.pt',state)
        save(folder/'progress.json',dict(status=status,epoch=progress['epoch'],step=progress['step'],
             cursor=progress['cursor'],best=progress['best'],best_epoch=progress['best_epoch'],
             history=progress['history'],test_read=False))
        beat(stage='training',arm=arm,arm_step=progress['step'],arm_epoch=progress['epoch'])
    store('running')
    try:
        while True:
            check()
            if progress['pending_monitor']:
                preserved=rng_state();value=monitor(model,monitor_rows,coefficients,check);restore_rng(preserved)
                improved=progress['best'] is None or value['total']<progress['best']-1e-5
                if improved:progress['best']=value['total'];progress['best_epoch']=progress['epoch']
                progress['history'].append(dict(epoch=progress['epoch'],step=progress['step'],train_monitor=value))
                progress['pending_monitor']=False
                store('running',best=improved)
                print(arm,'epoch',progress['epoch'],'step',progress['step'],'fixed train500 Loss',value['total'],flush=True)
                if progress['epoch']>=4:break
            if progress['order'] is None:
                progress['order']=np.random.default_rng(58+progress['epoch']).permutation(16000)
                progress['cursor']=0
            model.train()
            while progress['cursor']<16000:
                check();ids=progress['order'][progress['cursor']:progress['cursor']+32]
                pack.batch(ids,out=cap.inputs);values,norm=cap.replay()
                progress['step']+=1;progress['cursor']+=len(ids);progress['seen'][ids]=True
                if progress['step']%100==0:
                    store('running');print(arm,progress['step'],'/2000',flush=True)
            progress['epoch']+=1;progress['cursor']=0;progress['order']=None;progress['pending_monitor']=True
            store('monitor_pending')
        assert progress['step']==2000 and cap.projection.steps==31000
        store('complete',finished=True)
    except Halt:
        store('user_or_budget_stop');raise
    except BaseException:
        store('error');raise
    finally:
        del cap,opt,model;gc.collect();torch.cuda.empty_cache()
    return latest

def load_predictor(predictor,path):
    state=torch.load(path,map_location='cpu',weights_only=False)
    predictor.model.load_state_dict(state['model'],strict=True);predictor.model.eval()
    del state

@torch.inference_mode()
def feedback(paths,rows,predictor,check,beat):
    from ..qualification.metrics import module_overlap
    torch.set_num_threads(1)
    all_scores={name:[] for name in paths}
    with threadpool_limits(limits=1):
        for repetition in range(2):
            arms=list(paths) if repetition==0 else list(paths)[::-1]
            for arm in arms:
                check();load_predictor(predictor,paths[arm]);records=[]
                for i,row in enumerate(rows):
                    check();target=RUN/f'feedback/{arm}/repeat{repetition}/{i:03d}.json'
                    if target.exists():record=json.loads(target.read_text())
                    else:
                        draft,final,stats,hard,wall,draft_time=evaluator.request(predictor,row['case'].solver_inputs)
                        scored=evaluate_case(row['case'],final,wall)
                        assert scored['metrics']['is_feasible'] and hard['is_feasible']
                        record=dict(member=list(row['member']),n=int(row['case'].block_count),
                             metrics=scored['metrics'],seconds=wall,draft_seconds=draft_time,
                             raw_overlap=module_overlap(draft),draft_positions=draft.tolist(),positions=final.tolist(),
                             checkpoint_sha256=sha(paths[arm]),repeat=repetition,source_role='train_monitor500',
                             fallback=stats['row_retained'],test_read=False)
                        save(target,record)
                    assert record['member']==list(row['member']) and record['checkpoint_sha256']==sha(paths[arm])
                    records.append(record)
                q=official_evaluator().compute_total_score([r['metrics']['cost'] for r in records],[r['n'] for r in records])
                all_scores[arm].append(dict(repeat=repetition,weighted_Q=q,all_legal=all(r['metrics']['is_feasible'] for r in records),
                     mean_seconds=float(np.mean([r['seconds'] for r in records]))))
                beat(stage='train_feedback',arm=arm,repeat=repetition,feedback_completed=24)
                print('TRAIN FEEDBACK',arm,repetition,q,flush=True)
    # Normalize reversed-arm processing back to repeat order.
    for values in all_scores.values():values.sort(key=lambda r:r['repeat'])
    save(RUN/'feedback/results.json',dict(scores=all_scores,feedback_role='train only',unique_members=24,
         repeats=2,solver_calls=192,test_read=False))
    return all_scores

def choose(scores,paths):
    margin=PLAN['minimum_absolute_gain_per_repeat'];accepted=[]
    for arm in ('plus','minus'):
        if all(scores[arm][r]['all_legal'] and scores[arm][r]['weighted_Q']<scores['control'][r]['weighted_Q']-margin for r in range(2)):
            accepted.append(arm)
    selected=min(accepted,key=lambda a:np.mean([r['weighted_Q'] for r in scores[a]])) if accepted else 'control'
    result=dict(status='frozen_before_dev',selected_arm=selected,checkpoint=paths[selected].relative_to(ROOT).as_posix(),
         checkpoint_sha256=sha(paths[selected]),coefficients=PLAN['coefficients'][selected],
         coefficient_changed=selected!='control',accepted_candidates=accepted,
         feedback_source='24 frozen train members only, two repeats, no dev reads',
         selection_rule='candidate must beat control by >0.0001 on BOTH feedback repeats',
         validation_may_change_selection=False,test_read=False)
    save(RUN/'selected.json',result);return result

@torch.inference_mode()
def terminal(selected,paths,predictor,check,beat):
    from ..qualification.metrics import module_overlap
    torch.set_num_threads(1)
    # This is the first point in the controller that accesses dev payloads.
    selection_path=RUN/'selected.json';frozen_sha=sha(selection_path)
    rows=evaluator.load_rows('dev_monitor500')
    expected=registry()['sets']['dev_monitor500']['records']
    assert [list(r['member']) for r in rows]==[r['member'] for r in expected]
    for row in rows:
        case=row['case'];case.reference_positions=public_hard_reference(case.solver_inputs[1],case.solver_inputs[5],case.solver_inputs[6])
    arms=['control']+([selected['selected_arm']] if selected['selected_arm']!='control' else [])
    summaries={}
    with threadpool_limits(limits=1):
        for arm in arms:
            load_predictor(predictor,paths[arm]);records=[]
            for i,row in enumerate(rows):
                check();target=RUN/f'terminal/{arm}/{i:04d}.json'
                if target.exists():record=json.loads(target.read_text())
                else:
                    draft,final,stats,hard,wall,draft_time=evaluator.request(predictor,row['case'].solver_inputs)
                    score=evaluate_case(row['case'],final,wall);raw=evaluate_case(row['case'],draft,draft_time)
                    assert score['metrics']['is_feasible'] and hard['is_feasible']
                    record=dict(index=i,member=list(row['member']),n=int(row['case'].block_count),
                         complete_seconds=wall,draft_seconds=draft_time,repair_seconds=stats['seconds'],
                         metrics=score['metrics'],independent_hard_checks=hard,raw_metrics=raw['metrics'],
                         raw_overlap=module_overlap(draft),draft_positions=draft.tolist(),positions=final.tolist(),
                         fallback=stats['row_retained'],trace=stats['trace'],checkpoint_sha256=sha(paths[arm]))
                    save(target,record)
                assert record['member']==list(row['member']) and record['checkpoint_sha256']==sha(paths[arm])
                records.append(record)
                if (i+1)%25==0:beat(stage='terminal_dev_only',arm=arm,completed=i+1,total=500)
            summaries[arm]=evaluator.summary(records)
            save(RUN/f'terminal/{arm}/results.json',dict(terminal=summaries[arm],test_read=False,selection_already_frozen=True))
            print('FROZEN TERMINAL',arm,json.dumps(summaries[arm]),flush=True)
    assert sha(selection_path)==frozen_sha
    save(RUN/'terminal/results.json',dict(status='complete',arms=summaries,selected_sha256=frozen_sha,
         no_validation_feedback=True,test_read=False))
    # Render only after all timed solves; same IDs/scales, no rating controls.
    from visualization.layouts import plot_case,geometry_details
    for arm in arms:
        gallery=RUN/f'terminal/{arm}/gallery';gallery.mkdir(exist_ok=True)
        for i,row in enumerate(rows[:100]):
            check();record=json.loads((RUN/f'terminal/{arm}/{i:04d}.json').read_text())
            image=gallery/f'{i:03d}.png'
            if image.exists():continue
            c=np.asarray(row['case'].solver_inputs[5]).tolist()
            reference=(row['graph']['reference']*row['graph']['scale']).numpy().tolist()
            variants={k:dict(positions=p,soft_details=geometry_details(p,c)) for k,p in
                      [('reference',reference),('raw',record['draft_positions']),('final',record['positions'])]}
            plot_case(dict(case_id=f'dev500:{i}',panel=f'E98 weight round1 / {arm}',constraints=c,variants=variants),image,
                 stages=(('reference','Reference'),('raw',arm+' layout'),('final','Full repair')),show_metrics=False)
        (gallery/'index.html').write_text('<!doctype html><meta charset="utf-8"><style>img{width:100%}</style>'+''.join(f'<p>{i}</p><img loading="lazy" src="{i:03d}.png">' for i in range(100)),encoding='utf-8')
    return summaries

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    verify_preflight();parent=verify_parent()
    if (OWNER/'STOP').exists() or (RUN/'STOP').exists():raise RuntimeError('STOP present')
    with exclusive(OWNER/'training.lock'):
        started=time.perf_counter();identity=dict(plan=PLAN,sources=sources(),parent_sha256=PLAN['parent_sha256'])
        if args.resume:
            assert json.loads((RUN/'config.json').read_text())['identity']==identity
            if (RUN/'error.json').exists():raise RuntimeError('Error run requires diagnosed migration')
            prior=json.loads((RUN/'budget.json').read_text())['elapsed_seconds']
        else:
            RUN.mkdir(parents=True,exist_ok=False);prior=0.
            save(RUN/'config.json',dict(identity=identity,parent_full_budget_seconds=parent['elapsed_seconds'],
                 original_baseline_complete_seconds=3465.580786599996,test_read=False))
            with zipfile.ZipFile(RUN/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
                for name in identity['sources']:archive.write(ROOT/name,name)
        elapsed=lambda:prior+time.perf_counter()-started
        requested=[False];old_signal=signal.signal(signal.SIGINT,lambda *_:requested.__setitem__(0,True))
        hb=dict(status='running',stage='initializing',pid=os.getpid());closed=threading.Event();lock=threading.RLock()
        def beat(**fields):
            with lock:
                hb.update(fields)
                save(RUN/'budget.json',dict(elapsed_seconds=elapsed(),maximum_seconds=3600))
                save(RUN/'state.json',dict(**hb,elapsed_seconds=elapsed(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),test_read=False))
        def check():
            if requested[0] or (OWNER/'STOP').exists() or (RUN/'STOP').exists():raise Halt('user_stop')
            if elapsed()>=3600:raise Halt('budget')
        def heartbeats():
            while not closed.wait(5.):beat()
        worker=threading.Thread(target=heartbeats,daemon=True);worker.start()
        status='running'
        try:
            # Imported evaluator setup fixes CPU math threads to1 for feedback;
            # branch training restores torch intra-op4, inter-op remains1.
            evaluator.setup();beat(stage='loading frozen train16k')
            rows=load_rows('train');train_records=ReferenceLayouts('train').records
            assert [list(r['member']) for r in rows]==[r['member'] for r in train_records]
            pack=ResidentRows(rows,'cuda',max_nodes=120,max_pins=146);register_resident(pack)
            lookup={tuple(r['member']):i for i,r in enumerate(rows)}
            monitor_rows=[rows[lookup[tuple(r['member'])]] for r in ReferenceLayouts('train_monitor500').records]
            feedback_rows=[rows[r['train_index']] for r in PLAN['feedback_panel']]
            assert all(list(row['member'])==reg['member'] for row,reg in zip(feedback_rows,PLAN['feedback_panel']))
            for row in feedback_rows:
                case=row['case'];case.reference_positions=public_hard_reference(case.solver_inputs[1],case.solver_inputs[5],case.solver_inputs[6])
            paths={'base':ROOT/PLAN['parent_checkpoint']}
            if not (RUN/'selected.json').exists():
                for arm in PLAN['training_arms']:
                    check();paths[arm]=train_branch(arm,parent,pack,monitor_rows,identity,check,beat)
            else:
                for arm in PLAN['training_arms']:paths[arm]=RUN/f'branches/{arm}/checkpoints/latest.pt'
            check();torch.set_num_threads(1)
            with torch.inference_mode(),threadpool_limits(limits=1):
                evaluator.warmup();predictor=evaluator.Predictor(feedback_rows[0]['case'].solver_inputs)
            if not (RUN/'selected.json').exists():
                scores=feedback(paths,feedback_rows,predictor,check,beat);selected=choose(scores,paths)
            else:
                scores=json.loads((RUN/'feedback/results.json').read_text())['scores']
                selected=json.loads((RUN/'selected.json').read_text())
            beat(stage='selection_frozen',selected_arm=selected['selected_arm'],coefficient_changed=selected['coefficient_changed'])
            terminals=terminal(selected,paths,predictor,check,beat)
            verify_parent();status='complete'
            save(RUN/'results.json',dict(status=status,selected=selected,train_feedback=scores,terminal=terminals,
                 elapsed_seconds=elapsed(),extra_outer_rounds=0,test_read=False,no_dev_controller_feedback=True))
            (RUN/'report.md').write_text('# E98首轮训练集求解反馈调权\n\n'+json.dumps(dict(selected=selected,train_feedback=scores,terminal=terminals),ensure_ascii=False,indent=2)+'\n\n训练集反馈决定接纳；验证集只作冻结后的终末评价，不回流。原基底未确认充分训练收敛；保留等更新控制支区分继续拟合与调权收益。测试封存。\n',encoding='utf-8')
        except Halt as exc:
            status=str(exc);save(RUN/'results.json',dict(status=status,elapsed_seconds=elapsed(),test_read=False,next_stage='none'))
        except BaseException as exc:
            status='error';save(RUN/'error.json',dict(error=repr(exc),traceback=traceback.format_exc(),test_read=False,no_automatic_retry=True));traceback.print_exc()
        finally:
            closed.set();worker.join(timeout=6.);signal.signal(signal.SIGINT,old_signal)
            beat(status=status,stage='exited')
            print('STOPPED',status,'elapsed',elapsed(),flush=True)

if __name__=='__main__':main()
