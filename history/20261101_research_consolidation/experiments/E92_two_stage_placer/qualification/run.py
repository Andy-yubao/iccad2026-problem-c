"""Frozen F-best qualification only; resumable raw2000, one flash500 and gate."""
import argparse,json,multiprocessing as mp,os,signal,time,zipfile
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor,wait,FIRST_COMPLETED
import numpy as np
import torch
from ..io import HERE,ROOT,save,sha,exclusive
from ..data import load_rows,register_resident
from ..resident import ResidentRows
from ..model import GeometryModel
from ..batches import inputs_only,decode
from ..loss import per_case
from ..train import monitor as training_monitor
from ..dataset_view import ReferenceLayouts
from .common import OUTPUT,REGISTRATION,registration,ready,freeze,auxiliary_seconds,charge_auxiliary,seed_all
from .diagnostics import geometry_checks,distribution
from .metrics import module_overlap,union_area
from .worker import STEADY,summarize,initialize,evaluate

class EvaluationStop(Exception):pass

def diagnostic(payload):
    index,row,predicted,losses=payload
    p,checks,values=geometry_checks(predicted,row['case'].solver_inputs)
    overlap=module_overlap(p);bbox=float(np.prod((p[:,:2]+p[:,2:]).max(0)-p[:,:2].min(0)))
    return {'index':index,'member':list(row['member']),'n':len(p),'positions':p.tolist(),
            'losses':losses,'raw_hard_legal':bool(checks['is_feasible']),**values,**overlap,
            'empty_fraction_of_bbox':max(0.,1.-union_area(p)/bbox)}

def raw_dev(state,identity,halt,beat):
    target=OUTPUT/'full_dev.json'
    if target.exists():
        if json.loads(target.read_text(encoding='utf-8'))['checkpoint_sha256']!=identity['checkpoint_sha256']:
            raise ValueError('Raw result belongs to a different model')
        return
    seed_all(0,4);model=GeometryModel(**state['model_config']).cuda().eval();model.load_state_dict(state['model'])
    rows=load_rows('dev');lookup={tuple(r['member']):i for i,r in enumerate(rows)}
    panel=[rows[lookup[tuple(x['member'])]] for x in ReferenceLayouts('dev_monitor500').records]
    panelpack=ResidentRows(panel,'cuda',max_nodes=120);register_resident(panelpack)
    monitor_path=OUTPUT/'dev500_loss.json'
    if monitor_path.exists():
        saved=json.loads(monitor_path.read_text(encoding='utf-8'))
        if saved['checkpoint_sha256']!=identity['checkpoint_sha256']:raise ValueError('Monitor belongs to another model')
    else:
        beat('dev500_F_loss')
        monitor=training_monitor(model,panel,'F','cuda',halt)
        if monitor is None:raise EvaluationStop()
        selected=identity['selected_dev500_F_loss']
        if abs(monitor['total']-selected)>1e-5:
            raise ValueError('Frozen selected monitor was not reproduced within1e-5')
        save(monitor_path,{'checkpoint_sha256':identity['checkpoint_sha256'],'losses':monitor,
            'selected_training_monitor':selected,'test_read':False})
    pack=ResidentRows(rows,'cuda',max_nodes=120);register_resident(pack)
    chunks=OUTPUT/'raw_chunks';chunks.mkdir(exist_ok=True);records=[]
    with ThreadPoolExecutor(max_workers=4) as pool,torch.no_grad():
        for start in range(0,len(rows),32):
            if halt():raise EvaluationStop()
            path=chunks/f'{start:04d}.json'
            if path.exists():
                value=json.loads(path.read_text(encoding='utf-8'))
                if value['checkpoint_sha256']!=identity['checkpoint_sha256']:raise ValueError('Raw chunk model drift')
                block=value['records']
            else:
                selected_rows=rows[start:start+32];g=pack.batch(range(start,min(start+32,len(rows))))
                output=model(inputs_only(g));losses=per_case(output,g,'F')
                if not bool(torch.isfinite(output).all() & torch.isfinite(torch.stack(list(losses.values()))).all() & (losses['total'].amax()<1000)):
                    raise FloatingPointError('Full-dev numerical catastrophe; do not continue qualification')
                predicted=(decode(output,g)*g['scale'][:,None,None]).cpu().double().numpy()
                keys=list(losses);numbers=torch.stack([losses[k] for k in keys],-1).cpu().tolist()
                work=[(start+i,row,predicted[i,:row['graph']['n']],dict(zip(keys,numbers[i]))) for i,row in enumerate(selected_rows)]
                block=list(pool.map(diagnostic,work))
                save(path,{'checkpoint_sha256':identity['checkpoint_sha256'],'records':block})
            if [r['index'] for r in block]!=list(range(start,min(start+32,len(rows)))):
                raise ValueError('Invalid raw chunk cursor')
            if any(r['member']!=list(rows[r['index']]['member']) for r in block):raise ValueError('Raw member drift')
            records.extend(block);beat('raw2000',completed=len(records),count=len(rows))
    losses={key:float(np.mean([r['losses'][key] for r in records])) for key in records[0]['losses']}
    fields=[key for key,v in records[0].items() if isinstance(v,(int,float,bool)) and key not in ('index','n')]
    save(target,{'checkpoint_sha256':identity['checkpoint_sha256'],'count':len(records),'losses':losses,
                 'draft_diagnostics':distribution(records,fields),'raw_hard_legal':sum(r['raw_hard_legal'] for r in records),
                 'records':records,'postprocessing_calls':0,'test_read':False})
    del model
    torch.cuda.empty_cache()

def validate_terminal(records,metadata):
    indices=[r['index'] for r in records]
    if len(set(indices))!=len(indices) or any(not 0<=i<500 for i in indices):raise ValueError('Terminal index drift')
    if any(r['member']!=list(metadata[r['index']]['member']) for r in records):raise ValueError('Terminal member drift')

def terminal(identity,halt,beat):
    folder=OUTPUT/'terminal';folder.mkdir(exist_ok=True)
    metadata=ReferenceLayouts('dev_monitor500').records
    config={**identity,'steady':STEADY,'timing':'same E90 CPU feature+inference+decode+flash; initialization/JIT/scoring separate'}
    if (folder/'config.json').exists() and json.loads((folder/'config.json').read_text(encoding='utf-8'))!=config:
        raise ValueError('Frozen terminal configuration changed')
    save(folder/'config.json',config)
    target=folder/'results.json'
    if target.exists():
        value=json.loads(target.read_text(encoding='utf-8'));validate_terminal(value['records'],metadata)
        if len(value['records'])!=500 or value['checkpoint_sha256']!=identity['checkpoint_sha256']:
            raise ValueError('Existing terminal is incomplete or changed')
        if not (folder/'summary.json').exists():
            save(folder/'summary.json',{'checkpoint_sha256':identity['checkpoint_sha256'],**value['summary'],
                'workers':4,'threads_per_worker':2,'neutral_runtime_development_only':True,'test_read':False})
        return
    partial=folder/'partial.json'
    records=json.loads(partial.read_text(encoding='utf-8')) if partial.exists() else []
    validate_terminal(records,metadata);done={r['index'] for r in records}
    todo=iter(i for i in range(500) if i not in done);pending={}
    previous=os.environ.get('E92_EVAL_THREADS');os.environ['E92_EVAL_THREADS']='2'
    try:
        with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=initialize,
                                 initargs=(str(OUTPUT/'candidate_model.pt'),)) as pool:
            def submit():
                if halt():return
                index=next(todo,None)
                if index is not None:pending[pool.submit(evaluate,index)]=index
            for _ in range(4):submit()
            while pending:
                finished,_=wait(pending,timeout=1.,return_when=FIRST_COMPLETED)
                for future in finished:
                    pending.pop(future);records.append(future.result());save(partial,records);submit()
                beat('flash500',completed=len(records),count=500)
    finally:
        if previous is None:os.environ.pop('E92_EVAL_THREADS',None)
        else:os.environ['E92_EVAL_THREADS']=previous
    if len(records)!=500:raise EvaluationStop()
    records.sort(key=lambda r:r['index']);summary=summarize(records)
    save(target,{'checkpoint_sha256':identity['checkpoint_sha256'],'summary':summary,'records':records,'test_read':False})
    save(folder/'summary.json',{'checkpoint_sha256':identity['checkpoint_sha256'],**summary,
                              'workers':4,'threads_per_worker':2,'neutral_runtime_development_only':True,'test_read':False})
    if partial.exists():partial.unlink()

def gate(values,limits):
    return {'dev500_F_loss':values['dev500_F_loss']<=limits['dev500_F_loss_lte'],
            'weighted_Q':values['weighted_Q']<=limits['weighted_Q_lte'],
            'hard_legal':values['hard_legal']==limits['hard_legal_count'],
            'max_Q':values['max_Q']<=limits['max_Q_lte'],
            'module_pass10_pooled':values['module_pass10_pooled']>=limits['module_pass10_pooled_gte'],
            'worst_module_coverage_p95':values['worst_module_coverage_p95']<=limits['worst_module_coverage_p95_lte'],
            'numerical_stability':values['numerical_stability_pass'] is True,
            'F_checkpoint':values['F_phase_checkpoint_exists'] is True}

def assess(identity):
    reg=registration();data=json.loads((OUTPUT/'terminal/results.json').read_text(encoding='utf-8'))
    baseline=reg['baseline'];source=next(name for name in baseline['source_files'] if name.replace('\\','/').endswith('terminal/results.json'))
    old=json.loads((ROOT/source).read_text(encoding='utf-8'))['records'];records=data['records']
    if any(a['index']!=b['index'] or a['member']!=b['member'] or a['n']!=b['n'] for a,b in zip(records,old)) or len(records)!=500 or len(old)!=500:
        raise ValueError('Paired E90 comparison is not on identical members')
    def overlap(row):
        if 'draft_positions' not in row:raise ValueError('A failed draft cannot disappear from coverage aggregation')
        return module_overlap(np.asarray(row['draft_positions'],dtype=float)[:row['n']])
    with ThreadPoolExecutor(max_workers=4) as pool:geometry=list(pool.map(overlap,records))
    total_n=sum(r['n'] for r in records)
    values={**data['summary'],'dev500_F_loss':json.loads((OUTPUT/'dev500_loss.json').read_text(encoding='utf-8'))['losses']['total'],
            'module_pass10_pooled':1.-sum(r['N10'] for r in geometry)/total_n,
            'worst_module_coverage_p95':float(np.percentile([r['max_covered_fraction'] for r in geometry],95)),
            'numerical_stability_pass':data['summary']['errors']==0,'F_phase_checkpoint_exists':True}
    checks=gate(values,reg['qualification']['gates'])
    n=np.asarray([r['n'] for r in records]);weight=np.exp((n-n.max())/12)
    delta=np.asarray([a['metrics']['cost']-b['metrics']['cost'] for a,b in zip(records,old)])
    workers=sorted({r['member'][0].split('/')[0] for r in records});lookup={name:i for i,name in enumerate(workers)}
    groups=np.asarray([lookup[r['member'][0].split('/')[0]] for r in records])
    numerator=np.bincount(groups,weights=delta*weight,minlength=len(workers));denominator=np.bincount(groups,weights=weight,minlength=len(workers))
    sampled=np.random.default_rng(20261031).integers(0,len(workers),(5000,len(workers)))
    boot=numerator[sampled].sum(1)/denominator[sampled].sum(1)
    result={'status':'passed' if all(checks.values()) else 'failed','checks':checks,'values':values,
            'checkpoint_sha256':identity['checkpoint_sha256'],'registration_sha256':sha(REGISTRATION),
            'weighted_Q_delta':float(np.average(delta,weights=weight)),
            'worker_paired_95pct_interval':np.quantile(boot,[.025,.975]).tolist(),
            'interpretation':reg['qualification']['interpretation'],'test_read':False}
    save(OUTPUT/'geometry500.json',{'records':[dict(index=i,member=records[i]['member'],**g) for i,g in enumerate(geometry)],'test_read':False})
    save(OUTPUT/'assessment.json',result)
    study=json.loads((HERE/'study.json').read_text(encoding='utf-8'))
    study.update(status='layout_qualified' if result['status']=='passed' else 'terminated_layout_gate_failed',
                 qualification_result=(OUTPUT/'assessment.json').relative_to(ROOT).as_posix(),
                 next_action='Freeze qualified layout; register independent reusable corrector within remaining conditional budget' if result['status']=='passed' else 'End task; delete heartbeat; no corrector research')
    save(HERE/'study.json',study)
    deployment=json.loads((OUTPUT/'deployment.json').read_text(encoding='utf-8'))
    deployment['approved_for_corrector_training']=result['status']=='passed'
    save(OUTPUT/'deployment.json',deployment)
    return result

def run(stage='all'):
    ok,reason=ready()
    if not ok:raise RuntimeError(reason)
    begin=time.perf_counter();prior_aux=auxiliary_seconds();limit=7200.
    def halt():return (OUTPUT/'STOP').exists() or (HERE/'runs/STOP').exists() or prior_aux+time.perf_counter()-begin>=limit
    last=[0.]
    def beat(activity,**fields):
        now=time.perf_counter()-begin
        if now-last[0]>=5. or activity!=getattr(beat,'activity',None):
            save(OUTPUT/'heartbeat.json',{'activity':activity,**fields,'elapsed_this_invocation':now,
                 'auxiliary_remaining_seconds':max(0.,limit-prior_aux-now),'test_read':False})
            beat.activity=activity;last[0]=now
    try:
        with exclusive(HERE/'runs/training.lock'),exclusive(OUTPUT.with_suffix('.lock')):
            from ..train import RUN
            if not (RUN/'checkpoints/F_best.pt').exists():
                registration()
                reason='2h/resource endpoint reached without a scheduled F checkpoint; do not force a phase switch or select P by Q'
                save(OUTPUT/'assessment.json',{'status':'failed','checks':{'F_checkpoint':False},'reason':reason,
                     'postprocessing_calls':0,'test_read':False})
                study=json.loads((HERE/'study.json').read_text(encoding='utf-8'))
                study.update(status='terminated_layout_gate_failed',next_action='End task; delete heartbeat; no corrector research',qualification_failure=reason)
                save(HERE/'study.json',study)
                return
            state,identity=freeze()
            snapshot_path=OUTPUT/'source_snapshot.zip'
            if not snapshot_path.exists():
                with zipfile.ZipFile(snapshot_path,'w',zipfile.ZIP_DEFLATED) as archive:
                    for name in identity['sources']:archive.write(ROOT/name,name)
            if halt():raise EvaluationStop()
            if stage in ('all','raw'):raw_dev(state,identity,halt,beat)
            del state;torch.cuda.empty_cache()
            if stage in ('all','terminal'):terminal(identity,halt,beat)
            if stage in ('all','render'):
                if halt():raise EvaluationStop()
                from .render import run as render
                beat('same100_gallery');render(OUTPUT,halt)
            if stage in ('all','assess'):
                if halt():raise EvaluationStop()
                result=assess(identity);print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
            save(OUTPUT/'status.json',{'status':'completed','stage':stage,'test_read':False})
    except EvaluationStop:
        save(OUTPUT/'status.json',{'status':'stopped','stage':stage,'test_read':False})
    except BaseException as error:
        OUTPUT.mkdir(parents=True,exist_ok=True)
        save(OUTPUT/'status.json',{'status':'error','stage':stage,'error':repr(error),'test_read':False})
        raise
    finally:charge_auxiliary(time.perf_counter()-begin,'frozen qualification '+stage)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=('all','raw','terminal','render','assess'),default='all')
    p.add_argument('--check-ready',action='store_true');args=p.parse_args()
    if args.check_ready:
        registration();print(json.dumps(dict(zip(('ready','reason'),ready())),ensure_ascii=False));return
    OUTPUT.mkdir(parents=True,exist_ok=True)
    old=signal.signal(signal.SIGINT,lambda *_:(OUTPUT/'STOP').touch())
    try:run(args.stage)
    finally:signal.signal(signal.SIGINT,old)

if __name__=='__main__':main()
