"""Public-solver supervision, frozen v5 train16k/dev500, no reference targets."""
import datetime,hashlib,json,multiprocessing as mp,os,time,uuid,zipfile
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
import numpy as np
from .latency import torch,ROOT,HERE,DeploymentModel,decode85,save,sha
from .mechanism import FULL
from .train_io.data import load_rows,collate,LEGACY

FOLDER=HERE/'runs/20261101_public_teacher'

def checkpoint(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    torch.save(value,tmp);os.replace(tmp,path)

def init_worker():
    torch.set_num_threads(1)
    from .repair85.canonical_native import warmup
    warmup()

def solve(job):
    index,inputs,draft=job
    from .repair85.solver import optimize
    from .repair85.backend.frozen.packing import prepare_problem
    from .repair85.backend.frozen.feasible import Objective
    from .repair85.backend.frozen.solver import hard_valid
    np.random.seed(index);torch.manual_seed(index);start=time.perf_counter()
    positions,stats=optimize(draft,inputs,**FULL)
    problem=prepare_problem(*inputs)
    if not hard_valid(positions,problem):raise RuntimeError('Teacher final geometry failed hard validation')
    objective=Objective(problem)
    return dict(positions=positions,usable=not stats['row_retained'],seconds=time.perf_counter()-start,
        solver_seconds=stats['seconds'],public_proxy=objective(positions),soft_violations=objective.violations(positions),
        fallback=bool(stats['row_retained']),shape_calls=sum(t.get('phase')=='shape' for t in stats['trace']))

@torch.inference_mode()
def main():
    started=time.perf_counter();FOLDER.mkdir(exist_ok=False)
    torch.manual_seed(0);torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    study=json.loads((HERE/'study.json').read_text(encoding='utf-8'));deadline=datetime.datetime.fromisoformat(study['deadline_utc'])
    plan=json.loads((HERE/'candidate1_plan.json').read_text(encoding='utf-8'))
    expires=min(deadline,datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(seconds=plan['teacher']['wall_cap_seconds']))
    halt=lambda:(HERE/'runs/STOP').exists() or datetime.datetime.now(datetime.timezone.utc)>=expires
    path=ROOT/plan['source_checkpoint'];assert sha(path)==plan['source_sha256']
    loaded=torch.load(path,map_location='cpu',weights_only=False)
    model=DeploymentModel(**loaded['model_config']).cuda().eval();model.load_state_dict(loaded['model'],strict=True)
    roles={'train':load_rows('train'),'dev_monitor500':load_rows('dev_monitor500')}
    assert len(roles['train'])==16000 and len(roles['dev_monitor500'])==500
    binding=dict(revision='E97.public_solver_teacher.v1',candidate_plan=plan,checkpoint_sha256=sha(path),FULL=FULL,
        members={k:[list(r['member']) for r in v] for k,v in roles.items()},source_sha256={p.relative_to(HERE).as_posix():sha(p) for p in HERE.rglob('*.py') if 'runs' not in p.parts and '__pycache__' not in p.parts},
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),expires_utc=expires.isoformat(),test_read=False)
    save(FOLDER/'config.json',binding)
    with zipfile.ZipFile(FOLDER/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in binding['source_sha256']:z.write(HERE/name,name)
        z.write(HERE/'candidate1_plan.json','candidate1_plan.json')
    buffers={role:{} for role in roles};manifests={role:dict(count=len(rows),parts=[],completed=0,usable=0,fallbacks=0,worker_seconds=0.) for role,rows in roles.items()}
    pending={};completed=0;last=time.perf_counter()
    def flush(role,force=False):
        manifest=manifests[role];buffer=buffers[role]
        for start in range(0,len(roles[role]),128):
            ids=list(range(start,min(start+128,len(roles[role]))))
            if all(i in buffer for i in ids):
                file=FOLDER/role/f'{start:05d}.pt';checkpoint(file,dict(indices=ids,rows=[buffer.pop(i) for i in ids]))
                manifest['parts'].append(dict(path=file.relative_to(FOLDER).as_posix(),indices=ids,sha256=sha(file)))
        if force and buffer:checkpoint(FOLDER/role/'partial.pt',dict(indices=sorted(buffer),rows=[buffer[i] for i in sorted(buffer)]))
        save(FOLDER/role/'manifest.json',manifest)
    def progress(status='running'):
        save(FOLDER/'progress.json',dict(status=status,completed=completed,total=16500,roles=manifests,pending=len(pending),elapsed_seconds=time.perf_counter()-started,formal_training_started=False,test_read=False))
    def collect(block=True):
        nonlocal completed,last
        if not pending:return
        done,_=wait(pending,timeout=1 if block else 0,return_when=FIRST_COMPLETED)
        for future in done:
            role,index,base,source,scale,frame=pending.pop(future);value=future.result();p=value.pop('positions')
            state=np.c_[(p[:,:2]+p[:,2:]/2)/scale,np.log(p[:,2]/p[:,3])].astype(np.float32)
            # Target metadata diagnoses the bounded action-space limitation.
            lo,hi=frame;outside=np.maximum(lo-p[:,:2],0)+np.maximum(p[:,:2]+p[:,2:]-hi,0)
            value.update(index=index,member=list(roles[role][index]['member']),base=base,source=source,target=torch.from_numpy(state),teacher_outside_L_frame_max=float(outside.max()/scale))
            buffers[role][index]=value;manifest=manifests[role];manifest['completed']+=1;manifest['usable']+=int(value['usable']);manifest['fallbacks']+=int(value['fallback']);manifest['worker_seconds']+=value['seconds'];completed+=1
            if completed%128==0:flush(role)
        if time.perf_counter()-last>10:
            progress();print('teacher',completed,'/16500', {k:(v['completed'],v['fallbacks']) for k,v in manifests.items()},flush=True);last=time.perf_counter()
    try:
        from .repair85.canonical_native import warmup
        warmup()
        with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=init_worker) as pool:
            for role,rows in roles.items():
                for start in range(0,len(rows),32):
                    if halt():break
                    selected=rows[start:start+32]
                    host=collate(selected,'cpu');g={k:host[k].cuda() for k in LEGACY}
                    base=model.placer(g);source=model.refiner(base,g)
                    physical=(decode85(source,g)*g['scale'][:,None,None]).cpu().double().numpy()
                    original=(decode85(base,g)*g['scale'][:,None,None]).cpu().double().numpy()
                    base_cpu=base.cpu();source_cpu=source.cpu()
                    for j,row in enumerate(selected):
                        while len(pending)>=16:
                            collect()
                            if halt():break
                        if halt():break
                        n=row['graph']['n'];inputs=row['case'].solver_inputs;draft=physical[j,:n].copy();c,s=np.asarray(inputs[5]),np.asarray(inputs[6]);locked=c[:,:2].any(1)
                        draft[locked,2:]=s[locked,2:];draft[c[:,1]!=0]=s[c[:,1]!=0]
                        initial=original[j,:n];frame=(initial[:,:2].min(0), (initial[:,:2]+initial[:,2:]).max(0))
                        index=start+j;future=pool.submit(solve,(index,inputs,draft))
                        pending[future]=(role,index,base_cpu[j,:n].clone(),source_cpu[j,:n].clone(),float(host['scale'][j]),frame)
                    collect(False)
                if halt():break
            while pending:collect()
        for role in roles:flush(role,True)
        finished=all(m['completed']==m['count'] for m in manifests.values())
        status='complete' if finished else ('user_stop' if (HERE/'runs/STOP').exists() else 'budget')
        progress(status);save(FOLDER/'results.json',dict(status=status,finished=finished,roles=manifests,elapsed_seconds=time.perf_counter()-started,formal_training_started=False,test_read=False))
        print('Teacher preparation',status,'elapsed',time.perf_counter()-started,flush=True)
    except BaseException as error:
        for role in roles:flush(role,True)
        progress('error');save(FOLDER/'error.json',dict(error=repr(error),no_automatic_retry=True,test_read=False));raise

if __name__=='__main__':main()
