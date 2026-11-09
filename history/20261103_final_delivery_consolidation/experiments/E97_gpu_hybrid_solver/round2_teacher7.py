"""Distinct C5 full-solver teacher4000, actual WH; reuse completed dev finals."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
import json,time,datetime,zipfile
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from .latency import HERE,ROOT,IDS,Captured,sha
from .student import DeploymentModel
from .nn85.base_model import decode
from .latency_fixed import padded
from .train_io.data import load_rows
from .train_io.dataset_view import registry,REGISTRY
from .train_io.io import save,exclusive,checkpoint
from .repair85_one_thread.solver import optimize
from .repair85_one_thread.canonical_native import warmup
from .repair85_one_thread.backend.frozen.packing import prepare_problem
from .repair85_one_thread.backend.frozen.feasible import Objective
from scoring.checks import independent_hard_checks
from datasets.reference_layouts import public_hard_reference
OUTPUT=HERE/'runs/20261102_round2_C5_teacher4000_exactWH'
def read(p):return json.loads(p.read_text(encoding='utf-8'))

@torch.inference_mode()
def main():
    begin=time.perf_counter();OUTPUT.mkdir(exist_ok=False);plan=read(HERE/'round2_candidate7_plan.json');study=read(HERE/'round2_study.json');deadline=datetime.datetime.fromisoformat(study['deadline_utc'])
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(0);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    halt=lambda:(HERE/'runs/STOP').exists() or (OUTPUT/'STOP').exists() or datetime.datetime.now(datetime.timezone.utc)>=deadline or time.perf_counter()-begin>=1200
    with exclusive(HERE/'runs/round2_execution.lock'),exclusive(HERE/'runs/round2_teacher7.lock'):
        reg=registry();lookup={tuple(r['member']):i for i,r in enumerate(reg['sets']['train']['records'][:16000])};forced=set(IDS)|{lookup[tuple(r['member'])] for r in reg['sets']['train_monitor500']['records']};remaining=np.array(sorted(set(range(16000))-forced));chosen=sorted(forced|set(map(int,np.random.default_rng(730).choice(remaining,4000-len(forced),replace=False))))
        assert len(chosen)==4000 and set(IDS).issubset(chosen)
        rows={'train':load_rows('train',chosen),'dev_monitor500':load_rows('dev_monitor500')};indices={'train':chosen,'dev_monitor500':list(range(500))};full=read(HERE/'round2_control_plan.json')['full_repair'];path=ROOT/plan['source_checkpoint'];assert sha(path)==plan['source_sha256'];saved=torch.load(path,map_location='cpu',weights_only=False)
        model=DeploymentModel(**saved['model_config']).cuda().eval();model.load_state_dict(saved['model'],strict=True);warmup()
        g={k:v.cuda() for k,v in padded(rows['train'][0]['case'].solver_inputs,False).items()}
        def packet(gg):
            base=model.placer(gg);source=model.refiner(base,gg);physical=decode(source,gg)*gg['scale'][:,None,None]
            return torch.cat((base,source,physical),-1)
        cap=Captured(packet,g);config=dict(revision='E97.round2.C5_full_teacher4000_exactWH.v1',source_checkpoint=plan['source_checkpoint'],source_sha256=sha(path),plan_sha256=sha(HERE/'round2_candidate7_plan.json'),registry_sha256=sha(REGISTRY),indices=indices,members={k:[list(r['member']) for r in rs] for k,rs in rows.items()},full_repair=full,dev_final_source='runs/20261102_round2_candidate5_countweighted_seed0/evaluation',dev_solver_calls=0,cpu_threads=1,workers=1,deadline_utc=study['deadline_utc'],source_bindings={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),HERE/'student.py',HERE/'round2_candidate7_plan.json',HERE/'round2_control_plan.json']+list((HERE/'nn85').glob('*.py'))+list((HERE/'repair85_one_thread').rglob('*.py'))},test_read=False)
        save(OUTPUT/'config.json',config)
        with zipfile.ZipFile(OUTPUT/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
            for name in config['source_bindings']:z.write(ROOT/name,name)
        manifests={role:dict(count=len(rs),completed=0,parts=[],usable=0,fallbacks=0,teacher_aspect_outside5=0,max_nominal_area_relative_difference=0.) for role,rs in rows.items()}
        for role,rs in rows.items():
            buffer=[];buffer_ids=[]
            for local,(index,row) in enumerate(zip(indices[role],rs)):
                if halt():
                    if buffer:checkpoint(OUTPUT/role/'partial.pt',dict(indices=buffer_ids,rows=buffer))
                    save(OUTPUT/'results.json',dict(status='budget_or_stop',finished=False,roles=manifests,elapsed_seconds=time.perf_counter()-begin,test_read=False));return
                inputs=row['case'].solver_inputs;host=padded(inputs,False)
                for k,v in host.items():cap.g[k].copy_(v)
                values=cap.replay()[0,:int(inputs[0])].cpu();base=values[:,:3].clone();source=values[:,3:6].clone();draft=values[:,6:].double().numpy().copy();c,s=np.asarray(inputs[5]),np.asarray(inputs[6]);locked=c[:,:2].any(1);draft[locked,2:]=s[locked,2:];draft[c[:,1]!=0]=s[c[:,1]!=0]
                if role=='train':
                    final,stats=optimize(draft,inputs,**full);fallback=bool(stats['row_retained']);solver_seconds=stats['seconds']
                else:
                    record=read(HERE/config['dev_final_source']/'terminal'/f'{index:04d}.json');assert record['member']==list(row['member']);final=np.asarray(record['positions']);fallback=bool(record['fallback']);solver_seconds=0.
                hard=independent_hard_checks(final.tolist(),np.asarray(inputs[1]),c,public_hard_reference(inputs[1],inputs[5],inputs[6]));assert hard['is_feasible'] and (final[:,:2]>=0).all() and np.isfinite(final).all()
                scale=float(host['scale'][0]);state=np.c_[(final[:,:2]+final[:,2:]/2)/scale,np.log(final[:,2]/final[:,3])].astype(np.float32);wh=(final[:,2:]/scale).astype(np.float32);area=row['graph']['area'].numpy();relative=float(np.max(np.abs(np.prod(wh,1)-area)/np.maximum(area,1e-12)));p=prepare_problem(*inputs);obj=Objective(p)
                label=dict(index=index,member=list(row['member']),base=base,source=source,target=torch.from_numpy(state),teacher_wh=torch.from_numpy(wh),positions=torch.from_numpy(final.copy()),usable=not fallback,fallback=fallback,solver_seconds=solver_seconds,public_proxy=obj(final),soft_violations=obj.violations(final),teacher_aspect_outside5=int((np.abs(state[:,2])>5).sum()),nominal_area_relative_difference=relative)
                assert all(bool(torch.isfinite(label[k]).all()) for k in ('base','source','target','teacher_wh','positions'))
                buffer.append(label);buffer_ids.append(index);m=manifests[role];m['completed']+=1;m['usable']+=int(not fallback);m['fallbacks']+=int(fallback);m['teacher_aspect_outside5']+=label['teacher_aspect_outside5'];m['max_nominal_area_relative_difference']=max(m['max_nominal_area_relative_difference'],relative)
                if len(buffer)==128 or local+1==len(rs):
                    file=OUTPUT/role/f'{local+1-len(buffer):05d}.pt';checkpoint(file,dict(indices=buffer_ids,rows=buffer));m['parts'].append(dict(path=file.relative_to(OUTPUT).as_posix(),indices=buffer_ids,sha256=sha(file)));buffer=[];buffer_ids=[];save(OUTPUT/role/'manifest.json',m)
                if (local+1)%50==0:
                    save(OUTPUT/'progress.json',dict(status='running',role=role,completed=local+1,total=len(rs),roles=manifests,elapsed_seconds=time.perf_counter()-begin,test_read=False));print('C5 exact teacher',role,local+1,'/',len(rs),'seconds',round(time.perf_counter()-begin,1),flush=True)
        save(OUTPUT/'results.json',dict(status='complete',finished=True,roles=manifests,dev_solver_calls=0,elapsed_seconds=time.perf_counter()-begin,test_read=False));print('C5 exact teacher complete',round(time.perf_counter()-begin,1),flush=True)
if __name__=='__main__':
    try:
        with threadpool_limits(1):main()
    except BaseException as error:save(OUTPUT/'error.json',dict(error=repr(error),no_automatic_retry=True,test_read=False));raise
