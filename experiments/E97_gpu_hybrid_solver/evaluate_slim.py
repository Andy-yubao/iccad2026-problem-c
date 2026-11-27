"""Frozen two-arm dev500 single-request evaluation, no weight search/training."""
import collections,datetime,json,time,zipfile,subprocess
import numpy as np
from .mechanism import torch,ROOT,HERE,DeploymentModel,OpsOnly,Captured,padded,returned,decode85,optimize,SLIM,warmup,independent_hard_checks,public_hard_reference,save,sha
from datasets.expanded_reference import ExpandedReferenceLayouts
from scoring.evaluation import evaluate_case
from scoring.official import official_evaluator

FOLDER=HERE/'runs/20261101_slim3_dev500'

def summary(rows):
    q=[r['metrics']['cost'] for r in rows];n=[r['n'] for r in rows];wall=np.array([r['complete_seconds'] for r in rows])
    return dict(count=len(rows),legal=sum(r['metrics'].get('is_feasible',False) for r in rows),errors=sum('error' in r for r in rows),weighted_Q=official_evaluator().compute_total_score(q,n),mean_Q=float(np.mean(q)),max_Q=float(max(q)),mean_complete_seconds=float(wall.mean()),p95_complete_seconds=float(np.percentile(wall,95)),max_complete_seconds=float(wall.max()),cases_over_025s=int((wall>.25).sum()),fallbacks=sum(r.get('fallback',False) for r in rows))

@torch.inference_mode()
def main():
    begin=time.perf_counter();FOLDER.mkdir(exist_ok=False)
    torch.manual_seed(0);torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    study=json.loads((HERE/'study.json').read_text(encoding='utf-8'));deadline=datetime.datetime.fromisoformat(study['deadline_utc'])
    checkpoint=ROOT/'experiments/E85_cooperative_space_refinement/runs/20261030_P74_F81_cooperative_seed0/checkpoints/best.pt'
    loaded=torch.load(checkpoint,map_location='cpu',weights_only=False);model=DeploymentModel(**loaded['model_config']).cuda().eval();model.load_state_dict(loaded['model'],strict=True)
    data=ExpandedReferenceLayouts('dev_monitor500',train_size=16000);rows=[data[i] for i in range(len(data))]
    assert len(rows)==500
    for row in rows:
        case=row['case'];case.reference_positions=public_hard_reference(case.solver_inputs[1],case.solver_inputs[5],case.solver_inputs[6])
    warmup();g={k:v.cuda() for k,v in padded(rows[0]['case'].solver_inputs,False).items()}
    caps={'E85_slim3':Captured(model,g),'ops_only_slim3':Captured(OpsOnly(model).eval(),g)}
    binding=dict(checkpoint_sha256=sha(checkpoint),plan=json.loads((HERE/'plan.json').read_text(encoding='utf-8'))['stage2'],SLIM=SLIM,members=[list(r['member']) for r in rows],source_sha256={p.relative_to(HERE).as_posix():sha(p) for p in HERE.rglob('*.py') if 'runs' not in p.parts and '__pycache__' not in p.parts},cold_initialization_seconds=time.perf_counter()-begin,batch_size=1,threads=2,precision='FP32 no AMP/TF32',test_read=False)
    save(FOLDER/'config.json',binding)
    with zipfile.ZipFile(FOLDER/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in binding['source_sha256']:z.write(HERE/name,name)
        z.write(HERE/'plan.json','plan.json');z.write(HERE/'study.json','study_at_start.json')
    records={k:[] for k in caps}
    for i,row in enumerate(rows):
        if (HERE/'runs/STOP').exists() or datetime.datetime.now(datetime.timezone.utc)>=deadline:
            save(FOLDER/'results.json',dict(status='stopped',finished=False,counts={k:len(v) for k,v in records.items()},test_read=False));return
        arms=('ops_only_slim3','E85_slim3') if i%2==0 else ('E85_slim3','ops_only_slim3')
        for arm in arms:
            inputs=row['case'].solver_inputs;cap=caps[arm];torch.cuda.synchronize();start=time.perf_counter()
            try:
                host=padded(inputs,False)
                for k,v in host.items():cap.g[k].copy_(v)
                draft=returned(cap.replay(),cap.g,inputs,decode85);draft_seconds=time.perf_counter()-start
                final,trace=optimize(draft,inputs,**SLIM)
                hard=independent_hard_checks(final.tolist(),np.asarray(inputs[1]),np.asarray(inputs[5]),public_hard_reference(inputs[1],inputs[5],inputs[6]))
                if not hard['is_feasible'] or (final[:,:2]<0).any():raise RuntimeError('Final public hard check failed')
                elapsed=time.perf_counter()-start
                # Score computation is evaluator overhead, outside solve time.
                scored=evaluate_case(row['case'],final,elapsed)
                record=dict(index=i,member=list(row['member']),n=int(inputs[0]),complete_seconds=elapsed,draft_seconds=draft_seconds,repair_seconds=trace['seconds'],metrics=scored['metrics'],independent_hard_checks=scored['independent_hard_checks'],fallback=trace['row_retained'],trace=trace['trace'],positions=final.tolist(),draft_positions=draft.tolist())
            except Exception as error:
                record=dict(index=i,member=list(row['member']),n=int(inputs[0]),complete_seconds=time.perf_counter()-start,metrics=dict(cost=10.,is_feasible=False),error=repr(error))
                save(FOLDER/arm/f'{i:04d}.json',record)
                save(FOLDER/'error.json',dict(arm=arm,index=i,error=repr(error),no_automatic_retry=True,test_read=False));raise
            records[arm].append(record);save(FOLDER/arm/f'{i:04d}.json',record)
        if (i+1)%10==0:
            view={arm:summary(v) for arm,v in records.items()};save(FOLDER/'progress.json',dict(completed_members=i+1,arms=view,elapsed_seconds=time.perf_counter()-begin,test_read=False))
            print('completed',i+1,'/500',json.dumps(view),flush=True)
    result=dict(status='complete',finished=True,arms={arm:summary(v) for arm,v in records.items()},elapsed_seconds=time.perf_counter()-begin,source_sha256=binding['source_sha256'],test_read=False)
    save(FOLDER/'results.json',result);print(json.dumps(result['arms'],indent=2),flush=True)

if __name__=='__main__':main()
