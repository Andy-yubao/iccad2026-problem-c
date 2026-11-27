"""Round2 own-Loss best with original full singlethread repair and captured decode."""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[_key]='1'
import datetime,json,time,zipfile
from pathlib import Path
import numpy as np
from .round2_candidate1 import torch,ROOT,HERE,OUTPUT,source_identity,teacher_rows,seed
from .student import DeploymentModel
from .latency import Captured,sha
from .latency_fixed import padded,returned
from .round2_latency import restored
from threadpoolctl import threadpool_limits,threadpool_info
from .nn85.base_model import decode
from .repair85_one_thread.solver import optimize
from .repair85_one_thread.canonical_native import warmup
from scoring.checks import independent_hard_checks
from datasets.reference_layouts import public_hard_reference
from .train_io.io import save,exclusive
from .train_io.data import load_rows,collate,LEGACY
from .draft_metrics import module_overlap
from datasets.expanded_reference import ExpandedReferenceLayouts
from scoring.evaluation import evaluate_case
from scoring.official import official_evaluator
from visualization.layouts import plot_case,geometry_details

FOLDER=OUTPUT/'evaluation'
FULL=json.loads((HERE/'round2_control_plan.json').read_text(encoding='utf-8'))['full_repair']

def read(path):return json.loads(path.read_text(encoding='utf-8'))

def summary(records):
    q=[r['metrics']['cost'] for r in records];n=[r['n'] for r in records];wall=np.array([r['complete_seconds'] for r in records])
    return dict(count=len(records),legal=sum(r['metrics']['is_feasible'] for r in records),weighted_Q=official_evaluator().compute_total_score(q,n),mean_Q=float(np.mean(q)),max_Q=max(q),mean_complete_seconds=float(wall.mean()),p95_complete_seconds=float(np.percentile(wall,95)),max_complete_seconds=float(wall.max()),cases_over_025s=int((wall>.25).sum()),fallbacks=sum(r['fallback'] for r in records))

@torch.inference_mode()
def main():
    begin=time.perf_counter();seed();study=read(HERE/'round2_study.json');deadline=datetime.datetime.fromisoformat(study['deadline_utc'])
    halt=lambda:(HERE/'runs/STOP').exists() or (OUTPUT/'STOP').exists() or datetime.datetime.now(datetime.timezone.utc)>=deadline
    if halt():return
    with exclusive(HERE/'runs/round2_execution.lock'),exclusive(HERE/'runs/round2_candidate1_evaluation.lock'):
        best=OUTPUT/'checkpoints/best.pt';saved=torch.load(best,map_location='cpu',weights_only=False)
        assert saved['identity']['source_sha256']==source_identity()
        if saved['progress']['best_epoch']==0:
            plan=read(HERE/'round2_candidate1_plan.json');initial=torch.load(ROOT/plan['source_checkpoint'],map_location='cpu',weights_only=False)
            assert all(torch.equal(v,initial['model'][k]) for k,v in saved['model'].items());del initial
            source=HERE/'runs/20261102_round2_current_control/C3';control=read(source/'results.json');assert control['status']=='complete'
            raw_source=HERE/'runs/20261102_candidate3_clearance_seed0/evaluation';raw=read(raw_source/'summary.json')
            save(FOLDER/'summary.json',dict(terminal=control['terminal'],raw=raw['raw'],raw_count=2000,reused_initial=True,terminal_source=source.relative_to(HERE).as_posix(),raw_source=raw_source.relative_to(HERE).as_posix(),test_read=False))
            gallery=FOLDER/'gallery';gallery.mkdir(parents=True,exist_ok=True)
            (gallery/'index.html').write_text('<!doctype html><meta charset="utf-8"><p>Selected model exactly equals current C3 control; <a href="../../../20261102_round2_current_control/C3/gallery/index.html">same100 gallery reused</a></p>',encoding='utf-8')
            save(FOLDER/'results.json',dict(status='complete',finished=True,terminal=control['terminal'],raw_count=2000,gallery_count=100,reused_initial=True,reused_from=source.relative_to(HERE).as_posix(),test_read=False));return

        model=DeploymentModel(**saved['model_config']).cuda().eval();model.load_state_dict(saved['model'],strict=True)
        labels=teacher_rows('dev_monitor500');dataset=ExpandedReferenceLayouts('dev_monitor500',train_size=16000);rows=[dataset[i] for i in range(500)]
        for row in rows:
            case=row['case'];case.reference_positions=public_hard_reference(case.solver_inputs[1],case.solver_inputs[5],case.solver_inputs[6])
        warmup();g={k:v.cuda() for k,v in padded(rows[0]['case'].solver_inputs,False).items()};cap=Captured(lambda gg:decode(model(gg),gg)*gg['scale'][:,None,None],g)
        dependencies=[Path(__file__),HERE/'mechanism.py',HERE/'latency.py',HERE/'latency_fixed.py',HERE/'round2_latency.py',HERE/'round2_control_plan.json',HERE/'draft_metrics.py',ROOT/'visualization/layouts.py']+list((HERE/'repair85_one_thread').rglob('*.py'))+list((ROOT/'scoring').glob('*.py'))
        binding=dict(checkpoint_sha256=sha(best),source_sha256=source_identity(),evaluation_sources={p.relative_to(ROOT).as_posix():sha(p) for p in dependencies},student_class='E97.student.DeploymentModel',repair=FULL,batch_size=1,members=[list(r['member']) for r in rows],cold_initialization_seconds=time.perf_counter()-begin,test_read=False)
        FOLDER.mkdir(exist_ok=True);frozen=FOLDER/'frozen.json'
        if frozen.exists():
            old=read(frozen);assert all(old[k]==binding[k] for k in binding if k!='cold_initialization_seconds')
        else:
            save(frozen,binding)
            with zipfile.ZipFile(FOLDER/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
                for name in binding['evaluation_sources']:z.write(ROOT/name,name)
        records=[]
        for i,row in enumerate(rows):
            if halt():return
            path=FOLDER/'terminal'/f'{i:04d}.json'
            if path.exists():
                record=read(path);assert record['member']==list(row['member']);records.append(record);continue
            inputs=row['case'].solver_inputs;torch.cuda.synchronize();start=time.perf_counter()
            host=padded(inputs,False)
            for k,v in host.items():cap.g[k].copy_(v)
            draft=restored(cap.replay(),inputs);draft_seconds=time.perf_counter()-start
            final,trace=optimize(draft,inputs,**FULL)
            hard=independent_hard_checks(final.tolist(),np.asarray(inputs[1]),np.asarray(inputs[5]),public_hard_reference(inputs[1],inputs[5],inputs[6]))
            assert hard['is_feasible'] and bool((final[:,:2]>=0).all())
            elapsed=time.perf_counter()-start
            scored=evaluate_case(row['case'],final,elapsed)
            # Saved frozen-L geometry is display-only; the timed graph executed L.
            base=torch.zeros(1,120,3);base[0,:int(inputs[0])]=labels[i]['base']
            original=returned(base,host,inputs,decode)
            record=dict(index=i,member=list(row['member']),n=int(inputs[0]),complete_seconds=elapsed,draft_seconds=draft_seconds,repair_seconds=trace['seconds'],metrics=scored['metrics'],independent_hard_checks=scored['independent_hard_checks'],fallback=trace['row_retained'],trace=trace['trace'],positions=final.tolist(),draft_positions=draft.tolist(),L_positions=original.tolist())
            save(path,record);records.append(record)
            if (i+1)%10==0:
                save(FOLDER/'progress.json',dict(completed=i+1,summary=summary(records),elapsed_seconds=time.perf_counter()-begin,test_read=False));print('round2_candidate1 terminal',i+1,'/500',flush=True)
        terminal_summary=summary(records);save(FOLDER/'summary.json',dict(terminal=terminal_summary,selection='unique own-Loss best',elapsed_seconds=time.perf_counter()-begin,test_read=False))
        del cap;torch.cuda.empty_cache()
        # Complete raw development diagnostics reuse this same frozen best.
        raw_rows=load_rows('dev');raw_folder=FOLDER/'raw';raw_records=[]
        for start in range(0,2000,32):
            if halt():return
            path=raw_folder/f'{start:04d}.json';selected=raw_rows[start:start+32]
            if path.exists():
                chunk=read(path);assert [r['member'] for r in chunk]==[list(r['member']) for r in selected]
            else:
                host=collate(selected,'cpu');g={k:host[k].cuda() for k in LEGACY};base=model.placer(g);state=model.refiner(base,g)
                initial=(decode(base,g)*g['scale'][:,None,None]).cpu().double().numpy();corrected=(decode(state,g)*g['scale'][:,None,None]).cpu().double().numpy();chunk=[]
                for j,row in enumerate(selected):
                    n=row['graph']['n'];variants={};c,s=np.asarray(row['case'].solver_inputs[5]),np.asarray(row['case'].solver_inputs[6]);locked=c[:,:2].any(1)
                    for label,array in (('placer',initial[j,:n].copy()),('corrected',corrected[j,:n].copy())):
                        array[locked,2:]=s[locked,2:];array[c[:,1]!=0]=s[c[:,1]!=0]
                        variants[label]=dict(positions=array.tolist(),diagnostics=module_overlap(array))
                    chunk.append(dict(index=start+j,member=list(row['member']),n=n,**variants))
                save(path,chunk)
            raw_records.extend(chunk)
        raw_summary={stage:{k:float(np.mean([r[stage]['diagnostics'][k] for r in raw_records])) for k in raw_records[0][stage]['diagnostics']} for stage in ('placer','corrected')}
        save(FOLDER/'summary.json',dict(terminal=terminal_summary,raw_count=len(raw_records),raw=raw_summary,selection='unique own-Loss best',elapsed_seconds=time.perf_counter()-begin,test_read=False))
        gallery=FOLDER/'gallery';gallery.mkdir(exist_ok=True);manifest=[]
        stages=(('reference','Reference'),('placer','Frozen E74 draft'),('corrected','Small student F'),('final','Full E85 final / one thread'))
        for i,(row,record) in enumerate(zip(rows[:100],records[:100])):
            if halt():return
            c=np.asarray(row['case'].solver_inputs[5]).tolist();variants={key:dict(positions=p) for key,p in [('reference',row['positions'].tolist()),('placer',record['L_positions']),('corrected',record['draft_positions']),('final',record['positions'])]}
            for value in variants.values():value['soft_details']=geometry_details(value['positions'],c)
            image=gallery/f'{i:03d}.png'
            if not image.exists():plot_case(dict(case_id=f'dev500:{i}',panel='E97 round2_candidate1 / public-solver distilled student',constraints=c,variants=variants),image,stages=stages,show_metrics=False)
            manifest.append(dict(index=i,member=list(row['member']),image=image.name,sha256=sha(image),checkpoint_sha256=sha(best)))
        save(gallery/'manifest.json',dict(count=100,cases=manifest,score_controls=False,shared_IDs_and_scale=True,test_read=False))
        html='<!doctype html><meta charset="utf-8"><title>E97 round2_candidate1</title><style>body{font:16px system-ui}img{width:100%}</style><h1>Reference / E74 / student F / full E85</h1><p>同100例，同ID同尺度，无评分控件；reference只作展示。</p>'
        html+=''.join(f'<p>{x["index"]}</p><img loading="lazy" src="{x["image"]}">' for x in manifest)
        (gallery/'index.html').write_text(html,encoding='utf-8')
        save(FOLDER/'results.json',dict(status='complete',finished=True,terminal=terminal_summary,raw_count=2000,gallery_count=100,elapsed_seconds=time.perf_counter()-begin,test_read=False));print(json.dumps(terminal_summary,indent=2),flush=True)

if __name__=='__main__':
    try:
        with threadpool_limits(1):main()
    except BaseException as error:
        save(FOLDER/'error.json',dict(error=repr(error),no_automatic_retry=True,test_read=False));raise
