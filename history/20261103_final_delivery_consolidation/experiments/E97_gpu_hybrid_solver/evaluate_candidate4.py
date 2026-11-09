"""Unique own-Loss student best, unchanged slim3, actual single-case service."""
import argparse,datetime,json,time,zipfile
from pathlib import Path
import numpy as np
from .candidate4 import torch,ROOT,HERE,OUTPUT,source_identity,teacher_rows,seed
from .relation_model import DeploymentModel,relations
from .latency import Captured,sha
from .latency_fixed import padded,returned
from .nn85.base_model import decode
from .solver_relations import optimize
from .mechanism import SLIM,warmup,independent_hard_checks,public_hard_reference
from .train_io.io import save,exclusive
from .train_io.data import load_rows,collate,LEGACY
from .draft_metrics import module_overlap
from datasets.expanded_reference import ExpandedReferenceLayouts
from scoring.evaluation import evaluate_case
from scoring.official import official_evaluator
from visualization.layouts import plot_case,geometry_details

FOLDER=OUTPUT/'evaluation'

def read(path):return json.loads(path.read_text(encoding='utf-8'))

def summary(records):
    q=[r['metrics']['cost'] for r in records];n=[r['n'] for r in records];wall=np.array([r['complete_seconds'] for r in records])
    return dict(count=len(records),legal=sum(r['metrics']['is_feasible'] for r in records),weighted_Q=official_evaluator().compute_total_score(q,n),mean_Q=float(np.mean(q)),max_Q=max(q),mean_complete_seconds=float(wall.mean()),p95_complete_seconds=float(np.percentile(wall,95)),max_complete_seconds=float(wall.max()),cases_over_025s=int((wall>.25).sum()),fallbacks=sum(r['fallback'] for r in records))

@torch.inference_mode()
def main(control=False):
    global FOLDER
    FOLDER=OUTPUT/('control' if control else 'evaluation')
    begin=time.perf_counter();seed();study=read(HERE/'study.json');deadline=datetime.datetime.fromisoformat(study['deadline_utc'])
    halt=lambda:(HERE/'runs/STOP').exists() or (OUTPUT/'STOP').exists() or datetime.datetime.now(datetime.timezone.utc)>=deadline
    if halt():return
    with exclusive(HERE/'runs/candidate4_evaluation.lock'):
        best=OUTPUT/('checkpoints/untrained.pt' if control else 'checkpoints/best.pt');saved=torch.load(best,map_location='cpu',weights_only=False)
        assert saved['identity']['source_sha256']==source_identity()
        if control and (OUTPUT/'evaluation/results.json').exists():
            selected=torch.load(OUTPUT/'checkpoints/best.pt',map_location='cpu',weights_only=False)
            if all(torch.equal(v,selected['model'][k]) for k,v in saved['model'].items()):
                result=read(OUTPUT/'evaluation/results.json');result.update(control='identical initial and selected weights; reused completed identical solve, no repeat',reused_from='evaluation/results.json')
                save(FOLDER/'results.json',result);return
        model=DeploymentModel(**saved['model_config']).cuda().eval();model.load_state_dict(saved['model'],strict=True)
        labels=teacher_rows('dev_monitor500');dataset=ExpandedReferenceLayouts('dev_monitor500',train_size=16000);rows=[dataset[i] for i in range(500)]
        for row in rows:
            case=row['case'];case.reference_positions=public_hard_reference(case.solver_inputs[1],case.solver_inputs[5],case.solver_inputs[6])
        warmup();g={k:v.cuda() for k,v in padded(rows[0]['case'].solver_inputs,False).items()};cap=Captured(model,g)
        dependencies=[Path(__file__),HERE/'mechanism.py',HERE/'latency.py',HERE/'latency_fixed.py',HERE/'draft_metrics.py',ROOT/'visualization/layouts.py']+list((HERE/'repair85').rglob('*.py'))+list((ROOT/'scoring').glob('*.py'))
        binding=dict(checkpoint_sha256=sha(best),source_sha256=source_identity(),evaluation_sources={p.relative_to(ROOT).as_posix():sha(p) for p in dependencies},student_class='E97.relation_model.DeploymentModel',control=control,fixed_relation_graph=True,repair=SLIM,batch_size=1,members=[list(r['member']) for r in rows],cold_initialization_seconds=time.perf_counter()-begin,test_read=False)
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
            draft=returned(cap.replay(),cap.g,inputs,decode);draft_seconds=time.perf_counter()-start
            relation_edges=relations(cap.output[0,:int(inputs[0])].cpu().double().numpy(),int(inputs[0]))
            final,trace=optimize(draft,inputs,**SLIM,relations_override=relation_edges)
            hard=independent_hard_checks(final.tolist(),np.asarray(inputs[1]),np.asarray(inputs[5]),public_hard_reference(inputs[1],inputs[5],inputs[6]))
            assert hard['is_feasible'] and bool((final[:,:2]>=0).all())
            elapsed=time.perf_counter()-start
            scored=evaluate_case(row['case'],final,elapsed)
            # Saved frozen-L geometry is display-only; the timed graph executed L.
            base=torch.zeros(1,120,3);base[0,:int(inputs[0])]=labels[i]['base']
            original=returned(base,host,inputs,decode)
            record=dict(relation_edges=relation_edges.tolist(),index=i,member=list(row['member']),n=int(inputs[0]),complete_seconds=elapsed,draft_seconds=draft_seconds,repair_seconds=trace['seconds'],metrics=scored['metrics'],independent_hard_checks=scored['independent_hard_checks'],fallback=trace['row_retained'],trace=trace['trace'],positions=final.tolist(),draft_positions=draft.tolist(),L_positions=original.tolist())
            save(path,record);records.append(record)
            if (i+1)%10==0:
                save(FOLDER/'progress.json',dict(completed=i+1,summary=summary(records),elapsed_seconds=time.perf_counter()-begin,test_read=False));print('candidate4 terminal',i+1,'/500',flush=True)
        terminal_summary=summary(records);save(FOLDER/'summary.json',dict(terminal=terminal_summary,selection='unique own-Loss best',elapsed_seconds=time.perf_counter()-begin,test_read=False))
        if control:
            save(FOLDER/'results.json',dict(status='complete',finished=True,terminal=terminal_summary,control='same frozen C1 geometry, untrained native-prior relation head; same fixed-input3-step solve',elapsed_seconds=time.perf_counter()-begin,test_read=False));print(json.dumps(terminal_summary,indent=2),flush=True);return
        del cap;torch.cuda.empty_cache()
        # The physical draft is identically frozen C1; reuse its complete2000 raw
        # geometry instead of rerunning the already-completed identical inference.
        raw_source=HERE/'runs/20261102_candidate1_distill_seed0/evaluation'
        raw_summary=read(raw_source/'summary.json')['raw']
        save(FOLDER/'summary.json',dict(terminal=terminal_summary,raw_count=2000,raw=raw_summary,raw_evidence_reused_from=raw_source.relative_to(HERE).as_posix(),reason='physical draft identical frozen C1; only relation outputs learned',selection='unique own-Loss best',elapsed_seconds=time.perf_counter()-begin,test_read=False))
        gallery=FOLDER/'gallery';gallery.mkdir(exist_ok=True);manifest=[]
        stages=(('reference','Reference'),('placer','Frozen E74 draft'),('corrected','Frozen C1 geometry'),('final','Slim3 final'))
        for i,(row,record) in enumerate(zip(rows[:100],records[:100])):
            if halt():return
            c=np.asarray(row['case'].solver_inputs[5]).tolist();variants={key:dict(positions=p) for key,p in [('reference',row['positions'].tolist()),('placer',record['L_positions']),('corrected',record['draft_positions']),('final',record['positions'])]}
            for value in variants.values():value['soft_details']=geometry_details(value['positions'],c)
            image=gallery/f'{i:03d}.png'
            if not image.exists():plot_case(dict(case_id=f'dev500:{i}',panel='E97 C4 / learned relations, frozen C1 geometry',constraints=c,variants=variants),image,stages=stages,show_metrics=False)
            manifest.append(dict(index=i,member=list(row['member']),image=image.name,sha256=sha(image),checkpoint_sha256=sha(best)))
        save(gallery/'manifest.json',dict(count=100,cases=manifest,score_controls=False,shared_IDs_and_scale=True,test_read=False))
        html='<!doctype html><meta charset="utf-8"><title>E97 candidate4</title><style>body{font:16px system-ui}img{width:100%}</style><h1>Reference / E74 / student F / slim3</h1><p>同100例，同ID同尺度，无评分控件；reference只作展示。</p>'
        html+=''.join(f'<p>{x["index"]}</p><img loading="lazy" src="{x["image"]}">' for x in manifest)
        (gallery/'index.html').write_text(html,encoding='utf-8')
        save(FOLDER/'results.json',dict(status='complete',finished=True,terminal=terminal_summary,raw_count=2000,gallery_count=100,raw_evidence_reused_from=raw_source.relative_to(HERE).as_posix(),elapsed_seconds=time.perf_counter()-begin,test_read=False));print(json.dumps(terminal_summary,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--control',action='store_true');args=parser.parse_args()
    try:main(args.control)
    except BaseException as error:
        save(FOLDER/'error.json',dict(error=repr(error),no_automatic_retry=True,test_read=False));raise
