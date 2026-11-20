"""Frozen best29 on original dev500, with verified historical parent reuse."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import os
import signal
import threading
import time
import traceback
import zipfile
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from datasets.reference_layouts import ReferenceLayouts,public_hard_reference
from scoring.evaluation import evaluate_case
from scoring.checks import independent_hard_checks
from ..common import ROOT,OWNER as EXPERIMENT,read,save,sha,setup,q_summary
from ..finetune_learned_v1.data import load_rows
from ..postprocessor.solver import optimize
from ..postprocessor.repair.canonical_native import warmup
from .predictor import Predictor,model_hash,input_hash
from .metrics import module_overlap,union_area

OWNER=Path(__file__).resolve().parent
TRAIN=EXPERIMENT/'runs/20261104_learned_v1_lr1e3_seed0'
OUT=TRAIN/'evaluation_depth_dev500'

def sources():
    files=list(OWNER.glob('*.py'))+[OWNER/'plan.json',OWNER/'copy_provenance.json']
    files+=list((EXPERIMENT/'postprocessor').rglob('*.py'))+[EXPERIMENT/'postprocessor/budget.json']
    files+=[EXPERIMENT/n for n in ('__init__.py','common.py','features.py','batches.py','ports.py','cache.py',
        'sources/e98_generalization/model.py','finetune_learned_v1/data.py')]
    files+=list((ROOT/'datasets').glob('*.py'))+list((ROOT/'scoring').glob('*.py'))
    files+=[ROOT/n for n in ('datasets/expanded_reference/__init__.py','protocol/paths.json',
        'protocol/reference_workset_v5.json','protocol/reference_workset_v6.json','protocol/split_v1.json',
        'visualization/layouts.py','third_party/FloorSet/iccad2026contest/iccad2026_evaluate.py',
        'third_party/FloorSet/cost.py','third_party/FloorSet/utils.py')]
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(files))}

def geometry(positions):
    p=np.asarray(positions,dtype=float);a=p[:,2:].prod(1).sum()
    bbox=float(np.prod((p[:,:2]+p[:,2:]).max(0)-p[:,:2].min(0)))
    covered=union_area(p)
    return dict(**module_overlap(p),union_over_A=covered/a,
        empty_over_bbox=max(0.,1.-covered/bbox))

def summarize(records):
    qs=np.array([r['metrics']['cost'] for r in records]);ns=[r['n'] for r in records]
    seconds=np.array([r['complete_seconds'] for r in records]);raw=[r['raw_overlap'] for r in records]
    return dict(**q_summary(qs,ns),count=len(records),hard_legal=sum(r['metrics']['is_feasible'] for r in records),
        raw_hard_legal=sum(r['raw_metrics']['is_feasible'] for r in records),fallbacks=sum(r['fallback'] for r in records),
        p95_Q=float(np.percentile(qs,95)),max_Q=float(qs.max()),worst5pct_mean_Q=float(np.sort(qs)[-25:].mean()),
        mean_complete_seconds=float(seconds.mean()),p95_complete_seconds=float(np.percentile(seconds,95)),
        max_complete_seconds=float(seconds.max()),over250ms=int((seconds>.25).sum()),
        mean_module_pass10=float(np.mean([r['module_pass10_fraction'] for r in raw])),
        total_modules_above10=sum(r['N10'] for r in raw),worst_module_covered_fraction=max(r['max_covered_fraction'] for r in raw),
        mean_N10=float(np.mean([r['N10'] for r in raw])),mean_bbox_over_A=float(np.mean([r['bbox_over_A'] for r in raw])),
        mean_empty_over_bbox=float(np.mean([r['empty_over_bbox'] for r in raw])))

def main():
    setup();plan=read(OWNER/'plan.json');training=read(TRAIN/'results.json');progress=read(TRAIN/'progress.json')
    assert training['status']=='early_stop' and progress['finished'] and training['best_epoch']==29
    candidate=ROOT/plan['candidate_checkpoint'];parent=ROOT/plan['parent_checkpoint'];baseline=ROOT/plan['parent_evaluation']
    assert sha(candidate)==plan['candidate_sha256'] and sha(parent)==plan['parent_sha256']
    saved=torch.load(candidate,map_location='cpu',weights_only=False)
    assert saved['progress']['epoch']==saved['progress']['best_epoch']==29
    assert saved['coefficients']==training['frozen']['coefficients']
    mh=model_hash(saved['model']);del saved
    baseline_frozen=read(baseline/'frozen.json');assert baseline_frozen['checkpoint_sha256']==sha(parent)
    members=[r['member'] for r in ReferenceLayouts('dev_monitor500').records]
    assert members==baseline_frozen['members'] and len(members)==500
    # The entire repair implementation remains a byte-identical local copy.
    copied=read(EXPERIMENT/'implementation_provenance.json')
    for entry in copied:
        if '/postprocessor/' in entry['copy']:assert sha(ROOT/entry['copy'])==entry['sha256']
    current=sources();parent_records=[read(baseline/f'cases/{i:04d}.json') for i in range(500)]
    for i,r in enumerate(parent_records):assert r['member']==members[i] and r['checkpoint_sha256']==plan['parent_sha256']
    freeze=dict(plan=plan,candidate_model_hash=mh,sources=current,members=members,
        training_results_sha256=sha(TRAIN/'results.json'),baseline_results_sha256=sha(baseline/'results.json'),
        parent_case_hashes={f'{i:04d}':sha(baseline/f'cases/{i:04d}.json') for i in range(500)},test_read=False)
    OUT.mkdir(parents=True,exist_ok=True)
    import msvcrt
    with (OUT/'evaluation.lock').open('a+b') as handle:
        if handle.tell()==0:handle.write(b'0');handle.flush()
        handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        if (OUT/'frozen.json').exists():
            assert read(OUT/'frozen.json')==freeze
            if (OUT/'results.json').exists():print('Completed evaluation reused',flush=True);return
            if (OUT/'error.json').exists():raise RuntimeError('Existing evaluation error requires diagnosis')
        else:
            save(OUT/'frozen.json',freeze)
            with zipfile.ZipFile(OUT/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
                for name in current:z.write(ROOT/name,name)
        started=time.perf_counter();hb=dict(status='running',stage='loading_dev500',completed=0,total=500,test_read=False)
        mutex=threading.RLock();closed=threading.Event();stopped=[False]
        def beat(**fields):
            with mutex:
                hb.update(fields);save(OUT/'state.json',dict(**hb,pid=os.getpid(),elapsed_seconds=time.perf_counter()-started,
                    updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
        def heartbeat():
            while not closed.wait(5):beat()
        thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
        previous=signal.signal(signal.SIGINT,lambda *_:stopped.__setitem__(0,True))
        def check():
            if stopped[0] or (OUT/'STOP').exists():raise InterruptedError('user stop')
        try:
            rows=load_rows('dev_monitor500');assert [list(r['member']) for r in rows]==members
            for row in rows:
                case=row['case'];inp=case.solver_inputs
                case.reference_positions=public_hard_reference(inp[1],inp[5],inp[6])
            with torch.inference_mode(),threadpool_limits(limits=1):
                warmup();predictor=Predictor(parent,rows[0]['case'].solver_inputs)
                sizes=[r['case'].block_count for r in rows];bridge=[]
                for target in (21,60,90,110,120):
                    i=min((j for j in range(500) if j not in bridge),key=lambda j:(abs(sizes[j]-target),j));bridge.append(i)
                if not (OUT/'parent_reproduction.json').exists():
                    checks=[];beat(stage='verifying_parent_five_cases')
                    for i in bridge:
                        check();draft=predictor.predict(rows[i]['case'].solver_inputs)
                        np.testing.assert_array_equal(draft,parent_records[i]['draft_positions'])
                        final,info=optimize(draft,rows[i]['case'].solver_inputs)
                        np.testing.assert_array_equal(final,parent_records[i]['positions'])
                        q=evaluate_case(rows[i]['case'],final,info['seconds'])['metrics']['cost']
                        assert q==parent_records[i]['metrics']['cost']
                        checks.append(dict(index=i,member=members[i],n=sizes[i],same_draft=True,same_final=True,same_Q=True))
                    save(OUT/'parent_reproduction.json',dict(status='passed',checks=checks,parent500_reused=True))
                predictor.model.load_state_dict(torch.load(candidate,map_location='cpu',weights_only=False)['model'],strict=True)
                predictor.model.eval();records=[]
                beat(stage='evaluating_frozen_best29',completed=0)
                for i,row in enumerate(rows):
                    check();path=OUT/f'cases/{i:04d}.json'
                    if path.exists():
                        record=read(path);assert record['checkpoint_sha256']==plan['candidate_sha256'] and record['member']==members[i]
                    else:
                        inp=row['case'].solver_inputs;assert input_hash(inp)==parent_records[i]['input_hash']
                        torch.cuda.synchronize();begin=time.perf_counter()
                        draft=predictor.predict(inp);draft_seconds=time.perf_counter()-begin
                        final,info=optimize(draft,inp)
                        hard=independent_hard_checks(final.tolist(),np.asarray(inp[1]),np.asarray(inp[5]),public_hard_reference(inp[1],inp[5],inp[6]))
                        complete=time.perf_counter()-begin
                        scored=evaluate_case(row['case'],final,complete);raw=evaluate_case(row['case'],draft,draft_seconds)
                        record=dict(index=i,member=members[i],n=int(row['case'].block_count),checkpoint_sha256=plan['candidate_sha256'],
                            model_hash=mh,metrics=scored['metrics'],independent_hard_checks=hard,raw_metrics=raw['metrics'],
                            raw_overlap=geometry(draft),draft_positions=draft.tolist(),positions=final.tolist(),
                            complete_seconds=complete,draft_seconds=draft_seconds,repair_seconds=info['seconds'],
                            fallback=info['row_retained'],work=info['work'],trace=info['trace'],test_read=False)
                        save(path,record)
                    records.append(record)
                    if (i+1)%10==0:beat(completed=i+1);print('dev500',i+1,'/500',flush=True)
            for r in parent_records:r['raw_overlap']=geometry(r['draft_positions'])
            old=summarize(parent_records);new=summarize(records)
            delta=np.array([b['metrics']['cost']-a['metrics']['cost'] for a,b in zip(parent_records,records)])
            result=dict(status='complete',parent=old,candidate=new,
                Q_delta={k:new[k]-old[k] for k in ('weighted_Q','mean_Q','mixed_Q')},
                paired_cases=dict(improved=int((delta< -1e-6).sum()),worsened=int((delta>1e-6).sum()),unchanged=int((abs(delta)<=1e-6).sum())),
                training=dict(epochs=32,best_epoch=29,baseline_newLoss=progress['history'][0]['dev_monitor']['total'],best_newLoss=training['best_dev_loss'],
                    best_train_newLoss=next(h['train_monitor']['total'] for h in progress['history'] if h['epoch']==29)),
                frozen_checkpoint_sha256=plan['candidate_sha256'],parent500_reused=True,test_read=False,
                timing_boundary='Sequential complete candidate timing; parent historical timing, not same-session speed ablation',
                evidence_boundary='Adaptive development dev500; added training and changedLoss confounded without trained old-Loss control; not sealed test or contest score')
            save(OUT/'comparison.json',result)
            beat(stage='rendering_same100',completed=500)
            from .plots import render
            gallery=OUT/'gallery';gallery.mkdir(exist_ok=True)
            jobs=[]
            for i,row in enumerate(rows[:100]):
                destination=gallery/f'{i:03d}.png'
                if not destination.exists():
                    jobs.append((i,(row['graph']['reference']*row['graph']['scale']).numpy().tolist(),np.asarray(row['case'].solver_inputs[5]).tolist(),
                        parent_records[i]['draft_positions'],records[i]['draft_positions'],records[i]['positions'],str(destination)))
            with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
                for _ in pool.map(render,jobs):check()
            (gallery/'index.html').write_text('<!doctype html><meta charset="utf-8"><style>img{width:100%}</style>'+''.join(
                f'<p>{i}</p><img loading="lazy" src="{i:03d}.png">' for i in range(100)),encoding='utf-8')
            assert sha(candidate)==plan['candidate_sha256'] and sha(parent)==plan['parent_sha256']
            save(OUT/'results.json',result)
            lines=['# E103新Loss续训best29：固定dev500验证','',
                '|模型|加权Q|平均Q|混合Q|硬合法|','|---|---:|---:|---:|---:|']
            for name,r in (('增强基线best71',old),('新Loss续训best29',new)):
                lines.append(f"|{name}|{r['weighted_Q']:.9f}|{r['mean_Q']:.9f}|{r['mixed_Q']:.9f}|{r['hard_legal']}/500|")
            lines+=['',f"混合Q变化：{result['Q_delta']['mixed_Q']:+.9f}；逐例改善/变差/持平：{result['paired_cases']}。",
                f"训练32轮正常早停，best29；同一新验证Loss：{result['training']['baseline_newLoss']:.9f}→{training['best_dev_loss']:.9f}。",
                f"候选完整耗时均值/P95/最大值：{new['mean_complete_seconds']*1000:.3f}/{new['p95_complete_seconds']*1000:.3f}/{new['max_complete_seconds']*1000:.3f}ms；超过250ms：{new['over250ms']}/500。",
                f"草稿原始硬合法：{new['raw_hard_legal']}/500；模块重叠union≤10%平均通过率：{old['mean_module_pass10']:.2%}→{new['mean_module_pass10']:.2%}。",
                f"草稿bbox/A：{old['mean_bbox_over_A']:.6f}→{new['mean_bbox_over_A']:.6f}；框内精确空白比例：{old['mean_empty_over_bbox']:.2%}→{new['mean_empty_over_bbox']:.2%}。",
                '',result['timing_boundary'],result['evidence_boundary'],
                '唯一best在Q前冻结；基线五题逐位复现后复用旧500结果。没有重选checkpoint、改Loss或读测试。',
                '', '[同100例reference/增强基线草稿/新Loss草稿/最终结果](gallery/index.html)','']
            (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
            beat(status='complete',stage='validation_complete',completed=500)
            print(result,flush=True)
        except InterruptedError:beat(status='user_stop',stage='stopped')
        except BaseException as exc:
            save(OUT/'error.json',dict(error=repr(exc),traceback=traceback.format_exc(),test_read=False))
            beat(status='error',stage='failed_no_retry');raise
        finally:closed.set();thread.join(timeout=6);signal.signal(signal.SIGINT,previous)

if __name__=='__main__':main()
