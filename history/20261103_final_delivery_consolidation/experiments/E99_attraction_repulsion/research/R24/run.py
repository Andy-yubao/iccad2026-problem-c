"""Frozen AR reproducibility on disjoint training-source members, no tuning."""
import os
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_k]='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['MPLBACKEND']='Agg'
import argparse,datetime,json,time,zipfile,hashlib
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from datasets.reference_layouts import ReferenceLayouts,registry
from scoring.evaluation import evaluate_case
from scoring.official import official_evaluator
from ...planner.solver import optimize
from ..R23 import run as base
from ..R23.public_inputs import PublicShard,build_public
from ..R16 import replay

ROOT=base.ROOT;RESEARCH=base.RESEARCH;EXPERIMENT=base.EXPERIMENT;HERE=Path(__file__).resolve().parent
read=base.read;save=base.save;sha=base.sha;CFG=base.CFG
PLAN=read(HERE/'plan.json');PANEL=read(HERE/'panel.json');OUT=EXPERIMENT/PLAN['output'];PREFLIGHT=EXPERIMENT/PLAN['preflight_output']


class Probe:
    def __init__(self):
        self.data=ReferenceLayouts('train');self.records=PANEL['records'];reg=registry()
        assert sha(ROOT/'protocol/reference_workset_v5.json')==PANEL['source_registry_sha256']
        old={tuple(r['member']) for r in reg['sets']['train_monitor500']['records']}
        assert len(self.records)==500 and len({tuple(r['member']) for r in self.records})==500
        assert all(tuple(r['member']) not in old for r in self.records)
        for r in self.records:assert self.data.records[r['source_train_index']]['member']==r['member']
    def __getitem__(self,index):return self.data[self.records[index]['source_train_index']]


def sources():
    result=base.sources();result.update({p.relative_to(ROOT).as_posix():sha(p) for p in (HERE/'run.py',HERE/'plan.json',HERE/'panel.json',RESEARCH/'R23/run.py')});return result


def register(out):
    Probe();assert sha(ROOT/base.PLAN['checkpoint'])==base.CHECKPOINT_SHA
    base.replay.check_planner()
    identity=dict(plan=PLAN,panel_sha256=sha(HERE/'panel.json'),sources=sources(),frozen_AR_config=base.PLAN['configuration'],checkpoint_sha256=base.CHECKPOINT_SHA,
        R23_registration_sha256=sha(base.OUT/'registration.json'),R23_results_sha256=sha(base.OUT/'results.json'),registry_sha256=sha(ROOT/'protocol/reference_workset_v5.json'))
    if (out/'registration.json').exists():assert read(out/'registration.json')['identity']==identity
    else:
        out.mkdir(parents=True,exist_ok=True);save(out/'registration.json',dict(identity=identity,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        with zipfile.ZipFile(out/'source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for p in identity['sources']:z.write(ROOT/p,p)
    return identity


def calculate(index,arm,data,cache,model,out,options):
    meta=data.records[index];raw,io=cache.load(meta);torch.cuda.synchronize();start=time.perf_counter()
    inputs=build_public(raw,meta);p_end=time.perf_counter();draft=model.predict(inputs);d_end=time.perf_counter()
    if arm=='retained_AR':prefix,stats=base.preprocess(draft,inputs,CFG,capture=False)
    else:prefix=draft.copy();stats=dict(config=None,seconds=0.)
    a_end=time.perf_counter();final,trace=optimize(prefix,inputs,**options);s_end=time.perf_counter();h=base.hard(final,inputs);torch.cuda.synchronize();end=time.perf_counter();wall=end-start
    row=data[index];assert list(row['member'])==meta['member'];score=evaluate_case(row['case'],final,wall);assert score['metrics']['is_feasible']
    path=out/'coordinates'/arm/f'{index:04d}.npz';path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp')
    with temp.open('wb') as f:np.savez_compressed(f,raw=draft,prefix=prefix,final=final)
    temp.replace(path)
    record=dict(index=index,arm=arm,member=meta['member'],source_train_index=meta['source_train_index'],n=int(inputs[0]),metrics=score['metrics'],hard=h,complete_seconds=wall,
        stages_seconds=dict(public_construction=p_end-start,public_features_C3_transfers=d_end-p_end,AR=a_end-d_end,planner=s_end-a_end,hard_and_final_sync=end-s_end),excluded_shard_loading_seconds=io,preprocessor_stats=stats,planner_trace=trace,
        raw_draft_sha256=hashlib.sha256(draft.tobytes()).hexdigest(),coordinate_source=path.relative_to(ROOT).as_posix(),coordinates_sha256=sha(path))
    save(out/'cases'/arm/f'{index:04d}.json',record);return record


@torch.inference_mode()
def preflight():
    if PREFLIGHT.exists():raise FileExistsError('Preflight evidence protected')
    base.setup();identity=register(PREFLIGHT);data=Probe();cache=PublicShard('train');options=read(EXPERIMENT/'planner_config.json');records=[]
    with replay.process_lock(RESEARCH/'.numerical.lock'),threadpool_limits(limits=1):
        model,cold=base.warm_model(cache,data.records[0]);before=base.model_hash(model.model)
        for index in PLAN['preflight']['indices']:
            pair=[]
            for arm in PLAN['arms']:
                if base.stopped(PREFLIGHT):raise RuntimeError('STOP')
                r=calculate(index,arm,data,cache,model,PREFLIGHT,options);records.append(r);pair.append(r)
            assert pair[0]['raw_draft_sha256']==pair[1]['raw_draft_sha256']
        assert before==base.model_hash(model.model) and all(not p.requires_grad for p in model.model.parameters())
    assert sources()==identity['sources'];save(PREFLIGHT/'results.json',dict(status='passed',unique_cases=5,records=records,cold=cold,source_sha256=identity['sources'],all_matched_drafts_bit_exact=True,test_read=False))
    print(json.dumps(dict(status='passed',unique_cases=5,arm_cases=10,matched_drafts_bit_exact=True,all_hard_legal=True)))


@torch.inference_mode()
def run(resume=False):
    pre=read(PREFLIGHT/'results.json');assert pre['status']=='passed' and pre['source_sha256']==sources()
    if OUT.exists() and not resume:raise FileExistsError('Explicit same-source --resume required')
    base.setup();identity=register(OUT);data=Probe();cache=PublicShard('train');options=read(EXPERIMENT/'planner_config.json');records=[];begin=time.perf_counter()
    with replay.process_lock(RESEARCH/'.numerical.lock'),threadpool_limits(limits=1):
        if base.stopped(OUT):save(OUT/'state.json',dict(status='stopped',completed=0,total=1000));return
        model,cold=base.warm_model(cache,data.records[0]);before=base.model_hash(model.model)
        for index in range(500):
            arms=PLAN['arms'];order=arms[index%2:]+arms[:index%2];pair=[]
            for arm in order:
                if base.stopped(OUT):save(OUT/'state.json',dict(status='stopped',completed=len(records),total=1000));return
                path=OUT/'cases'/arm/f'{index:04d}.json'
                if path.exists():record=read(path)
                elif index in PLAN['preflight']['indices']:
                    source=PREFLIGHT/'cases'/arm/f'{index:04d}.json';record=dict(read(source),reused_same_identity_preflight=True,preflight_record_sha256=sha(source));save(path,record)
                else:record=calculate(index,arm,data,cache,model,OUT,options)
                assert record['member']==data.records[index]['member'] and record['arm']==arm and sha(ROOT/record['coordinate_source'])==record['coordinates_sha256']
                pair.append(record);records.append(record);save(OUT/'state.json',dict(status='running',pid=os.getpid(),completed=len(records),total=1000,elapsed_seconds=time.perf_counter()-begin,test_read=False))
            assert pair[0]['raw_draft_sha256']==pair[1]['raw_draft_sha256']
            if (index+1)%25==0:print('R24 disjoint training probe',index+1,'/500 x2',flush=True)
        assert sources()==identity['sources'] and before==base.model_hash(model.model)
        summaries={arm:base.summarize([r for r in records if r['arm']==arm]) for arm in PLAN['arms']}
        lookup={r['index']:r for r in records if r['arm']=='original'};ar=[r for r in records if r['arm']=='retained_AR'];delta=np.array([r['metrics']['cost']-lookup[r['index']]['metrics']['cost'] for r in ar])
        paired=dict(better=int((delta<-1e-10).sum()),equal=int((np.abs(delta)<=1e-10).sum()),worse=int((delta>1e-10).sum()),weighted_delta_Q=summaries['retained_AR']['weighted_neutral_Q']-summaries['original']['weighted_neutral_Q'])
        # base summary's strict_pass is a numerical predicate only; this is TRAIN,
        # so it must never be used to claim formal dev acceptance.
        for s in summaries.values():s['training_probe_only']=True;s.pop('strict_acceptance_pass',None)
        save(OUT/'results.json',dict(status='complete',panel=PANEL['revision'],summary=summaries,paired=paired,records=records,cold=cold,source_sha256=identity['sources'],frozen_model_state_sha256=before,new_training=False,test_read=False,new_dev_evaluation=False,all500_matched_drafts_bit_exact=True,fresh_arm_solves=990,reused_preflight_arm_cases=10))
        save(OUT/'state.json',dict(status='complete',completed=1000,total=1000,test_read=False))
        study=read(RESEARCH/'study.json');study.update(status='R24_disjoint_training_probe_complete_review',current_run=None,active_pid=None);study['R24_probe']=dict(plan='research/R24/plan.json',run=PLAN['output'],summary=summaries,paired=paired);save(RESEARCH/'study.json',study)
        lines=['# Frozen AR paired reproducibility: disjoint training-source500','',f"Original weightedQ={summaries['original']['weighted_neutral_Q']:.12f}; retainedAR={summaries['retained_AR']['weighted_neutral_Q']:.12f}; paired {paired}.",'', 'All500 members are v5 training members seen by C3 training, excluded from E99 original train_monitor500 coefficient development. No parameter change on this probe. This is not a sealed test or dev acceptance. Both arms use bit-exact same fresh drafts and original full planner. All bad/slow cases retained.10 identical preflight arm-cases reused,990 other fresh arm solves. Test sealed; no new network training.']
        (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(dict(status='complete',summary=summaries,paired=paired),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');p.add_argument('--resume',action='store_true');args=p.parse_args()
    try:preflight() if args.preflight else run(args.resume)
    except BaseException as e:
        out=PREFLIGHT if args.preflight else OUT
        save(out/('error_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.json'),dict(error=repr(e),pid=os.getpid(),no_automatic_retry=True,test_read=False));raise
