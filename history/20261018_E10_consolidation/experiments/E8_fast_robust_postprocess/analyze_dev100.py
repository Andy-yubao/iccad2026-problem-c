"""Audit saved dev100 and replay only the changed output guard on five cases."""
import hashlib
import json
from pathlib import Path
import time
import zipfile
import numpy as np
from datasets.worksets import LayoutReader,workset_manifest
from datasets.cases import _build_case
from scoring.evaluation import evaluate_case
from scoring.official import official_evaluator
from .frozen import solver
from .frozen.packing import prepare_problem
from .guard import repair_tiny_negative,guarded_refine

HERE=Path(__file__).resolve().parent
BASE=HERE/'runs/20261018_stageA_dev100_seed0'
OUT=HERE/'runs/20261018_dev100_guard_replay_seed0'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(metrics,counts):
    q=np.array([m['cost'] for m in metrics])
    return {'cases':len(q),'Q_mean':float(q.mean()),'Q_weighted':float(official_evaluator().compute_total_score(q.tolist(),counts)),
            'Q_p95':float(np.quantile(q,.95)),'Q_max':float(q.max()),'worst10_mean':float(np.sort(q)[-10:].mean()),
            'Q_above1_5':int((q>1.5).sum()),'feasible':sum(m['is_feasible'] for m in metrics)}


def main():
    started=time.perf_counter()
    rows=[json.loads(s) for s in (BASE/'cases.jsonl').read_text(encoding='utf-8').splitlines()]
    config=json.loads((BASE/'config.json').read_text(encoding='utf-8'))
    result=json.loads((BASE/'results.json').read_text(encoding='utf-8'))
    reader=LayoutReader(workset_manifest(),'dev_quality')
    assert result['status']=='complete' and len(rows)==100
    assert config['members']==[list(pair) for pair in reader.members]
    assert len({r['case_id'] for r in rows})==100
    OUT.mkdir(exist_ok=False)
    source_files=[Path(__file__),HERE/'guard.py',HERE/'frozen/solver.py']
    metadata={'baseline':str(BASE),'baseline_sha256':{n:sha(BASE/n) for n in ('config.json','cases.jsonl','drafts.jsonl')},
              'source_sha256':{p.name:sha(p) for p in source_files},
              'target_workers':[37,45,59,65,73],
              'selection':'Retrospective two guard failures and three small soft-tail controls, all in fixed dev_quality100.',
              'scope':'Terminal-candidate cache replay; not new inference or an online speed benchmark.',
              'workset_sha256':reader.manifest['manifest_sha256'],'test_payload_reads':0}
    (OUT/'config.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    with zipfile.ZipFile(OUT/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in source_files:archive.write(p,p.relative_to(HERE))
    old_metrics=[r['variants']['control120']['final']['metrics'] for r in rows]
    new_metrics=list(old_metrics)
    changed=[];controls=[];lower_bounds=[]
    for idx,row in enumerate(rows):
        worker=int(row['source']['shard'].split('/')[0].split('_')[1])
        if worker not in metadata['target_workers']:continue
        source=row['source'];i=reader.members.index((source['shard'],source['sample_index']))
        raw=reader.case(i);p=prepare_problem(*raw.solver_inputs)
        case=_build_case(raw.case_id,raw.source,raw.solver_inputs[1:6],raw.reference_positions,raw.stored_metrics)
        value=row['variants']['control120']
        prior=value['stats']['observation']['pre_guard_validation']
        original=np.asarray(prior['rejected_positions'] if not prior['strict_valid'] else value['final']['positions'])
        repaired,repair=repair_tiny_negative(original,p)
        assert solver.hard_valid(repaired,p)
        scored=evaluate_case(case,repaired,value['solve_seconds'])
        if not prior['strict_valid']:
            # Real cached counterexample through the same refine adapter used online.
            adapter=guarded_refine(lambda unused_raw,unused_inputs,**unused: (original.copy(),{}))
            adapted,stats=adapter(original,raw.solver_inputs)
            assert np.array_equal(adapted,repaired) and stats['tiny_negative_cleanup']['applied']
            new_metrics[idx]=scored['metrics']
            changed.append({'case_id':row['case_id'],'source':source,'repair':repair,
                            'before_metrics':old_metrics[idx],'after':scored,'cached_adapter_check':True})
        else:
            assert np.array_equal(repaired,original) and scored['metrics']['cost']==old_metrics[idx]['cost']
            controls.append({'case_id':row['case_id'],'repair':repair,'unchanged_positions':True,
                             'unchanged_Q':True,'Q':scored['metrics']['cost']})
        # Public-input lower bound: disjoint allowed-area intervals cannot share one shape.
        groups=[]
        for group in np.unique(p.constraints[:,2]):
            if group<=0:continue
            ids=np.flatnonzero(p.constraints[:,2]==group)
            intervals=sorted([(float(p.areas[j]*.99),float(p.areas[j]*1.01),int(j)) for j in ids],key=lambda t:t[1])
            points=[]
            for lo,hi,j in intervals:
                if not any(lo<=point<=hi for point in points):points.append(hi)
            # Locked shapes may force additional classes; this remains a lower bound.
            groups.append({'group':int(group),'members':ids.tolist(),'areas':p.areas[ids].tolist(),
                           'minimum_classes_area_only':len(points),'minimum_violations_area_only':len(points)-1,
                           'fixed_members':[int(j) for j in ids if p.fixed[j] or p.preplaced[j]]})
        lower_bounds.append({'case_id':row['case_id'],'MIB_actual':old_metrics[idx]['mib_violations'],'groups':groups})
        print(json.dumps({'case_id':row['case_id'],'repair':repair,'Q_before':old_metrics[idx]['cost'],'Q_after':scored['metrics']['cost']}),flush=True)
    assert len(changed)==2 and len(controls)==3
    old=summarize(old_metrics,[r['block_count'] for r in rows]);new=summarize(new_metrics,[r['block_count'] for r in rows])
    parts={}
    total=0.;earliest=0.;initial_infeasible=0
    for row in rows:
        v=row['variants']['control120'];o=v['stats']['observation'];total+=v['solve_seconds']
        for key,seconds in o['seconds_by_component'].items():parts[key]=parts.get(key,0.)+seconds
        earliest+=v['solve_seconds'] if v['stats']['guard_fallback'] else v['quality_curve_offline'][-1]['elapsed']
        initial_infeasible+=any(t.get('seed')==0 and 'infeasible' in t.get('status','').lower() for t in v['stats']['trace'])
    native=parts['fixed_solver_only']+parts['shape_solver_only']
    trace_cuts=[]
    for limit in (1.,1.3,2.,5.,10.):
        scores=[];missing=0
        for row in rows:
            events=[e for e in row['variants']['control120']['quality_curve_offline'] if e['elapsed']<=limit]
            if events:scores.append(events[-1]['Q'])
            else:scores.append(10.);missing+=1
        trace_cuts.append({'seconds':limit,'Q_mean_missing_as10':float(np.mean(scores)),'no_observed_strict_candidate':missing,
                           'scope':'Offline original-trajectory prefix; not an actual short-budget solver.'})
    metrics={'status':'complete_cached_guard_replay','before':old,'after_cached_guard_replay':new,
             'changed_cases':len(changed),'unchanged_reused_cases':98,'checked_controls':controls,
             'solver_calls':0,'model_calls':0,'dev_cases_loaded':5,'test_payload_reads':0,
             'wall_seconds':time.perf_counter()-started,'timing_claim':'No new online solve timing; baseline time must not be claimed as guard-repair speed.',
             'baseline_analysis':{'seconds_by_component':parts,'native_solver_fraction':native/total,
                'ideal_speedup_eliminate_all_non_solver_time':total/native,
                'initial_fixed_infeasible_cases':initial_infeasible,
                'oracle_stop_speedup_to_last_strict_proxy_improvement':total/earliest,
                'speed8x_mean_target_seconds':total/100/8,'speed10x_mean_target_seconds':total/100/10,
                'original_trace_cutoffs_diagnostic_only':trace_cuts},'targeted_MIB_area_lower_bounds':lower_bounds}
    (OUT/'changed_cases.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in changed),encoding='utf-8')
    (OUT/'results.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'before':old,'after_cached_guard_replay':new,'baseline_analysis':metrics['baseline_analysis']}),flush=True)


if __name__=='__main__':main()
