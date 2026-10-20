"""Fixed dev100 validation of legal-native retention; shared 900s compute cap."""
import argparse
import json
from pathlib import Path
import random
import sys
import time
import zipfile

from .run import (ROOT, HERE, DRAFTS, TEACHERS, Worker, sha, save, record, np, torch,
                  LayoutReader, workset_manifest, _build_case, prepare_problem, topology, contact_tree)
from .representation import decoder_oracle
from .panel import summarize as panel_summary

ARMS = ('raw_short_safe', 'teacher_relations_safe', 'decoder_relations_safe')


def summaries(rows):
    # Reuse the experiment's established metrics without duplicating coordinates.
    from . import panel
    original = panel.ARMS
    try:
        panel.ARMS = {a: .4 for a in ARMS}
        result = panel_summary(rows)
    finally:
        panel.ARMS = original
    for arm in ARMS:
        trials = [t for r in rows for t in r['trials'] if t['arm'] == arm]
        if arm in result:
            result[arm]['legal_native_retained_count'] = sum(t.get('stats', {}).get('acceptance', {}).get('legal_native_retained_count', 0) for t in trials)
            result[arm]['cases_using_native_retention'] = sum(t.get('stats', {}).get('acceptance', {}).get('legal_native_retained_count', 0)>0 for t in trials)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-run', type=Path, required=True)
    parser.add_argument('--representation-run', type=Path, required=True)
    parser.add_argument('--audit-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); begin=time.perf_counter(); torch.set_num_threads(2)
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    audit = json.loads((args.audit_run/'results.json').read_text(encoding='utf-8'))
    assert audit['status']=='completed' and len(audit['cases'])==5
    parent_elapsed=audit['shared_elapsed_seconds']
    if parent_elapsed>=700:raise ValueError('Insufficient shared budget for this validation; do not start automatically')
    args.output.mkdir(parents=True,exist_ok=False);(args.output/'cases').mkdir()
    anchors={r['case_id']:r for r in (json.loads(p.read_text(encoding='utf-8')) for p in (args.main_run/'cases').glob('*.json'))}
    drafts={r['case_id']:r for r in map(json.loads,DRAFTS.read_text(encoding='utf-8').splitlines())}
    teachers={r['case_id']:r for r in map(json.loads,TEACHERS.read_text(encoding='utf-8').splitlines())}
    config={'command':[sys.executable,*sys.argv],'seed':0,'threads':2,'arms':ARMS,
            'source_runs':{str(p):sha(p/'results.json') for p in [args.main_run,args.representation_run,args.audit_run]},
            'parent_elapsed_seconds':parent_elapsed,'shared_cap_seconds':900,
            'hypothesis':'A legal native candidate must not be discarded solely because canonicalization made it illegal. Same strict hard guards; no tolerance or scoring changes.',
            'local_test_read':False,'training':False,'scope':'Adaptive development follow-up after a verified numerical counterexample, all100 members retained; original baseline unmodified.',
            'sources':{p.relative_to(ROOT).as_posix():sha(p) for p in HERE.rglob('*.py')}}
    save(args.output/'config.json',config)
    with zipfile.ZipFile(args.output/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in HERE.rglob('*.py'):archive.write(path,path.relative_to(ROOT).as_posix())
        archive.write(HERE/'README.md',(HERE/'README.md').relative_to(ROOT).as_posix())
    reader=LayoutReader(workset_manifest(),'dev_quality');assert len(reader.members)==100
    worker=Worker(args.output/'worker.log',module='experiments.E11_short_budget_diagnosis.worker_safe')
    rows=[];status='running';rng=random.Random(0);warmed=False
    try:
        for index in range(100):
            raw=reader.case(index);case=_build_case(raw.case_id,raw.source,raw.solver_inputs[1:6],raw.reference_positions,raw.stored_metrics)
            draft=np.asarray(drafts[raw.case_id]['draft']);teacher=np.asarray(teachers[raw.case_id]['variants']['conic_robust']['final']['positions'])
            inputs=case.solver_inputs;p=prepare_problem(*inputs);full=topology(teacher,p.constraints)
            guidance={'edges':[[int(v) for v in e] for e in full],'contacts':[[int(v) for v in e] for e in contact_tree(teacher,p,full)]}
            projected=decoder_oracle(teacher,inputs);serialized=[int(inputs[0]),*[np.asarray(v).tolist() for v in inputs[1:]]]
            row={'case_id':raw.case_id,'source':raw.source,'panel_index':index,'block_count':case.block_count,
                 'baseline_q':anchors[raw.case_id]['baseline_q'],'layouts':{},'trials':[]};rows.append(row)
            if not warmed:
                warmup=worker.call({'inputs':serialized,'draft':draft.tolist(),'seconds':.1});config['warmup_wall_seconds']=warmup['worker_wall_seconds'];warmed=True
            arms=list(ARMS);rng.shuffle(arms)
            for arm in arms:
                if parent_elapsed+time.perf_counter()-begin>=900 or (args.output/'STOP').exists():status='budget_or_user_stop';break
                response=worker.call({'inputs':serialized,'draft':(projected if arm=='decoder_relations_safe' else draft).tolist(),
                                      'seconds':.4,'guidance':None if arm=='raw_short_safe' else guidance})
                trial=record(case,response,drafts[raw.case_id]['inference_seconds'],row['layouts'])
                if 'charged_total_seconds' not in trial:trial['charged_total_seconds']=drafts[raw.case_id]['inference_seconds']+response['worker_wall_seconds']
                trial.update(arm=arm,budget_seconds=.4,Q_delta_vs_live_full_raw=trial['metrics']['cost']-row['baseline_q'])
                row['trials'].append(trial);save(args.output/'cases'/f'{index:03d}.json',row)
            print(f"safe {index+1}/100",' '.join(f"{t['arm']}={t['metrics']['cost']:.4f}" for t in row['trials']),flush=True)
            save(args.output/'results.json',{'status':status,'elapsed_seconds':time.perf_counter()-begin,'shared_elapsed_seconds':parent_elapsed+time.perf_counter()-begin,
                'case_count':len(rows),'trial_count':sum(len(r['trials']) for r in rows),'summary':summaries(rows)})
            if status!='running':break
        if status=='running':status='completed'
    except Exception as exc:status='error';config['fatal_error']=repr(exc);raise
    finally:
        worker.close();save(args.output/'config.json',config)
        save(args.output/'results.json',{'status':status,'elapsed_seconds':time.perf_counter()-begin,'shared_elapsed_seconds':parent_elapsed+time.perf_counter()-begin,
            'case_count':len(rows),'trial_count':sum(len(r['trials']) for r in rows),'summary':summaries(rows),
            'scope':'All100 development follow-up; strict guards unchanged, preserve native candidate only if canonical form fails and original passes. Oracle is not a trained model.'})


if __name__=='__main__':main()
