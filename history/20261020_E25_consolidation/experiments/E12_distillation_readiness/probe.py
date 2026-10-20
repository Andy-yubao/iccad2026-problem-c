"""Prespecified gate/sweep/confirmation for the fixed-shape student interface."""
import argparse
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import zipfile

from .io import ROOT,HERE,HISTORY,np,sha,save,Worker
from .backend.frozen.packing import prepare_problem
from .backend.frozen.global_refine import topology,contact_tree
from datasets.worksets import LayoutReader,workset_manifest
from datasets.cases import _build_case
from datasets.paths import resolved_data_paths
from scoring.evaluation import evaluate_case


def decode_teacher(teacher,p,family,noise,seed,certificate=None):
    rng=np.random.default_rng(seed)
    center=(np.asarray((teacher[:,:2]+teacher[:,2:]/2)/p.scale,dtype=np.float32).astype(float)*p.scale)
    logs=np.asarray(np.log(teacher[:,2]/teacher[:,3]),dtype=np.float32).astype(float)
    center+=rng.normal(size=center.shape)*noise*np.sqrt(p.areas)[:,None]
    logs=np.clip(logs+rng.normal(size=p.n)*noise,-5,5)
    if family=='exact':areas=p.areas.copy()
    elif family=='lower':areas=p.areas*.9900002
    elif family=='teacher':areas=p.areas*np.clip(teacher[:,2:].prod(1)/p.areas+rng.normal(size=p.n)*noise*.02,.9900002,1.0099998)
    else:raise ValueError(family)
    ratio=np.exp(logs);wh=np.column_stack((np.sqrt(areas*ratio),np.sqrt(areas/ratio)))
    locked=p.fixed|p.preplaced;wh[locked]=p.specs[locked,2:]
    if certificate is not None:
        from .certificate import coordinate_shapes
        wh=coordinate_shapes(wh,p,certificate['mib_labels'],family,areas,logs)
    # Public-only compatible MIB coordination is part of this candidate decoder.
    for group in ([] if certificate is not None else np.unique(p.constraints[:,2])):
        if group<=0:continue
        ids=np.flatnonzero(p.constraints[:,2]==group);rigid=ids[locked[ids]]
        if len(rigid):
            for i in ids[~locked[ids]]:
                compatible=[j for j in rigid if abs(p.specs[j,2:].prod()/p.areas[i]-1)<.0099998]
                if compatible:
                    closest=min(compatible,key=lambda j:abs(np.log(p.specs[j,2]/p.specs[j,3])-logs[i]))
                    wh[i]=p.specs[closest,2:]
        else:
            lower=float(np.max(p.areas[ids]*.9900002));upper=float(np.min(p.areas[ids]*1.0099998))
            groups=[ids] if lower<=upper else [ids[p.areas[ids]==a] for a in np.unique(p.areas[ids])]
            for subset in groups:
                if len(subset)<2:continue
                low=float(np.max(p.areas[subset]*.9900002));high=float(np.min(p.areas[subset]*1.0099998))
                area=float(np.clip(np.exp(np.log(areas[subset]).mean()),low,high));r=float(np.exp(logs[subset].mean()))
                wh[subset]=[np.sqrt(area*r),np.sqrt(area/r)]
    out=np.column_stack((np.maximum(0.,center-wh/2),wh));out[p.preplaced]=p.specs[p.preplaced]
    return out


def geometry(pos):
    area=pos[:,2:].prod(1).sum();lo=pos[:,:2].min(0);hi=(pos[:,:2]+pos[:,2:]).max(0);bbox=float(np.prod(hi-lo))
    xs=np.unique(np.r_[pos[:,0],pos[:,0]+pos[:,2]]);union=0.
    for a,b in zip(xs[:-1],xs[1:]):
        active=pos[(pos[:,0]<b)&(pos[:,0]+pos[:,2]>a)]
        intervals=sorted((r[1],r[1]+r[3]) for r in active);covered=0.;end=-np.inf
        for c,d in intervals:covered+=max(0.,d-max(c,end));end=max(end,d)
        union+=(b-a)*covered
    i,j=np.triu_indices(len(pos),1)
    overlap=np.minimum(pos[i,:2]+pos[i,2:],pos[j,:2]+pos[j,2:])-np.maximum(pos[i,:2],pos[j,:2])
    return {'overlap_excess_area_fraction':float(max(0.,area-union)/area),
            'internal_whitespace_fraction':float(max(0.,bbox-union)/bbox),
            'overlap_pairs':int((overlap>1e-6).all(1).sum()),'bbox_area':bbox}


def arms_for(phase,family):
    if phase=='focused':return [(family+'_geo0',family,0.,False,'separated'),
            (family+'_geo005',family,.005,False,'separated'),(family+'_geo02',family,.02,False,'separated'),
            (family+'_geo05',family,.05,False,'separated'),(family+'_teacher02',family,.02,True,'separated')]
    if phase=='gate':return [('exact_geo0','exact',0.,False,'separated'),('lower_legacy','lower',0.,False,'legacy'),
                            ('lower_geo0','lower',0.,False,'separated'),('teacher_geo0','teacher',0.,False,'separated')]
    if phase=='sweep':return [('exact_geo0','exact',0.,False,'separated'),('lower_geo0','lower',0.,False,'separated'),
          ('teacher_geo0','teacher',0.,False,'separated'),('lower_teacher0','lower',0.,True,'separated'),
          ('lower_geo005','lower',.005,False,'separated'),('lower_geo02','lower',.02,False,'separated'),
          ('lower_geo05','lower',.05,False,'separated'),('lower_teacher02','lower',.02,True,'separated'),
          ('teacher_geo02','teacher',.02,False,'separated')]
    return [(family+'_geo0',family,0.,False,'separated'),(family+'_geo02',family,.02,False,'separated'),
            (family+'_teacher02',family,.02,True,'separated')]


def summarize(rows,arms):
    out={}
    for name,*_ in arms:
        pairs=[(r,t) for r in rows for t in r['trials'] if t['arm']==name]
        if not pairs:continue
        trials=[t for _,t in pairs];q=np.array([t['metrics']['cost'] for t in trials]);times=np.array([t['worker_wall_seconds'] for t in trials])
        def avg(key):
            values=[t['metrics'][key] for t in trials if key in t['metrics']];return float(np.mean(values)) if values else None
        out[name]={'count':len(trials),'Q_mean':float(q.mean()),'Q_p95':float(np.quantile(q,.95)),'Q_max':float(q.max()),
             'hard_legal_count':sum(t['metrics']['is_feasible'] for t in trials),'error_count':sum('error' in t for t in trials),
             'Q_delta_mean_vs_teacher':float(np.mean([t['metrics']['cost']-r['teacher_q'] for r,t in pairs])),
             'within_002_count':sum(t['metrics']['cost']<=r['teacher_q']+.002 and t['metrics']['is_feasible'] for r,t in pairs),
             'row_retained_count':sum(t.get('stats',{}).get('row_retained',False) for t in trials),
             'worker_seconds_mean':float(times.mean()),'worker_seconds_p95':float(np.quantile(times,.95)),'worker_seconds_max':float(times.max()),
             'mean_plus_010_network_reserve':float(times.mean())+.1,
             'charged_original_inference_mean':(float(np.mean([t['charged_original_total_seconds'] for t in trials if t['charged_original_total_seconds'] is not None])) if any(t['charged_original_total_seconds'] is not None for t in trials) else None),
             'original_inference_available_count':sum(t['charged_original_total_seconds'] is not None for t in trials),
             'hpwl_gap_mean':avg('hpwl_gap'),'area_gap_mean':avg('area_gap'),'boundary_mean':avg('boundary_violations'),
             'grouping_mean':avg('grouping_violations'),'mib_mean':avg('mib_violations'),
             'draft_overlap_excess_mean':float(np.mean([t['draft_geometry']['overlap_excess_area_fraction'] for t in trials])),
             'draft_whitespace_mean':float(np.mean([t['draft_geometry']['internal_whitespace_fraction'] for t in trials]))}
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--phase',choices=['gate','audit','sweep','focused','confirm'],required=True)
    parser.add_argument('--family',choices=['exact','lower','teacher'],default='lower')
    parser.add_argument('--joint',action='store_true')
    parser.add_argument('--certified',action='store_true')
    parser.add_argument('--polished',action='store_true')
    parser.add_argument('--prior-runs',type=Path,nargs='*',default=[])
    args=parser.parse_args();begin=time.perf_counter()
    import torch
    torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0);random.seed(0)
    args.output.mkdir(parents=True,exist_ok=False);(args.output/'cases').mkdir()
    prior_elapsed=20.+sum(json.loads((p/'results.json').read_text(encoding='utf-8'))['elapsed_seconds'] for p in args.prior_runs)
    manifest=workset_manifest()
    if args.phase=='confirm':
        origin=HISTORY/'20261018_robust_confirmation100_seed0';origin_config=json.loads((origin/'config.json').read_text(encoding='utf-8'))
        reader=LayoutReader(manifest,'dev');lookup={pair:i for i,pair in enumerate(reader.members)}
        indices=[lookup[(shard,int(row))] for shard,row in origin_config['members']]
        draft_path=origin/'drafts.jsonl'
    else:
        origin=HISTORY/'20261018_robust_dev100_seed0';reader=LayoutReader(manifest,'dev_quality')
        indices=[i for i in range(100) if args.phase!='gate' or i in (20,29,52,66,86)]
        if args.phase=='audit':indices=[1,2,3,4,7]
        draft_path=HISTORY/'20261018_stageA_dev100_seed0/drafts.jsonl'
    teacher_path=origin/'cases.jsonl'
    teachers={r['case_id']:r for r in map(json.loads,teacher_path.read_text(encoding='utf-8').splitlines())}
    drafts={r['case_id']:r for r in map(json.loads,draft_path.read_text(encoding='utf-8').splitlines())}
    arms=arms_for(args.phase,args.family)
    sources=[*HERE.rglob('*.py'),HERE/'README.md',HERE/'provenance.json']
    config={'command':[sys.executable,*sys.argv],'phase':args.phase,'arms':arms,'seed':0,'threads':2,
       'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
       'git_status':subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True).strip(),
       'members':[reader.members[i] for i in indices],'sources':{p.relative_to(ROOT).as_posix():sha(p) for p in sources},
       'cache_sources':{str(p):sha(p) for p in [teacher_path,draft_path]},'paths':resolved_data_paths(),
       'split_sha256':sha(ROOT/'protocol/split_v1.json'),'workset_sha256':sha(ROOT/'protocol/workset_v2.json'),
       'local_test_read':False,'training':False,'prior_runs':[str(p) for p in args.prior_runs],
       'prior_elapsed_seconds':prior_elapsed,'shared_compute_cap_seconds':1100,'joint':args.joint,'certified':args.certified,'polished':args.polished,'single_case_diagnostic_charge_seconds':20.,
       'timing_scope':'Live worker includes decoding inputs, projection and IPC; original inference charged separately, 0.10s future-network reserve explicit. Oracle construction/scoring outside worker.'}
    save(args.output/'config.json',config)
    with zipfile.ZipFile(args.output/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in sources:archive.write(path,path.relative_to(ROOT).as_posix())
    worker=Worker(args.output/'worker.log');rows=[];status='running';rng=random.Random(0);warmed=False
    try:
        for panel_index,index in enumerate(indices):
            raw=reader.case(index);case=_build_case(raw.case_id,raw.source,raw.solver_inputs[1:6],raw.reference_positions,raw.stored_metrics)
            teacher_record=teachers[raw.case_id]['variants']['conic_robust']['final'];teacher=np.asarray(teacher_record['positions'])
            p=prepare_problem(*case.solver_inputs);edges=topology(teacher,p.constraints)
            guidance={'edges':edges,'contacts':contact_tree(teacher,p,edges)}
            guidance={key:[[int(v) for v in t] for t in values] for key,values in guidance.items()}
            certificate=None
            if args.certified:
                from .certificate import extract
                certificate=extract(teacher,p);guidance=certificate
            inputs=[int(case.solver_inputs[0]),*[np.asarray(v).tolist() for v in case.solver_inputs[1:]]]
            row={'case_id':raw.case_id,'source':raw.source,'panel_index':panel_index,'reader_index':index,'block_count':case.block_count,
                 'teacher_q':teacher_record['metrics']['cost'],'teacher_metrics':teacher_record['metrics'],
                 'teacher_geometry':geometry(teacher),'layouts':{},'trials':[]};rows.append(row)
            order=list(arms);rng.shuffle(order)
            for name,family,noise,use_teacher,snap_mode in order:
                if prior_elapsed+time.perf_counter()-begin>=1100 or (args.output/'STOP').exists():status='budget_or_user_stop';break
                seed=decode_teacher(teacher,p,family,noise,20261018+index,certificate=certificate)
                active_guidance=guidance if use_teacher else (None if certificate is None else {**certificate,'edges':topology(seed,p.constraints)})
                if not warmed:
                    warmup=worker.call({'inputs':inputs,'draft':seed.tolist(),'guidance':guidance,'seconds':.35,'snap_mode':snap_mode,'joint':args.joint,'polished':args.polished})
                    config['warmup_wall_seconds']=warmup['worker_wall_seconds'];warmed=True
                response=worker.call({'inputs':inputs,'draft':seed.tolist(),'guidance':active_guidance,'seconds':.35,'snap_mode':snap_mode,'joint':args.joint,'polished':args.polished})
                trial={'arm':name,'family':family,'noise':noise,'teacher_relations':use_teacher,'snap_mode':snap_mode,
                       'worker_wall_seconds':response['worker_wall_seconds'],'solve_seconds':response['solve_seconds'],
                       'charged_original_total_seconds':(response['worker_wall_seconds']+drafts[raw.case_id]['inference_seconds'] if raw.case_id in drafts else None),
                       'draft_geometry':geometry(seed)}
                if 'error' in response:trial.update(error=response['error'],metrics={'cost':10.,'is_feasible':False})
                else:
                    positions=response['positions'];evaluated=evaluate_case(case,positions,response['solve_seconds']);evaluated.pop('positions')
                    key=__import__('hashlib').sha256(np.asarray(positions,dtype='<f8').tobytes()).hexdigest()[:20]
                    row['layouts'].setdefault(key,positions)
                    trial.update(**evaluated,stats=response['stats'],layout_key=key,final_geometry=geometry(np.asarray(positions)))
                row['trials'].append(trial);save(args.output/'cases'/f'{panel_index:03d}.json',row)
            print(f"{args.phase} {panel_index+1}/{len(indices)}",raw.case_id,' '.join(f"{t['arm']}={t['metrics']['cost']:.4f}" for t in row['trials']),flush=True)
            save(args.output/'results.json',{'status':status,'elapsed_seconds':time.perf_counter()-begin,'shared_elapsed_seconds':prior_elapsed+time.perf_counter()-begin,
                  'case_count':len(rows),'trial_count':sum(len(r['trials']) for r in rows),'summary':summarize(rows,arms)})
            if status!='running':break
        if status=='running':status='completed'
    except Exception as exc:status='error';config['fatal_error']=repr(exc);raise
    finally:
        worker.close();save(args.output/'config.json',config)
        save(args.output/'results.json',{'status':status,'elapsed_seconds':time.perf_counter()-begin,'shared_elapsed_seconds':prior_elapsed+time.perf_counter()-begin,
             'case_count':len(rows),'trial_count':sum(len(r['trials']) for r in rows),'summary':summarize(rows,arms),
             'scope':'Development oracle/noise diagnostic, not trained-model performance; all cases retained.'})


if __name__=='__main__':main()
