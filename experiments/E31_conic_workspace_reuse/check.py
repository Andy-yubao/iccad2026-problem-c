"""At most five known dev cases; fixed numerical problems then complete solver."""
from .resources import configure
RESOURCE=configure()
import time
import json
import sys
import numpy as np
import torch
from .cases import prepare_cache,graph,predict,seed_all
from .model import GeometryModel
from .fast_geometry import step,topology
from .native_reduce import reduce_edges,warmup
from .canonical_native import warmup as canonical_warmup
from .backend.frozen.packing import prepare_problem,_dimensions
from .solver import optimize
from .io import ROOT,HERE,sha,sources,snapshot,save
from scoring.evaluation import evaluate_case

POLICY=dict(relaxed=False,mode=0,adaptive=True,fast_budget=False,late_extension=True,plateau=True,robust_seeds=True,certified=True,quick_project=True,quadratic_project=True,incumbent_project=True,snap=1e-4,native_reduce=True,tail_refine=True)
CHECKPOINT=ROOT/'experiments/E30_stable_fast_geometry/runs/20261022_training_seed0/geometry_control/checkpoints/best.pt'

def load_model(rows):
    state=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    model=GeometryModel(**state['model_config']).eval();model.load_state_dict(state['model'])
    with torch.inference_mode():
        model=torch.jit.optimize_for_inference(torch.jit.freeze(torch.jit.trace(model,(graph(rows[0]['case'].solver_inputs),),strict=False)))
    return model

def main():
    started=time.perf_counter();warmup();canonical_warmup();rows=prepare_cache()['rows'];model=load_model(rows)
    folder=HERE/'runs/20261022_real_case_check';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'results.json').exists():raise ValueError('Completed evidence immutable')
    identity={'sources':sources(),'checkpoint_sha256':sha(CHECKPOINT),'policy':POLICY,'panel':[14,414,105,143,21],'command':sys.argv,'resource':RESOURCE}
    save(folder/'config.json',identity);snapshot(folder/'source_snapshot.zip',identity)
    records=[]
    for index in identity['panel']:
        row=rows[index];case=row['case'];seed_all(index);draft,_=predict(model,case.solver_inputs)
        p=prepare_problem(*case.solver_inputs);w,h=_dimensions(p,draft[:,2]/draft[:,3],True);state=np.column_stack((draft[:,:2],w,h))
        full=topology(state,p.constraints,array=True);edges=reduce_edges(full,p.n);workspace={};numeric=[]
        for repetition in range(6):
            pair={}
            for arm in (('cold','reuse') if repetition%2==0 else ('reuse','cold')):
                x,stats=step(p,state,edges,seconds=2.,full_edges=full,workspace=workspace if arm=='reuse' else None)
                pair[arm]={'stats':stats,'positions':None if x is None else x.tolist()}
            a,b=pair['cold']['positions'],pair['reuse']['positions']
            pair['max_abs_difference']=None if a is None or b is None else float(np.max(np.abs(np.asarray(a)-b)))
            numeric.append(pair)
        complete={}
        for arm in (('cold','reuse') if len(records)%2==0 else ('reuse','cold')):
            seed_all(index);begin=time.perf_counter();draft,inference=predict(model,case.solver_inputs)
            x,stats=optimize(draft,case.solver_inputs,**POLICY,workspace_reuse=arm=='reuse');elapsed=time.perf_counter()-begin
            complete[arm]={'seconds':elapsed,'metrics':evaluate_case(case,x.tolist(),elapsed)['metrics'],'positions':x.tolist(),'trace':stats['trace']}
        records.append({'index':index,'member':row['member'],'n':p.n,'numeric':numeric,'complete':complete})
        save(folder/'partial.json',records);print(index,p.n,'hits',sum(v['reuse']['stats']['workspace_hit'] for v in numeric),'Q',*[complete[k]['metrics']['cost'] for k in ('cold','reuse')],flush=True)
    cold=[v['cold']['stats']['setup_seconds'] for r in records for v in r['numeric'][1:]]
    reuse=[v['reuse']['stats']['setup_seconds'] for r in records for v in r['numeric'][1:]]
    report={'records':records,'summary':{'mean_cold_setup':float(np.mean(cold)),'mean_reuse_setup':float(np.mean(reuse)),
        'max_numeric_difference':max(v['max_abs_difference'] or 0 for r in records for v in r['numeric']),
        'numeric_hits':sum(v['reuse']['stats']['workspace_hit'] for r in records for v in r['numeric']),
        'full_hits':sum(t.get('workspace_hit',False) for r in records for t in r['complete']['reuse']['trace']),
        'hard_legal':{k:sum(r['complete'][k]['metrics']['is_feasible'] for r in records) for k in ('cold','reuse')}},
        'seconds':time.perf_counter()-started,'test_read':False}
    save(folder/'results.json',report);print(json.dumps(report['summary']),flush=True)

if __name__=='__main__':main()
