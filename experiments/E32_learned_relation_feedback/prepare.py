"""Resumable input-only replay trajectories; reference tensors are never consulted."""
from .resources import configure
RESOURCE=configure()
import argparse
import datetime as dt
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from .data import load_rows
from .cases import WEIGHTS,predict,graph,seed_all
from .model import GeometryModel
from .backend.frozen.packing import prepare_problem,_dimensions
from .fast_geometry import topology,step
from .native_reduce import reduce_edges,warmup
from .relations import encode_positions
from .io import HERE,ROOT,save,checkpoint,sha,sources,snapshot,exclusive
from .cache import CACHE,verify

FOLDER=HERE/'runs/prepared_relation_replay_v1'

def replay(base,row):
    inputs=row['case'].solver_inputs;draft,_=predict(base,inputs);p=prepare_problem(*inputs)
    w,h=_dimensions(p,draft[:,2]/draft[:,3],True);current=np.column_stack((draft[:,:2],w,h));states=[draft.copy()];trace=[]
    for iteration in range(3):
        full=topology(current,p.constraints,array=True);edges=reduce_edges(full,p.n)
        result,stats=step(p,current,edges,seconds=.10,full_edges=full,elastic=100000.*(1+iteration/3))
        if result is not None:current=result
        trace.append({k:stats[k] for k in ('status','seconds','iterations')})
        if iteration in (0,2):states.append(current.copy())
    scale=float(row['graph']['scale']);states=torch.as_tensor(np.stack(states)/scale,dtype=torch.float32)
    return torch.cat((states[...,:2]+states[...,2:]/2,(states[...,2]/states[...,3].clamp_min(1e-12)).log()[...,None]),-1),trace

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    started=time.perf_counter();deadline=dt.datetime.fromisoformat('2026-10-22T19:39:00+08:00').timestamp()
    seed_all(0);warmup();FOLDER.mkdir(parents=True,exist_ok=True)
    keys=['model.py','cases.py','features.py','batches.py','fast_geometry.py','native_reduce.py','prepare.py','relations.py']
    identity={'revision':'E32.input_only_three_conic_steps.v1','local_sources':{k:sha(HERE/k) for k in keys},
        'base_checkpoint_sha256':sha(WEIGHTS),'cache_identity':verify('train')['identity'],'resources':{k:v for k,v in RESOURCE.items() if k!='pid'},'steps':3,'step_seconds':.10,'slots':[0,1,3]}
    with exclusive(FOLDER/'prepare.lock'):
        prior=json.loads((FOLDER/'progress.json').read_text()).get('cumulative_seconds',0.) if (FOLDER/'progress.json').exists() else 0.
        elapsed=lambda:prior+time.perf_counter()-started
        if (FOLDER/'identity.json').exists():
            if not args.resume:raise ValueError('Use --resume for existing trajectory preparation')
            if json.loads((FOLDER/'identity.json').read_text())!=identity:raise ValueError('Replay source identity changed')
        else:
            save(FOLDER/'identity.json',identity);snapshot(FOLDER/'source_snapshot.zip',{'sources':sources()})
        base_state=torch.load(WEIGHTS,map_location='cpu',weights_only=False);base=GeometryModel(**base_state['model_config']).eval();base.load_state_dict(base_state['model'])
        from .cases import prepare_cache
        with torch.inference_mode():base=torch.jit.optimize_for_inference(torch.jit.freeze(torch.jit.trace(base,(graph(prepare_cache()['rows'][0]['case'].solver_inputs),),strict=False)))
        for role in ('train','dev'):
            rows=load_rows(role);folder=FOLDER/role;folder.mkdir(exist_ok=True);parts=[]
            for start in range(0,len(rows),128):
                path=folder/f'part_{start//128:04d}.pt'
                if path.exists():
                    piece=torch.load(path,map_location='cpu',weights_only=False)
                    if piece['identity']!=identity or piece['members']!=[list(r['member']) for r in rows[start:start+128]]:raise ValueError('Replay chunk mismatch')
                else:
                    if (FOLDER/'STOP').exists() or time.time()>=deadline or elapsed()>=7200:
                        save(FOLDER/'progress.json',{'status':'paused','role':role,'next':start,'cumulative_seconds':elapsed(),'test_read':False});return
                    values=[];traces=[]
                    for offset,row in enumerate(rows[start:start+128]):
                        seed_all(start+offset);state,trace=replay(base,row);values.append(state);traces.append(trace)
                    piece={'identity':identity,'members':[list(r['member']) for r in rows[start:start+128]],'states':values,'traces':traces}
                    checkpoint(path,piece)
                parts.append({'name':path.name,'start':start,'count':len(piece['members']),'size':path.stat().st_size,'mtime_ns':path.stat().st_mtime_ns})
                save(FOLDER/'progress.json',{'status':'running','role':role,'completed':start+len(piece['members']),'cumulative_seconds':elapsed(),'test_read':False})
                if len(parts)%5==0:print(role,start+len(piece['members']),len(rows),round(time.perf_counter()-started,1),flush=True)
            save(folder/'manifest.json',{'identity':identity,'members':[list(r['member']) for r in rows],'count':len(rows),'parts':parts})
        save(FOLDER/'results.json',{'status':'complete','train':16000,'dev':2000,'cumulative_seconds':elapsed(),'test_read':False,'reference_used_by_generation':False})
        print('Replay preparation complete',flush=True)

def load_states(role):
    if role not in ('train','dev'):raise PermissionError('Tests remain sealed')
    folder=FOLDER/role;manifest=json.loads((folder/'manifest.json').read_text());out={}
    for part in manifest['parts']:
        path=folder/part['name'];st=path.stat()
        if (st.st_size,st.st_mtime_ns)!=(part['size'],part['mtime_ns']):raise ValueError('Trajectory artifact changed')
        piece=torch.load(path,map_location='cpu',weights_only=False)
        if piece['identity']!=manifest['identity']:raise ValueError('Trajectory identity mismatch')
        for member,state in zip(piece['members'],piece['states']):out[tuple(member)]=state
    if len(out)!=manifest['count']:raise ValueError('Replay duplicate/missing members')
    return out,sha(folder/'manifest.json')

if __name__=='__main__':main()
