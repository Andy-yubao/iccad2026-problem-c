"""Parameter-identity cache and single depth-budget evaluation on current members."""
import hashlib
import json
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from ..generalization_baseline.model import GeometryModel
from ..batches import decode
from ..qualification.evaluate import public_graph
from ..depth_evaluation.solver import optimize
from ..depth_evaluation.preflight import sources as depth_sources
from ..io import ROOT,sha,save
from scoring.evaluation import evaluate_case
from .common import RUN,read,sources,q_summary

def model_hash(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        a=value.detach().cpu().contiguous().numpy()
        h.update(name.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()

def input_hash(inputs):
    h=hashlib.sha256()
    for value in inputs:
        a=np.ascontiguousarray(value)
        h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()

def fingerprint():
    import scipy,clarabel
    value=dict(sources=sources(),numpy=np.__version__,scipy=scipy.__version__,
        clarabel=clarabel.__version__,torch=str(torch.__version__),cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),threads=1)
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

class Predictor:
    def __init__(self,checkpoint_path,inputs):
        state=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
        self.model=GeometryModel(**state['model_config']).cuda().eval()
        self.model.load_state_dict(state['model'],strict=True)
        self.g={k:v.cuda() for k,v in public_graph(inputs).items()}
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):decode(self.model(self.g),self.g)
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=stream):
            self.output=decode(self.model(self.g),self.g)
        torch.cuda.synchronize()

    def predict(self,inputs):
        for key,value in public_graph(inputs).items():
            self.g[key].copy_(value)
        self.graph.replay()
        p=(self.output[0,:int(inputs[0])]*self.g['scale'][0]).cpu().numpy().astype(float)
        constraints,specs=np.asarray(inputs[5]),np.asarray(inputs[6])
        locked=constraints[:,:2].any(1);pre=constraints[:,1]!=0
        p[locked,2:]=specs[locked,2:];p[pre]=specs[pre]
        if not np.isfinite(p).all():raise RuntimeError('Nonfinite draft')
        return p

@torch.inference_mode()
def evaluate_panel(predictor,checkpoint_path,rows,folder,check,beat,stage,**fields):
    assert len(rows)==100 and len({tuple(row['member']) for row in rows})==100
    state=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    mh=model_hash(state['model']);fp=fingerprint();digest=sha(checkpoint_path)
    predictor.model.load_state_dict(state['model'],strict=True);predictor.model.eval()
    del state
    records=[];hits=0
    with threadpool_limits(limits=1):
        for i,row in enumerate(rows):
            check()
            ih=input_hash(row['case'].solver_inputs)
            key=hashlib.sha256((mh+ih+fp).encode()).hexdigest()
            cache_path=RUN/'evaluation_cache'/key[:2]/(key+'.json')
            if cache_path.exists():
                record=read(cache_path)
                assert record['model_hash']==mh and record['input_hash']==ih and record['fingerprint']==fp
                hits+=1
            else:
                torch.cuda.synchronize();start=time.perf_counter()
                draft=predictor.predict(row['case'].solver_inputs)
                draft_seconds=time.perf_counter()-start
                final,stats=optimize(draft,row['case'].solver_inputs)
                wall=time.perf_counter()-start
                scored=evaluate_case(row['case'],final,wall)
                if not scored['metrics']['is_feasible'] or not scored['independent_hard_checks']['is_feasible']:
                    raise RuntimeError('Hard legality failed; no automatic retry')
                record=dict(member=list(row['member']),n=int(row['case'].block_count),
                    metrics=scored['metrics'],independent_hard_checks=scored['independent_hard_checks'],
                    complete_seconds=wall,draft_seconds=draft_seconds,repair_seconds=stats['seconds'],
                    draft_positions=draft.tolist(),positions=final.tolist(),fallback=stats['row_retained'],
                    trace=stats['trace'],work=stats['work'],model_hash=mh,input_hash=ih,fingerprint=fp,
                    original_checkpoint_sha256=digest,test_read=False)
                save(cache_path,record)
            assert record['member']==list(row['member']) and record['metrics']['is_feasible']
            save(folder/f'{i:04d}.json',dict(cache_key=key,cache_record=cache_path.relative_to(ROOT).as_posix(),
                 model_hash=mh,current_checkpoint_sha256=digest,member=list(row['member'])))
            records.append(record)
            if (i+1)%10==0:beat(stage=stage,completed=i+1,total=100,**fields)
    result=q_summary(records)
    result.update(model_hash=mh,cache_hits=hits,actual_solver_calls=100-hits,
        latency_scope='Cached rows retain original measured times; feedback summaries are not a fresh latency benchmark')
    save(folder/'results.json',result)
    print('SCORE',stage,fields,json.dumps(result),flush=True)
    return result

def choose(scores,plan):
    margin=plan['minimum_absolute_gain_per_repeat']
    accepted=[a for a in ('plus','minus') if scores[a][0]['hard_legal']==100 and
              scores[a][0]['mixed_Q']<scores['control'][0]['mixed_Q']-margin]
    if not accepted:return 'control',accepted
    if len(accepted)==2 and abs(scores['plus'][0]['mixed_Q']-scores['minus'][0]['mixed_Q'])<=1e-12:
        return 'control',accepted
    return min(accepted,key=lambda a:scores[a][0]['mixed_Q']),accepted
