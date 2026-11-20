"""Same-depth paired evaluation; cache identical models without timing-based choices."""
import hashlib
import json
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from ..generalization_baseline.model import GeometryModel
from ..batches import decode
from ..qualification.evaluate import public_graph
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


from ..depth_evaluation.solver import optimize
from ..qualification.metrics import module_overlap
from scoring.evaluation import evaluate_case
from scoring.checks import independent_hard_checks
from datasets.reference_layouts import public_hard_reference
from .common import RUN,sources,read,save,sha,q_summary

def fingerprint():
    import scipy,clarabel
    env=dict(sources=sources(),numpy=np.__version__,scipy=scipy.__version__,
        clarabel=clarabel.__version__,torch=str(torch.__version__),cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),threads=1)
    return hashlib.sha256(json.dumps(env,sort_keys=True).encode()).hexdigest()

@torch.inference_mode()
def evaluate_panel(predictor,checkpoint_path,rows,folder,check,beat,stage,**fields):
    count=500
    assert len(rows)==count and count in (100,500)
    assert len({tuple(r['member']) for r in rows})==count
    state=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    mh=model_hash(state['model']);digest=sha(checkpoint_path);fp=fingerprint()
    predictor.model.load_state_dict(state['model'],strict=True);predictor.model.eval()
    del state
    records=[];hits=0
    with threadpool_limits(limits=1):
        for i,row in enumerate(rows):
            check()
            inputs=row['case'].solver_inputs
            ih=input_hash(inputs)
            key=hashlib.sha256((mh+ih+fp+repr(tuple(row['member']))).encode()).hexdigest()
            cache=RUN/'evaluation_cache'/key[:2]/(key+'.json')
            if cache.exists():
                record=read(cache);hits+=1
                assert record['model_hash']==mh and record['input_hash']==ih and record['fingerprint']==fp
            else:
                torch.cuda.synchronize();start=time.perf_counter()
                draft=predictor.predict(inputs)
                draft_seconds=time.perf_counter()-start
                final,info=optimize(draft,inputs)
                hard=independent_hard_checks(final.tolist(),np.asarray(inputs[1]),np.asarray(inputs[5]),
                    public_hard_reference(inputs[1],inputs[5],inputs[6]))
                complete=time.perf_counter()-start
                if not hard['is_feasible']:raise RuntimeError('Independent hard legality failed')
                score=evaluate_case(row['case'],final,complete)
                raw=evaluate_case(row['case'],draft,draft_seconds)
                assert score['metrics']['is_feasible']
                record=dict(member=list(row['member']),n=int(row['case'].block_count),model_hash=mh,
                    input_hash=ih,fingerprint=fp,original_checkpoint_sha256=digest,
                    metrics=score['metrics'],independent_hard_checks=hard,raw_metrics=raw['metrics'],
                    raw_overlap=module_overlap(draft),draft_positions=draft.tolist(),positions=final.tolist(),
                    complete_seconds=complete,draft_seconds=draft_seconds,repair_seconds=info['seconds'],
                    fallback=info['row_retained'],trace=info['trace'],work=info['work'],test_read=False)
                save(cache,record)
            assert record['member']==list(row['member'])
            save(folder/f'{i:04d}.json',dict(cache_key=key,cache_record=cache.as_posix(),model_hash=mh,
                current_checkpoint_sha256=digest,member=record['member']))
            records.append(record)
            if (i+1)%10==0:
                beat(stage=stage,completed=i+1,total=count,**fields)
                print(stage,i+1,'/',count,flush=True)
    summary=q_summary(records)
    summary.update(model_hash=mh,cache_hits=hits,actual_solver_calls=count-hits,
        raw_hard_legal=sum(r['raw_metrics']['is_feasible'] for r in records),
        latency_scope='Fresh rows include input/transfer/network/repair/independent hard check; cache hits retain original timing, not a new latency measurement')
    save(folder/'results.json',summary)
    return summary


def choose(scores,plan):
    margin=plan['minimum_absolute_gain_per_repeat']
    accepted=[a for a in ('plus','minus') if scores[a][0]['hard_legal']==plan['feedback_cases_per_round'] and
              scores[a][0]['mixed_Q']<scores['control'][0]['mixed_Q']-margin]
    if not accepted:return 'control',accepted
    if len(accepted)==2 and abs(scores['plus'][0]['mixed_Q']-scores['minus'][0]['mixed_Q'])<=1e-12:
        return 'control',accepted
    return min(accepted,key=lambda a:scores[a][0]['mixed_Q']),accepted
