"""C5 public-input resident GPU service and byte-preserved full CPU1 repair."""
import time
import numpy as np
import torch
from .common import PLAN,seed
from .model import DeploymentModel
from .nn.base_model import decode
from .nn.inference import graph
from .repair.solver import optimize
from .repair.canonical_native import warmup
from datasets.reference_layouts import mean_area_inputs,public_hard_reference
from scoring.checks import independent_hard_checks

def public_request(inputs):
    n=int(inputs[0]);values=[v.detach().cpu().clone() if isinstance(v,torch.Tensor) else torch.as_tensor(np.array(v,copy=True)) for v in inputs[1:]]
    values=[v.to(torch.float32) for v in values];c,s=values[4],values[5]
    locked=c[:,:2].bool().any(1);pre=c[:,1]!=0
    s[~locked]=-1;s[locked&~pre,:2]=-1
    return mean_area_inputs((n,*values))

def padded(inputs):
    original=graph(inputs);n=int(inputs[0]);assert n<=120
    result={}
    for k,v in original.items():
        if k=='scale':result[k]=v;continue
        shape=list(v.shape);shape[1]=120
        if k in ('adjacency','mib','group'):shape[2]=120
        result[k]=torch.zeros(shape,dtype=v.dtype);result[k][tuple(slice(0,x) for x in v.shape)].copy_(v)
    return result

def restored(output,inputs):
    n=int(inputs[0]);p=output[0,:n].cpu().numpy().astype(float);c,s=np.asarray(inputs[5]),np.asarray(inputs[6]);locked=c[:,:2].any(1)
    p[locked,2:]=s[locked,2:];p[c[:,1]!=0]=s[c[:,1]!=0];return p

class Captured:
    def __init__(self,fn,g):
        self.g={k:v.clone() for k,v in g.items()};s=torch.cuda.Stream();s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):fn(self.g)
        torch.cuda.current_stream().wait_stream(s);torch.cuda.synchronize();self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=s):self.output=fn(self.g)
        torch.cuda.synchronize()
    def replay(self):self.graph.replay();return self.output

class Service:
    @torch.inference_mode()
    def __init__(self,checkpoint,example):
        begin=time.perf_counter();seed();saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        self.model=DeploymentModel(**saved['model_config']).cuda().eval();self.model.load_state_dict(saved['model'],strict=True)
        public=public_request(example);g={k:v.cuda() for k,v in padded(public).items()};warmup()
        def fn(x):
            base=self.model.placer(x);state=self.model.refiner(base,x)
            return torch.stack((decode(base,x),decode(state,x)),dim=0)*x['scale'][None,:,None,None]
        self.cap=Captured(fn,g);self.cold_seconds=time.perf_counter()-begin
    @torch.inference_mode()
    def solve(self,raw_public_inputs):
        torch.cuda.synchronize();start=time.perf_counter();inputs=public_request(raw_public_inputs);host=padded(inputs)
        for k,v in host.items():self.cap.g[k].copy_(v)
        outputs=self.cap.replay();draft=restored(outputs[1],inputs);draft_seconds=time.perf_counter()-start
        positions,trace=optimize(draft,inputs,**PLAN['repair'])
        hard=independent_hard_checks(positions.tolist(),np.asarray(inputs[1]),np.asarray(inputs[5]),public_hard_reference(inputs[1],inputs[5],inputs[6]))
        assert hard['is_feasible'] and bool((positions[:,:2]>=0).all())
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        # Display-only L output transfer is outside the returned-request timer.
        original=restored(outputs[0],inputs)
        return positions,dict(complete_seconds=elapsed,draft_seconds=draft_seconds,repair_seconds=trace['seconds'],fallback=bool(trace['row_retained']),hard=hard,L_positions=original.tolist(),draft_positions=draft.tolist(),trace=trace['trace']),inputs
