"""Frozen public-input C3 batch1 CUDA capture, independently owned by E99."""
import hashlib,time
from pathlib import Path
import numpy as np
import torch
from datasets.paths import ROOT
from .frozen_c3.student import DeploymentModel
from .frozen_c3.nn85.inference import graph
from .frozen_c3.nn85.base_model import decode

CHECKPOINT=ROOT/'experiments/E97_gpu_hybrid_solver/runs/20261102_candidate3_clearance_seed0/checkpoints/best.pt'
CHECKPOINT_SHA='d784e96f46decf7a487c25bde2ab16db8cf6ebae1bc0632434cc5e2f78173fce'


def padded(inputs):
    public=graph(inputs);n=int(inputs[0])
    if n>120 or len(inputs[4])>480:raise ValueError('Outside frozen C3 public shape bounds')
    result={}
    for k,v in public.items():
        if k=='scale':result[k]=v;continue
        shape=list(v.shape);shape[1]=120
        if k in ('adjacency','group','mib'):shape[2]=120
        result[k]=torch.zeros(shape,dtype=v.dtype)
        result[k][tuple(slice(0,s) for s in v.shape)].copy_(v)
    return result


def restore(rectangles,inputs):
    p=rectangles[0,:int(inputs[0])].cpu().numpy().astype(float)
    c,s=np.asarray(inputs[5]),np.asarray(inputs[6]);locked=c[:,:2].any(1)
    p[locked,2:]=s[locked,2:];p[c[:,1]!=0]=s[c[:,1]!=0]
    return p


class FrozenC3:
    @torch.inference_mode()
    def __init__(self,example):
        start=time.perf_counter()
        if hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()!=CHECKPOINT_SHA:raise ValueError('C3 checkpoint changed')
        saved=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
        self.model=DeploymentModel(**saved['model_config']).cuda().eval();self.model.load_state_dict(saved['model'],strict=True)
        self.model.requires_grad_(False);del saved
        self.g={k:v.cuda() for k,v in padded(example).items()}
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        def forward():return decode(self.model(self.g),self.g)*self.g['scale'][:,None,None]
        with torch.cuda.stream(stream):
            for _ in range(3):forward()
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        self.capture=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.capture,stream=stream):self.output=forward()
        torch.cuda.synchronize();self.cold_seconds=time.perf_counter()-start

    @torch.inference_mode()
    def predict(self,inputs):
        # Caller starts complete timer before this public feature constructor.
        host=padded(inputs)
        for k,v in host.items():self.g[k].copy_(v)
        self.capture.replay()
        return restore(self.output,inputs)

    @torch.inference_mode()
    def eager(self,inputs):
        g={k:v.cuda() for k,v in padded(inputs).items()}
        return restore(decode(self.model(g),g)*g['scale'][:,None,None],inputs)
