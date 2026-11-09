"""Unchanged frozen-baseline public graph / CUDA inference path, locally copied."""
import hashlib
import numpy as np
import torch
from ...sources.e98_generalization.model import GeometryModel
from ...features import graph_features
from ...batches import decode
def public_graph(inputs,pad=120):
    g=graph_features(inputs);n=int(inputs[0]);scale=g['scale']
    c=torch.as_tensor(inputs[5]).long();spec=g['specs'];locked=g['locked']
    area=torch.where(locked,spec[:,2:].prod(-1),g['areas'])/scale.square()
    known=torch.cat(((spec[:,:2]+spec[:,2:]/2)/scale,
                     (spec[:,2].clamp_min(1e-8)/spec[:,3].clamp_min(1e-8)).log()[:,None]),1)*(1-g['mask'])
    data=dict(features=g['features'],adjacency=g['adjacency'],mib=g['relations'][0],group=g['relations'][1],
              mask=g['mask'],area=area,known=known,specs=torch.where(locked[:,None],spec/scale,torch.zeros_like(spec)),
              constraints=c)
    out={}
    for key,value in data.items():
        shape=(pad,pad) if key in ('adjacency','mib','group') else (pad,*value.shape[1:])
        out[key]=torch.zeros((1,*shape),dtype=value.dtype)
        if key in ('adjacency','mib','group'):out[key][0,:n,:n]=value
        else:out[key][0,:n]=value
    out['valid']=torch.arange(pad)[None,:]<n
    out['scale']=scale[None]
    return out

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
