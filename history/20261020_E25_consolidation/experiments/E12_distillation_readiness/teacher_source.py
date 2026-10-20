"""Independent E12 adapter for the frozen E10 teacher and shared checkpoint."""
import time
import torch
from .io import ROOT,sha,np
from .backend.frozen.model import GeometryModel
from .backend.frozen.inference import prepare_inference
from .backend.frozen.batches import decode
from .backend.optimizer import optimize

CHECKPOINT=ROOT/'experiments/E10_integrated_baseline/runs/20261017_factorial_seed0/attention_geometry/checkpoints/best.pt'
CHECKPOINT_SHA256='47df667ed0b3994b60e554012255bcef1f7ec5cc76874b9e7493a30216f26c28'

class Teacher:
    def __init__(self,device='cpu'):
        if sha(CHECKPOINT)!=CHECKPOINT_SHA256:raise ValueError('Frozen teacher checkpoint mismatch')
        self.device=device;state=torch.load(CHECKPOINT,map_location=device,weights_only=False)
        self.model=GeometryModel(**state['model_config']).to(device)
        self.model.load_state_dict(state['model']);self.model.eval()
    def solve(self,inputs):
        begin=time.perf_counter();g=prepare_inference(inputs,self.device)
        with torch.no_grad():draft=(decode(self.model(g),g)[0]*g['scale'][0]).cpu().numpy().astype(float)
        c,spec=np.asarray(inputs[5]),np.asarray(inputs[6]);locked=c[:,:2].any(1)
        draft[locked,2:]=spec[locked,2:];draft[c[:,1]!=0]=spec[c[:,1]!=0]
        inference=time.perf_counter()-begin
        positions,stats=optimize(draft,inputs,seconds=5.)
        return positions,{'inference_seconds':inference,'end_to_end_seconds':time.perf_counter()-begin,
          'rescue_used':stats.get('rescue_used',False),'initial_row_retained':stats.get('initial_row_retained',False)}
