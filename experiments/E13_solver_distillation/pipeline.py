"""Actual student inference + structure decode + E12-validated repair copy."""
import time
import numpy as np
import torch
from .student import Student
from .backend.frozen.inference import prepare_inference
from .backend.frozen.packing import prepare_problem
from .backend.frozen.solver import hard_valid
from .decode import decode
from .targets import decode_prediction
from .certificate import extract
from .backend.frozen.global_refine import topology
from .io import Worker

class Pipeline:
    def __init__(self,checkpoint,device,log):
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        self.device=device;self.structure=state['arm']=='structure';self.model=Student().to(device)
        self.model.load_state_dict(state['model']);self.model.eval();self.worker=Worker(log)
    def solve(self,inputs,oracle=None,teacher=None):
        if oracle and teacher is None:raise ValueError('Oracle diagnostic needs explicit teacher labels')
        begin=time.perf_counter();g=prepare_inference(inputs,self.device)
        with torch.no_grad():out={k:v[0].cpu().numpy() for k,v in self.model(g,self.structure).items()}
        inference=time.perf_counter()-begin;p=prepare_problem(*inputs)
        if oracle=='geometry':out['geometry']=np.asarray(teacher['geometry'])
        seed,cert,stats=decode(out,p,self.structure)
        if oracle=='structure':
            cert=extract(np.asarray(teacher['positions']),p)
            seed=decode_prediction(out['geometry'],p,cert['mib_labels']);cert['edges']=topology(seed,p.constraints)
        decode_seconds=time.perf_counter()-begin-inference
        request={'inputs':[int(inputs[0]),*[np.asarray(v).tolist() for v in inputs[1:]]],
            'draft':seed.tolist(),'guidance':cert,'seconds':.35,'joint':True,'polished':True}
        response=self.worker.call(request)
        if 'error' in response:raise RuntimeError(response['error'])
        final=np.asarray(response['positions']);assert hard_valid(final,p)
        return final,{'inference_seconds':inference,'decode_seconds':decode_seconds,
          'repair_ipc_seconds':response['worker_wall_seconds'],'end_to_end_seconds':time.perf_counter()-begin,
          'draft':seed.tolist(),'raw_strict_legal':hard_valid(seed,p),'decode':stats,'repair':response['stats'],
          'teacher_information_used':bool(oracle),'oracle':oracle}
    def close(self):self.worker.close()
