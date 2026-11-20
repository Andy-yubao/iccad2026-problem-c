"""Five existing train members; profile actual scratch updates, no dev/test."""
import time,gc,json
import numpy as np
import torch
from .common import RUN,HERE,save,seed,restore_rng
from .train import baseline,configure,Engine
from .data import Data,Packer
IDS=[1683,10027,15001,32000,63999]

def main():
    seed();saved=torch.load(RUN/'L/checkpoints/latest.pt',map_location='cpu',weights_only=False)
    model,cfg=baseline();part=configure('L',model);part.load_state_dict(saved['model']);model=model.cuda();opt=torch.optim.AdamW([p for p in part.parameters() if p.requires_grad],lr=.00025,weight_decay=.0003);opt.load_state_dict(saved['optimizer'])
    data=Data('train');packer=Packer(data);ids=(IDS*7)[:32];host,_=packer.pack(ids);sample={k:v.cuda() for k,v in host.items()};restore_rng(saved['rng']);engine=Engine('L',part,opt,sample);records=[]
    for repeat in range(5):
        torch.cuda.synchronize();start=time.perf_counter();host,_=packer.pack(ids);packed=time.perf_counter()
        for k,v in host.items():engine.g[k].copy_(v,non_blocking=True)
        torch.cuda.synchronize();copied=time.perf_counter();event0=torch.cuda.Event(enable_timing=True);event1=torch.cuda.Event(enable_timing=True);event0.record();engine.graph.replay();event1.record();torch.cuda.synchronize();computed=time.perf_counter()
        gate=torch.stack((engine.finite.float(),engine.values['total'].mean(),engine.norm)).cpu().tolist();assert gate[0]==1 and gate[1]<10 and gate[2]<10000;checked=time.perf_counter()
        engine.projection.before();before=time.perf_counter();opt.step();adam=time.perf_counter();engine.projection.after();torch.cuda.synchronize();projected=time.perf_counter()
        assert bool(torch.isfinite(torch.stack(torch._foreach_norm(engine.parameters))).all());done=time.perf_counter()
        records.append(dict(repeat=repeat,pack_ms=(packed-start)*1000,H2D_ms=(copied-packed)*1000,forward_backward_clip_ms=(computed-copied)*1000,graph_cuda_ms=event0.elapsed_time(event1),gate_ms=(checked-computed)*1000,projection_before_host_ms=(before-checked)*1000,adam_host_ms=(adam-before)*1000,projection_after_and_wait_ms=(projected-adam)*1000,postcheck_ms=(done-projected)*1000,complete_ms=(done-start)*1000))
    result=dict(unique_train_members=IDS,scratch_updates=5,formal_run_advanced=False,test_read=False,records=records,mean={k:float(np.mean([r[k] for r in records])) for k in records[0] if k!='repeat'})
    save(RUN/'engineering/profile_v1.json',result);print(json.dumps(result['mean'],indent=2),flush=True)
if __name__=='__main__':main()
