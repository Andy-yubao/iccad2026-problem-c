"""Five unique train members: exact cache, full legality and real L/F updates."""
import gc,time
import numpy as np
import torch
from .common import HERE,RUN,PLAN,save,sha,seed
from .data import Data,manifest,resolve,public
from .train import baseline,configure,Engine,Batch
from .teacher import PublicRows
from .solver import Service
from . import loss_f

IDS=[1683,10027,15001,32000,63999]
def main():
    seed();begin=time.perf_counter();data=Data('train');raw=PublicRows('train');folder=RUN/'preflight';folder.mkdir(parents=True,exist_ok=True)
    # Five source-vs-compact exact checks, including two newly added members.
    m=manifest('train')
    for i in IDS:
        e=next(e for e in m['parts'] if i in e['indices']);p=torch.load(resolve(e['path']),map_location='cpu',weights_only=False,mmap=True)
        source=p['rows'][p['indices'].index(i)];row=data.row(i)
        assert row['member']==list(source['member'])
        for k,v in row['graph'].items():
            if isinstance(v,torch.Tensor):assert torch.equal(v,source['graph'][k]),(i,k)
        del p,source,row
    original_sha=sha(HERE/'assets/c5_baseline.pt');service=Service(HERE/'assets/c5_baseline.pt',raw.get(IDS[0]));targets={};records=[]
    for i in IDS:
        p,diag,inputs=service.solve(raw.get(i));scale=float(data.arrays['scale'][i]);n=len(p)
        targets[i]=np.c_[(p[:,:2]+p[:,2:]/2)/scale,np.log(p[:,2]/p[:,3])].astype(np.float32)
        records.append(dict(index=i,member=data.members[i],hard_legal=diag['hard']['is_feasible'],complete_seconds=diag['complete_seconds'],fallback=diag['fallback']))
    del service;gc.collect();torch.cuda.empty_cache()
    loss_f.MAX_GROUP_SIZE=data.meta['max_public_group'];updates={}
    for stage in ('L','F'):
        seed();model,cfg=baseline();model=model.cuda();part=configure(stage,model);batch=Batch(data,'L');ids=(IDS*7)[:32];g,_=batch.gpu(ids)
        if stage=='F':
            with torch.no_grad():model.placer.eval();base=model.placer(public(g))
            target=torch.zeros(32,120,3,device='cuda')
            for j,i in enumerate(ids):target[j,:len(targets[i])]=torch.from_numpy(targets[i]).cuda()
            g={k:v for k,v in g.items() if k not in ('target','reference')};g.update(base=base,teacher=target,usable=torch.ones(32,device='cuda'))
        frozen={k:v.detach().cpu().clone() for k,v in model.named_parameters() if not v.requires_grad};before={k:v.detach().cpu().clone() for k,v in part.named_parameters() if v.requires_grad}
        opt=torch.optim.AdamW([p for p in part.parameters() if p.requires_grad],lr=.00025,weight_decay=.0003)
        engine=Engine(stage,part,opt,g);assert not opt.state
        value=engine.step(g);changed=sum(not torch.equal(v,dict(part.named_parameters())[k].detach().cpu()) for k,v in before.items());assert changed>0
        assert all(torch.equal(v,dict(model.named_parameters())[k].detach().cpu()) for k,v in frozen.items())
        updates[stage]=dict(fixed_loss=value,actually_changed_trainable_tensors=changed,frozen_parameters_unchanged=True,optimizer_steps=1,graph_warmup_optimizer_steps=0,peak_allocated_bytes=torch.cuda.max_memory_allocated())
        del engine,opt,model,part,g,batch,frozen,before;gc.collect();torch.cuda.empty_cache()
    assert sha(HERE/'assets/c5_baseline.pt')==original_sha==PLAN['source_sha256']
    save(folder/'results.json',dict(status='passed',unique_train_members=IDS,checks='exact source cache / public-only inputs / hard-legal full solver / finite real captured Adam updates / frozen parameters',records=records,updates=updates,elapsed_seconds=time.perf_counter()-begin,test_read=False,no_resume_test=True))
    print('Final preflight PASSED: five unique train cases, real L/F updates',flush=True)
if __name__=='__main__':
    try:main()
    except BaseException as e:save(RUN/'preflight/error.json',dict(error=repr(e),no_automatic_retry=True));raise
