"""Finite GPU batch rigid translations with exact public CPU certificates."""
import time,math
import numpy as np
import torch
from .repair85_one_thread.backend.frozen.feasible import Objective
from .repair85_one_thread.backend.frozen.solver import hard_valid
from .repair85_one_thread.canonical import canonicalize
from .repair85_one_thread.solver import raw_measures

def component_groups(x,p):
    comps=[];mapping=np.eye(p.n,dtype=bool)
    for group in Objective(p).groups:
        q=x[group];ov=np.minimum((q[:,:2]+q[:,2:])[:,None],(q[:,:2]+q[:,2:])[None])-np.maximum(q[:,:2][:,None],q[:,:2][None])
        touch=((ov[...,0]>=0)&(ov[...,1]>0))|((ov[...,0]>0)&(ov[...,1]>=0));unseen=set(range(len(group)));parts=[]
        while unseen:
            first=min(unseen);unseen.remove(first);todo=[first];ids=[]
            while todo:
                i=todo.pop();ids.append(i);near=unseen.intersection(np.flatnonzero(touch[i]).tolist());unseen-=near;todo+=sorted(near)
            ids=np.sort(group[ids]);parts.append(ids);mapping[np.ix_(ids,ids)]=True
        comps.append(parts)
    return comps,mapping

def proposals(x,p):
    groups,connections=component_groups(x,p);lo=x[:,:2].min(0);hi=(x[:,:2]+x[:,2:]).max(0);boundary=[];contact=[]
    for i in np.flatnonzero(p.constraints[:,4]):
        ids=np.flatnonzero(connections[i]);code=int(p.constraints[i,4]);shift=np.zeros(2)
        if p.preplaced[ids].any():continue
        if code&1:shift[0]=lo[0]-x[i,0]
        elif code&2:shift[0]=hi[0]-x[i,0]-x[i,2]
        if code&8:shift[1]=lo[1]-x[i,1]
        elif code&4:shift[1]=hi[1]-x[i,1]-x[i,3]
        if np.abs(shift).max()>1e-8:boundary.append((ids,shift))
    for parts in groups:
        if len(parts)<2:continue
        for ai,ids in enumerate(parts):
            if p.preplaced[ids].any():continue
            for bi,other in enumerate(parts):
                if ai==bi:continue
                for axis in (0,1):
                    t=1-axis
                    for direction in (0,1):
                        best=None
                        for i in ids:
                            for j in other:
                                eps=.001*min(x[i,t+2],x[j,t+2]);shift=np.zeros(2)
                                shift[t]=float(np.clip(0,x[j,t]-x[i,t]-x[i,t+2]+eps,x[j,t]+x[j,t+2]-x[i,t]-eps))
                                shift[axis]=x[j,axis]-x[i,axis]-x[i,axis+2] if direction==0 else x[j,axis]+x[j,axis+2]-x[i,axis]
                                key=(float(np.abs(shift).sum()),int(i),int(j))
                                if best is None or key<best[0]:best=(key,shift)
                        contact.append((ids,best[1]))
    contact.sort(key=lambda item:(float(np.abs(item[1]).sum()),tuple(item[0]),tuple(item[1])))
    all_proposals=(boundary+contact)[:256];mask=np.zeros((len(all_proposals),p.n),dtype=bool);shift=np.zeros((len(all_proposals),2))
    for k,(ids,delta) in enumerate(all_proposals):mask[k,ids]=True;shift[k]=delta
    return mask,shift,connections

class BatchGeometry:
    def __init__(self,p):
        self.p=p;self.obj=Objective(p);self.device='cuda'
        self.preplaced=torch.as_tensor(p.preplaced,device='cuda');self.codes=torch.as_tensor(p.constraints[:,4].astype(np.int64),device='cuda')
        self.bi=torch.as_tensor(self.obj.bi,device='cuda');self.pi=torch.as_tensor(self.obj.pi,device='cuda')
        self.bw=torch.as_tensor(p.b2b[:,2],device='cuda');self.pw=torch.as_tensor(p.p2b[:,2],device='cuda');self.pins=torch.as_tensor(p.pins[:,:2],device='cuda')
        self.groups=[torch.as_tensor(g,device='cuda') for g in self.obj.groups]

    @torch.no_grad()
    def evaluate(self,x,mask,shift,connections):
        original=torch.as_tensor(x,device='cuda');moved=torch.as_tensor(mask,device='cuda');delta=torch.as_tensor(shift,device='cuda');links=torch.as_tensor(connections.astype(np.float32),device='cuda')
        for _ in range(8):
            xy=original[None,:,:2]+moved[...,None]*delta[:,None];hi=xy+original[None,:,2:]
            overlap=torch.minimum(hi[:,:,None],hi[:,None,:])-torch.maximum(xy[:,:,None],xy[:,None,:])
            blocking=((overlap>1e-8).all(-1)&moved[:,:,None]&~moved[:,None,:]).any(1)
            moved=moved|((blocking.float()@links)>0)
        xy=original[None,:,:2]+moved[...,None]*delta[:,None];hi=xy+original[None,:,2:]
        overlap=torch.minimum(hi[:,:,None],hi[:,None,:])-torch.maximum(xy[:,:,None],xy[:,None,:]);n=len(x);ix=torch.arange(n,device='cuda')
        collision=(overlap>1e-8).all(-1)&(ix[:,None]!=ix[None,:])[None]
        legal=~collision.any((1,2))&~(xy<0).any((1,2))&~(moved&self.preplaced[None]&(delta.abs().sum(-1)>0)[:,None]).any(-1)
        low=xy.amin(1);high=hi.amax(1);area=(high-low).prod(-1);center=(xy+hi)/2;wire=torch.zeros_like(area)
        if len(self.bi):wire+=((center[:,self.bi[:,0]]-center[:,self.bi[:,1]]).abs().sum(-1)*self.bw[None]).sum(-1)
        if len(self.pi):wire+=((self.pins[self.pi[:,0]][None]-center[:,self.pi[:,1]]).abs().sum(-1)*self.pw[None]).sum(-1)
        distances=torch.stack((xy[...,0]-low[:,None,0],high[:,None,0]-hi[...,0],high[:,None,1]-hi[...,1],xy[...,1]-low[:,None,1]),-1)
        required=torch.stack(tuple((self.codes&bit)!=0 for bit in (1,2,4,8)),-1)
        soft=((distances.abs()>=1e-6)&required[None]).any(-1).sum(-1)
        for group in self.groups:
            o=overlap[:,group[:,None],group[None,:]];touch=((o[...,0]>=-1e-6)&(o[...,1]>1e-6))|((o[...,0]>1e-6)&(o[...,1]>=-1e-6));k=len(group);ids=torch.arange(k,device='cuda');labels=ids[None].expand(len(mask),-1)
            for _ in range(k-1):labels=torch.where(touch,labels[:,None,:],k).amin(-1)
            soft+=(labels==ids[None]).sum(-1)-1
        # WH are unchanged, so MIB is exactly the original constant in this rank proxy.
        soft+=self.obj.violations(x)[2]
        candidates=torch.cat((xy,original[None,:,2:].expand(len(mask),-1,-1)),-1)
        return candidates,area,wire,soft,legal

def polish(x,p):
    start=time.perf_counter();x=np.asarray(x).copy();obj=Objective(p);soft=sum(obj.violations(x));area,wire=raw_measures(x,p);trace=[];tested=0
    if not soft:return x,dict(seconds=time.perf_counter()-start,accepted=0,tested=0,rounds=[])
    batch=BatchGeometry(p)
    for _ in range(4):
        mask,delta,links=proposals(x,p)
        if not len(mask):break
        candidates,a,w,v,legal=batch.evaluate(x,mask,delta,links)
        factor=torch.maximum(torch.ones_like(a),torch.maximum(a/max(area,1e-15),w/max(wire,1e-15)))*torch.exp(2*(v-soft)/obj.nsoft)
        factor=factor.masked_fill(~legal|(v>=soft),float('inf'));rank=np.argsort(factor.cpu().numpy(),kind='stable')[:4];selected=candidates[torch.as_tensor(rank,device='cuda')].cpu().numpy();accepted=None
        for index,candidate in zip(rank,selected):
            tested+=1;candidate=canonicalize(candidate,p,tolerance=1e-6)
            if not hard_valid(candidate,p) or (candidate[:,:2]<0).any():continue
            new_soft=sum(obj.violations(candidate));new_area,new_wire=raw_measures(candidate,p)
            cert=max(1.,new_area/max(area,1e-15),new_wire/max(wire,1e-15))*math.exp(2*(new_soft-soft)/obj.nsoft)
            if new_soft<soft and cert<1-1e-10:
                accepted=(candidate,new_area,new_wire,new_soft);trace.append(dict(proposals=len(mask),rank_index=int(index),old_soft=int(soft),new_soft=int(new_soft),certificate=cert));break
        if accepted is None:break
        x,area,wire,soft=accepted
        if not soft:break
    torch.cuda.synchronize()
    return x,dict(seconds=time.perf_counter()-start,accepted=len(trace),tested=tested,rounds=trace)
