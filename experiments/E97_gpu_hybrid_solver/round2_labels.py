"""Public-teacher group contact trees are Loss labels, excluded from NN inputs."""
from types import SimpleNamespace
import numpy as np
import torch
from .candidate3 import Packer as BasePacker,attach as base_attach
from .train_io.data import LEGACY
from .repair85_one_thread.fast_geometry import topology,contact_tree

def network_inputs(g):return {k:g[k] for k in LEGACY}

def attach(rows,labels,indices=None):
    base_attach(rows,labels,indices)
    for row in rows:
        g=row['graph'];t=row['teacher'].cpu().double().numpy()
        if not np.isfinite(t).all():raise ValueError('Nonfinite public teacher contact labels')
        area=g['area'].cpu().double().numpy();mask=g['mask'].cpu().numpy();spec=g['specs'].cpu().double().numpy()
        ratio=t[:,2].clip(-5,5);wh=np.c_[np.sqrt(area*np.exp(ratio)),np.sqrt(area/np.exp(ratio))]
        locked=~mask[:,2].astype(bool);placed=~mask[:,0].astype(bool);wh[locked]=spec[locked,2:]
        xy=np.maximum(t[:,:2]-wh/2,0);xy[placed]=spec[placed,:2];geometry=np.c_[xy,wh]
        c=np.asarray(row['case'].solver_inputs[5]);n=len(c);problem=SimpleNamespace(n=n,constraints=c)
        edges=np.asarray(contact_tree(geometry,problem,topology(geometry,c,array=True)),dtype=np.int64).reshape(-1,3)
        expected=sum(max(0,int((c[:,3]==v).sum())-1) for v in set(c[:,3])-{0})
        assert len(edges)==expected
        mib=sum(max(0,int((c[:,2]==v).sum())-1) for v in set(c[:,2])-{0})
        row['public_contact_edges']=torch.from_numpy(edges)
        row['public_soft_count']=int((c[:,4]>0).sum())+expected+mib
    return rows

class Packer(BasePacker):
    def __init__(self,row):
        super().__init__(row)
        self.contact_mask=torch.zeros(32,120,120,dtype=torch.bool,pin_memory=True)
        self.contact_x=torch.zeros_like(self.contact_mask,pin_memory=True)
        self.soft_count=torch.zeros(32,pin_memory=True)
    def pack(self,rows):
        g,d,t,u,count=super().pack(rows);self.contact_mask.zero_();self.contact_x.zero_();self.soft_count.zero_()
        for i,row in enumerate(list(rows)+[rows[-1]]*(32-count)):
            e=row['public_contact_edges']
            if len(e):
                a,b,axis=e.T;self.contact_mask[i,a,b]=True;self.contact_x[i,a,b]=axis==0
            self.soft_count[i]=row['public_soft_count']
        return dict(g,label_contact_mask=self.contact_mask,label_contact_x=self.contact_x,public_soft_count=self.soft_count),d,t,u,count
