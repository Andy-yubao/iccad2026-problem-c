"""Exact teacher WH is Loss-only; strict legacy public NN input filter."""
import torch
from .candidate3 import Packer as BasePacker,attach as base_attach
from .round2_connectivity import network_inputs

def attach(rows,labels,indices=None):
    base_attach(rows,labels,indices)
    selected=list(range(len(rows))) if indices is None else indices
    for row,index in zip(rows,selected):
        wh=labels[index]['teacher_wh'];assert wh.shape==(row['graph']['n'],2) and bool(torch.isfinite(wh).all()) and bool((wh>0).all())
        groups=row['graph']['constraints'][:,3]
        assert all(int((groups==value).sum())<=10 for value in groups.unique() if value>0)
        row['teacher_wh']=wh
    return rows

class Packer(BasePacker):
    def __init__(self,row):
        super().__init__(row);self.wh=torch.zeros(32,120,2,pin_memory=True)
    def pack(self,rows):
        g,d,t,u,count=super().pack(rows);self.wh.zero_()
        for i,row in enumerate(list(rows)+[rows[-1]]*(32-count)):self.wh[i,:row['graph']['n']].copy_(row['teacher_wh'])
        return dict(g,teacher_wh=self.wh),d,t,u,count
