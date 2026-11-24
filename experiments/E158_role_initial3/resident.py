"""Original pinned batching plus compact static structural/port fields."""
import torch
from .base_resident import ResidentRows as Base

class ResidentRows(Base):
    def __init__(self,rows,device,max_nodes=None,max_pins=None,max_ports=None):
        super().__init__(rows,device,max_nodes,max_pins)
        self.max_ports=max_ports or max(1,max(len(r['role']['port_features']) for r in rows))
        n=self.max_nodes;m=self.max_ports;t=n+m
        self.extra_shapes=dict(role_features=(n,16),port_features=(m,4),port_valid=(m,),role_hops=(n,t),role_edges=(n,t),role_group=(n,t))
        self.extra_types={k:torch.float32 for k in self.extra_shapes};self.extra_types.update(port_valid=torch.bool,role_hops=torch.int16)
        self.extra_buffers={};self.extra_events={}
        self.row_bytes+=sum(torch.empty((),dtype=self.extra_types[k]).element_size()*__import__('math').prod(s) for k,s in self.extra_shapes.items())

    def batch(self,indices,out=None):
        ids=indices.tolist() if isinstance(indices,torch.Tensor) else list(indices);count=len(ids)
        result=super().batch(ids,out=out)
        if count not in self.extra_buffers:
            self.extra_buffers[count]={k:torch.empty((count,*shape),dtype=self.extra_types[k],pin_memory=self.device.type=='cuda') for k,shape in self.extra_shapes.items()}
        if count in self.extra_events:self.extra_events[count].synchronize()
        host=self.extra_buffers[count]
        for k,v in host.items():v.fill_(-1) if k=='role_hops' else v.zero_()
        limit=self.max_nodes
        for j,index in enumerate(ids):
            row=self.rows[index];g=row['graph'];r=row['role'];n=g['n'];m=len(r['port_features'])
            if m>self.max_ports:raise ValueError('Ports may not be truncated')
            host['role_features'][j,:n]=torch.from_numpy(r['role_features'])
            host['port_features'][j,:m]=torch.from_numpy(r['port_features']);host['port_valid'][j,:m]=True
            host['role_edges'][j,:n,:n]=g['wire_b2b']*2
            host['role_edges'][j,:n,limit:limit+m]=torch.from_numpy(r['port_edges'])
            host['role_group'][j,:n,:n]=g['group']
            hops=torch.from_numpy(r['hops'])
            for a,b,c,d in ((0,n,0,n),(0,n,limit,limit+m)):
                ra=0 if a==0 else n;rc=0 if c==0 else n
                host['role_hops'][j,a:b,c:d]=hops[ra:ra+b-a,rc:rc+d-c]
        for k,v in host.items():
            if out is None:result[k]=v.to(self.device,non_blocking=self.device.type=='cuda') if self.device.type=='cuda' else v.clone()
            else:result[k].copy_(v,non_blocking=self.device.type=='cuda')
        if self.device.type=='cuda':
            event=self.extra_events.setdefault(count,torch.cuda.Event());event.record(torch.cuda.current_stream(self.device))
        return result
