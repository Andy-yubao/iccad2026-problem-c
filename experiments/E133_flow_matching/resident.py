"""Only fields needed by the FM network, augmentation and supervision are staged."""
import torch

class ResidentRows:
    def __init__(self,rows,device,max_nodes=120,**unused):
        self.rows=rows;self.device=torch.device(device);self.max_nodes=max_nodes
        self.keys=('features','mask','area','known','target','specs','reference','constraints','adjacency','group')
        self._buffers={};self._events={};self.row_bytes=0
    def batch(self,indices,out=None):
        indices=list(indices);count=len(indices)
        if count not in self._buffers:
            host={}
            for k in self.keys:
                v=self.rows[0]['graph'][k]
                shape=(self.max_nodes,self.max_nodes) if k in ('adjacency','group') else (self.max_nodes,*v.shape[1:])
                host[k]=torch.empty((count,*shape),dtype=v.dtype,pin_memory=self.device.type=='cuda')
            host['valid']=torch.empty((count,self.max_nodes),dtype=torch.bool,pin_memory=self.device.type=='cuda')
            host['scale']=torch.empty(count,dtype=torch.float32,pin_memory=self.device.type=='cuda')
            self._buffers[count]=host
            self.row_bytes=sum(v.numel()*v.element_size() for v in host.values())//count
        if count in self._events:self._events[count].synchronize()
        host=self._buffers[count]
        for v in host.values():v.zero_()
        for i,index in enumerate(indices):
            g=self.rows[int(index)]['graph'];n=g['n']
            for k in self.keys:
                if k in ('adjacency','group'):host[k][i,:n,:n].copy_(g[k])
                else:host[k][i,:n].copy_(g[k])
            host['valid'][i,:n]=True;host['scale'][i]=g['scale']
        if out is None:out={k:v.to(self.device,non_blocking=True) if self.device.type=='cuda' else v.clone() for k,v in host.items()}
        else:
            for k,v in host.items():out[k].copy_(v,non_blocking=self.device.type=='cuda')
        if self.device.type=='cuda':
            event=self._events.setdefault(count,torch.cuda.Event());event.record(torch.cuda.current_stream())
        return out
