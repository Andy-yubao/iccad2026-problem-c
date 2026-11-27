"""CPU rows plus small pinned minibatches; never put the full32k on CUDA."""
import math
import torch


class ResidentRows:
    def __init__(self, rows, device, max_nodes=None, max_pins=None):
        self.rows = rows
        self.device = torch.device(device)
        self.max_nodes = max_nodes or max(r['graph']['n'] for r in rows)
        self.max_pins=max_pins or max(1,max(r['graph']['pin_coordinates'].shape[1] for r in rows))
        self.indices = {id(row): i for i, row in enumerate(rows)}
        self.shapes = {}
        self.dtypes = {}
        for key in ('features', 'mask', 'area', 'known', 'target', 'specs', 'reference', 'constraints',
                    'adjacency', 'group', 'wire_b2b'):
            first = rows[0]['graph'][key]
            self.shapes[key]=((self.max_nodes,self.max_nodes) if key in ('adjacency','group','wire_b2b')
                              else (self.max_nodes,*first.shape[1:]))
            self.dtypes[key]=first.dtype
        self.shapes.update(pin_coordinates=(self.max_nodes,self.max_pins,2),
                           pin_weights=(self.max_nodes,self.max_pins),pin_valid=(self.max_nodes,self.max_pins),
                           scale=(),valid=(self.max_nodes,))
        self.dtypes.update(pin_coordinates=torch.float32,pin_weights=torch.float32,pin_valid=torch.bool,
                           scale=rows[0]['graph']['scale'].dtype,valid=torch.bool)
        self.row_bytes=sum(torch.empty((),dtype=self.dtypes[k]).element_size()*math.prod(s)
                           for k,s in self.shapes.items())
        self.bytes=len(rows)*self.row_bytes  # hypothetical old padded allocation, for diagnostics only
        self._buffers={}
        self._transfers={}

    def batch(self, indices, out=None):
        if isinstance(indices,torch.Tensor):
            if indices.device.type!='cpu':raise ValueError('Host minibatches require CPU indices')
            indices=indices.tolist()
        else:indices=list(indices)
        count=len(indices)
        if count not in self._buffers:
            self._buffers[count]={k:torch.empty((count,*shape),dtype=self.dtypes[k],
                pin_memory=self.device.type=='cuda') for k,shape in self.shapes.items()}
        if count in self._transfers:self._transfers[count].synchronize()
        host=self._buffers[count]
        for value in host.values():value.zero_()
        for i,index in enumerate(indices):
            g=self.rows[index]['graph'];n=g['n'];pins=g['pin_coordinates'].shape[1]
            for key,value in host.items():
                if key=='valid':value[i,:n]=True
                elif key=='scale':value[i].copy_(g[key])
                elif key in ('adjacency','group','wire_b2b'):value[i,:n,:n].copy_(g[key])
                elif key in ('pin_coordinates','pin_weights','pin_valid'):value[i,:n,:pins].copy_(g[key])
                else:value[i,:n].copy_(g[key])
        if out is None:
            out={k:v.to(self.device,non_blocking=True) if self.device.type=='cuda' else v.clone()
                 for k,v in host.items()}
        else:
            for k,value in host.items():out[k].copy_(value,non_blocking=self.device.type=='cuda')
        if self.device.type=='cuda':
            event=self._transfers.setdefault(count,torch.cuda.Event())
            event.record(torch.cuda.current_stream(self.device))
        return out

    def row_indices(self, rows):
        return [self.indices[id(r)] for r in rows]
