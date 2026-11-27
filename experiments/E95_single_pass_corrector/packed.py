"""Reusable pinned public-only batches; preserve source order and batch32."""
import torch

NODE=('features','mask','area','known','specs','constraints')
PAIR=('adjacency','mib','group')

class PublicPacker:
    def __init__(self,example,n=120,p=512,batch=32):
        self.n,self.p,self.batch=n,p,batch
        self.cpu={}
        for key in NODE:
            x=example['graph'][key]
            self.cpu[key]=torch.zeros((batch,n)+x.shape[1:],dtype=x.dtype,pin_memory=True)
        for key in PAIR:self.cpu[key]=torch.zeros(batch,n,n,pin_memory=True)
        self.cpu.update(valid=torch.zeros(batch,n,dtype=torch.bool,pin_memory=True),
            scale=torch.ones(batch,pin_memory=True),port_xy=torch.zeros(batch,p,2,pin_memory=True),
            port_valid=torch.zeros(batch,p,dtype=torch.bool,pin_memory=True),
            p2b_weights=torch.zeros(batch,n,p,pin_memory=True),b2b_weights=torch.zeros(batch,n,n,pin_memory=True))
        self.draft=torch.zeros(batch,n,3,pin_memory=True)

    def pack(self,rows):
        if not 0<len(rows)<=self.batch:raise ValueError('Invalid public batch')
        count=len(rows)
        for tensor in self.cpu.values():tensor.zero_()
        self.draft.zero_()
        # Tail monitor rows repeat the last valid row; only first count rows
        # contribute to arithmetic monitor means. No all-invalid fake case.
        selected=list(rows)+[rows[-1]]*(self.batch-count)
        for i,row in enumerate(selected):
            g=row['graph'];ports=row['public_ports'];n=g['n'];p=len(ports['port_xy'])
            if n>self.n or p>self.p:raise ValueError('Public shape exceeds registered packer')
            for key in NODE:self.cpu[key][i,:n].copy_(g[key])
            for key in PAIR:self.cpu[key][i,:n,:n].copy_(g[key])
            self.cpu['valid'][i,:n]=True;self.cpu['scale'][i]=g['scale']
            self.cpu['port_xy'][i,:p].copy_(ports['port_xy']);self.cpu['port_valid'][i,:p]=True
            self.cpu['p2b_weights'][i,:n,:p].copy_(ports['p2b_weights'])
            self.cpu['b2b_weights'][i,:n,:n].copy_(ports['b2b_weights'])
            self.draft[i,:n].copy_(row['draft'])
        return self.cpu,self.draft,count

    def gpu(self,rows):
        values,draft,count=self.pack(rows)
        n=max(r['graph']['n'] for r in rows);p=max(1,max(len(r['public_ports']['port_xy']) for r in rows))
        # Clone contiguous slices on the CPU: no transfers of global padding.
        sliced={}
        for key,v in values.items():
            if key in PAIR or key=='b2b_weights':v=v[:count,:n,:n]
            elif key=='p2b_weights':v=v[:count,:n,:p]
            elif key in ('port_xy','port_valid'):v=v[:count,:p]
            elif key=='scale':v=v[:count]
            else:v=v[:count,:n]
            sliced[key]=v.to('cuda',non_blocking=True)
        return sliced,draft[:count,:n].to('cuda',non_blocking=True),count
