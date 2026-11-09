"""Public-only full port distribution, normalized by the unchanged area scale."""
import torch


def public_ports(inputs, scale):
    n,_,_,p2b,pins,_,_=inputs
    all_coordinates=torch.as_tensor(pins,dtype=torch.float32)[:,:2]/scale
    edges=torch.as_tensor(p2b,dtype=torch.float32)
    edges=edges[~(edges==-1).all(1)]
    dense=torch.zeros(int(n),len(all_coordinates),dtype=torch.float32)
    if edges.numel():
        dense.index_put_((edges[:,1].long(),edges[:,0].long()),edges[:,2],accumulate=True)
    count=max(1,int((dense>0).sum(1).max()))
    coordinates=torch.zeros(int(n),count,2);weights=torch.zeros(int(n),count)
    valid=torch.zeros(int(n),count,dtype=torch.bool)
    for i in range(int(n)):
        ids=torch.nonzero(dense[i]>0).flatten();k=len(ids)
        coordinates[i,:k]=all_coordinates[ids];weights[i,:k]=dense[i,ids];valid[i,:k]=True
    if not torch.isfinite(coordinates).all() or not torch.isfinite(weights).all() or (weights<0).any():
        raise ValueError('Invalid public pin distribution')
    # Half each undirected contribution: summing the symmetric matrix counts
    # every original edge weight exactly once, including reversed duplicates.
    b2b=torch.as_tensor(inputs[2],dtype=torch.float32)
    b2b=b2b[~(b2b==-1).all(1)]
    wire=torch.zeros(int(n),int(n),dtype=torch.float32)
    if b2b.numel():
        i,j=b2b[:,:2].long().T;w=b2b[:,2]*.5
        wire.index_put_((i,j),w,accumulate=True);wire.index_put_((j,i),w,accumulate=True)
    return {'pin_coordinates':coordinates,'pin_weights':weights,
            'pin_valid':valid,'wire_b2b':wire}


def attach(row):
    if 'pin_coordinates' not in row['graph']:
        row['graph'].update(public_ports(row['case'].solver_inputs,row['graph']['scale']))
    return row
