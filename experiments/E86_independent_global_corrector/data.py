"""Full public connectivity/ports added beside byte-identical legacy cache features."""
import torch
from .dataset_view import ReferenceLayouts
from .batches import collate as legacy_collate, sample

LEGACY=('features','adjacency','mib','group','mask','area','known','specs','constraints','scale','valid')
EXTRA=('port_xy','port_valid','p2b_weights','b2b_weights')
PUBLIC=(*LEGACY,*EXTRA)

def inputs_only(g):return {k:g[k] for k in PUBLIC}
def placer_inputs(g):return {k:g[k] for k in LEGACY}

def augment(row):
    n,_,b2b,p2b,pins,_,_=row['case'].solver_inputs
    n=int(n);pins=torch.as_tensor(pins,dtype=torch.float32)
    b2b=torch.as_tensor(b2b,dtype=torch.float32);p2b=torch.as_tensor(p2b,dtype=torch.float32)
    xy=pins[:,:2]/row['graph']['scale'];pw=torch.zeros(n,len(pins));bw=torch.zeros(n,n)
    for edges,port in ((b2b,False),(p2b,True)):
        edges=edges[~(edges==-1).all(1)]
        if not len(edges):continue
        a,b=edges[:,0].long(),edges[:,1].long();w=edges[:,2]
        if port:pw.index_put_((b,a),w,accumulate=True)
        else:
            bw.index_put_((a,b),w,accumulate=True);bw.index_put_((b,a),w,accumulate=True)
    return dict(row,public_ports=dict(port_xy=xy,p2b_weights=pw,b2b_weights=bw))

def load_rows(role,indices=None,*,use_cache=True):
    dataset=ReferenceLayouts(role)
    chosen=list(range(len(dataset))) if indices is None else list(indices)
    if use_cache:
        from .cache import load_rows as cached
        rows=cached(role,chosen)
    else:
        from datasets.reference_layouts import public_hard_reference
        rows=[]
        for index in chosen:
            item=dataset[index];case=item['case'];g=sample(case.solver_inputs,torch.as_tensor(item['positions']).float())
            case.reference_positions=public_hard_reference(case.solver_inputs[1],case.solver_inputs[5],case.solver_inputs[6])
            rows.append(dict(graph=g,case=case,member=item['member']))
    return [augment(row) for row in rows]

def collate(rows,device):
    g=legacy_collate([r['graph'] for r in rows],list(range(len(rows))),device)
    b,n=g['valid'].shape;p=max(1,max(len(r['public_ports']['port_xy']) for r in rows))
    xy=torch.zeros(b,p,2);valid=torch.zeros(b,p,dtype=torch.bool)
    pw=torch.zeros(b,n,p);bw=torch.zeros(b,n,n)
    for i,row in enumerate(rows):
        values=row['public_ports'];nn=row['graph']['n'];pp=len(values['port_xy'])
        xy[i,:pp]=values['port_xy'];valid[i,:pp]=True
        pw[i,:nn,:pp]=values['p2b_weights'];bw[i,:nn,:nn]=values['b2b_weights']
    g.update(port_xy=xy.to(device),port_valid=valid.to(device),p2b_weights=pw.to(device),b2b_weights=bw.to(device))
    return g
