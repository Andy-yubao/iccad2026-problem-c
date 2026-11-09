"""Saved500 local-structure diagnostics; no network forward or parameter gradients."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
from .io import HERE,read,save
from .collect import RUNS,key
from .data import load_rows
from .diagnostic_metrics import coverage_label

OUT=HERE/'runs/20261028_sparse_anchor_diagnosis'


def local_pairs(payload):
    member,inputs,reference,p=payload
    n,area,b2b,p2b,pins,c,s=inputs;c=np.asarray(c);s=np.asarray(s)
    a=np.where(c[:,:2].any(1),s[:,2:].prod(1),area);local=np.sqrt(a)
    pin=np.zeros(n,bool);adj=np.zeros((n,n),bool)
    for pi,bi,w in p2b:
        if w>0:pin[int(bi)]=True
    for i,j,w in b2b:
        if w>0:adj[int(i),int(j)]=adj[int(j),int(i)]=True
    rc=reference[:,:2]+reference[:,2:]/2;pc=p[:,:2]+p[:,2:]/2
    i,j=np.triu_indices(n,1);mutable=(c[i,1]==0)|(c[j,1]==0);i=i[mutable];j=j[mutable]
    size=(local[i]+local[j])/2
    rd=rc[i]-rc[j];pd=pc[i]-pc[j]
    margin=np.abs(rd)-(reference[i,2:]+reference[j,2:])/2
    near=np.linalg.norm(np.maximum(margin,0),axis=1)<=.25*size
    lo=np.maximum(p[i,:2],p[j,:2]);hi=np.minimum(p[i,:2]+p[i,2:],p[j,:2]+p[j,2:])
    overlap=np.maximum(hi-lo,0).prod(1)
    result={'member':member,'pin_coverage':float(pin.mean()),'pairs':{},
            'reference_near_count':int(near.sum()),
            'reference_near_public_edge_fraction':float(adj[i,j][near].mean()) if near.any() else None}
    for name,ids in (('pin_pin',pin[i]&pin[j]),('mixed',pin[i]^pin[j]),('no_pin_no_pin',~pin[i]&~pin[j])):
        z=ids&near;clear=(np.abs(rd)>.25*size[:,None])&z[:,None]
        flips=(pd*rd<0)&clear
        result['pairs'][name]={'pair_count':int(ids.sum()),'near_count':int(z.sum()),
            'near_relative_local_error':float((np.linalg.norm(pd-rd,axis=1)/size)[z].mean()) if z.any() else None,
            'near_clear_axis_count':int(clear.sum()),'near_clear_axis_flip_count':int(flips.sum()),
            'overlap_area_share':float(overlap[ids].sum()/max(overlap.sum(),1e-12))}
    return result


def main():
    if (OUT/'local_structure.json').exists():raise ValueError('Do not overwrite completed supplemental analysis')
    rows=load_rows('dev_monitor500');terminal={key(r['member']):r for r in read(RUNS['E65_geometry']/'terminal/results.json')['records']}
    payload=[]
    for r in rows:
        inputs=tuple(int(v) if i==0 else np.asarray(v) for i,v in enumerate(r['case'].solver_inputs))
        payload.append((list(r['member']),inputs,(r['graph']['reference']*r['graph']['scale']).numpy(),
                        np.asarray(terminal[key(r['member'])]['draft_positions'])))
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
        records=list(pool.map(local_pairs,payload,chunksize=8))
    summary={}
    for label in ('low','middle','high','all'):
        rs=[r for r in records if label=='all' or coverage_label(r['pin_coverage'])==label]
        values=[r['reference_near_public_edge_fraction'] for r in rs if r['reference_near_public_edge_fraction'] is not None]
        group={'count':len(rs),'reference_near_public_edge_fraction':float(np.mean(values)),'pairs':{}}
        for name in ('pin_pin','mixed','no_pin_no_pin'):
            pairs=[r['pairs'][name] for r in rs];values=[r['near_relative_local_error'] for r in pairs if r['near_relative_local_error'] is not None]
            group['pairs'][name]={'near_relative_local_error_case_mean':float(np.mean(values)) if values else None,
                 'near_axis_flip_fraction':sum(r['near_clear_axis_flip_count'] for r in pairs)/max(sum(r['near_clear_axis_count'] for r in pairs),1),
                 'overlap_area_share_case_mean':float(np.mean([r['overlap_area_share'] for r in pairs])),
                 'cases_with_near_pairs':len(values)}
        summary[label]=group
    save(OUT/'local_structure.json',{'records':records,'summary':summary,'new_inference':False,'test_read':False,
         'scope':'Saved E65 dev500 decoded drafts; reference-near pairs are label-assisted diagnosis, never network inputs'})
    print(summary)


if __name__=='__main__':main()
