"""Exact bbox and rectangle union on saved drafts only; no inference or repair."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np

def geometry(p):
    """Exact axis-aligned rectangle union by x slabs; no raster approximation."""
    p=np.asarray(p,dtype=float);lo=p[:,:2];hi=lo+p[:,2:]
    xs=np.unique(np.r_[lo[:,0],hi[:,0]]);union=0.
    for left,right in zip(xs[:-1],xs[1:]):
        active=(lo[:,0]<right)&(hi[:,0]>left)
        y=np.column_stack((lo[active,1],hi[active,1]));y=y[np.argsort(y[:,0])]
        if not len(y):continue
        prev=np.r_[-np.inf,np.maximum.accumulate(y[:,1])[:-1]]
        length=np.maximum(0,y[:,1]-np.maximum(y[:,0],prev)).sum()
        union+=(right-left)*length
    area=p[:,2:].prod(1).sum();extent=hi.max(0)-lo.min(0);bbox=extent.prod()
    return {'A':float(area),'B':float(bbox),'U':float(union),
            'B_over_A':float(bbox/area),'U_over_A':float(union/area),
            'empty_over_A':float((bbox-union)/area),'empty_over_B':float((bbox-union)/bbox),
            'redundant_over_A':float((area-union)/area),
            'width':float(extent[0]),'height':float(extent[1])}


def summarize(rows):
    keys=('B_over_A','empty_over_A','empty_over_B','redundant_over_A')
    return {'count':len(rows),**{k:float(np.mean([r[k] for r in rows])) for k in keys}}


def compare(terminal,baseline,*,current_records=None):
    def one(row):
        return {'member':row['member'],'index':row['index'],**geometry(row['draft_positions'])}
    reference={tuple(r['member']):r for r in baseline}
    if current_records is None:
        with ThreadPoolExecutor(max_workers=4) as pool:
            current=list(pool.map(one,[r for r in terminal['records'] if 'draft_positions' in r and tuple(r['member']) in reference]))
    else:
        current=[r for r in current_records if tuple(r['member']) in reference]
    paired=[]
    for r in current:
        b=reference[tuple(r['member'])]
        if abs(r['A']/b['A']-1)>1e-5:raise ValueError('Saved geometry public area mismatch')
        paired.append({'member':r['member'],'bbox_relative_change':r['B']/b['B']-1,
                       'empty_over_A_delta':r['empty_over_A']-b['empty_over_A'],
                       'redundant_over_A_delta':r['redundant_over_A']-b['redundant_over_A']})
    return {'current':summarize(current),'reference':summarize([reference[tuple(r['member'])] for r in current]),
            'bbox_relative_change_mean':float(np.mean([r['bbox_relative_change'] for r in paired])),
            'bbox_larger':sum(r['bbox_relative_change']>1e-8 for r in paired),
            'current_records':current,'paired':paired,'workers':4,'scope':'Matched saved drafts, exact union; no new inference'}
