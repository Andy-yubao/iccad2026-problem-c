"""Select once from frozen train public structure, before model inference."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from .io import HERE,save
from .cache import load_rows
from .ports import attach
import json

def public_stats(item):
    index,row=item
    n,_,b2b,p2b,_,_,_=row['case'].solver_inputs
    n=int(n)
    b=np.asarray(b2b); b=b[~(b==-1).all(1)]
    p=np.asarray(p2b); p=p[~(p==-1).all(1)]
    port=np.zeros(n,dtype=bool)
    if len(p):port[p[:,1].astype(int)]=True
    degree=np.zeros(n,dtype=int)
    if len(b):
        positive=b[b[:,2]>0]
        degree=np.bincount(positive[:,:2].astype(int).ravel(),minlength=n)
    return {'index':index,'member':list(row['member']),'n':n,
            'port_fraction':float(port.mean()),'isolated_fraction':float((degree==0).mean()),
            'max_degree':int(degree.max())}

def select():
    frozen=HERE/'runs/sentinels.json'
    if frozen.exists():
        from .data import load_rows as prepared_rows
        value=json.loads(frozen.read_text(encoding='utf-8'))
        ids=[r['index'] for r in value['records']]
        if value['status']!='frozen_before_inference' or len(ids)!=5 or len(set(ids))!=5 or ids[0]!=1032:
            raise ValueError('Invalid frozen sentinel panel')
        rows=prepared_rows('train',ids)
        for record,row in zip(value['records'],rows):
            actual=public_stats((record['index'],row))
            if any(actual[k]!=record[k] for k in actual):raise ValueError('Frozen public structure drift')
        return rows
    rows=load_rows('train')  # Cached immutable rows; only public solver_inputs scanned.
    def chunk(items):return [public_stats(item) for item in items]
    with ThreadPoolExecutor(max_workers=4) as pool:
        chunks=list(pool.map(chunk,[list(enumerate(rows))[i::4] for i in range(4)]))
    stats=sorted((r for part in chunks for r in part),key=lambda r:r['index'])
    chosen=[dict(stats[1032],reason='known E91 forward amplification counterexample')]
    for reason,key in [
        ('lowest port coverage; larger n breaks ties',lambda x:(x['port_fraction'],-x['n'],x['index'])),
        ('most isolated modules; larger n breaks ties',lambda x:(-x['isolated_fraction'],-x['n'],x['index'])),
        ('largest unnormalised graph degree',lambda x:(-x['max_degree'],-x['n'],x['index'])),
        ('largest module count',lambda x:(-x['n'],x['index']))]:
        used={x['index'] for x in chosen}
        row=min((r for r in stats if r['index'] not in used),key=key)
        chosen.append(dict(row,reason=reason))
    save(HERE/'runs/sentinels.json',{'status':'frozen_before_inference','selection':'public-only; first32k; no Q/free-reference/model-based selection','workers':4,'records':chosen,'test_read':False})
    return [attach(rows[x['index']]) for x in chosen]
