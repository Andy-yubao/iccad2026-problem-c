"""Read-only C4 saved-edge diagnostic: unique native labels versus admissible axes."""
import json
import numpy as np
from .candidate4 import HERE,teacher_rows,attach,decode85
from .train_io.data import load_rows,collate
from .train_io.io import save

def main():
    labels=teacher_rows('dev_monitor500');rows=attach(load_rows('dev_monitor500'),labels);folder=HERE/'runs/20261102_candidate4_relations_seed0';records=[]
    for i,row in enumerate(rows):
        g=collate([row],'cpu');n=row['graph']['n'];t=row['teacher'][None];p=decode85(t,g)[0,:n].numpy();center=t[0,:n,:2].numpy();wh=p[:,2:]
        local=g['area'][0,:n].numpy()**.5;free_shape=g['mask'][0,:n,2].numpy().astype(bool);free_center=g['mask'][0,:n,0].numpy().astype(bool)
        lo_wh=wh*np.where(free_shape,np.sqrt(.99),1)[:,None];a,b=np.tril_indices(n,-1);active=free_center[a]|free_center[b]
        delta=center[b]-center[a];extent=(wh[a]+wh[b])/2;min_extent=(lo_wh[a]+lo_wh[b])/2
        score=np.abs(delta)/np.maximum(extent,1e-8)-1;native=np.argmax(score,1);allowed=np.abs(delta)>=min_extent
        near=(np.maximum(np.abs(delta)-extent,0)**2).sum(1)**.5<=.25*(local[a]+local[b])/2
        record=dict(index=i,n=n,ambiguous_fraction=float(allowed[active].all(1).mean()),near_ambiguous_fraction=float(allowed[active&near].all(1).mean()) if (active&near).any() else 0.,no_admissible_pairs=int((~allowed.any(1)&active).sum()),arms={})
        for arm in ['evaluation','control']:
            r=json.loads((folder/arm/'terminal'/f'{i:04d}.json').read_text(encoding='utf-8'));edge=np.array(r['relation_edges']);u,v,axis=edge.T
            # Stored edges follow the same unordered triangular pair order.
            assert np.array_equal(np.maximum(u,v),a) and np.array_equal(np.minimum(u,v),b)
            physical=np.abs(delta)[np.arange(len(a)),axis]>=min_extent[np.arange(len(a)),axis]
            signed=(center[v,axis]-center[u,axis])>=min_extent[np.arange(len(a)),axis]
            rate=lambda z,m:float(z[m].mean()) if m.any() else 0.
            record['arms'][arm]=dict(fallback=r['fallback'],Q=r['metrics']['cost'],wrong_unique_axis_rate=rate(axis!=native,active),inadmissible_axis_rate=rate(~physical,active),bad_signed_separation_rate=rate(~signed,active),near_bad_signed_separation_rate=rate(~signed,active&near))
        records.append(record)
    summary=dict(count=500,mean_ambiguous_fraction=float(np.mean([r['ambiguous_fraction'] for r in records])),mean_near_ambiguous_fraction=float(np.mean([r['near_ambiguous_fraction'] for r in records])),pairs_without_admissible_axis=sum(r['no_admissible_pairs'] for r in records),arms={arm:{k:float(np.mean([r['arms'][arm][k] for r in records])) for k in ['wrong_unique_axis_rate','inadmissible_axis_rate','bad_signed_separation_rate','near_bad_signed_separation_rate']} for arm in ['evaluation','control']})
    save(HERE/'runs/candidate4_relation_label_audit.json',dict(summary=summary,records=records,scope='saved geometry/edges and existing public teacher only; no new NN or solver calls; area bound sqrt(.99) from official free-area tolerance, locked WH exact',test_read=False));print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
