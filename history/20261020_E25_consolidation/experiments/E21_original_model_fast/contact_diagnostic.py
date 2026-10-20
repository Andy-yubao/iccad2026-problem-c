"""Public-geometry diagnostic of residual disconnected groups in saved outputs."""
import json
import torch
import numpy as np
from scipy.sparse.csgraph import connected_components
from .io import ROOT,HERE,save


def main():
    rows=torch.load(ROOT/'experiments/E18_fast_postprocess/runs/public_dev500_v1.pt',weights_only=False)['rows']
    records=json.loads((ROOT/'experiments/E18_fast_postprocess/runs/20261020_full4_robust_fast_dev500/results.json').read_text())['records']
    distances=[]
    for row,record in zip(rows,records):
        x=np.array(record['positions']);groups=np.asarray(row['case'].solver_inputs[5])[:,3]
        for g in np.unique(groups):
            if g<=0:continue
            ids=np.flatnonzero(groups==g);rect=x[ids];lo=rect[:,:2];hi=lo+rect[:,2:]
            overlap=np.minimum(hi[:,None],hi[None,:])-np.maximum(lo[:,None],lo[None,:])
            touch=((overlap[:,:,0]>=0)&(overlap[:,:,1]>0))|((overlap[:,:,0]>0)&(overlap[:,:,1]>=0))
            count,labels=connected_components(touch,directed=False)
            if count<2:continue
            other=labels[:,None]!=labels[None,:]
            for axis in (0,1):
                candidate=other&(overlap[:,:,1-axis]>1e-4)
                if candidate.any():distances.append(float(np.min(np.abs(overlap[:,:,axis][candidate]))))
    values=np.array(distances)
    result={'disconnected_group_axis_minima':len(values),'distance_percentiles':np.percentile(values,[0,10,50,90,100]).tolist(),
            'near_1e_4':int((values<1e-4).sum()),'near_1e_3':int((values<1e-3).sum()),'near_1e_2':int((values<1e-2).sum()),
            'scope':'public geometry only, not a solver timing or reference score comparison','test_read':False}
    save(HERE/'runs/20261020_contact_gap_diagnostic/results.json',result);print(result)


if __name__=='__main__':main()
