"""Same empty area, different ability to remove it: a constructive counterexample."""
from pathlib import Path
import sys
import json
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
from experiments.E42_joint_regularization.feedback_geometry import candidate,geometry,official_metrics,witness,synthetic_checks
torch.set_num_threads(1);torch.set_num_interop_threads(1)
p=np.array([[0.,0.,1.,1.],[1.1,0.,1.,1.],[0.,1.1,1.,1.]])
values=[]
for anchored in (False,True):
    c=torch.zeros((3,5));specs=torch.full((3,4),-1.)
    if anchored:
        c[:,1]=1;specs=torch.tensor(p,dtype=torch.float32)
    inputs=(3,torch.ones(3),torch.empty((0,3)),torch.empty((0,3)),torch.empty((0,2)),c,specs)
    out,info=candidate(p,inputs,.2,0.,'balanced')
    assert out is not None,info
    w=witness(p,out,inputs,geometry(p),official_metrics(p,inputs),.2,0.)
    assert w['accepted'],w
    values.append({'anchored':anchored,'initial_empty_fraction':geometry(p)['empty_fraction'],
                   'bbox_reduction_fraction':w['bbox_reduction_fraction'],'witness':w,'solver':info})
assert abs(values[0]['initial_empty_fraction']-values[1]['initial_empty_fraction'])<1e-12
assert values[0]['bbox_reduction_fraction']>.02 and abs(values[1]['bbox_reduction_fraction'])<1e-6
result={'passed':True,'pileup_checks':synthetic_checks(),'same_empty_area_different_removability':values}
Path(__file__).with_name('verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print([(r['anchored'],r['initial_empty_fraction'],r['bbox_reduction_fraction']) for r in values])
