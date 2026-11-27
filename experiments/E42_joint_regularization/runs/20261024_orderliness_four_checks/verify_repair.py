"""Two mechanism controls: movable overlap, and a tiny overlap pinned against an anchor."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from experiments.E42_joint_regularization.orderliness import fixed_shape,shape_repair,verify,save

def inputs(p,anchor=False):
    c=torch.zeros((len(p),5));spec=torch.full((len(p),4),-1.)
    if anchor:c[0,1]=1;spec[0]=torch.as_tensor(p[0],dtype=torch.float32)
    return (len(p),torch.as_tensor(p[:,2:].prod(1),dtype=torch.float32),torch.empty((0,3)),torch.empty((0,3)),torch.empty((0,2)),c,spec)

p=np.array([[0.,0.,1.,1.],[.9,0.,1.,1.],[3.,2.,1.,1.]])
first=[]
for flexible in (False,True):
    out,stats=fixed_shape(p,inputs(p),flexible)
    assert out is not None,stats
    out[:,:2]=np.maximum(out[:,:2],0.)
    proof=verify(p,out,inputs(p));assert proof['hard_legal'] and proof['quality_guard'],proof
    assert proof['max_move']<=.10001
    first.append({'flexible':flexible,'stats':stats,'proof':proof})
p=np.array([[0.,1.,1.,1.],[0.,0.,1.,1.02],[3.,3.,1.,1.]])
out,stats=fixed_shape(p,inputs(p,True),True);assert out is None and stats['status']==2
changed,shape_stats=shape_repair(p,inputs(p,True));assert changed is not None,shape_stats
changed[:,:2]=np.maximum(changed[:,:2],0.)
proof=verify(p,changed,inputs(p,True));assert proof['hard_legal'] and proof['quality_guard'],proof
save(Path(__file__).with_name('verification.json'),{'passed':True,'fixed_and_flexible_repair_positive_control':first,
     'anchor_case_fixed_shape_infeasible':stats,'anchor_case_shape_change_witness':proof,'shape_solver':shape_stats})
print('PASS: joint movement succeeds in positive control; anchored tiny overlap requires shape or larger displacement; conic shape witness is legal.')
