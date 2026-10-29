"""Saved-coordinate coverage decomposition, not a causal allocation of Loss gains."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.E84_loss_control.io import HERE,save
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def union(p,clip=None):
    p=np.asarray(p,dtype=float);lo=p[:,:2];hi=lo+p[:,2:]
    if clip is not None:
        lo=np.maximum(lo,clip[0]);hi=np.minimum(hi,clip[1]);mask=(hi>lo).all(1);lo=lo[mask];hi=hi[mask]
    xs=np.unique(np.r_[lo[:,0],hi[:,0]]);value=0.
    for l,r in zip(xs[:-1],xs[1:]):
        active=(lo[:,0]<r)&(hi[:,0]>l);end=-np.inf;height=0.
        for b,t in sorted(zip(lo[active,1],hi[active,1])):height+=max(0.,t-max(b,end));end=max(end,t)
        value+=(r-l)*height
    return float(value)
def main():
    a=read(HERE/'runs/20261029_same_initialization_G_only_seed0/terminal/results.json')['records']
    b=read(ROOT/'experiments/E81_multisource_refinement/runs/20261029_e42_multisource_seed0/terminal/results.json')['records']
    assert [r['member'] for r in a]==[r['member'] for r in b];rows=[]
    for g,f in zip(a,b):
        p=np.array(g['draft_positions']);q=np.array(f['draft_positions']);area=p[:,2:].prod(1).sum();lo=p[:,:2].min(0);hi=(p[:,:2]+p[:,2:]).max(0)
        ug=union(p);uf=union(q);within=union(q,(lo,hi));inside=(within-ug)/area;outside=(uf-within)/area
        rows.append({'index':g['index'],'member':g['member'],'union_gain_over_A':(uf-ug)/area,'within_G_only_actual_bbox_gain_over_A':inside,'outside_G_only_actual_bbox_coverage_over_A':outside})
    total=sum(r['union_gain_over_A'] for r in rows);inside=sum(r['within_G_only_actual_bbox_gain_over_A'] for r in rows)
    save(HERE/'runs/completion/coverage_decomposition.json',{'scope':'Existing500 saved geometry; full E81 minus G-only E84. Both constrained to identical source P42 prediction frames.',
      'mean_union_gain_over_A':total/500,'mean_within_G_only_bbox_gain_over_A':inside/500,'mean_outside_G_only_bbox_coverage_over_A':(total-inside)/500,
      'fraction_summed_union_gain_within_G_only_actual_bbox':inside/total,'positive_within_bbox_gain_cases':sum(r['within_G_only_actual_bbox_gain_over_A']>1e-8 for r in rows),
      'limits':'Descriptive rectangle-union decomposition, not causal Loss attribution. Outside coverage uses additional parts of the same allowed source frame. Union gain is not identical to pairwise-intersection reduction.',
      'records':rows,'new_model_solver_Q_calls':0,'test_read':False})
    print({'union_gain_mean':total/500,'inside_gain_mean':inside/500,'outside_gain_mean':(total-inside)/500,'inside_fraction':inside/total})
if __name__=='__main__':main()
