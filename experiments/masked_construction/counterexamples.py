"""Small direct counterexamples to the four E3 failure mechanisms."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.masked_construction.construction import (Problem, candidate_pool, grouping_components,
                                                        boundary_violations, Optimizer)
from scoring.checks import independent_hard_checks


def inputs(areas, c, specs):
    return (len(areas), np.array(areas), np.empty((0,3)), np.empty((0,3)), np.empty((0,2)), np.array(c), np.array(specs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    tests = {}
    c = np.zeros((4,5)); c[:,3]=1
    p = Problem(*inputs([1]*4, c, np.full((4,4),-1)))
    disconnected = [[0,0,1,1],[1,0,1,1],[4,0,1,1],[5,0,1,1]]
    comp = grouping_components(p, disconnected)
    assert comp == {'1': 2}
    tests['disconnected_tight_subgroups'] = dict(components=comp, expected_violation=1)
    c = np.zeros((2,5)); c[:,2]=1
    p = Problem(*inputs([1,2], c, np.full((2,4),-1)))
    assert not any(np.allclose(a,b) for a in p.shapes(0) for b in p.shapes(1))
    tests['incompatible_mib_areas'] = dict(same_shape_possible=False, original_areas=p.areas.tolist(),
                                        policy='soft mismatch remains; never change target area')
    c=np.zeros((2,5)); c[0,4]=2
    p=Problem(*inputs([1,1],c,np.full((2,4),-1)))
    violations=boundary_violations(p,[[0,0,1,1],[2,0,1,1]])
    assert violations==1
    tests['near_pin_not_final_right_boundary'] = dict(pin=[1,0], final_bbox_right=3, violations=violations)
    c=np.zeros((3,5)); c[1:,1]=1
    specs=np.array([[-1,-1,-1,-1],[0,0,1,2],[3,0,1,2]])
    p=Problem(*inputs([4,2,2],c,specs))
    # This prefix consumes the only 2x2 gap. Future fixed 2x2 can go on the shelf:
    # candidate framework is unbounded; it cannot detect a bounded-die deadlock.
    state={0:np.array([1,0,2,2]),1:specs[1],2:specs[2]}
    c2=np.zeros((4,5)); c2[:3]=c; c2[3,0]=1
    specs2=np.concatenate((specs,[[0,0,2,2]]))
    p2=Problem(*inputs([4,2,2,4],c2,specs2))
    pool,mask=candidate_pool(p2,3,state)
    bounded=mask & (pool[:,0]+pool[:,2]<=4) & (pool[:,1]+pool[:,3]<=2)
    assert not np.any(bounded) and np.any(mask)
    tests['greedy_prefix_blocks_bounded_gap'] = dict(bounded_die_candidate_count=int(bounded.sum()),
          unbounded_candidate_count=int(mask.sum()), conclusion='future bounded-envelope deadlock possible; shelf avoids hard infeasibility but may inflate bbox')
    c=np.zeros((3,5)); c[0,0]=1; c[1,1]=1
    specs=np.array([[-1,-1,3,1],[1,1,1,2],[-1,-1,-1,-1]])
    inp=inputs([3,2,2],c,specs)
    sol=Optimizer().solve(*inp)
    check=independent_hard_checks(sol,inp[1],inp[5],[[0,0,3,1],[1,1,1,2],[0,0,1,2]])
    assert check['is_feasible']
    tests['immutable_shape_anchor_obstacle'] = check
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(tests,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(tests,indent=2))


if __name__=='__main__':
    main()
