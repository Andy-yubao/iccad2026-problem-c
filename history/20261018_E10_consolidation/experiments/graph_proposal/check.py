"""Three small E1 counterexamples, independent of any dataset tensor."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from experiments.graph_proposal.pipeline import correct, Optimizer
from experiments.graph_proposal.packing import prepare_problem
from experiments.graph_proposal.feasible import is_legal
from scoring.checks import independent_hard_checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    n=3
    areas=torch.tensor([6.,4.,1.])
    constraints=torch.zeros((n,5));constraints[0,0]=1;constraints[1,1]=1
    specs=torch.tensor([[-1.,-1.,3.,2.],[2.,2.,2.,2.],[-1.,-1.,-1.,-1.]])
    inputs=(n,areas,torch.empty((0,3)),torch.empty((0,3)),torch.empty((0,2)),constraints,specs)
    p=prepare_problem(*inputs)
    raw=np.array([[2.,2.,3.,2.],[2.,2.,2.,2.],[2.,2.,1.,1.]])
    revised=correct(raw,p)
    final=np.array(Optimizer(seconds=2).solve(*inputs))
    evidence={
        'nonsquare_fixed_dimensions_preserved':bool(np.array_equal(final[0,2:],[3.,2.])),
        'crowded_preplaced_anchor_preserved_in_correction':bool(np.array_equal(revised[1],specs[1].numpy())),
        'crowded_preplaced_anchor_preserved_in_final':bool(np.array_equal(final[1],specs[1].numpy())),
        'final_hard_legal':is_legal(final,p),
        'independent_hard_check':independent_hard_checks(final,areas,constraints,final),
    }
    # Two nonoverlapping modes differing only in left/right assignment.
    mode_a=np.array([[0.,0.,1.,1.],[1.,0.,1.,1.]])
    mode_b=np.array([[1.,0.,1.,1.],[0.,0.,1.,1.]])
    c=torch.zeros((2,5));a=torch.ones(2)
    average=(mode_a+mode_b)/2
    check_a=independent_hard_checks(mode_a,a,c,mode_a)
    check_b=independent_hard_checks(mode_b,a,c,mode_b)
    check_average=independent_hard_checks(average,a,c,mode_a)
    evidence['two_legal_modes']=check_a['is_feasible'] and check_b['is_feasible']
    evidence['coordinate_average_is_infeasible']=not check_average['is_feasible']
    evidence['averaged_overlap_pairs']=check_average['overlap_pairs']
    required=[v for v in evidence.values() if isinstance(v,bool)]
    assert all(required),evidence
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():raise FileExistsError(args.output)
    args.output.write_text(json.dumps(evidence,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(evidence,indent=2))


if __name__=='__main__':main()
