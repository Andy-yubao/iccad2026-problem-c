"""Mechanism check on at most five previously exposed development cases."""
import argparse
import json
import time
import torch
import numpy as np
from .data import ROOT, HERE, reader, derived_case, save
from .pipeline import Baseline
from .optimizer import optimize
from .rescue import refine, seed_pack, measures
from .frozen.packing import prepare_problem
from scoring.evaluation import evaluate_case


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--count',type=int,default=1)
    ap.add_argument('--name',default='20261019_mechanisms_dev5_seed0')
    args=ap.parse_args();assert 1<=args.count<=5
    torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0)
    legacy=json.loads((ROOT/'experiments/E10_integrated_baseline/runs/20261018_mean_area_mib100_seed0/results.json').read_text(encoding='utf-8'))
    rows=sorted(legacy['cases'],key=lambda x:-x['variants']['mean_area']['result']['metrics']['cost'])[:args.count]
    members=[]
    for row in rows:
        _,s,r=row['case_id'].split(':');members.append((s,int(r)))
    source=reader(members);solver=Baseline();out=HERE/'runs'/args.name;out.mkdir(parents=True,exist_ok=False)
    result=[]
    for i in range(len(source)):
        case,_=derived_case(source.case(i));p=prepare_problem(*case.solver_inputs)
        begin=time.perf_counter();draft=solver.draft(case.solver_inputs)
        baseline,stats=optimize(draft,case.solver_inputs,seconds=5.)
        stats['end_to_end_seconds']=time.perf_counter()-begin
        variants={'baseline':{'metrics':evaluate_case(case,baseline,stats['end_to_end_seconds'])['metrics'],'stats':stats}}
        for name,seed,mode,boundary,contact,hard in [
            ('draft_boundary',draft,1,40.,20.,False),
            ('incumbent_boundary',baseline,1,40.,20.,False),
            ('incumbent_hard',baseline,0,40.,20.,True),
            *[(f'pack{k}',seed_pack(draft,case.solver_inputs,k),1,40.,20.,False) for k in range(3)]]:
            if seed is None:
                variants[name]={'error':'invalid_seed'};continue
            x,ss=refine(seed,case.solver_inputs,seconds=5.,mode=mode,boundary=boundary,contact=contact,hard=hard)
            if x is None:
                variants[name]={'error':'no_legal_candidate','stats':ss}
            else:
                ev=evaluate_case(case,x,ss['seconds'])
                variants[name]={'metrics':ev['metrics'],'stats':ss,'positions':ev['positions'],
                                'public_measures':measures(x,p).tolist()}
            print(case.case_id,name,variants[name].get('metrics',{}).get('cost'),flush=True)
        result.append({'case_id':case.case_id,'variants':variants})
        save(out/'results.json',{'cases':result,'local_test_read':False,'scope':'Q-selected exposed failures; mechanism diagnosis only'})


if __name__=='__main__':main()
