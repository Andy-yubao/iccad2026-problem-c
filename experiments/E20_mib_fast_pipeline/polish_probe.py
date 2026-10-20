"""Public certified block-cluster moves on five saved legal E17 outputs."""
import json
import numpy as np
import torch
from .io import ROOT,HERE,save,sources,snapshot
from .cases import seed_all
from .backend.frozen.packing import prepare_problem
from .certify_polish import polish
from scoring.evaluation import evaluate_case


def main():
    seed_all()
    rows=torch.load(ROOT/'experiments/E18_fast_postprocess/runs/public_dev500_v1.pt',weights_only=False)['rows']
    previous=json.loads((ROOT/'experiments/E18_fast_postprocess/runs/20261020_full4_robust_fast_dev500/results.json').read_text())['records']
    folder=HERE/'runs/20261020_push_polish5_e17_diagnostic'
    identity={'sources':sources(),'indices':[21,127,275,447,456],
              'seconds_budget':.03,'test_read':False,'timing':'diagnostic under concurrent training'}
    save(folder/'config.json',identity);snapshot(folder/'source_snapshot.zip',identity)
    records=[]
    for index in identity['indices']:
        case=rows[index]['case'];p=prepare_problem(*case.solver_inputs)
        x,stats=polish(np.array(previous[index]['positions']),p,seconds=.03,max_accept=8,push=True)
        metrics=evaluate_case(case,x.tolist(),stats['seconds'])['metrics']
        records.append({'index':index,'before':previous[index]['metrics'],'after':metrics,'stats':stats})
        print(index,previous[index]['metrics']['cost'],metrics['cost'],stats,flush=True)
    save(folder/'results.json',{'records':records,'test_read':False})


if __name__=='__main__':main()
