"""Five-case mechanism diagnostic on old public E17 drafts while E19 trains.

These shared-load measurements are not acceptance timing evidence. Reference
scores are read only by the external scorer after optimization has returned.
"""
import time
import torch
from .cases import seed_all
from .io import ROOT,HERE,save,sources,snapshot
from .solver import optimize
from scoring.evaluation import evaluate_case


def main():
    seed_all()
    rows=torch.load(ROOT/'experiments/E18_fast_postprocess/runs/public_dev500_v1.pt',weights_only=False)['rows']
    policies={'qp_first':{'quick_project':True,'quadratic_project':True,'incumbent_project':True,'warm_project':True}}
    folder=HERE/'runs/20261020_qpfirst5_e17_diagnostic';folder.mkdir(parents=True,exist_ok=True)
    identity={'sources':sources(),'indices':[21,127,275,447,456],
              'policies':policies,'timing':'diagnostic only; concurrent E19 training','test_read':False}
    save(folder/'config.json',identity);snapshot(folder/'source_snapshot.zip',identity)
    records=[]
    for index in identity['indices']:
        row=rows[index]
        for name,config in policies.items():
            seed_all(index)
            x,stats=optimize(row['draft'],row['case'].solver_inputs,adaptive=True,fast_budget=True,
                late_extension=True,robust_seeds=True,plateau=True,certified=True,snap=1e-4,**config)
            metrics=evaluate_case(row['case'],x.tolist(),stats['seconds'])['metrics']
            records.append({'index':index,'policy':name,'seconds':stats['seconds'],'metrics':metrics,'trace':stats['trace']})
            save(folder/'results.json',{'records':records,'test_read':False})
            print(index,name,round(metrics['cost'],6),round(stats['seconds'],3),flush=True)


if __name__=='__main__':main()
