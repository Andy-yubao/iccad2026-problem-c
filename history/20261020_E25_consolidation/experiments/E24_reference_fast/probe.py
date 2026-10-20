"""Five public E17 drafts check native legality and public-stop branches."""
import torch
from .cases import seed_all
from .io import ROOT,HERE,save,sources,snapshot
from .canonical_native import warmup
from .solver import optimize
from .run import CONFIGS
from scoring.evaluation import evaluate_case


def main():
    seed_all();warmup()
    cached=torch.load(ROOT/'experiments/E18_fast_postprocess/runs/public_dev500_v1.pt',weights_only=False)
    assert cached['checkpoint_sha256']=='3de22809e831edf11b56a85647a0bd9fe625e8b7617abb6eb1f4b27969cfe00a'
    folder=HERE/'runs/20261020_public_gate5';identity={'sources':sources(),'indices':[0,83,275,447,462],'test_read':False}
    save(folder/'config.json',identity);snapshot(folder/'source_snapshot.zip',identity);records=[]
    for index in identity['indices']:
        row=cached['rows'][index]
        for config in ['qp_gate','public_stop','steady','selection','steady_select']:
            seed_all(index)
            x,stats=optimize(row['draft'],row['case'].solver_inputs,**CONFIGS[config])
            metrics=evaluate_case(row['case'],x.tolist(),stats['seconds'])['metrics']
            assert metrics['is_feasible']
            for event in stats['trace']:
                if event['phase']=='public_stop':assert event['area_over_total']<=1.025 and event['wire_over_draft']<=1.
            records.append({'index':index,'config':config,'stats':stats,'metrics':metrics})
            print(index,config,round(metrics['cost'],5),round(stats['seconds'],3),stats['public_stopped'],flush=True)
            save(folder/'results.json',{'records':records,'test_read':False})


if __name__=='__main__':main()
