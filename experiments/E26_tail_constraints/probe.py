"""Five already exposed dev cases, saved incumbent control, no network updates."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[name]='1'
import json,time,argparse
import torch,numpy as np
from .io import ROOT,HERE,save,sources,snapshot,sha
from .backend.frozen.packing import prepare_problem
from .backend.frozen.feasible import Objective
from .canonical_native import warmup
from .tail_repair import repair
from scoring.evaluation import evaluate_case

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--flexible',action='store_true');ap.add_argument('--rewire',action='store_true');ap.add_argument('--shape',action='store_true');ap.add_argument('--compact',action='store_true');args=ap.parse_args()
    torch.set_num_threads(1)
    source=ROOT/'experiments/E25_integrated_baseline/runs/20261020_controls_steady_monitor500/results.json'
    saved=json.loads(source.read_text(encoding='utf-8'))['records']
    cache=torch.load(ROOT/'experiments/E25_integrated_baseline/runs/public_dev500_v1.pt',map_location='cpu',weights_only=False)
    folder=HERE/('runs/20261020_compact5' if args.compact else ('runs/20261020_shape5' if args.shape else ('runs/20261020_rewire5' if args.rewire else ('runs/20261020_flexible5' if args.flexible else 'runs/20261020_coordinated5'))));folder.mkdir(parents=True,exist_ok=False)
    identity={'sources':sources(),'indices':[164,21,447,456,496],'source_sha256':sha(source),'seconds_per_case':.25,
              'solver_threads':1,'E23_concurrent_training':True,'test_read':False,'cached_incumbent_only':True,'flexible_contact':args.flexible,'rewire':args.rewire,'shape':args.shape,'compact':args.compact}
    save(folder/'config.json',identity);snapshot(folder/'source_snapshot.zip',identity)
    begin=time.perf_counter();warmup();save(folder/'initialization.json',{'native_warmup_seconds':time.perf_counter()-begin})
    records=[]
    for index in identity['indices']:
        case=cache['rows'][index]['case'];p=prepare_problem(*case.solver_inputs);old=saved[index]
        x=np.asarray(old['positions']);counts=list(Objective(p).violations(x))
        assert counts==[old['metrics'][k] for k in ['boundary_violations','grouping_violations','mib_violations']]
        candidate,stats=repair(x,p,seconds=.25,flexible=args.flexible,rewire=args.rewire,shape=args.shape,compact=args.compact)
        metrics=evaluate_case(case,candidate.tolist(),old['seconds']+stats['seconds'])['metrics']
        assert metrics['is_feasible'] and metrics['cost']<=old['metrics']['cost']+1e-10
        records.append({'index':index,'n':p.n,'before':old['metrics'],'after':metrics,'stats':stats,'positions':candidate.tolist()})
        save(folder/'results.json',{'status':'running','records':records,'test_read':False})
        print(index,round(old['metrics']['cost'],6),'->',round(metrics['cost'],6),'accepted',stats['accepted'],'attempts',stats['attempts'],'seconds',round(stats['seconds'],4),flush=True)
    save(folder/'results.json',{'status':'complete','records':records,'test_read':False,
        'scope':'Mechanism check on five selected exposed dev cases; not a full-panel quality or speed claim'})
if __name__=='__main__':main()
