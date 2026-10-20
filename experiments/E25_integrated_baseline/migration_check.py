"""At most five development cases; no training or sealed-test access."""
import json,time
import numpy as np
import torch
from .io import ROOT,HERE,sha,save
from .cases import prepare_cache,seed_all,predict,WEIGHTS
from .model import GeometryModel
from .run import CONFIGS
from .solver import optimize
from .canonical_native import warmup
from scoring.evaluation import evaluate_case

def main():
    from experiments.E23_capacity_geometry.cache import verify,CACHE
    # Reusing a registered data artifact is allowed; no E23 algorithm is imported.
    manifests={role:verify(role) for role in ['train','dev']}
    assert not (CACHE/'STOP').exists()
    formal=ROOT/'experiments/E23_capacity_geometry/runs/20261020_capacity160_v5_seed0'
    assert not (formal/'checkpoints/latest.pt').exists(), 'Do not silently migrate an active formal training run'
    assert not (formal/'STOP').exists()
    origin=ROOT/'history/20261020_E25_consolidation/experiments/E24_reference_fast'
    compared=[]
    for p in HERE.rglob('*.py'):
        relative=p.relative_to(HERE)
        if 'runs' in relative.parts or relative.as_posix() in {'cases.py','run.py','audit.py','migration_check.py'}:continue
        assert sha(p)==sha(origin/relative),str(relative)
        compared.append(relative.as_posix())
    saved=json.loads((HERE/'runs/20261020_controls_steady_monitor500/results.json').read_text())
    config=json.loads((HERE/'runs/20261020_controls_steady_monitor500/config.json').read_text())
    assert CONFIGS['steady']==config['config']
    assert sha(WEIGHTS)=='3de22809e831edf11b56a85647a0bd9fe625e8b7617abb6eb1f4b27969cfe00a'
    seed_all();cache=prepare_cache();state=torch.load(WEIGHTS,map_location='cpu',weights_only=False)
    model=GeometryModel(**state['model_config']).eval();model.load_state_dict(state['model'])
    started=time.perf_counter();warmup();initialization=time.perf_counter()-started
    records=[]
    for index in [0,83,275,447,462]:
        row=cache['rows'][index];draft,seconds=predict(model,row['case'].solver_inputs)
        difference=float(np.max(np.abs(draft-row['draft'])))
        assert np.allclose(draft,row['draft'],rtol=0,atol=1e-5),(index,difference)
        original=saved['records'][index]
        assert original['member']==row['member']
        metric=evaluate_case(row['case'],original['positions'],original['seconds'])['metrics']
        assert metric['is_feasible'] and abs(metric['cost']-original['metrics']['cost'])<1e-10
        seed_all(index);x,stats=optimize(draft,row['case'].solver_inputs,**CONFIGS['steady'])
        current=evaluate_case(row['case'],x.tolist(),stats['seconds']+seconds)['metrics']
        assert current['is_feasible']
        records.append({'index':index,'draft_max_abs_difference':difference,'saved_result_Q':metric['cost'],
                        'current_hard_legal':True,'current_Q':current['cost']})
        print('Migration case passed',index,flush=True)
    save(HERE/'runs/20261020_migration_check/results.json',{'status':'passed','cases':5,
        'records':records,'numeric_sources_exact':compared,'selected_config_exact':True,
        'checkpoint_sha256':sha(WEIGHTS),'E23_cache_verified':{role:len(m['members']) for role,m in manifests.items()},
        'E23_formal_training_started':False,'native_initialization_seconds':initialization,
        'scope':'Migration functionality only; not a replacement performance measurement','test_read':False})

if __name__=='__main__':main()
