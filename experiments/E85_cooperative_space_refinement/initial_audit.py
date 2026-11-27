"""Public frozen initial-model diagnostics; no optimizer, solver, or Q scoring."""
import json,time
import torch
from .io import HERE,save,sources
from .data import load_rows
from .initialization import load,verify_protected
from .model import GeometryModel
from .arms import model_config
from .train import monitor_losses,seed_all
from .source_compatibility import reviewed_sources

def main():
    started=time.perf_counter();seed_all(0);rows=load_rows('dev');protected=verify_protected()
    for arm in ('control','cooperative'):
        path=HERE/('runs/initial_'+arm+'.json')
        if path.exists():
            if not reviewed_sources(json.loads(path.read_text(encoding='utf-8'))['sources'],sources()):raise ValueError('Initial audit source changed')
            continue
        model=GeometryModel(**model_config(arm)).cuda().eval();load(model,model_config(arm))
        values=monitor_losses(model,rows,'cuda',arm,0,32,return_records=True)
        records=values.pop('diagnostic_records')
        if arm=='control':
            from .io import ROOT
            old=json.loads((ROOT/'experiments/E83_strategy_attribution/runs/P74_frozen_F81/full_dev.json').read_text(encoding='utf-8'))['records']
            assert [r['member'] for r in old]==[r['member'] for r in records]
            error=max(abs(a['overlap_area_fraction']-b['overlap_area_fraction']) for a,b in zip(old,records))
            if error>2e-7:raise ValueError('Control does not reproduce saved P74+F81 raw geometry: '+str(error))
            values['prior_P74_F81_max_overlap_error']=error
        save(path,dict(role='zero-training operator effect; raw dev2000 only',arm=arm,losses=values,records=records,sources=sources(),seconds=time.perf_counter()-started,test_read=False,solver_calls=0))
        del model;torch.cuda.empty_cache();print('Frozen initial full-dev complete:',arm,flush=True)
    assert verify_protected()==protected
if __name__=='__main__':main()
