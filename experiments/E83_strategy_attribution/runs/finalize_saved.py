"""Recover report/gallery serialization only; do not evaluate a model or solver."""
import sys,json,time
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from experiments.E83_strategy_attribution.io import HERE,sha,sources,save
from experiments.E83_strategy_attribution import analysis,render_drafts

def normalize(x):
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,np.ndarray):return normalize(x.tolist())
    if isinstance(x,dict):return {str(k):normalize(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [normalize(v) for v in x]
    return x
def safe_save(path,x):return save(path,normalize(x))

def main():
    start=time.perf_counter();state=json.loads((HERE/'state.json').read_text(encoding='utf-8'))
    assert all(a['status']=='terminal_complete' for a in state['arms'].values())
    immutable={}
    for name in state['arms']:
        folder=HERE/'runs'/name;meta=json.loads((folder/'frozen.json').read_text(encoding='utf-8'))
        assert meta['sources']==sources() and sha(folder/'composition.pt')==meta['checkpoint_sha256']
        for path in (folder/'full_dev.json',folder/'terminal/results.json',folder/'composition.pt'):immutable[str(path)]=sha(path)
    analysis.save=safe_save;render_drafts.save=safe_save
    analysis.main();render_drafts.main()
    assert all(sha(Path(p))==v for p,v in immutable.items())
    safe_save(HERE/'runs/serialization_recovery.json',{'reason':'NumPy int32 report serialization, followed by GBK console print error after report was saved; both terminal evaluations were already complete',
      'change':'Only JSON scalar conversion in runs wrapper; frozen production source/weights/results untouched',
      'new_model_forward_calls':0,'new_solver_calls':0,'new_Q_evaluations':0,'immutable_evidence_sha256':immutable,
      'seconds':time.perf_counter()-start,'utc':datetime.now(timezone.utc).isoformat()})
    state['pipeline_error_retained']=state.get('error');state['status']='complete_pending_visual_audit';state['postprocessing_recovery']='runs/serialization_recovery.json'
    state['recovery_finished_utc']=datetime.now(timezone.utc).isoformat();safe_save(HERE/'state.json',state)

if __name__=='__main__':main()
