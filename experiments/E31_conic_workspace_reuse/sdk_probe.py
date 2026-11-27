"""Tiny convex toy only; test whether update preserves an enforceable deadline."""
from .resources import configure
RESOURCE=configure()
import json
from pathlib import Path
import time
import importlib.metadata as metadata
import numpy as np
import clarabel
from scipy.sparse import csc_matrix,eye

HERE=Path(__file__).resolve().parent

def main():
    folder=HERE/'runs/20261022_sdk_probe';folder.mkdir(parents=True,exist_ok=True)
    path=folder/'results.json'
    if path.exists():raise ValueError('Completed SDK evidence is immutable')
    settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_threads=2
    settings.time_limit=1.;settings.tol_feas=settings.tol_gap_abs=settings.tol_gap_rel=1e-10
    solver=clarabel.DefaultSolver(eye(2,format='csc'),np.array([-1.,-1.]),-eye(2,format='csc'),np.zeros(2),[clarabel.NonnegativeConeT(2)],settings)
    first=solver.solve();report={'clarabel':metadata.version('clarabel'),'resource':RESOURCE,
        'toy_only':True,'dataset_cases':0,'test_read':False,'first_status':str(first.status),'first_x':list(first.x),
        'available_methods':[x for x in dir(solver) if any(k in x for k in ('setting','update'))]}
    try:
        solver.update(q=np.array([-2.,-1.]))
        second=solver.solve();report.update(data_update_status=str(second.status),updated_x=list(second.x))
    except Exception as error:report['data_update_error']=repr(error)
    newer=clarabel.DefaultSettings();newer.verbose=False;newer.max_threads=2;newer.time_limit=.125
    newer.tol_feas=newer.tol_gap_abs=newer.tol_gap_rel=1e-10
    try:
        solver.update(settings=newer)
        actual=solver.get_settings();report['time_limit_after_update']=actual.time_limit
        report['settings_update_verified']=abs(actual.time_limit-.125)<1e-12 and actual.max_threads==2
    except Exception as error:report.update(settings_update_error=repr(error),settings_update_verified=False)
    updated=report.get('updated_x',[])
    report['continue_to_real_cases']=bool(report.get('settings_update_verified') and len(updated)==2 and np.allclose(updated,[2.,1.],atol=1e-7))
    if not report['continue_to_real_cases']:report['decision']='stop: cannot safely preserve per-step deadline through the verified update interface'
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');temporary.replace(path)
    print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
