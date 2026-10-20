"""Five-case exact native/Python parity, including an int64 overflow guard."""
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import torch
from .canonical_py import canonicalize as python_canonicalize
from .canonical_native import canonicalize,warmup

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent


def main():
    torch.set_num_threads(2)
    folder=HERE/'runs/20261020_native_parity5_r2';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'results.json').exists():raise ValueError('Parity evidence already exists; use a new implementation version')
    rows=torch.load(ROOT/'experiments/E18_fast_postprocess/runs/public_dev500_v1.pt',weights_only=False)['rows']
    saved=json.loads((ROOT/'experiments/E18_fast_postprocess/runs/20261020_full4_robust_fast_dev500/results.json').read_text())['records']
    started=time.perf_counter();warmup();startup=time.perf_counter()-started;records=[]
    for index in [0,164,366,104,447]:
        inputs=rows[index]['case'].solver_inputs;c=np.asarray(inputs[5]);p=SimpleNamespace(n=int(inputs[0]),specs=np.asarray(inputs[6]),fixed=c[:,0]!=0,preplaced=c[:,1]!=0)
        for stage,x in [('raw',rows[index]['draft']),('legal',np.array(saved[index]['positions']))]:
            started=time.perf_counter();expected=python_canonicalize(x,p,1e-4);old_seconds=time.perf_counter()-started
            started=time.perf_counter();actual=canonicalize(x,p,1e-4);new_seconds=time.perf_counter()-started
            error=float(np.abs(expected-actual).max());equal=bool(np.array_equal(expected,actual))
            records.append({'index':index,'stage':stage,'exact':equal,'maximum_absolute_error':error,'python_seconds':old_seconds,'native_seconds':new_seconds})
            if not equal:
                (folder/'failure.json').write_text(json.dumps({'records':records,'jit_startup_seconds':startup},indent=2));raise AssertionError((index,stage,error))
    # Reuse case0 with huge public-input coordinates to exercise guarded fallback.
    x=rows[0]['draft'].copy();x[:,:2]+=1e10;inputs=rows[0]['case'].solver_inputs;c=np.asarray(inputs[5]);p=SimpleNamespace(n=int(inputs[0]),specs=np.asarray(inputs[6]),fixed=c[:,0]!=0,preplaced=c[:,1]!=0)
    assert np.array_equal(canonicalize(x,p,1e-4),python_canonicalize(x,p,1e-4))
    result={'passed':True,'case_count':5,'records':records,'jit_startup_seconds':startup,'overflow_fallback_exact':True,
            'test_read':False,'timing_scope':'single-call mechanism diagnostic, not final solver acceptance',
            'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}}
    (folder/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
