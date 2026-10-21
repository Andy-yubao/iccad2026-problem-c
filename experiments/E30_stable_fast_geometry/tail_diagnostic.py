"""One disclosed worst-case counterexample; no sealed data or solver labels."""
from .resources import configure
RESOURCE=configure()
import json
import time
from .io import HERE,save,sources,snapshot
from .cases import prepare_cache
from .solver import optimize
from .backend.robust import fast
from scoring.evaluation import evaluate_case

def main():
    folder=HERE/'runs/20261022_tail_counterexample'
    if (folder/'results.json').exists():raise ValueError('Completed diagnostic is immutable')
    saved=json.loads((HERE/'runs/20261022_matched_steady_monitor500/results.json').read_text())
    worst=max(saved['records'],key=lambda r:r['metrics']['cost']);row=prepare_cache()['rows'][worst['index']]
    if row['member']!=worst['member']:raise ValueError('Member mismatch')
    params={'constant':dict(iterations=32,seconds=.7,reweighted=False,boundary=12.,contact=6.,mode=0),
        'boundary':dict(iterations=32,seconds=.7,reweighted=False,boundary=40.,contact=20.,mode=1)}
    identity={'sources':sources(),'member':row['member'],'selected_by':'worst Q on already adaptive monitor500',
        'hypothesis':'Reweighted geometry can stall on a poor topology; constant penalties or the original baseline geometry provide a counterexample control',
        'parameters':params,'resource':RESOURCE,'test_read':False}
    save(folder/'config.json',identity);snapshot(folder/'source_snapshot.zip',identity);records=[]
    from .native_reduce import warmup
    warmup()
    for name in ('constant','boundary','original_fast'):
        started=time.perf_counter()
        if name=='original_fast':x,stats=fast(row['draft'],row['case'].solver_inputs,1.3)
        else:x,stats=optimize(row['draft'],row['case'].solver_inputs,**params[name],native_reduce=True,
            projection=True,quadratic_project=True,periodic_project=True,snap=1e-4)
        seconds=time.perf_counter()-started;metrics=evaluate_case(row['case'],x.tolist(),seconds)['metrics']
        records.append({'variant':name,'seconds':seconds,'metrics':metrics,'positions':x.tolist(),'proxy':stats['proxy'],'trace':stats['trace']})
        print(name,seconds,metrics['cost'],stats['proxy'],flush=True)
    save(folder/'results.json',{'member':row['member'],'baseline_record_index':worst['index'],'records':records,'test_read':False,
        'scope':'One selected development counterexample; no population performance claim'})

if __name__=='__main__':main()
