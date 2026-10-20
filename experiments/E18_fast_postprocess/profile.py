"""Five module-count strata: profile frozen E17 postprocessing."""
import cProfile
import io
import json
import pstats
import time
import numpy as np
from .cases import prepare_cache
from .io import HERE,save,sources,snapshot
from .backend.optimizer import optimize
from .backend.stable import polish
from scoring.evaluation import evaluate_case


def main():
    rows=prepare_cache()['rows']; order=sorted(range(len(rows)),key=lambda i:rows[i]['case'].block_count)
    selected=[order[i] for i in [0,124,249,374,499]]
    folder=HERE/'runs/20261020_profile5_seed0';folder.mkdir(exist_ok=False)
    profiler=cProfile.Profile();records=[]
    for index in selected:
        row=rows[index];case=row['case'];profiler.enable();start=time.perf_counter()
        x,stats=optimize(row['draft'],case.solver_inputs)
        x,rescue=polish(row['draft'],x,case.solver_inputs,stats)
        seconds=time.perf_counter()-start;profiler.disable()
        metrics=evaluate_case(case,x.tolist(),seconds)['metrics']
        trace=stats.get('trace',[])
        record={'index':index,'n':case.block_count,'Q':metrics['cost'],'seconds':seconds,
            'native_seconds':sum(t.get('native_seconds',0.) for t in trace),'iterations':len(trace),
            'rescue_used':stats.get('rescue_used'),'tail_rescue':rescue['triggered']}
        records.append(record);print(json.dumps(record),flush=True)
    profiler.dump_stats(str(folder/'baseline.prof'))
    output=io.StringIO();pstats.Stats(profiler,stream=output).sort_stats('cumtime').print_stats(28)
    (folder/'hotspots.txt').write_text(output.getvalue(),encoding='utf-8')
    save(folder/'results.json',{'records':records,'purpose':'Profiler adds overhead; not deployment timing','test_read':False})
    save(folder/'config.json',{'sources':sources(),'selected':selected,'checkpoint':'frozen E17 best51'})
    print(output.getvalue(),flush=True)


if __name__=='__main__':main()
