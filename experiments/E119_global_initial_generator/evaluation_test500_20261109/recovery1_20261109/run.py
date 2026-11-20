"""Explicitly authorized one-shot E119 best55 on a frozen 500/2000 test subset."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[name]='1'
import hashlib
import json
from pathlib import Path
import signal
import sys
import time
import traceback
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
ORIGINAL=HERE.parent
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from experiments.E119_global_initial_generator.evaluation.run import Predictor,repair_inputs
from experiments.E119_global_initial_generator.evaluation.solver import optimize
from experiments.E119_global_initial_generator.evaluation.repair.canonical_native import warmup
from experiments.E119_global_initial_generator.io import sha,save,exclusive
from datasets.random_reference import registry
from datasets.cases import _build_case
from datasets.reference_layouts import public_hard_reference
from scoring.research_no_mib import evaluate_case
from scoring.checks import independent_hard_checks
from scoring.official import official_evaluator
from visualization.layer_comparison import measures

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def now():return datetime.now(timezone.utc).isoformat()

def main():
    plan=read(HERE/'plan.json');out=HERE/'output'
    assert plan['authorized_test_evaluation'] and plan['count']==500 and plan['model']=='E119 best55'
    assert plan['test_payload_scope']=='Only the frozen selected500; other1500 remain sealed'
    for path,digest in plan['sources'].items():assert sha(ROOT/path)==digest,('Source changed',path)
    assert sha(ROOT/plan['checkpoint'])==plan['checkpoint_sha256']
    assert sha(ROOT/plan['registry'])==plan['registry_sha256']
    assert not out.exists(),'Existing result/error retained; no automatic retry'
    if (HERE/'STOP').exists() or (ORIGINAL/'STOP').exists():raise InterruptedError('STOP before dispatch')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    import numba
    numba.set_num_threads(1)
    with exclusive(HERE/'evaluation.lock'):
        started=time.perf_counter();requested=[False]
        signal.signal(signal.SIGINT,lambda *_:requested.__setitem__(0,True))
        state=dict(status='running',pid=os.getpid(),completed=0,total=500,started_at=now(),test_read=True,training=False)
        def beat(**kw):state.update(kw);save(out/'state.json',dict(**state,updated_at=now()))
        def check():
            if requested[0] or (HERE/'STOP').exists() or (ORIGINAL/'STOP').exists() or (out/'STOP').exists():raise InterruptedError('Test evaluation STOP; no retry')
        try:
            beat(stage='loading_existing_authorized_test500_NPZ_no_data_generation')
            reg=registry();paths=read(ROOT/'protocol/paths.json');folder=(ROOT/paths['random_reference40k_root']).resolve()
            assert folder.is_relative_to(ROOT)
            assert read(folder/'report.json')['manifest_sha256']==reg['manifest_sha256']
            assert sha(folder/'checksums.jsonl')==plan['payload_checksums_sha256']
            ledger={r['id']:r['payload_sha256'] for line in (folder/'checksums.jsonl').read_text(encoding='utf-8').splitlines() for r in json.loads(line)['records']}
            def load_item(i):
                check();record=plan['records'][i]
                beat(stage='loading_existing_authorized_test_NPZ',loading_index=i,loaded_payloads=i)
                assert {k:v for k,v in record.items() if k!='frozen_test_index'}==reg['sets']['test']['records'][record['frozen_test_index']]
                path=(folder/record['payload']).resolve()
                assert path.is_relative_to(folder/'test') and sha(path)==ledger[record['id']]
                with np.load(path,allow_pickle=False) as raw:
                    inputs=(len(raw['areas']),*[raw[k].copy() for k in ('areas','b2b','p2b','pins','constraints','specs')])
                    supervision={k:raw[k].copy() for k in ('reference_xywh','stored_metrics')}
                item=dict(member=record['member'],solver_inputs=inputs,supervision=supervision)
                beat(stage='building_scoring_case_from_existing_reference',loading_index=i,loaded_payloads=i+1)
                case=_build_case('authorized_random_test:'+str(item['member']),dict(member=item['member'],role='authorized_test500'),
                    [torch.as_tensor(v).clone() for v in inputs[1:6]],supervision['reference_xywh'].tolist(),supervision['stored_metrics'])
                np.testing.assert_array_equal(case.solver_inputs[6],inputs[6])
                return item,case
            first_item,first_case=load_item(0)
            save(out/'frozen.json',dict(recovery_authorized=True,original_plan_sha256=plan['original_plan_sha256'],plan_sha256=sha(HERE/'plan.json'),checkpoint_sha256=plan['checkpoint_sha256'],
                best_epoch=55,members=[r['member'] for r in plan['records']],source_identity=plan['sources'],
                sampling=plan['sampling'],postprocessor='E98.depth_budget.v1 original work caps',
                test_read=True,selected_test_count=500,unselected_test_payloads_read=0,selection_frozen_before_test=True))
            tick=time.perf_counter()
            with torch.inference_mode(),threadpool_limits(limits=1):
                beat(stage='model_initialization_and_CUDA_graph_capture')
                warmup();predictor=Predictor(ROOT/plan['checkpoint'],first_item['solver_inputs'])
            save(out/'initialization.json',dict(seconds=time.perf_counter()-tick,raw_data_generated=False,
                training_static_cache_built=False,existing_NPZ_loaded_at_initialization=1,torch_threads=1,numba_threads=1))
            records=[]
            with torch.inference_mode(),threadpool_limits(limits=1):
                for i in range(500):
                    item,case=(first_item,first_case) if i==0 else load_item(i)
                    check();torch.cuda.synchronize();begin=time.perf_counter()
                    inputs=tuple(int(v) if j==0 else np.asarray(v).copy() for j,v in enumerate(item['solver_inputs']))
                    draft=predictor.predict(inputs);draft_seconds=time.perf_counter()-begin
                    final,info=optimize(draft,repair_inputs(inputs))
                    hard=independent_hard_checks(final.tolist(),inputs[1],inputs[5],public_hard_reference(inputs[1],inputs[5],inputs[6]))
                    complete_seconds=time.perf_counter()-begin
                    scored=evaluate_case(case,final,complete_seconds);raw=evaluate_case(case,draft,draft_seconds)
                    record=dict(index=i,frozen_test_index=plan['records'][i]['frozen_test_index'],member=item['member'],n=inputs[0],
                        checkpoint_sha256=plan['checkpoint_sha256'],metrics=scored['metrics'],raw_metrics=raw['metrics'],
                        independent_hard_checks=hard,raw_geometry=measures(draft),draft_positions=draft.tolist(),positions=final.tolist(),
                        draft_seconds=draft_seconds,complete_seconds=complete_seconds,repair_seconds=info['seconds'],
                        fallback=info['row_retained'],trace=info['trace'],work=info['work'],test_read=True)
                    save(out/f'cases/{i:04d}.json',record);records.append(record)
                    if not hard['is_feasible'] or not scored['metrics']['is_feasible']:raise RuntimeError(f'Hard legality failed at case{i}')
                    beat(stage='evaluating_frozen_test500_original_budget',completed=i+1,last_Q=scored['metrics']['cost'])
                    if i<3 or (i+1)%25==0:print(f'E119 test500 {i+1}/500 Q={scored["metrics"]["cost"]:.6f}',flush=True)
            costs=np.array([r['metrics']['cost'] for r in records]);seconds=np.array([r['complete_seconds'] for r in records])
            weighted=official_evaluator().compute_total_score(costs.tolist(),[r['n'] for r in records])
            result=dict(status='complete',model='E119 best55',count=500,checkpoint_sha256=plan['checkpoint_sha256'],
                mean_Q=float(costs.mean()),weighted_Q=weighted,mixed_Q=.5*(float(costs.mean())+weighted),
                hard_legal=sum(r['metrics']['is_feasible'] for r in records),raw_hard_legal=sum(r['raw_metrics']['is_feasible'] for r in records),
                fallbacks=sum(r['fallback'] for r in records),max_Q=float(costs.max()),p95_Q=float(np.percentile(costs,95)),
                mean_complete_seconds=float(seconds.mean()),p95_complete_seconds=float(np.percentile(seconds,95)),max_complete_seconds=float(seconds.max()),
                over500ms=int((seconds>.5).sum()),elapsed_seconds=time.perf_counter()-started,
                postprocessor='E98.depth_budget.v1 original work caps, not3x',
                scoring_policy='Original areas/specs; local research Q without MIB; neutral runtime factor1',
                selection_frozen_before_test=True,checkpoint_reselected=False,test_read=True,unselected_test_payloads_read=0,automatic_retry=False)
            assert sha(ROOT/plan['checkpoint'])==plan['checkpoint_sha256']
            save(out/'results.json',result)
            dev=read(ROOT/plan['dev_result'])
            lines=['# E119 best55：冻结test500原预算评价','',
                '测试500例在读取载荷与评价前从原test2000随机无放回抽取、冻结；模型/预算/评分口径保持。不按本次测试结果重选模型或调整后处理。','',
                '|面板|平均Q|加权Q|混合Q|硬合法|','|---|---:|---:|---:|---:|',
                f'|原dev500|{dev["mean_Q"]:.9f}|{dev["weighted_Q"]:.9f}|{dev["mixed_Q"]:.9f}|{dev["hard_legal"]}/500|',
                f'|本次test500|{result["mean_Q"]:.9f}|{weighted:.9f}|{result["mixed_Q"]:.9f}|{result["hard_legal"]}/500|','',
                '两行成员不同，属于跨面板描述性对照。其他1500例测试载荷未读取；本次500例自此记为已曝光。完整时长含公开输入/特征/传输/网络/全部修复与必要硬检查，读盘/评分/预热在计时外。','']
            (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
            beat(status='complete',stage='complete',completed=500,completed_at=now());print(json.dumps(result),flush=True)
        except BaseException as error:
            save(out/'error.json',dict(error=repr(error),traceback=traceback.format_exc()))
            beat(status='user_stop' if isinstance(error,InterruptedError) else 'error',stage='stopped_no_retry',error=repr(error));raise

if __name__=='__main__':
    main()
