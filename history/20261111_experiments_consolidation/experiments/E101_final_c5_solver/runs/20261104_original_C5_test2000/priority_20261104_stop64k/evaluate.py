"""Only the original frozen C5, only the already registered test2000."""
from pathlib import Path
import sys,time,json
ROOT=Path(__file__).resolve().parents[4];BUNDLE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.E101_final_c5_solver.common import read,save,sha,seed,utc
from experiments.E101_final_c5_solver.solver import Service
from experiments.E101_final_c5_solver.evaluate import summary
from datasets.expanded_reference import registry
from datasets.worksets import FrozenMemberReader
from datasets.cases import _build_case
from datasets.reference_layouts import public_hard_reference
from scoring.evaluation import evaluate_case

def stopped():
    if (BUNDLE/'STOP').exists():raise InterruptedError('Original C5 evaluation STOP; other runs unaffected')

def main():
    stopped();plan=read(BUNDLE/'plan.json');model=Path(plan['checkpoint'])
    assert read(Path(plan['serial_after']))['status']=='complete','Finish serial training/evaluation before baseline test'
    assert sha(model)==plan['checkpoint_sha256']
    for name,digest in read(BUNDLE/'sources.json').items():assert sha(ROOT/name)==digest,'Frozen source changed: '+name
    assert read(BUNDLE/'preflight.json')['status']=='passed'
    assert [r['member'] for r in registry()['sets']['local_test']['records']]==plan['members']
    seed();binding=dict(checkpoint_sha256=sha(model),members=plan['members'],source_snapshot_sha256=sha(BUNDLE/'source_snapshot.zip'),plan_sha256=sha(BUNDLE/'plan.json'),frozen_utc=plan['frozen_utc'],repair=plan['repair'],role='local_test',batch_size=1,CPU_threads=1,test_read=True)
    if (BUNDLE/'frozen.json').exists():assert read(BUNDLE/'frozen.json')==binding
    else:save(BUNDLE/'frozen.json',binding)
    reader=FrozenMemberReader(plan['members'],'local_test',identity=binding['plan_sha256'],allow_local_test=True)
    first=reader.case(0);service=Service(model,first.solver_inputs);save(BUNDLE/'cold_start.json',dict(seconds=service.cold_seconds,excluded_from_resident_request_latency=True))
    records=[];begin=time.perf_counter()
    for i,member in enumerate(plan['members']):
        stopped();path=BUNDLE/'cases'/f'{i:04d}.json'
        if path.exists():
            r=read(path);assert r['member']==member and r['checkpoint_sha256']==binding['checkpoint_sha256'];records.append(r);continue
        start=time.perf_counter();raw=reader.case(i);disk=time.perf_counter()-start
        positions,diagnostic,inputs=service.solve(raw.solver_inputs)
        case=_build_case(raw.case_id,raw.source,tuple(inputs[1:6]),raw.reference_positions,raw.stored_metrics);case.solver_inputs=inputs;case.reference_positions=public_hard_reference(inputs[1],inputs[5],inputs[6])
        scored=evaluate_case(case,positions,diagnostic['complete_seconds'])
        r=dict(index=i,member=member,n=int(inputs[0]),checkpoint_sha256=binding['checkpoint_sha256'],disk_load_seconds=disk,positions=positions.tolist(),metrics=scored['metrics'],independent_hard_checks=scored['independent_hard_checks'],**diagnostic)
        save(path,r);records.append(r)
        if (i+1)%25==0:
            save(BUNDLE/'progress.json',dict(completed=i+1,total=2000,summary=summary(records),elapsed_seconds=time.perf_counter()-begin));print('Original C5 test',i+1,'/2000',flush=True)
    value=summary(records);save(BUNDLE/'results.json',dict(status='complete',summary=value,elapsed_seconds=time.perf_counter()-begin,checkpoint_sha256=binding['checkpoint_sha256'],test_read=True,scope=plan['test_scope'],no_test_feedback=True))
    lines=['# 原E97第二轮C5＋完整后处理：registered test2000','',f"权重SHA256：`{binding['checkpoint_sha256']}`",'',plan['test_scope'],'','| 指标 | 数值 |','|---|---:|']+[f'| {k} | {v} |' for k,v in value.items()]
    lines+=['','原开发集dev500已完成，结果只读复用；本次未重训或重选权重。完整时间协议与E101重训评价一致，历史dev计时不作严格跨协议速度比较。']
    (BUNDLE/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Original C5 test complete',value,flush=True)

if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    try:
        with threadpool_limits(1):main()
    except BaseException as e:save(BUNDLE/'error.json',dict(error=repr(e),at_utc=utc(),no_automatic_retry=True));raise
