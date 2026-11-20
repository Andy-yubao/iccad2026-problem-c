"""Separate frozen dev500 insertions: one complete S0 or S1 at its own stage end."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import csv
import json
import time
import traceback
import numpy as np
import torch
from experiments.E148_shared_functional_shape.io import ROOT,HERE,sha,save,checkpoint
from experiments.E148_shared_functional_shape.model import GeometryModel
from experiments.E148_shared_functional_shape.train import seed_all
from experiments.E148_shared_functional_shape.data import load_rows,records
from experiments.E148_shared_functional_shape.resident import ResidentRows
from experiments.E148_shared_functional_shape.batches import inputs_only
from experiments.E148_shared_functional_shape.loss import base_per_case,TOTAL_COMPONENTS

OUT=Path(__file__).resolve().parent
RUN=OUT.parent
ORDERS={'baseline':[0,0,1,1,2,2], 'insert_S0':[0,0,0,1,1,2,2], 'insert_S1':[0,0,1,1,1,2,2]}
ENDPOINTS={'baseline':{'S0':4,'S1':8,'S2':12},'insert_S0':{'S0':6,'S1':10,'S2':14},'insert_S1':{'S0':4,'S1':10,'S2':14}}

def read(path):return json.loads(path.read_text(encoding='utf-8'))

def paired(a,b):
    a=np.array(a);b=np.array(b);d=b-a
    return dict(before=float(a.mean()),after=float(b.mean()),change=float(d.mean()),relative_change=float(d.mean()/a.mean()),
        improved=int((d < -1e-8).sum()),worsened=int((d > 1e-8).sum()),unchanged=int((abs(d)<=1e-8).sum()),median_change=float(np.median(d)))

def main():
    started=time.perf_counter()
    if (OUT/'results.json').exists():
        existing=read(OUT/'results.json');assert existing['status']=='complete'
        print(json.dumps(existing,ensure_ascii=False));return
    frozen=read(HERE/'evaluation/frozen_e148_shared_functional_shape.json')
    weight=ROOT/frozen['checkpoint'];digest=sha(weight);assert digest==frozen['checkpoint_sha256']
    seed_all(0);state=torch.load(weight,map_location='cpu',weights_only=False)
    assert state['progress']['best_epoch']==46
    for path,expected in state['identity']['sources'].items():assert sha(ROOT/path)==expected,path
    plan=dict(authorization='2026-11-17 user requested separate one-S0 and one-S1 insertions at corresponding positions, comparing stage and final Loss on same500',
        checkpoint=str(weight.relative_to(ROOT)),checkpoint_sha256=digest,best_epoch=46,orders=ORDERS,stage_endpoints=ENDPOINTS,
        insertion='One whole existing two-layer FunctionalUnit; same instance/weights, fresh hidden from original public-only condition; never carry hidden across units',
        comparison='S0 inserted after original layer4 then rerun both S1 and both S2; S1 inserted after original layer8 then rerun both S2. Never combine insertions.',
        loss='Original full base_per_case and coefficients. Additionally S0 stage objective excludes entire direct overlap group exactly as training.',
        panel='Original fixed dev_monitor500, same order/cache/batch64/padding as original monitoring',
        model_selection='Fixed original own-Loss best46; no new checkpoint or depth selection',
        test_read=False,optimizer_updates=0,postprocessing_calls=0,Q_calls=0,
        original_source_identity=state['identity']['sources'],diagnostic_source_sha256=sha(Path(__file__)))
    save(OUT/'plan.json',plan)
    def stopped():return any(p.exists() for p in (HERE/'STOP',HERE/'QUEUE_STOP',RUN/'STOP',OUT/'STOP'))
    def beat(**fields):save(OUT/'state.json',dict(status='running',pid=os.getpid(),elapsed_seconds=time.perf_counter()-started,**fields))
    if stopped():raise RuntimeError('STOP present; not cleared')
    beat(activity='loading original fixed dev500',completed=0)
    rows=load_rows('dev_monitor500',halted=stopped);assert rows is not None and len(rows)==500
    members=[list(r['member']) for r in rows]
    assert members==[r['member'] for r in records('dev_monitor500')]==read(RUN/'panels.json')['dev_monitor_members']
    resources=read(RUN/'resources.json')
    pack=ResidentRows(rows,'cuda',max_nodes=resources['max_nodes'],max_pins=resources['max_loss_ports_per_block'])
    model=GeometryModel(**state['model_config']).cuda().eval()
    model.load_state_dict(state['model'],strict=True);model.requires_grad_(False)
    before_parameters={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    condition=[];hook=model.problem_encoder.register_forward_hook(lambda module,args,value:condition.append(value))
    values={arm:{stage:[] for stage in ('S0','S1','S2')} for arm in ORDERS}
    insert_intermediate={arm:[] for arm in ('insert_S0','insert_S1')}
    saved_states={arm:[] for arm in ORDERS}
    def record(output,stage):
        result=base_per_case(output,batch)
        result['stage_objective']=sum(result[k] for k in TOTAL_COMPONENTS if k!='overlap_weighted') if stage=='S0' else result['total']
        keys=list(result)
        matrix=torch.stack([result[k] for k in keys],-1).cpu().double().numpy()
        assert np.isfinite(matrix).all()
        return [dict(zip(keys,row.tolist())) for row in matrix]
    try:
        with torch.inference_mode():
            for start in range(0,500,64):
                if stopped():raise RuntimeError('User STOP during diagnostic')
                batch=pack.batch(range(start,min(start+64,500)));public=inputs_only(batch)
                condition.clear();original=model(public);assert original.shape[0]==12 and len(condition)==1
                cases={'baseline':original}
                for arm,index,prefix in (('insert_S0',0,4),('insert_S1',1,8)):
                    # Reuse unchanged prefix outputs, then recompute every downstream unit.
                    extra=model.units[index](public,original[prefix-1],condition[0])
                    output=[*original[:prefix].unbind(0),*extra.unbind(0)]
                    insert_intermediate[arm].extend(record(extra[0],f'S{index}'))
                    remaining=(1,1,2,2) if index==0 else (2,2)
                    for downstream in remaining:
                        emitted=model.units[downstream](public,output[-1],condition[0]);output.extend(emitted.unbind(0))
                    cases[arm]=torch.stack(output,0)
                    assert cases[arm].shape[0]==14
                    torch.testing.assert_close(cases[arm][:prefix],original[:prefix],rtol=0,atol=0)
                for arm,output in cases.items():
                    endpoints=[]
                    for stage,depth in ENDPOINTS[arm].items():
                        values[arm][stage].extend(record(output[depth-1],stage));endpoints.append(output[depth-1])
                    saved_states[arm].append(torch.stack(endpoints,1).cpu())
                beat(activity='separate S0 and S1 insertions; downstream rerun',completed=min(start+64,500))
                print(f'completed={min(start+64,500)}/500',flush=True)
    finally:hook.remove()
    assert sha(weight)==digest
    assert all(torch.equal(v.detach().cpu(),before_parameters[k]) for k,v in model.state_dict().items())
    for path,expected in state['identity']['sources'].items():assert sha(ROOT/path)==expected,path
    summary={arm:{stage:{k:float(np.mean([r[k] for r in entries])) for k in entries[0]} for stage,entries in stages.items()} for arm,stages in values.items()}
    baseline=summary['baseline'];best=next(r for r in state['progress']['history'] if r['epoch']==46)['dev_monitor']
    for stage,depth in ENDPOINTS['baseline'].items():
        assert abs(baseline[stage]['stage_objective']-best[f'layer{depth:02d}_total'])<2e-7,(stage,'Baseline mismatch')
    assert abs(baseline['S2']['total']-frozen['own_new_final_dev_loss'])<2e-7
    changes={}
    for arm in ('insert_S0','insert_S1'):
        changes[arm]={}
        for stage in ('S0','S1','S2'):
            changes[arm][stage]={metric:paired([r[metric] for r in values['baseline'][stage]],[r[metric] for r in values[arm][stage]]) for metric in ('total','stage_objective')}
    previous=read(RUN/'append_s2_dev500_v1/results.json')
    assert previous['status']=='complete' and previous['count']==500 and previous['checkpoint_sha256']==digest
    result=dict(status='complete',count=500,best_epoch=46,checkpoint_sha256=digest,orders=ORDERS,stage_endpoints=ENDPOINTS,
        summaries=summary,paired_changes=changes,
        inserted_first_sublayer={arm:{k:float(np.mean([r[k] for r in entries])) for k in entries[0]} for arm,entries in insert_intermediate.items()},
        final_component_changes={arm:{k:summary[arm]['S2'][k]-baseline['S2'][k] for k in TOTAL_COMPONENTS} for arm in ('insert_S0','insert_S1')},
        previous_append_S2=dict(full_loss_after=previous['full_loss_after'],change=previous['absolute_change'],relative_change=previous['relative_change'],improved=previous['improved_cases'],worsened=previous['worsened_cases'],source='append_s2_dev500_v1/results.json'),
        baseline_reproduction_error=baseline['S2']['total']-frozen['own_new_final_dev_loss'],
        original_weights_and_sources_unchanged=True,test_read=False,optimizer_updates=0,postprocessing_calls=0,Q_calls=0,
        elapsed_seconds=time.perf_counter()-started)
    save(OUT/'results.json',result)
    save(OUT/'per_case.json',[dict(index=i,member=members[i],**{arm:{stage:entries[i] for stage,entries in stages.items()} for arm,stages in values.items()}) for i in range(500)])
    checkpoint(OUT/'stage_end_states.pt',dict(members=members,checkpoint_sha256=digest,stage_endpoints=ENDPOINTS,
        states={arm:torch.cat(entries,0) for arm,entries in saved_states.items()}))
    with (OUT/'paired_loss.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['index','shard','row','arm','stage','original_depth','new_depth','full_loss_before','full_loss_after','full_change','stage_objective_before','stage_objective_after'])
        for i in range(500):
            for arm in ('insert_S0','insert_S1'):
                for stage in ('S0','S1','S2'):
                    a=values['baseline'][stage][i];b=values[arm][stage][i]
                    writer.writerow([i,*members[i],arm,stage,ENDPOINTS['baseline'][stage],ENDPOINTS[arm][stage],a['total'],b['total'],b['total']-a['total'],a['stage_objective'],b['stage_objective']])
    lines=['# E148 best46: insert one S0 or one S1 separately','',
        'Original fixed dev500, frozen original checkpoint; each additional unit has two geometry sublayers. All downstream units are rerun. No training/postprocessing/Q/test.','',
        'S0 stage objective excludes the entire direct overlap group exactly as original supervision; full Loss includes it. S1/S2 stage objectives are full Loss. No cross-layer weighting here.','',
        '|Insertion|Stage objective before|Stage objective after|Full stage Loss before|Full stage Loss after|Final full Loss|Final relative change|','|---|---:|---:|---:|---:|---:|---:|']
    for arm,stage in (('insert_S0','S0'),('insert_S1','S1')):
        a=baseline[stage];b=summary[arm][stage];c=changes[arm]['S2']['total']
        lines.append(f'|{arm}|{a["stage_objective"]:.9f}|{b["stage_objective"]:.9f}|{a["total"]:.9f}|{b["total"]:.9f}|{c["after"]:.9f}|{c["relative_change"]:+.3%}|')
    lines+=['',f'Original final full Loss: {baseline["S2"]["total"]:.9f}. Prior appended S2 final: {previous["full_loss_after"]:.9f} ({previous["relative_change"]:+.3%}), reused without rerunning.','',
        '|Arm|S0 stage full Loss|S1 stage full Loss|S2 final full Loss|Final improved/worsened/unchanged|','|---|---:|---:|---:|---|']
    for arm in ORDERS:
        s=summary[arm];c=changes[arm]['S2']['total'] if arm!='baseline' else None
        count=f'{c["improved"]}/{c["worsened"]}/{c["unchanged"]}' if c else '-'
        lines.append(f'|{arm}|{s["S0"]["total"]:.9f}|{s["S1"]["total"]:.9f}|{s["S2"]["total"]:.9f}|{count}|')
    lines+=['','|Weighted final component|Extra S0 change|Extra S1 change|','|---|---:|---:|']
    for k in TOTAL_COMPONENTS:lines.append(f'|{k}|{result["final_component_changes"]["insert_S0"][k]:+.9f}|{result["final_component_changes"]["insert_S1"][k]:+.9f}|')
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(OUT/'state.json',dict(status='complete',pid=os.getpid(),completed=500,elapsed_seconds=time.perf_counter()-started))
    print(json.dumps(dict(status='complete',summaries=summary,paired_changes=changes),ensure_ascii=False),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        save(OUT/'state.json',dict(status='failed',pid=os.getpid(),error=repr(exc),traceback=traceback.format_exc()))
        raise
