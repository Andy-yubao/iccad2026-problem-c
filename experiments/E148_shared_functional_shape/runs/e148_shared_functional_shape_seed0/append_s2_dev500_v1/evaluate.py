"""Frozen best46 dev500: original twelve readouts, then exactly one extra S2 unit."""
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
from experiments.E148_shared_functional_shape.loss import base_per_case,per_case,TOTAL_COMPONENTS
from experiments.E148_shared_functional_shape.loss_config import ARMS

OUT=Path(__file__).resolve().parent
RUN=OUT.parent

def read(path):return json.loads(path.read_text(encoding='utf-8'))

def main():
    started=time.perf_counter()
    if (OUT/'results.json').exists():
        result=read(OUT/'results.json')
        if result['status']!='complete':raise RuntimeError('Existing incomplete result; inspect first')
        print(json.dumps(result,ensure_ascii=False));return
    frozen=read(HERE/'evaluation/frozen_e148_shared_functional_shape.json')
    weight=ROOT/frozen['checkpoint'];weight_sha=sha(weight)
    assert weight_sha==frozen['checkpoint_sha256']
    seed_all(0)
    state=torch.load(weight,map_location='cpu',weights_only=False)
    assert state['progress']['best_epoch']==46
    for path,digest in state['identity']['sources'].items():
        assert sha(ROOT/path)==digest,'Frozen original source changed: '+path
    plan=dict(authorization='2026-11-17 user requested exactly one additional trained S2 at the end and loss comparison on500; continued after interruption',
        checkpoint=str(weight.relative_to(ROOT)),checkpoint_sha256=weight_sha,best_epoch=46,
        original_unit_order=[0,0,1,1,2,2],extended_unit_order=[0,0,1,1,2,2,2],
        reuse='Same model.units[2] instance and weights; fresh hidden from same pure-P condition; exactly two extra geometry layers',
        loss='Identical complete base_per_case total at layers12,13,14; no cross-layer weighting',
        dataset='Original fixed dev_monitor500, existing compatible CPU cache, batch64 and same padded ports',
        selection='Fixed original own-Loss-best before inference; exactly one appended unit, no new checkpoint or step selection',
        train=False,optimizer_updates=0,postprocessing_calls=0,Q_calls=0,test_read=False,
        original_source_identity=state['identity']['sources'],diagnostic_source_sha256=sha(Path(__file__)))
    save(OUT/'plan.json',plan)
    def beat(**fields):
        save(OUT/'state.json',dict(status='running',pid=os.getpid(),elapsed_seconds=time.perf_counter()-started,**fields))
    def stopped():
        return any(p.exists() for p in (HERE/'STOP',HERE/'QUEUE_STOP',RUN/'STOP',OUT/'STOP'))
    if stopped():raise RuntimeError('STOP present; not cleared')
    beat(activity='loading frozen dev500 cache',completed=0)
    rows=load_rows('dev_monitor500',halted=stopped)
    assert rows is not None and len(rows)==500
    members=[list(r['member']) for r in rows]
    assert members==[r['member'] for r in records('dev_monitor500')]
    assert members==read(RUN/'panels.json')['dev_monitor_members']
    resources=read(RUN/'resources.json')
    pack=ResidentRows(rows,'cuda',max_nodes=resources['max_nodes'],max_pins=resources['max_loss_ports_per_block'])
    model=GeometryModel(**state['model_config']).cuda().eval()
    model.load_state_dict(state['model'],strict=True);model.requires_grad_(False)
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    condition=[]
    hook=model.problem_encoder.register_forward_hook(lambda module,args,value:condition.append(value))
    series={12:[],13:[],14:[]};saved_states=[]
    try:
        with torch.inference_mode():
            for start in range(0,500,64):
                if stopped():raise RuntimeError('User STOP during diagnostic')
                batch=pack.batch(range(start,min(start+64,500)));public=inputs_only(batch)
                condition.clear();original=model(public)
                assert len(condition)==1 and original.shape[0]==12
                extra=model.units[2](public,original[-1],condition[0])
                assert extra.shape[0]==2
                outputs={12:base_per_case(original[-1],batch),13:base_per_case(extra[0],batch),14:base_per_case(extra[1],batch)}
                expected=per_case(original,batch,ARMS[0])['final_total']
                torch.testing.assert_close(outputs[12]['total'],expected,rtol=0,atol=0)
                for depth,values in outputs.items():
                    assert all(bool(torch.isfinite(v).all()) for v in values.values())
                    series[depth].extend([{k:float(v[i]) for k,v in values.items()} for i in range(len(batch['valid']))])
                saved_states.append(torch.stack((original[-1],extra[0],extra[1]),1).cpu())
                beat(activity='fixed appended-S2 frozen inference',completed=min(start+64,500))
                print(f"completed={min(start+64,500)}/500",flush=True)
    finally:hook.remove()
    assert sha(weight)==weight_sha
    assert all(torch.equal(v.detach().cpu(),initial[k]) for k,v in model.state_dict().items())
    summary={str(depth):{key:float(np.mean([r[key] for r in values])) for key in values[0]} for depth,values in series.items()}
    baseline=summary['12']['total'];after=summary['14']['total']
    assert abs(baseline-frozen['own_new_final_dev_loss'])<2e-7,'Saved best46 baseline did not reproduce'
    before=np.array([r['total'] for r in series[12]]);tail=np.array([r['total'] for r in series[14]])
    delta=tail-before
    result=dict(status='complete',count=500,best_epoch=46,checkpoint_sha256=weight_sha,
        full_loss_before=baseline,full_loss_after=after,absolute_change=after-baseline,
        relative_change=(after-baseline)/baseline,
        intermediate_layer13_full_loss=summary['13']['total'],
        improved_cases=int((delta < -1e-8).sum()),worsened_cases=int((delta > 1e-8).sum()),unchanged_cases=int((abs(delta)<=1e-8).sum()),
        median_paired_change=float(np.median(delta)),mean_paired_change=float(delta.mean()),
        components=summary,weighted_component_changes={k:summary['14'][k]-summary['12'][k] for k in TOTAL_COMPONENTS},
        reproduction_error=baseline-frozen['own_new_final_dev_loss'],
        original_weights_and_sources_unchanged=True,optimizer_updates=0,postprocessing_calls=0,Q_calls=0,test_read=False,
        elapsed_seconds=time.perf_counter()-started)
    save(OUT/'results.json',result)
    save(OUT/'per_case.json',[dict(index=i,member=members[i],layer12=series[12][i],layer13=series[13][i],layer14=series[14][i],loss_change=float(delta[i])) for i in range(500)])
    checkpoint(OUT/'states_12_13_14.pt',dict(depths=[12,13,14],members=members,states=torch.cat(saved_states,0),checkpoint_sha256=weight_sha))
    with (OUT/'paired_loss.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['index','shard','row','layer12_loss','layer13_loss','layer14_loss','change'])
        for i in range(500):writer.writerow([i,*members[i],before[i],series[13][i]['total'],tail[i],delta[i]])
    lines=['# E148 frozen best46: append one trained S2 unit','',
        'Original fixed dev500. Original12 layers versus14 layers. No retraining, postprocessing, Q or test reading.','',
        '|Output|Complete final Loss|','|---|---:|',f'|Original layer12|{baseline:.9f}|',
        f'|Appended S2, first sublayer13|{summary["13"]["total"]:.9f}|',f'|Appended S2, complete layer14|{after:.9f}|','',
        f'Change: {after-baseline:+.9f} ({(after-baseline)/baseline:+.3%}). Improved/worsened/unchanged: {result["improved_cases"]}/{result["worsened_cases"]}/{result["unchanged_cases"]}.','',
        'Same full base Loss and coefficients, per-case arithmetic mean. Original baseline reproduced within2e-7; checkpoint/model parameters unchanged.','',
        '|Weighted component|Layer12|Layer14|Change|','|---|---:|---:|---:|']
    for key in TOTAL_COMPONENTS:lines.append(f'|{key}|{summary["12"][key]:.9f}|{summary["14"][key]:.9f}|{result["weighted_component_changes"][key]:+.9f}|')
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(OUT/'state.json',dict(status='complete',pid=os.getpid(),completed=500,elapsed_seconds=time.perf_counter()-started))
    print(json.dumps({k:v for k,v in result.items() if k!='components'},ensure_ascii=False),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        save(OUT/'state.json',dict(status='failed',error=repr(exc),traceback=traceback.format_exc(),pid=os.getpid()))
        raise
