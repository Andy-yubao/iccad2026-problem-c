"""Frozen E149 initial state -> frozen E148 geometry operators; dev500 Loss only."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import json
import time
import traceback
import numpy as np
import torch
from experiments.E149_role_initial.io import ROOT,HERE,sha,save,checkpoint
from experiments.E149_role_initial.train import seed_all
from experiments.E149_role_initial.model import GeometryModel as InitialModel
from experiments.E149_role_initial.public import inputs_only,BASE
from experiments.E149_role_initial.data import load_rows
from experiments.E149_role_initial.resident import ResidentRows
from experiments.E149_role_initial.loss import base_per_case,TOTAL_COMPONENTS
from .frozen_e148.model import GeometryModel as RefinementModel

OUT=Path(__file__).resolve().parent
RUN=OUT.parent
OLD=ROOT/'experiments/E148_shared_functional_shape'
OLD_RUN=OLD/'runs/e148_shared_functional_shape_seed0'

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def paired(a,b):
    a=np.array(a);b=np.array(b);d=b-a
    return dict(before=float(a.mean()),after=float(b.mean()),change=float(d.mean()),relative_change=float(d.mean()/a.mean()),
        improved=int((d < -1e-8).sum()),worsened=int((d > 1e-8).sum()),unchanged=int((abs(d)<=1e-8).sum()))

def main():
    started=time.perf_counter();assert not (OUT/'results.json').exists()
    new_frozen=read(HERE/'frozen_best.json');old_frozen=read(OLD/'evaluation/frozen_e148_shared_functional_shape.json')
    new_path=ROOT/new_frozen['checkpoint'];old_path=ROOT/old_frozen['checkpoint']
    assert sha(new_path)==new_frozen['checkpoint_sha256'] and sha(old_path)==old_frozen['checkpoint_sha256']
    seed_all(0)
    new_state=torch.load(new_path,map_location='cpu',weights_only=False)
    old_state=torch.load(old_path,map_location='cpu',weights_only=False)
    for state in (new_state,old_state):
        for p,digest in state['identity']['sources'].items():assert sha(ROOT/p)==digest,p
    for name,entry in read(OUT/'frozen_e148_sources.json').items():assert sha(OUT/'frozen_e148'/name)==entry['sha256']==sha(ROOT/entry['source'])
    plan=dict(authorization='2026-11-18 user requested directly attaching trained initial generator to existing downstream networks and examining Loss',
        initial_checkpoint=new_frozen,refiner_checkpoint=old_frozen,
        routes={'standard':[0,0,1,1,2,2],'extended':[0,0,1,1,1,2,2,2]},
        interface='Only raw normalized center_x/center_y/log_aspect from E149; E148 uses its own public-only encoder, resets dynamic hidden at every two-layer unit',
        dataset='Original fixed dev500, exact same order, CPU role cache reused, batch64',
        metrics='Complete original single-output Loss and no-overlap Loss at initial and every real sublayer; same coefficients, no auxiliary weighting',
        selection='Original frozen own-Loss-best E149 best52 and E148 best46, no new weight/depth selection',
        diagnostic_source_sha256=sha(Path(__file__)),isolated_refiner_sources=read(OUT/'frozen_e148_sources.json'),
        optimizer_updates=0,postprocessing_calls=0,Q_calls=0,test_read=False)
    save(OUT/'plan.json',plan)
    def stopped():return any(p.exists() for p in (HERE/'STOP',HERE/'QUEUE_STOP',RUN/'STOP',OUT/'STOP',OLD/'STOP',OLD_RUN/'STOP'))
    def beat(**fields):save(OUT/'state.json',dict(status='running',pid=os.getpid(),elapsed_seconds=time.perf_counter()-started,**fields))
    if stopped():raise RuntimeError('STOP present')
    beat(activity='loading existing fixed dev500 role cache',completed=0)
    rows=load_rows('dev_monitor500',halted=stopped);assert rows is not None and len(rows)==500
    members=[list(r['member']) for r in rows]
    prior=read(OLD_RUN/'insert_s0_s1_dev500_v1/per_case.json');prior_extended=read(OLD_RUN/'add_s1_s2_dev500_v1/per_case.json')
    assert members==[r['member'] for r in prior]==[r['member'] for r in prior_extended]==read(RUN/'panels.json')['dev_monitor_members']
    resources=read(RUN/'resources.json')
    pack=ResidentRows(rows,'cuda',max_nodes=resources['max_nodes'],max_pins=resources['max_loss_ports_per_block'])
    initial=InitialModel(**new_state['model_config']).cuda().eval();initial.load_state_dict(new_state['model']);initial.requires_grad_(False)
    refiner=RefinementModel(**old_state['model_config']).cuda().eval();refiner.load_state_dict(old_state['model']);refiner.requires_grad_(False)
    initial_copy={k:v.detach().cpu().clone() for k,v in initial.state_dict().items()}
    refiner_copy={k:v.detach().cpu().clone() for k,v in refiner.state_dict().items()}
    labels=['initial']+[f'prefix_L{i}' for i in range(1,9)]+[f'standard_L{i}' for i in range(9,13)]+[f'extended_L{i}' for i in range(9,17)]
    values={label:[] for label in labels};stored=[]
    def measure(output):
        result=base_per_case(output,batch)
        result['no_overlap_total']=sum(result[k] for k in TOTAL_COMPONENTS if k!='overlap_weighted')
        keys=list(result);array=torch.stack([result[k] for k in keys],-1).cpu().double().numpy()
        assert np.isfinite(array).all()
        return [dict(zip(keys,row.tolist())) for row in array]
    with torch.inference_mode():
        for start in range(0,500,64):
            if stopped():raise RuntimeError('User STOP during frozen integration')
            end=min(start+64,500);batch=pack.batch(range(start,end));new_public=inputs_only(batch)
            public={key:batch[key] for key in BASE}
            state=initial(new_public)[0];condition=refiner.problem_encoder(public)
            outputs={'initial':state}
            for call,index in enumerate((0,0,1,1)):
                emitted=refiner.units[index](public,state,condition)
                for sub in range(2):outputs[f'prefix_L{2*call+sub+1}']=emitted[sub]
                state=emitted[-1]
            common=state
            for route,sequence in (('standard',(2,2)),('extended',(1,2,2,2))):
                state=common
                for call,index in enumerate(sequence):
                    emitted=refiner.units[index](public,state,condition)
                    for sub in range(2):outputs[f'{route}_L{9+2*call+sub}']=emitted[sub]
                    state=emitted[-1]
            if start==0:
                # A separately composed full16-layer trajectory must equal the shared-prefix route.
                state=initial(new_public)[0]
                for index in (0,0,1,1,1,2,2,2):state=refiner.units[index](public,state,condition)[-1]
                torch.testing.assert_close(state,outputs['extended_L16'],rtol=0,atol=0)
                assert not initial.training and not refiner.training
            for label in labels:values[label].extend(measure(outputs[label]))
            stored.append(torch.stack([outputs[label] for label in labels],1).cpu())
            beat(activity='frozen initial-to-refiner trajectories',completed=end)
            print(f'completed={end}/500',flush=True)
    for model,original in ((initial,initial_copy),(refiner,refiner_copy)):
        assert all(torch.equal(v.detach().cpu(),original[k]) for k,v in model.state_dict().items())
    assert sha(new_path)==new_frozen['checkpoint_sha256'] and sha(old_path)==old_frozen['checkpoint_sha256']
    summary={label:{k:float(np.mean([row[k] for row in records])) for k in records[0]} for label,records in values.items()}
    expected=read(HERE/'loss_results.json')
    assert abs(summary['initial']['no_overlap_total']-expected['dev_initial_no_overlap_loss'])<2e-7
    assert abs(summary['initial']['total']-expected['dev_initial_full_loss_diagnostic'])<2e-7
    old12=[r['baseline']['S2']['total'] for r in prior];old16=[r['layer16']['total'] for r in prior_extended]
    base=[r['total'] for r in values['initial']]
    endpoints={'initial':'initial','after_S0x1':'prefix_L2','after_S0x2':'prefix_L4','after_S1x1':'prefix_L6','after_S1x2':'prefix_L8',
        'standard_after_S2x1':'standard_L10','standard_after_S2x2':'standard_L12',
        'extended_after_S1x3':'extended_L10','extended_after_S2x1':'extended_L12','extended_after_S2x2':'extended_L14','extended_after_S2x3':'extended_L16'}
    result=dict(status='complete',count=500,initial_best_epoch=new_frozen['best_epoch'],refiner_best_epoch=old_frozen['best_epoch'],
        initial_checkpoint_sha256=new_frozen['checkpoint_sha256'],refiner_checkpoint_sha256=old_frozen['checkpoint_sha256'],
        summaries=summary,unit_endpoints=endpoints,
        standard_vs_new_initial=paired(base,[r['total'] for r in values['standard_L12']]),
        extended_vs_new_initial=paired(base,[r['total'] for r in values['extended_L16']]),
        standard_vs_original_E148=paired(old12,[r['total'] for r in values['standard_L12']]),
        extended_vs_original_E148_extended=paired(old16,[r['total'] for r in values['extended_L16']]),
        first_S0_change=paired(base,[r['total'] for r in values['prefix_L2']]),
        stage_transitions={route:[dict(before=a,after=b,full_loss_change=summary[b]['total']-summary[a]['total'],no_overlap_change=summary[b]['no_overlap_total']-summary[a]['no_overlap_total'])
            for a,b in zip(sequence[:-1],sequence[1:])] for route,sequence in (
                ('standard',('initial','prefix_L2','prefix_L4','prefix_L6','prefix_L8','standard_L10','standard_L12')),
                ('extended',('initial','prefix_L2','prefix_L4','prefix_L6','prefix_L8','extended_L10','extended_L12','extended_L14','extended_L16')))},
        original_models_sources_and_weights_unchanged=True,no_hidden_transfer=True,direct_route_reproduction_bitwise=True,
        optimizer_updates=0,postprocessing_calls=0,Q_calls=0,test_read=False,elapsed_seconds=time.perf_counter()-started)
    save(OUT/'results.json',result)
    save(OUT/'per_case.json',[dict(index=i,member=members[i],outputs={label:values[label][i] for label in labels}) for i in range(500)])
    checkpoint(OUT/'states.pt',dict(labels=labels,members=members,states=torch.cat(stored,0),initial_checkpoint_sha256=new_frozen['checkpoint_sha256'],refiner_checkpoint_sha256=old_frozen['checkpoint_sha256']))
    lines=['# E149 initial best52 attached to E148 frozen best46 refiners','',
        'Original fixed dev500; only layout passed between models and between units. E148 public-only condition encoder stays unchanged. No training, postprocessing, Q or test.','',
        '|Output|Complete Loss|No-overlap Loss|Weighted overlap|','|---|---:|---:|---:|']
    for name,label in endpoints.items():
        v=summary[label];lines.append(f'|{name}|{v["total"]:.9f}|{v["no_overlap_total"]:.9f}|{v["overlap_weighted"]:.9f}|')
    lines+=['','|Comparison|Old final Loss|Attached final Loss|Change|Improved/worsened/unchanged|','|---|---:|---:|---:|---|']
    for key in ('standard_vs_original_E148','extended_vs_original_E148_extended'):
        v=result[key];lines.append(f'|{key}|{v["before"]:.9f}|{v["after"]:.9f}|{v["relative_change"]:+.3%}|{v["improved"]}/{v["worsened"]}/{v["unchanged"]}|')
    lines+=['','Initial no-overlap and complete losses reproduce the saved E149 monitoring record. Models are eval/frozen; first64 shared-prefix composition matched a separately composed16-layer trajectory bitwise. No checkpoint or depth reselection.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(OUT/'state.json',dict(status='complete',pid=os.getpid(),completed=500,elapsed_seconds=time.perf_counter()-started))
    print(json.dumps({k:v for k,v in result.items() if k not in ('summaries','stage_transitions')},ensure_ascii=False),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        save(OUT/'state.json',dict(status='failed',pid=os.getpid(),error=repr(exc),traceback=traceback.format_exc()))
        raise
