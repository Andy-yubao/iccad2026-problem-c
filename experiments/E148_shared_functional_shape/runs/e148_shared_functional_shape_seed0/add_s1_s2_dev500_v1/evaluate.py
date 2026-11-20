"""Frozen best46: S0x2,S1x3,S2x3; reuse verified exact prior S1-insertion states."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
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
PREVIOUS=RUN/'insert_s0_s1_dev500_v1'

def read(path):return json.loads(path.read_text(encoding='utf-8'))

def paired(a,b):
    a=np.asarray(a);b=np.asarray(b);d=b-a
    return dict(before=float(a.mean()),after=float(b.mean()),change=float(d.mean()),relative_change=float(d.mean()/a.mean()),
        improved=int((d < -1e-8).sum()),worsened=int((d > 1e-8).sum()),unchanged=int((abs(d)<=1e-8).sum()),median_change=float(np.median(d)))

def main():
    started=time.perf_counter()
    if (OUT/'results.json').exists():
        result=read(OUT/'results.json');assert result['status']=='complete';print(json.dumps(result,ensure_ascii=False));return
    frozen=read(HERE/'evaluation/frozen_e148_shared_functional_shape.json')
    weight=ROOT/frozen['checkpoint'];digest=sha(weight);assert digest==frozen['checkpoint_sha256']
    seed_all(0);state=torch.load(weight,map_location='cpu',weights_only=False)
    for path,expected in state['identity']['sources'].items():assert sha(ROOT/path)==expected,path
    prior=read(PREVIOUS/'results.json');s2=read(RUN/'append_s2_dev500_v1/results.json')
    assert all(r['status']=='complete' and r['count']==500 and r['checkpoint_sha256']==digest for r in (prior,s2))
    saved=torch.load(PREVIOUS/'stage_end_states.pt',map_location='cpu',weights_only=False)
    cases=read(PREVIOUS/'per_case.json')
    assert saved['checkpoint_sha256']==digest and len(cases)==500
    save(OUT/'plan.json',dict(authorization='User requested adding one S1 and one S2 together, at their corresponding stage ends, same frozen dev500 loss evaluation',
        checkpoint_sha256=digest,best_epoch=46,unit_order=[0,0,1,1,1,2,2,2],geometry_layers=16,
        reuse='Verified cached14-layer outputs from prior extra-S1 arm, then exactly one additional existing S2 two-layer unit, fresh hidden. First batch checked against direct full16-layer composition.',
        loss='Identical complete base_per_case; arithmetic mean across fixed500; no cross-layer weighting or new model/depth selection',
        cached_inputs={str(p.relative_to(ROOT)):sha(p) for p in (PREVIOUS/'results.json',PREVIOUS/'per_case.json',PREVIOUS/'stage_end_states.pt')},
        original_source_identity=state['identity']['sources'],diagnostic_source_sha256=sha(Path(__file__)),
        test_read=False,optimizer_updates=0,postprocessing_calls=0,Q_calls=0))
    def stopped():return any(p.exists() for p in (HERE/'STOP',HERE/'QUEUE_STOP',RUN/'STOP',OUT/'STOP'))
    def beat(**fields):save(OUT/'state.json',dict(status='running',pid=os.getpid(),elapsed_seconds=time.perf_counter()-started,**fields))
    if stopped():raise RuntimeError('STOP present')
    beat(activity='loading original fixed dev500',completed=0)
    rows=load_rows('dev_monitor500',halted=stopped);assert rows is not None and len(rows)==500
    members=[list(r['member']) for r in rows]
    assert members==saved['members']==[r['member'] for r in cases]==[r['member'] for r in records('dev_monitor500')]
    resources=read(RUN/'resources.json')
    pack=ResidentRows(rows,'cuda',max_nodes=resources['max_nodes'],max_pins=resources['max_loss_ports_per_block'])
    model=GeometryModel(**state['model_config']).cuda().eval();model.load_state_dict(state['model'],strict=True);model.requires_grad_(False)
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    cached=saved['states']['insert_S1'];assert cached.shape==(500,3,120,3)
    values={14:[],15:[],16:[]};outputs=[]
    with torch.inference_mode():
        for start in range(0,500,64):
            if stopped():raise RuntimeError('User STOP during diagnostic')
            end=min(start+64,500);batch=pack.batch(range(start,end));public=inputs_only(batch)
            condition=model.problem_encoder(public);current=cached[start:end,2].cuda()
            appended=model.units[2](public,current,condition)
            if start==0:
                original=model(public)
                extra_s1=model.units[1](public,original[7],condition);direct=extra_s1[-1]
                for _ in range(2):direct=model.units[2](public,direct,condition)[-1]
                torch.testing.assert_close(direct,current,rtol=0,atol=0)
                direct_s2=model.units[2](public,direct,condition)
                torch.testing.assert_close(direct_s2,appended,rtol=0,atol=0)
            for depth,output in ((14,current),(15,appended[0]),(16,appended[1])):
                scores=base_per_case(output,batch);keys=list(scores)
                array=torch.stack([scores[k] for k in keys],-1).cpu().double().numpy();assert np.isfinite(array).all()
                values[depth].extend([dict(zip(keys,r.tolist())) for r in array])
            expected=torch.tensor([r['insert_S1']['S2']['total'] for r in cases[start:end]],device='cuda')
            torch.testing.assert_close(torch.tensor([r['total'] for r in values[14][-len(expected):]],device='cuda'),expected,rtol=0,atol=0)
            outputs.append(torch.stack((current,appended[0],appended[1]),1).cpu())
            beat(activity='appending S2 after extra-S1 exact cached prefix',completed=end)
            print(f'completed={end}/500',flush=True)
    assert sha(weight)==digest and all(torch.equal(v.detach().cpu(),initial[k]) for k,v in model.state_dict().items())
    for path,expected in state['identity']['sources'].items():assert sha(ROOT/path)==expected,path
    summary={str(depth):{k:float(np.mean([r[k] for r in entries])) for k in entries[0]} for depth,entries in values.items()}
    baseline=[r['baseline']['S2']['total'] for r in cases];extra_s1=[r['insert_S1']['S2']['total'] for r in cases]
    combined=[r['total'] for r in values[16]]
    extra_s2_cases=read(RUN/'append_s2_dev500_v1/per_case.json')
    assert [r['member'] for r in extra_s2_cases]==members
    extra_s2=[r['layer14']['total'] for r in extra_s2_cases]
    result=dict(status='complete',count=500,best_epoch=46,checkpoint_sha256=digest,unit_order=[0,0,1,1,1,2,2,2],
        final_full_loss=summary['16']['total'],baseline_final_loss=float(np.mean(baseline)),
        extra_S1_final_loss=float(np.mean(extra_s1)),extra_S2_final_loss=float(np.mean(extra_s2)),
        versus_baseline=paired(baseline,combined),versus_extra_S1=paired(extra_s1,combined),versus_extra_S2=paired(extra_s2,combined),
        stage_losses=dict(S0=prior['summaries']['baseline']['S0'],S1=prior['summaries']['insert_S1']['S1'],S2_intermediate=summary),
        weighted_final_component_changes={k:summary['16'][k]-prior['summaries']['baseline']['S2'][k] for k in TOTAL_COMPONENTS},
        original_weights_and_sources_unchanged=True,cached_prefix_direct_reproduction_bitwise=True,
        test_read=False,optimizer_updates=0,postprocessing_calls=0,Q_calls=0,elapsed_seconds=time.perf_counter()-started)
    save(OUT/'results.json',result)
    save(OUT/'per_case.json',[dict(index=i,member=members[i],baseline_loss=baseline[i],extra_S1_loss=extra_s1[i],extra_S2_loss=extra_s2[i],layer14=values[14][i],layer15=values[15][i],layer16=values[16][i]) for i in range(500)])
    checkpoint(OUT/'states_14_15_16.pt',dict(members=members,depths=[14,15,16],states=torch.cat(outputs,0),checkpoint_sha256=digest))
    lines=['# E148 best46: one extra S1 and one extra S2 together','',
        'Original fixed dev500. S0x2,S1x3,S2x3:16 geometry layers, frozen original weights. No retraining, postprocessing, Q or test.','',
        '|Composition|Final complete Loss|Change vs original|','|---|---:|---:|',
        f'|Original12 layers|{result["baseline_final_loss"]:.9f}|-|',
        f'|Extra S1 only14 layers|{result["extra_S1_final_loss"]:.9f}|{(result["extra_S1_final_loss"]/result["baseline_final_loss"]-1):+.3%}|',
        f'|Extra S2 only14 layers|{result["extra_S2_final_loss"]:.9f}|{(result["extra_S2_final_loss"]/result["baseline_final_loss"]-1):+.3%}|',
        f'|Extra S1 and S2,16 layers|{result["final_full_loss"]:.9f}|{result["versus_baseline"]["relative_change"]:+.3%}|','',
        f'Final improved/worsened/unchanged vs baseline: {result["versus_baseline"]["improved"]}/{result["versus_baseline"]["worsened"]}/{result["versus_baseline"]["unchanged"]}.','',
        f'S1 stage full Loss: {prior["summaries"]["baseline"]["S1"]["total"]:.9f} -> {result["stage_losses"]["S1"]["total"]:.9f}.',
        f'S2 stage complete Loss after two calls (layer14): {summary["14"]["total"]:.9f}; third-call first sublayer15: {summary["15"]["total"]:.9f}; complete layer16: {summary["16"]["total"]:.9f}.','',
        'Exact cached14-layer prefix was reproduced bitwise against direct composition on first64 cases. All500 prefix Loss values matched previous record exactly; original weights/source identities unchanged.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(OUT/'state.json',dict(status='complete',completed=500,pid=os.getpid(),elapsed_seconds=time.perf_counter()-started))
    print(json.dumps({k:v for k,v in result.items() if k!='stage_losses'},ensure_ascii=False),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        save(OUT/'state.json',dict(status='failed',pid=os.getpid(),error=repr(exc),traceback=traceback.format_exc()))
        raise
