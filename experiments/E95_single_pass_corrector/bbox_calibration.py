"""One analytic bbox coefficient, using only the original five train cases."""
import json,statistics,time
import torch
from .io import HERE,ROOT,save,sha
from .runtime import CONFIG,seed,charge
from .data import load_rows,collate,inputs_only
from .model import GeometryModel,decode
from .initialization import load
from .loss import per_case

FOLDER=HERE/'runs/20261101_bbox_preparation'

def box(state,g):
    p=decode(state,g);valid=g['valid'][...,None]
    lo=p[...,:2].masked_fill(~valid,float('inf')).amin(1)
    hi=(p[...,:2]+p[...,2:]).masked_fill(~valid,-float('inf')).amax(1)
    size=hi-lo
    return size[:,0]*size[:,1]/(g['area']*g['valid']).sum(1)

def main():
    rule=json.loads((FOLDER/'calibration_rule.json').read_text(encoding='utf-8'))
    seed();rows=load_rows('train',rule['ids']);g=inputs_only(collate(rows,'cuda'))
    model=GeometryModel(**CONFIG).cuda().eval();load(model)
    with torch.no_grad():draft=model.draft(g)
    state=draft.detach().clone().requires_grad_();v=per_case(state,g,draft);b=box(state,g)
    evidence=[]
    for i,index in enumerate(rule['ids']):
        go=torch.autograd.grad(v['overlap_total'][i],state,retain_graph=True)[0][i]*g['mask'][i]
        gb=torch.autograd.grad(b[i],state,retain_graph=True)[0][i]*g['mask'][i]
        no=float(go.norm());nb=float(gb.norm())
        assert no>0 and nb>1e-10
        evidence.append(dict(index=index,overlap_gradient_norm=no,bbox_gradient_norm=nb,
            ratio=no/nb,cosine=float((go*gb).sum()/(go.norm()*gb.norm())),bbox=float(b[i])))
    coefficient=float(format(statistics.median(v['ratio'] for v in evidence),'.3g'))
    assert coefficient>.05 and coefficient<100
    # Inspect one fixed old checkpoint; these five train cases do not select it.
    old=HERE/'runs/20261101_normalized_local_seed0/checkpoints/best.pt'
    saved=torch.load(old,map_location='cpu',weights_only=False);model.load_state_dict(saved['model'])
    with torch.no_grad():
        corrected=model.corrector(g,draft);after=per_case(corrected,g,draft);after_box=box(corrected,g)
        nonbox=lambda x:x['overlap_total']+.05*x['wirelength']+.2*x['soft']+.1*x['intent']
        ranking=[dict(index=index,initial_bbox=float(b[i]),old_corrected_bbox=float(after_box[i]),
            initial_overlap=float(v['overlap_total'][i]),old_corrected_overlap=float(after['overlap_total'][i]),
            new_loss_initial=float(nonbox(v)[i]+coefficient*b[i]),
            new_loss_old_corrected=float(nonbox(after)[i]+coefficient*after_box[i])) for i,index in enumerate(rule['ids'])]
    result=dict(status='passed',box_weight=coefficient,previous_effective_box_weight=.05,
        multiplier=coefficient/.05,rule_sha256=sha(FOLDER/'calibration_rule.json'),
        initial_gradients=evidence,ranking_diagnostic=ranking,old_checkpoint_sha256=sha(old),
        no_dev_or_test_loaded=True,reference_in_objective=False,optimizer_updates=0,
        note='analytic gradient scale hypothesis, not an optimum or a guarantee of nonexpansion')
    save(FOLDER/'calibration.json',result);print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    started=time.perf_counter()
    try:main()
    finally:charge(time.perf_counter()-started)
