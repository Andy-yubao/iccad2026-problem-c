"""Four independent train-only diagnostics; no reuse of check200 for tuning."""
from pathlib import Path
import itertools
import json
import numpy as np
import torch
import torch.nn.functional as F
from ..common import OWNER,RUN as BANK,KEYS,read,save,sha,setup,case_weights,q_summary
from ..analysis import fit
from ..batches import decode
from ..simple_loss import components
from ..loss_config import A,B,COEFFICIENTS
from ..worker import loss_graph

HERE=Path(__file__).resolve().parent
RUN=HERE/'runs/20261104_mechanism_v1'

def data():
    a=np.load(RUN/'fit_tune.npz')
    return a['x'],a['q'],a['n'],a['indices']

def prediction_metrics(pred,q,n):
    chosen=np.argmin(pred,axis=1);cost=q[np.arange(len(q)),chosen];oracle=q.min(1)
    per=[];available=[]
    for i in range(len(q)):
        correct=[]
        for a,b in itertools.combinations(range(5),2):
            if abs(q[i,a]-q[i,b])<=1e-4:continue
            dl=pred[i,a]-pred[i,b]
            correct.append(.5 if abs(dl)<=1e-12 else float(dl*(q[i,a]-q[i,b])>0))
        if correct:per.append(np.mean(correct));available.append(i)
    w=case_weights(n);aw=w[available];aw/=aw.sum()
    return dict(**q_summary(cost,n),mixed_regret=float((cost-oracle)@w),
        pair_accuracy=float(np.array(per)@aw),selected_source_counts=np.bincount(chosen,minlength=5).tolist())

def predict_linear(x,co):return x@np.array([co[k] for k in KEYS])

def original_prediction_checks(x,q,n):
    learned=read(BANK/'analysis/proposal.json')['physical_coefficients']
    return {name:prediction_metrics(predict_linear(x,co),q,n) for name,co in (
        ('original',COEFFICIENTS),('learned_v1',learned))}

def capacity():
    x,q,n,_=data();tr=slice(0,600);tu=slice(600,800)
    difference=np.concatenate([x[tr,a]-x[tr,b] for a,b in itertools.combinations(range(5),2)])
    scale=np.maximum(np.quantile(np.abs(difference),.75,axis=0),1e-6)
    z=x/scale
    result=dict(task='capacity_and_source_confound',tune=original_prediction_checks(x[tu],q[tu],n[tu]))
    cw=case_weights(n[tr]);y=q[tr]-q[tr].mean(1,keepdims=True)
    for degree in (1,2):
        features=z if degree==1 else np.concatenate([z,np.stack([z[...,a]*z[...,b]
            for a in range(10) for b in range(a,10)],axis=-1)],axis=-1)
        centered=features[tr]-features[tr].mean(1,keepdims=True)
        normal=np.sqrt(np.einsum('i,imk->k',cw,centered**2)/5).clip(1e-8)
        design=centered/normal
        lhs=np.einsum('i,imk,iml->kl',cw,design,design)/5+.01*np.eye(design.shape[-1])
        rhs=np.einsum('i,imk,im->k',cw,design,y)/5
        coefficients=np.linalg.solve(lhs,rhs)
        pred=(features[tu]/normal)@coefficients
        result['tune'][f'signed_ridge_degree{degree}']=prediction_metrics(pred,q[tu],n[tu])
        result[f'degree{degree}_details']=dict(features=design.shape[-1],fixed_ridge=.01,
            fitted_case_count=600,negative_coefficients=int((coefficients<0).sum()))
    source_offset=cw@y
    result['tune']['source_only_control']=prediction_metrics(np.broadcast_to(source_offset,(200,5)),q[tu],n[tu])
    result['tune']['always_e42_control']=prediction_metrics(np.broadcast_to([0.,1.,1.,1.,1.],(200,5)),q[tu],n[tu])
    result['interpretation']='Signed/quadratic fits and source-ID controls diagnose expressivity/confounding only; they are not adopted as training objectives and may be unbounded. No check200 is used.'
    save(RUN/'capacity.json',result);return result

def stability():
    x,q,n,_=data();rng=np.random.default_rng(2026110441)
    lam=read(BANK/'analysis/proposal.json')['regularization']
    values=[];scores=[]
    for _ in range(32):
        sample=rng.integers(0,600,size=600)
        r=fit(x[sample],q[sample],n[sample],lam,list(range(5)))
        co=r['physical_coefficients'];values.append([co[k] for k in KEYS])
        scores.append(prediction_metrics(predict_linear(x[600:],co),q[600:],n[600:]))
    values=np.array(values)
    result=dict(task='fixed_lambda_case_bootstrap',samples=32,seed=2026110441,regularization=lam,
        coefficients={k:dict(p10=float(np.quantile(values[:,j],.1)),median=float(np.median(values[:,j])),
            p90=float(np.quantile(values[:,j],.9)),near_zero_fraction=float(np.mean(values[:,j]<1e-8)))
            for j,k in enumerate(KEYS)},
        tune_accuracy_interval=np.quantile([s['pair_accuracy'] for s in scores],[.1,.5,.9]).tolist(),
        tune_regret_interval=np.quantile([s['mixed_regret'] for s in scores],[.1,.5,.9]).tolist(),
        interpretation='Descriptive case-bootstrap stability; no bootstrap winner selected, no coefficient update or new hyperparameter tuning.')
    save(RUN/'stability.json',result);return result

def geometry_supervision(state,g):
    p=decode(state,g)
    local=g['area'].clamp_min(1e-12).sqrt()[...,None]
    center=p[...,:2]+p[...,2:]/2
    mask=g['mask']*g['valid'][...,None]
    target=(g['target'][...,:2]/local).expand_as(center)
    position=F.smooth_l1_loss(center/local,target,reduction='none',beta=.1)
    position=(position*mask[...,:2]).sum((1,2))/mask[...,:2].sum((1,2)).clamp_min(1)
    shape=((p[...,2].clamp_min(1e-12)/p[...,3].clamp_min(1e-12)).log()-g['target'][...,2])**2
    shape=(shape*mask[...,2]).sum(1)/mask[...,2].sum(1).clamp_min(1)
    return position,shape

def canonical_view():
    x,q,n,indices=data();canonical=x.copy();affected=0;movable=0
    for local,index in enumerate(indices):
        row=torch.load(BANK/f'inputs/{index:04d}.pt',map_location='cpu',weights_only=False)
        core=read(BANK/f'bank/{index:04d}.json');g=loss_graph(row)
        state=torch.tensor([d['state'] for d in core['drafts']],dtype=torch.float32)
        # Broadcast only public geometry and supervision tensors for five drafts.
        with torch.no_grad():
            pos,shape=geometry_supervision(state,g)
            canonical[local,:,0]=pos.numpy();canonical[local,:,1]=shape.numpy()
            p=decode(state,g);before=state[...,:2]-p[...,2:]/2
            affected+=int(((before<0)&g['mask'][...,:2].bool()).sum())
            movable+=int(g['mask'][...,:2].sum())*5
    lam=read(BANK/'analysis/proposal.json')['regularization']
    fitted=fit(canonical[:600],q[:600],n[:600],lam,list(range(5)))
    evaluation=prediction_metrics(predict_linear(canonical[600:],fitted['physical_coefficients']),q[600:],n[600:])
    baseline=original_prediction_checks(x[600:],q[600:],n[600:])['learned_v1']
    gate=(evaluation['pair_accuracy']>=baseline['pair_accuracy']+.02 and
        evaluation['mixed_regret']<=baseline['mixed_regret']-.0005)
    result=dict(task='decoded_geometry_supervision',fitted=fitted,tune=evaluation,raw_v1_tune=baseline,
        clipped_center_coordinate_fraction=affected/max(1,movable),
        absolute_position_term_change=dict(mean=float(np.abs(canonical[:,:,0]-x[:,:,0]).mean()),max=float(np.abs(canonical[:,:,0]-x[:,:,0]).max())),
        geometry_candidate_gate_passed=bool(gate),
        gate='tune ranking +2 percentage points and mixed regret reduction>.0005 versus frozen raw-v1; fixed previously selected lambda; no check200',
        interpretation='Only position/shape supervision is recomputed from actual decoded geometry; eight public geometry terms remain the original saved terms. This diagnoses hidden raw-state/clipping effects; not adopted automatically.')
    np.savez_compressed(RUN/'canonical_fit_tune.npz',x=canonical,q=q,n=n,indices=indices)
    save(RUN/'canonical_view.json',result);return result

def descent_diagnosis():
    panel=read(OWNER/'panel.json');co=read(BANK/'analysis/proposal.json')['physical_coefficients']
    records=[]
    for index in panel['direction_indices']:
        row=torch.load(BANK/f'inputs/{index:04d}.pt',map_location='cpu',weights_only=False)
        core=read(BANK/f'bank/{index:04d}.json');old=read(BANK/f'direction/{index:04d}.json')
        d=next(d for d in core['drafts'] if d['model_id']=='e98_generalization');g=loss_graph(row)
        original=torch.tensor([d['state']],dtype=torch.float32)
        unit=torch.cat((g['area'].sqrt()[...,None].expand(-1,-1,2),torch.ones_like(g['area'][...,None])),-1)
        active=g['mask']*g['valid'][...,None]
        for name,weights in (('baseline',COEFFICIENTS),('learned',co)):
            u=(original/unit).detach().requires_grad_(True)
            values=components(u*unit,g,dict(overlap=A,task=B))
            before=sum(weights[k]*values[k].mean() for k in KEYS)
            grad=torch.autograd.grad(before,u)[0]*active
            rms=(grad.square().sum()/active.sum().clamp_min(1)).sqrt().clamp_min(1e-12)
            for step in (.002,.01):
                existing=next(v for v in old['results'] if v['objective']==name and v['step']==step)
                assert np.isclose(float(rms),existing['gradient_rms'],rtol=1e-6,atol=1e-10)
                after_state=(u.detach()-step*(grad/rms).clamp(-3,3))*unit
                with torch.no_grad():
                    after_values=components(after_state,g,dict(overlap=A,task=B))
                    after=sum(weights[k]*after_values[k].mean() for k in KEYS)
                records.append(dict(index=index,n=core['n'],objective=name,step=step,
                    loss_before=float(before.detach()),loss_after=float(after),loss_delta=float(after-before.detach()),
                    Q_delta=existing['Q_delta'],
                    identical_final_geometry=bool(np.array_equal(existing['positions'],d['positions'])),
                    component_delta={k:float(after_values[k]-values[k].detach()) for k in KEYS}))
    summary={}
    for name in ('baseline','learned'):
        for step in (.002,.01):
            r=[v for v in records if v['objective']==name and v['step']==step]
            w=case_weights([v['n'] for v in r])
            summary[f'{name}_{step}']=dict(loss_decreased=sum(v['loss_delta']<0 for v in r),
                loss_increased=sum(v['loss_delta']>1e-9 for v in r),
                lower_loss_but_worse_Q=sum(v['loss_delta']<0 and v['Q_delta']>1e-4 for v in r),
                identical_final_geometry=sum(v['identical_final_geometry'] for v in r),
                mixed_Q_delta=float(np.array([v['Q_delta'] for v in r])@w),
                worst_Q_deltas=sorted([dict(index=v['index'],Q_delta=v['Q_delta'],loss_delta=v['loss_delta']) for v in r],key=lambda v:v['Q_delta'],reverse=True)[:3])
    result=dict(task='descent_failure_mechanism',summary=summary,cases=40,records=records,
        postprocessor_reexecutions=0,interpretation='Recomputed actual fixed-Loss changes against the existing160 perturbation labels; unchanged final geometry identifies repair plateaus; lower Loss with higherQ identifies objective misalignment.')
    save(RUN/'descent_diagnosis.json',result);return result

TASKS={'capacity':capacity,'stability':stability,'canonical_view':canonical_view,'descent':descent_diagnosis}
def run_task(name):
    setup()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        result=TASKS[name]()
    return dict(task=name,status='complete')
