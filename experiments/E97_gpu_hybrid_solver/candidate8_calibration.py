"""One pre-registered train500 state-gradient calibration; no updates or Q calls."""
import datetime,json,hashlib
from pathlib import Path
import numpy as np
from .candidate7 import HERE,ROOT,torch,seed,teacher_rows,attach,load_rows,registry,sha
from .normalized_geometry import DeploymentModel
from .candidate8_io import Packer,network_inputs
from .candidate8_objective import raw_terms
from .train_io.io import save

FOLDER=HERE/'runs/20261102_candidate8_calibration'
def weighted_median(v,w):
    order=np.argsort(v);values=np.asarray(v)[order];weights=np.asarray(w)[order]
    return float(values[np.searchsorted(np.cumsum(weights),weights.sum()/2,side='left')])

def main():
    FOLDER.mkdir(exist_ok=False);seed();plan=json.loads((HERE/'candidate8_calibration_plan.json').read_text(encoding='utf-8'))
    assert sha(HERE/'candidate8_objective.py')==plan['objective_source_sha256'];path=HERE/plan['source_checkpoint'];assert sha(path)==plan['source_checkpoint_sha256']
    saved=torch.load(path,map_location='cpu',weights_only=False);model=DeploymentModel(**saved['model_config']).cuda().eval();model.load_state_dict(saved['model'],strict=True);model.requires_grad_(False)
    labels=teacher_rows('train');rows=load_rows('train_monitor500');lookup={tuple(v['member']):i for i,v in enumerate(registry()['sets']['train']['records'][:16000])};indices=[lookup[tuple(r['member'])] for r in rows];attach(rows,labels,indices)
    counts=np.array([r['n'] for r in registry()['sets']['train']['records'][:16000]]);count_normalizer=float(np.exp((counts-120)/12).mean())
    packer=Packer(rows[0]);records=[];keys=['teacher','physics','corridor','quality']
    deadline=datetime.datetime.fromisoformat(json.loads((HERE/'study.json').read_text(encoding='utf-8'))['deadline_utc'])
    for start in range(0,500,32):
        if (HERE/'runs/STOP').exists() or datetime.datetime.now(datetime.timezone.utc)>=deadline:raise RuntimeError('STOP/deadline')
        selected=rows[start:start+32];g,d,t,u,count=packer.gpu(selected)
        assert set(network_inputs(g)).isdisjoint({'port_xy','port_valid','p2b_weights','b2b_weights'})
        with torch.no_grad():prediction=model.refiner(d,network_inputs(g))
        state=prediction.detach().requires_grad_(True);values=raw_terms(state,g,t,u)
        local=g['area'].clamp_min(1e-12).sqrt();unit=torch.stack((local,local,torch.ones_like(local)),-1)*g['mask']*g['valid'][...,None]
        gradients=[]
        for j,key in enumerate(keys):gradients.append(torch.autograd.grad(values[key].sum(),state,retain_graph=j<len(keys)-1)[0]*unit)
        norms=torch.stack([v.square().sum((1,2)).sqrt() for v in gradients]);scalar_keys=list(values);scalars=torch.stack([values[k].detach() for k in scalar_keys])
        cos=torch.stack([(gradients[0]*x).sum((1,2))/(norms[0]*norms[j]).clamp_min(1e-20) for j,x in enumerate(gradients[1:],1)])
        n=g['valid'].sum(1).cpu().tolist();norms=norms[:,:count].detach().cpu().numpy();scalars=scalars[:,:count].cpu().numpy();cos=cos[:,:count].detach().cpu().numpy()
        if not np.isfinite(norms).all() or not np.isfinite(scalars).all():raise RuntimeError('Nonfinite calibration')
        for j,row in enumerate(selected):records.append(dict(index=start+j,member=list(row['member']),n=int(n[j]),count_weight=float(np.exp((n[j]-120)/12)),values={k:float(scalars[i,j]) for i,k in enumerate(scalar_keys)},local_state_gradient_norms={k:float(norms[i,j]) for i,k in enumerate(keys)},teacher_gradient_cosine={k:float(cos[i,j]) for i,k in enumerate(keys[1:])}))
        save(FOLDER/'progress.json',dict(completed=len(records),formal_training_started=False,test_read=False));print('train calibration',len(records),'/500',flush=True)
        del values,state,prediction,gradients
    weights={};details={}
    for key,share in plan['shares'].items():
        active=[r for r in records if r['local_state_gradient_norms'][key]>1e-10 and r['local_state_gradient_norms']['teacher']>1e-10]
        if len(active)<5:raise RuntimeError('Fewer than5 active calibration members: '+key)
        w=[r['count_weight'] for r in active];anchor=weighted_median([r['local_state_gradient_norms']['teacher'] for r in active],w);component=weighted_median([r['local_state_gradient_norms'][key] for r in active],w)
        coefficient=share*anchor/component
        if not np.isfinite(coefficient) or not 0<coefficient<=1000:raise RuntimeError('Coefficient scale gate: '+key)
        weights[key]=coefficient;details[key]=dict(active_count=len(active),weighted_median_teacher_gradient=anchor,weighted_median_component_gradient=component,registered_share=share)
    binding=dict(source_checkpoint=plan['source_checkpoint'],source_checkpoint_sha256=sha(path),calibration_plan_sha256=sha(HERE/'candidate8_calibration_plan.json'),sources={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),HERE/'candidate8_io.py',HERE/'candidate8_objective.py',HERE/'candidate7.py',HERE/'normalized_geometry.py']},member_sha256=hashlib.sha256(json.dumps([r['member'] for r in records],separators=(',',':')).encode()).hexdigest())
    result=dict(status='complete',weights=weights,count_normalizer=count_normalizer,count_weight='exp((n-120)/12)/fixed first16k arithmetic mean',details=details,binding=binding,records=records,no_optimizer_updates=True,no_dev_or_Q_calls=True,formal_slots_consumed=0,test_read=False)
    save(FOLDER/'results.json',result);save(FOLDER/'weights.json',{k:v for k,v in result.items() if k!='records'});print(json.dumps(dict(weights=weights,count_normalizer=count_normalizer,details=details),indent=2),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as error:save(FOLDER/'error.json',dict(error=repr(error),no_automatic_retry=True));raise
