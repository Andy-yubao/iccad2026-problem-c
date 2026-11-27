"""Same-state E42 gradient ablations and actual AdamW finite-step outcomes."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
import copy,json,random,time
from pathlib import Path
import numpy as np
import torch
from .io import ROOT,HERE,OUT,save,sha
from .cache import load_rows
from .batches import collate,inputs_only,decode
from .model import GeometryModel
from .losses import components,WEIGHTS
from .diagnostics import geometry_checks

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def flat(grads,params):return torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten() for g,p in zip(grads,params)])
def cos(a,b):
    denom=float(a.norm()*b.norm())
    return float(a.dot(b)/denom) if denom>1e-20 else None

def area_values(state,g):
    p=decode(state,g);lo=torch.maximum(p[:,:,None,:2],p[:,None,:,:2]);hi=torch.minimum((p[...,:2]+p[...,2:])[:,:,None],(p[...,:2]+p[...,2:])[:,None])
    inter=(hi-lo).clamp_min(0).prod(-1);valid=g['valid'];mask=valid[:,:,None]&valid[:,None,:]&~torch.eye(p.shape[1],dtype=torch.bool,device=p.device)[None]
    return (inter*mask).sum((1,2))/(2*g['area'].sum(1).clamp_min(1e-12))

def restore_rng(rng):
    random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch'])
    if rng['cuda'] is not None:torch.cuda.set_rng_state_all(rng['cuda'])

def main():
    started=time.perf_counter();torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    OUT.mkdir(parents=True,exist_ok=True)
    def guard():
        if time.perf_counter()-started>900:raise RuntimeError('900-second diagnostic budget reached')
    checkpoint=ROOT/'experiments/E42_joint_regularization/runs/20261023_rectangle_w96d12_ab_seed0/checkpoints/best.pt'
    digest=sha(checkpoint);assert digest=='12878108e9a58b8c195b314ba6586686c2a6e2796c1f4b6443e102b4db4224a6'
    cp=torch.load(checkpoint,map_location='cpu',weights_only=False)
    gallery=read(ROOT/'experiments/E61_explosion_replay/runs/20261027_large64000_graph_step_seed0/train_draft_comparison100_seed20261027/manifest.json')
    shared=[r for r in gallery['cases'] if r['shared_training_member']];ids=[]
    for n in (25,45,70,95,115):
        r=min((r for r in shared if r['train_index'] not in ids),key=lambda r:(abs(r['n']-n),r['train_index']));ids.append(r['train_index'])
    assert len(set(ids))==5 and all(i<16000 and cp['progress']['seen'][i] for i in ids)
    rows=load_rows('train',ids);g=collate([r['graph'] for r in rows],list(range(5)),'cuda')
    train_rows=[rows[i%5] for i in range(32)];g_train=collate([r['graph'] for r in train_rows],list(range(32)),'cuda')
    config={'checkpoint':str(checkpoint.relative_to(ROOT)),'checkpoint_sha256':digest,'source_epoch':cp['progress']['epoch'],'source_step':cp['progress']['step'],
        'selection':'nearest public module counts25/45/70/95/115 among27 reviewed shared-training members; no diagnostic quality used',
        'train_indices':ids,'members':[list(r['member']) for r in rows],'unique_cases':5,'training_batch_repetitions':[7,7,6,6,6],
        'optimizer':{k:v for k,v in cp['optimizer']['param_groups'][0].items() if k!='params'},
        'grad_clip':1.,'fixed_weights':WEIGHTS,'Dropout':'one shared original-checkpoint RNG forward; all candidate gradients use exactly this realization',
        'evaluation':'five unique eval/no_grad layouts; no dropout noise in finite-difference outcomes',
        'budget_seconds':900,'formal_training_runs':0,'postprocessing_calls':0,'test_read':False,
        'sources':{p.name:sha(p) for p in HERE.glob('*.py')}}
    for name in ('model.py','features.py','batches.py','losses.py'):
        assert sha(HERE/name)==sha(ROOT/'experiments/E42_joint_regularization'/name)
    save(OUT/'config.json',config)
    model=GeometryModel(**cp['model_config']).cuda();model.load_state_dict(cp['model']);params=list(model.parameters())
    slices=[];cursor=0
    for p in params:slices.append((cursor,cursor+p.numel()));cursor+=p.numel()
    theta0=torch.cat([p.detach().flatten() for p in params]).clone()
    def outcomes(state):
        with torch.no_grad():
            cc=components(state,g);positions=(decode(state,g)*g['scale'][:,None,None]).cpu().double().numpy();rr=[]
            for j,(row,pos) in enumerate(zip(rows,positions)):
                pos,checks,diag=geometry_checks(pos[:row['graph']['n']],row['case'].solver_inputs)
                rr.append({'train_index':ids[j],'member':list(row['member']),'n':row['graph']['n'],
                    'losses':{k:float(cc[k][j]) for k in WEIGHTS},'G':sum(w*float(cc[k][j]) for k,w in WEIGHTS.items()),
                    'hard_legal':bool(checks['is_feasible']),'positions':pos.tolist(),**diag})
            return rr
    model.eval();eval_state=model(inputs_only(g));baseline=outcomes(eval_state)
    overlap_eval=[];area_eval=[];eval_area=area_values(eval_state,g)
    for j in range(5):
        overlap_eval.append(flat(torch.autograd.grad(.5*components(eval_state,g)['overlap'][j],params,retain_graph=True,allow_unused=True),params))
        area_eval.append(flat(torch.autograd.grad(eval_area[j],params,retain_graph=True,allow_unused=True),params))
    assert all(abs(float(eval_area[j])-baseline[j]['overlap_area_fraction'])<1e-6 for j in range(5))
    target=outcomes(g['target'])
    model.train();restore_rng(cp['rng']);state=model(inputs_only(g_train));values=components(state,g_train)
    terms={k:w*values[k].mean() for k,w in WEIGHTS.items()};vectors={}
    output_vectors={}
    for k,term in terms.items():
        vectors[k]=flat(torch.autograd.grad(term,params,retain_graph=True,allow_unused=True),params)
        output_vectors[k]=torch.autograd.grad(term,state,retain_graph=True)[0].detach().flatten()
    direct=flat(torch.autograd.grad(sum(terms.values()),params,retain_graph=False,allow_unused=True),params)
    sum_error=float((direct-sum(vectors.values())).norm()/direct.norm().clamp_min(1e-12));assert sum_error<1e-5
    baseline_train={k:float(v.detach()) for k,v in terms.items()}
    psummary={'weighted_grad_norm':{k:float(v.norm()) for k,v in vectors.items()},
        'parameter_cosine_with_overlap':{k:cos(v,vectors['overlap']) for k,v in vectors.items()},
        'output_cosine_with_overlap':{k:cos(v,output_vectors['overlap']) for k,v in output_vectors.items()},
        'full_parameter_cosine_with_overlap':cos(direct,vectors['overlap']),'sum_relative_error':sum_error}
    arms={'full_G':direct,**{'without_'+k:direct-v for k,v in vectors.items()},
          'overlap_only':vectors['overlap'],'momentum_and_decay_only':torch.zeros_like(direct),'full_G_repeat':direct}
    results={};full_theta=None
    for label,gradient in arms.items():
        guard();model.load_state_dict(cp['model']);optimizer=torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=.0003)
        optimizer.load_state_dict(copy.deepcopy(cp['optimizer']))
        for p,(a,b) in zip(params,slices):p.grad=gradient[a:b].view_as(p).clone()
        preclip=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
        optimizer.step();theta=torch.cat([p.detach().flatten() for p in params]);delta=theta-theta0
        if label=='full_G':full_theta=theta.clone()
        if label=='full_G_repeat':assert torch.equal(theta,full_theta)
        model.eval()
        with torch.no_grad():post=outcomes(model(inputs_only(g)))
        rr=[]
        for j,(a,b) in enumerate(zip(baseline,post)):
            p0=np.asarray(a['positions']);p1=np.asarray(b['positions']);area=np.asarray(rows[j]['case'].solver_inputs[1])
            move=np.linalg.norm((p1[:,:2]+p1[:,2:]/2)-(p0[:,:2]+p0[:,2:]/2),axis=1)/np.sqrt(area)
            centres0=p0[:,:2]+p0[:,2:]/2;centres1=p1[:,:2]+p1[:,2:]/2;i,k=np.triu_indices(len(p0),1)
            dif0=centres0[i]-centres0[k];dif1=centres1[i]-centres1[k];scale=(np.sqrt(area[i])+np.sqrt(area[k]))/2
            reliable=abs(dif0)>.25*scale[:,None];flips=(dif0*dif1<0)&reliable
            rr.append({'train_index':ids[j],'overlap_delta':b['losses']['overlap']-a['losses']['overlap'],
                'area_delta':b['overlap_area_fraction']-a['overlap_area_fraction'],
                'small_load_delta':b['module_load_small_mean']-a['module_load_small_mean'],
                'G_delta':b['G']-a['G'],'mean_center_move_local':float(move.mean()),'max_center_move_local':float(move.max()),
                'coarse_order_flips':int(flips.sum()),'coarse_order_axes_count':int(reliable.sum()),
                'first_order_weighted_overlap_change':float(overlap_eval[j].dot(delta)),
                'first_order_area_change':float(area_eval[j].dot(delta)),
                'loss_deltas':{q:b['losses'][q]-a['losses'][q] for q in WEIGHTS}})
        results[label]={'gradient_norm_before_clip':preclip,'gradient_clip_factor':min(1.,1./(preclip+1e-6)),
            'parameter_step_norm':float(delta.norm()),'cosine_step_with_overlap_descent':cos(delta,-vectors['overlap']),
            'mean_area_delta':float(np.mean([x['area_delta'] for x in rr])),
            'mean_overlap_delta':float(np.mean([x['overlap_delta'] for x in rr])),
            'area_improved_cases':sum(x['area_delta']<-2e-7 for x in rr),
            'area_regressed_cases':sum(x['area_delta']>2e-7 for x in rr),'cases':rr,
            'post':post,'optimizer_steps_after':sorted(set(int(s['step']) for s in optimizer.state.values()))}
        print(json.dumps({'arm':label,'area_delta':results[label]['mean_area_delta'],'overlap_delta':results[label]['mean_overlap_delta'],
            'step_norm':results[label]['parameter_step_norm'],'area_improved_cases':results[label]['area_improved_cases']},ensure_ascii=False),flush=True)
        save(OUT/'partial.json',{'baseline':baseline,'gradient_summary':psummary,'arms':results,'test_read':False})
    # Common full-gradient preconditioner explains current-term versus historical-momentum forces.
    model.load_state_dict(cp['model']);temporary=torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=.0003)
    temporary.load_state_dict(copy.deepcopy(cp['optimizer']));group=temporary.param_groups[0];beta1,beta2=group['betas'];coef=min(1.,1./(float(direct.norm())+1e-6))
    denominator=[];momentum=[];decay=[];current_factor=[]
    for p,(a,b) in zip(params,slices):
        st=temporary.state[p];step=int(st['step'])+1;v=st['exp_avg_sq'].flatten();m=st['exp_avg'].flatten();grad=direct[a:b]*coef
        v_new=beta2*v+(1-beta2)*grad.square();denom=(v_new/(1-beta2**step)).sqrt()+group['eps']
        scale=-group['lr']/((1-beta1**step)*denom)
        momentum.append(scale*beta1*m);current_factor.append(scale*(1-beta1)*coef)
        decay.append(-group['lr']*group['weight_decay']*p.detach().flatten())
    factor=torch.cat(current_factor);directions={'historical_momentum':torch.cat(momentum),'weight_decay':torch.cat(decay)}
    directions.update({k:factor*v for k,v in vectors.items()})
    forces={label:{'mean_first_order_area_change':float(torch.stack([v.dot(direction) for v in area_eval]).mean()),
        'mean_first_order_weighted_overlap_change':float(torch.stack([v.dot(direction) for v in overlap_eval]).mean()),
        'per_case_area_change':[float(v.dot(direction)) for v in area_eval]} for label,direction in directions.items()}
    analytic=sum(directions.values());quant_error=float((analytic-(full_theta-theta0)).norm()/(full_theta-theta0).norm().clamp_min(1e-12))
    assert sha(checkpoint)==digest
    assert sorted(set(int(s['step']) for s in cp['optimizer']['state'].values()))==[cp['progress']['step']]
    output={'status':'complete','baseline':baseline,'reference_state_decoded':target,'baseline_train_weighted_terms':baseline_train,
        'gradient_summary':psummary,'arms':results,'full_preconditioner_direction_decomposition':forces,
        'analytic_vs_actual_step_relative_rounding_residual':quant_error,'elapsed_seconds':time.perf_counter()-started,
        'source_checkpoint_unchanged':True,'source_optimizer_in_memory_unchanged':True,'full_repeat_bit_exact':True,
        'unique_training_cases':5,'formal_training_runs':0,'test_read':False,'postprocessing_calls':0,
        'limits':['Five adaptive-development training cases, repeated into32; not representative training minibatch or generalization evidence',
            'One-step causal interventions retain historical Adam moments; they do not establish long-term removal benefits',
            'Every leave-one-out keeps original clipping policy, so changes include its normalization effect',
            'Common-preconditioner decomposition is explanatory only; leave-one-out recomputes actual Adam denominator',
            'Eval finite changes use deterministic drafts; the shared training gradient includes one fixed Dropout realization',
            'No new model candidate, checkpoint selection, formal training, solver or Q evaluation']}
    save(OUT/'results.json',output)
    print(json.dumps({'complete':True,'seconds':output['elapsed_seconds'],'rounding_residual':quant_error}),flush=True)

if __name__=='__main__':main()
