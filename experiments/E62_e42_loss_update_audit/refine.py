"""Resolve clipping/step-length confounds using only the same five cases."""
import copy,json,os,time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from .io import ROOT,HERE,OUT,save,sha
from .cache import load_rows
from .batches import collate,inputs_only,decode
from .model import GeometryModel
from .losses import components,WEIGHTS
from .diagnostics import geometry_checks
from .check import restore_rng,flat

def read(p):return json.loads(p.read_text(encoding='utf-8'))

def main():
    started=time.perf_counter();torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    config=read(OUT/'config.json');old=read(OUT/'results.json');ids=config['train_indices']
    cp_path=ROOT/config['checkpoint'];assert sha(cp_path)==config['checkpoint_sha256']
    cp=torch.load(cp_path,map_location='cpu',weights_only=False);rows=load_rows('train',ids)
    g=collate([r['graph'] for r in rows],list(range(5)),'cuda');rr=[rows[i%5] for i in range(32)]
    train=collate([r['graph'] for r in rr],list(range(32)),'cuda')
    model=GeometryModel(**cp['model_config']).cuda();model.load_state_dict(cp['model']);params=list(model.parameters());slices=[];cursor=0
    for p in params:slices.append((cursor,cursor+p.numel()));cursor+=p.numel()
    theta0=torch.cat([p.detach().flatten() for p in params]).clone()
    model.train();restore_rng(cp['rng']);state=model(inputs_only(train));values=components(state,train)
    total=sum(w*values[k].mean() for k,w in WEIGHTS.items())
    center=flat(torch.autograd.grad(values['center_local'].mean(),params,retain_graph=True,allow_unused=True),params)
    full=flat(torch.autograd.grad(total,params,allow_unused=True),params);without=full-center
    def measure():
        model.eval()
        with torch.no_grad():
            state=model(inputs_only(g));cc=components(state,g);positions=(decode(state,g)*g['scale'][:,None,None]).cpu().double().numpy()
            records=[]
            for j,(row,p) in enumerate(zip(rows,positions)):
                p,_,diagnostic=geometry_checks(p[:row['graph']['n']],row['case'].solver_inputs)
                records.append({'train_index':ids[j],'area_delta':diagnostic['overlap_area_fraction']-old['baseline'][j]['overlap_area_fraction'],
                    'overlap_delta':float(cc['overlap'][j])-old['baseline'][j]['losses']['overlap'],
                    'center_loss_delta':float(cc['center_local'][j])-old['baseline'][j]['losses']['center_local'],'positions':p})
            return records
    def update(gradient,clip_factor=None):
        model.load_state_dict(cp['model']);opt=torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=.0003)
        opt.load_state_dict(copy.deepcopy(cp['optimizer']))
        for p,(a,b) in zip(params,slices):p.grad=gradient[a:b].view_as(p).clone()
        if clip_factor is None:torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
        else:
            for p in params:p.grad.mul_(clip_factor)
        opt.step();return torch.cat([p.detach().flatten() for p in params]).clone()
    full_theta=update(full);full_result=measure()
    without_theta=update(without);without_result=measure()
    for name,records in [('full_G',full_result),('without_center_local',without_result)]:
        for j,r in enumerate(records):assert np.array_equal(r['positions'],np.asarray(old['arms'][name]['post'][j]['positions']))
    full_clip=old['arms']['full_G']['gradient_clip_factor'];fixed_theta=update(without,full_clip);fixed_result=measure()
    fraction=float((full_theta-theta0).norm()/(without_theta-theta0).norm())
    matched=theta0+fraction*(without_theta-theta0)
    with torch.no_grad():
        for p,(a,b) in zip(params,slices):p.copy_(matched[a:b].view_as(p))
    matched_result=measure()
    def summary(records,theta):
        return {'mean_area_delta':float(np.mean([r['area_delta'] for r in records])),
            'mean_overlap_delta':float(np.mean([r['overlap_delta'] for r in records])),
            'area_improved_cases':sum(r['area_delta']<-2e-7 for r in records),'area_regressed_cases':sum(r['area_delta']>2e-7 for r in records),
            'step_norm':float((theta-theta0).norm()),'cases':[{k:v for k,v in r.items() if k!='positions'} for r in records]}
    result={'status':'complete','same_five_cases':ids,'original_two_updates_bit_exact':True,
        'full_G':summary(full_result,full_theta),'without_center_original_clip':summary(without_result,without_theta),
        'without_center_fixed_full_clip':summary(fixed_result,fixed_theta),'without_center_matched_full_step_norm':summary(matched_result,matched),
        'matched_step_fraction':fraction,'fixed_clip_factor':full_clip,
        'scope':'Fixed-clip and matched-length variants are causal geometry diagnostics, not proposed training policies or new model candidates',
        'elapsed_seconds':time.perf_counter()-started,'unique_training_cases':5,'test_read':False,'formal_training_runs':0}
    assert sha(cp_path)==config['checkpoint_sha256']
    save(OUT/'refinement.json',result)
    print(json.dumps({k:{'area_delta':v['mean_area_delta'],'improved':v['area_improved_cases'],'step_norm':v['step_norm']} for k,v in result.items() if isinstance(v,dict) and 'mean_area_delta' in v}),flush=True)

if __name__=='__main__':main()
