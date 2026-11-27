"""Same-case focused checks against Dropout noise and adjacent checkpoint time."""
import copy,json,os,time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from .io import ROOT,OUT,save,sha
from .cache import load_rows
from .batches import collate,inputs_only,decode
from .model import GeometryModel
from .losses import components,WEIGHTS
from .diagnostics import geometry_checks
from .check import flat,restore_rng

def main():
    begin=time.perf_counter();torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    cfg=json.loads((OUT/'config.json').read_text(encoding='utf-8'));ids=cfg['train_indices'];rows=load_rows('train',ids)
    g=collate([r['graph'] for r in rows],list(range(5)),'cuda');rr=[rows[i%5] for i in range(32)]
    batch=collate([r['graph'] for r in rr],list(range(32)),'cuda');result={}
    for label,file,training in [('best51_eval_gradient','best.pt',False),('latest54_train_gradient','latest.pt',True)]:
        path=ROOT/'experiments/E42_joint_regularization/runs/20261023_rectangle_w96d12_ab_seed0/checkpoints'/file;digest=sha(path)
        cp=torch.load(path,map_location='cpu',weights_only=False);model=GeometryModel(**cp['model_config']).cuda();model.load_state_dict(cp['model']);params=list(model.parameters())
        slices=[];cursor=0
        for p in params:slices.append((cursor,cursor+p.numel()));cursor+=p.numel()
        def metrics():
            model.eval()
            with torch.no_grad():
                state=model(inputs_only(g));c=components(state,g);pos=(decode(state,g)*g['scale'][:,None,None]).cpu().double().numpy();out=[]
                for j,row in enumerate(rows):
                    _,_,d=geometry_checks(pos[j,:row['graph']['n']],row['case'].solver_inputs)
                    out.append({'area':d['overlap_area_fraction'],'overlap':float(c['overlap'][j]),'G':sum(w*float(c[k][j]) for k,w in WEIGHTS.items())})
                return out
        baseline=metrics();model.train(training);restore_rng(cp['rng']);state=model(inputs_only(batch));v=components(state,batch)
        center=flat(torch.autograd.grad(v['center_local'].mean(),params,retain_graph=True,allow_unused=True),params)
        full=flat(torch.autograd.grad(sum(w*v[k].mean() for k,w in WEIGHTS.items()),params,allow_unused=True),params)
        arms={}
        for arm,grad in [('full_G',full),('without_center',full-center)]:
            model.load_state_dict(cp['model']);optimizer=torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=.0003)
            optimizer.load_state_dict(copy.deepcopy(cp['optimizer']))
            for p,(a,b) in zip(params,slices):p.grad=grad[a:b].view_as(p).clone()
            norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True));optimizer.step();post=metrics()
            delta=[p['area']-b['area'] for b,p in zip(baseline,post)]
            arms[arm]={'mean_area_delta':float(np.mean(delta)),'area_improved_cases':sum(x<-2e-7 for x in delta),
                'area_regressed_cases':sum(x>2e-7 for x in delta),'per_case_area_deltas':delta,'gradient_norm_before_clip':norm}
        assert sha(path)==digest
        result[label]={'checkpoint_sha256':digest,'step':cp['progress']['step'],'gradient_Dropout_active':training,'arms':arms,'source_unchanged':True}
        print(json.dumps({label:arms}),flush=True)
    save(OUT/'robustness.json',{'status':'complete','checks':result,'same_training_indices':ids,'unique_cases':5,
        'scope':'Focused temporal/stochastic controls, not additional seeds or new training candidates','elapsed_seconds':time.perf_counter()-begin,'test_read':False,'formal_training_runs':0})

if __name__=='__main__':main()
