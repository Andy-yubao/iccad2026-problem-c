"""Fixed train500 local-state gradient interaction, zero parameter updates."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import json
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from .latency import HERE,ROOT,sha
from .candidate3 import teacher_rows,registry
from .train_io.data import load_rows
from .round2_connectivity import attach,Packer,network_inputs,raw_terms
from .student import DeploymentModel
from .train_io.io import save,exclusive
OUTPUT=HERE/'runs/20261102_round2_group_gradient_conflict'
def main():
    OUTPUT.mkdir(exist_ok=False);torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(0)
    with exclusive(HERE/'runs/round2_execution.lock'):
        path=HERE/'runs/20261102_candidate3_clearance_seed0/checkpoints/best.pt';saved=torch.load(path,map_location='cpu',weights_only=False);model=DeploymentModel(**saved['model_config']).cuda().eval();model.load_state_dict(saved['model'],strict=True);model.requires_grad_(False)
        rows=load_rows('train_monitor500');labels=teacher_rows('train');lookup={tuple(r['member']):i for i,r in enumerate(registry()['sets']['train']['records'][:16000])};attach(rows,labels,[lookup[tuple(r['member'])] for r in rows]);packer=Packer(rows[0]);records=[]
        for start in range(0,500,32):
            g,d,t,u,count=packer.gpu(rows[start:start+32])
            with torch.no_grad():pred=model.refiner(d,network_inputs(g))
            state=pred.detach().requires_grad_(True);values=raw_terms(state,g,t,u);local=g['area'].clamp_min(1e-12).sqrt();unit=torch.stack((local,local,torch.ones_like(local)),-1)*g['mask']*g['valid'][...,None]
            group=((g['constraints'][...,3]>0)&g['valid'])[...,None]
            gradients={key:torch.autograd.grad(values[key].sum(),state,retain_graph=True)[0]*unit*group for key in ('teacher_relative','teacher_clearance_tail','public_group_contact')}
            target=gradients['public_group_contact'];tnorm=target.square().sum((1,2)).sqrt();stats={}
            for key in ('teacher_relative','teacher_clearance_tail'):
                a=gradients[key];norm=a.square().sum((1,2)).sqrt();dot=(a*target).sum((1,2));cos=dot/(norm*tnorm).clamp_min(1e-20);stats[key]=torch.stack((norm,tnorm,dot,cos),-1).cpu().numpy()
            for j,row in enumerate(rows[start:start+32]):records.append(dict(index=start+j,member=list(row['member']),**{key:dict(zip(('base_norm','contact_norm','dot','cosine'),map(float,value[j]))) for key,value in stats.items()}))
        summary={}
        for key in ('teacher_relative','teacher_clearance_tail'):
            active=[r[key] for r in records if r[key]['base_norm']>1e-10 and r[key]['contact_norm']>1e-10]
            summary[key]=dict(active=len(active),zero_or_inactive=500-len(active),negative_dot_fraction=float(np.mean([r['dot']<0 for r in active])) if active else None,median_cosine=float(np.median([r['cosine'] for r in active])) if active else None)
        save(OUTPUT/'results.json',dict(status='complete',summary=summary,records=records,plan_sha256=sha(HERE/'round2_gradient_conflict_plan.json'),no_optimizer_solver_dev_or_Q=True,test_read=False));print(json.dumps(summary,indent=2))
if __name__=='__main__':
    with threadpool_limits(1):main()
