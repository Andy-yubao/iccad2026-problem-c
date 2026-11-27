"""One preregistered train500 calibration for the first quality-first candidate."""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import json,datetime
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from .latency import HERE,ROOT,sha
from .candidate3 import teacher_rows,load_rows,registry
from .student import DeploymentModel
from .round2_connectivity import attach,Packer,network_inputs
from .round2_connectivity import raw_terms
from .train_io.io import save

OUTPUT=HERE/'runs/20261102_round2_candidate2_calibration'
def main():
    OUTPUT.mkdir(exist_ok=False);torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(0)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    plan=json.loads((HERE/'round2_candidate2_calibration_plan.json').read_text(encoding='utf-8'));assert plan['gradient_share']==.5
    study=json.loads((HERE/'round2_study.json').read_text(encoding='utf-8'));deadline=datetime.datetime.fromisoformat(study['deadline_utc'])
    with threadpool_limits(1):
        path=HERE/'runs/20261102_candidate3_clearance_seed0/checkpoints/best.pt';saved=torch.load(path,map_location='cpu',weights_only=False)
        model=DeploymentModel(**saved['model_config']).cuda().eval();model.load_state_dict(saved['model'],strict=True);model.requires_grad_(False)
        labels=teacher_rows('train');rows=load_rows('train_monitor500');lookup={tuple(r['member']):i for i,r in enumerate(registry()['sets']['train']['records'][:16000])};attach(rows,labels,[lookup[tuple(r['member'])] for r in rows])
        packer=Packer(rows[0]);records=[]
        for start in range(0,500,32):
            if datetime.datetime.now(datetime.timezone.utc)>=deadline or (HERE/'runs/STOP').exists():raise RuntimeError('STOP/deadline')
            g,d,t,u,count=packer.gpu(rows[start:start+32]);assert set(network_inputs(g))==set(__import__('experiments.E97_gpu_hybrid_solver.train_io.data',fromlist=['LEGACY']).LEGACY)
            with torch.no_grad():pred=model.refiner(d,network_inputs(g))
            state=pred.detach().requires_grad_(True);values=raw_terms(state,g,t,u)
            local=g['area'].clamp_min(1e-12).sqrt();unit=torch.stack((local,local,torch.ones_like(local)),-1)*g['mask']*g['valid'][...,None]
            base=(torch.autograd.grad(values['total'].sum(),state,retain_graph=True)[0]*unit).square().sum((1,2)).sqrt()
            task=(torch.autograd.grad(values['public_soft'].sum(),state)[0]*unit).square().sum((1,2)).sqrt()
            numbers=torch.stack((base,task,values['total'].detach(),values['public_soft'].detach())).detach().cpu().numpy()
            assert np.isfinite(numbers).all()
            for j,row in enumerate(rows[start:start+32]):records.append(dict(index=start+j,member=list(row['member']),base_gradient=float(numbers[0,j]),task_gradient=float(numbers[1,j]),base_loss=float(numbers[2,j]),task_loss=float(numbers[3,j])))
            print('public-soft calibration',len(records),'/500',flush=True)
        active=[r for r in records if r['base_gradient']>1e-10 and r['task_gradient']>1e-10];assert len(active)>=5
        base=float(np.median([r['base_gradient'] for r in active]));task=float(np.median([r['task_gradient'] for r in active]));weight=.5*base/task
        assert np.isfinite(weight) and 0<weight<=1000
        result=dict(status='complete',coefficient=weight,gradient_share=.5,active_count=len(active),median_base_gradient=base,median_task_gradient=task,records=records,binding=dict(source_checkpoint=path.relative_to(ROOT).as_posix(),source_checkpoint_sha256=sha(path),calibration_plan_sha256=sha(HERE/'round2_candidate2_calibration_plan.json'),sources={p.relative_to(ROOT).as_posix():sha(p) for p in [HERE/'round2_candidate2_calibrate.py',HERE/'round2_connectivity.py',HERE/'round2_connectivity.py',HERE/'candidate3.py',HERE/'student.py',HERE/'repair85_one_thread/fast_geometry.py']}),no_optimizer_solver_dev_or_Q=True,test_read=False)
        save(OUTPUT/'results.json',result);save(OUTPUT/'weights.json',{k:v for k,v in result.items() if k!='records'});print('fixed public-soft coefficient',weight,flush=True)

if __name__=='__main__':main()
