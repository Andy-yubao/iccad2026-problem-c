import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import json
import platform
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import torch
from experiments.graph_proposal.model import GraphProposal

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
torch.manual_seed(0)
out_dir=Path('experiments/graph_proposal/runs/20261016_flow_train_smoke_seed0')
report={'purpose':'synthetic CUDA forward/backward/optimizer preflight only; no dataset reads', 'torch_version':torch.__version__, 'python':platform.python_version(), 'CUBLAS_WORKSPACE_CONFIG':os.environ['CUBLAS_WORKSPACE_CONFIG'], 'deterministic_algorithms':True, 'cpu_threads':1, 'seed':0, 'synthetic_blocks':3, 'dataset_tensor_reads':0, 'branches':{}}
if not torch.cuda.is_available():
 report.update(status='cuda_unavailable',fallback='--device cpu')
else:
 report['device_name']=torch.cuda.get_device_name(0)
 device=torch.device('cuda')
 a=torch.tensor([4.,6.,9.]);c=torch.zeros(3,5);c[0,0]=1;c[1,1]=1
 s=torch.full((3,4),-1.);s[0,2:]=torch.tensor([1.,4.]);s[1]=torch.tensor([2.,2.,3.,2.])
 inputs=(3,a,torch.tensor([[0.,1.,1.]]),torch.tensor([[0.,2.,1.]]),torch.tensor([[5.,5.]]),c,s)
 reference=torch.tensor([[0.,0.,1.,4.],[2.,2.,3.,2.],[6.,0.,3.,3.]])
 for kind in ('deterministic','flow'):
  torch.manual_seed(0);torch.cuda.manual_seed_all(0);torch.cuda.reset_peak_memory_stats(device)
  model=GraphProposal(kind).to(device)
  optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
  optimizer.zero_grad(set_to_none=True)
  loss=model.training_loss(inputs,reference)
  assert torch.isfinite(loss)
  loss.backward()
  assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
  torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
  optimizer.step();torch.cuda.synchronize(device)
  assert all(torch.isfinite(p).all() for p in model.parameters())
  report['branches'][kind]={'status':'passed','loss':float(loss.detach()),'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(device),'optimizer_steps':1,'finite_loss_gradients_parameters':True}
  del model,optimizer,loss
  torch.cuda.empty_cache()
 report['status']='passed'
(out_dir/'gpu_preflight.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,ensure_ascii=False))
