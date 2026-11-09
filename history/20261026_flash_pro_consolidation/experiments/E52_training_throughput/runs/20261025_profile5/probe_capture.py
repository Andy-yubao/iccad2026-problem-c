import os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
import torch
from datasets.reference_layouts import ReferenceLayouts
from experiments.E52_training_throughput.train import seed_all
from experiments.E52_training_throughput.data import load_rows,collate
from experiments.E52_training_throughput.model import GeometryModel
from experiments.E52_training_throughput.batches import inputs_only
from experiments.E52_training_throughput.loss import losses
seed_all(0)
mode=sys.argv[1]
if mode=='nondet':torch.use_deterministic_algorithms(False)
meta=ReferenceLayouts('train').records
index=min(range(len(meta)),key=lambda i:abs(meta[i]['n']-25))
row=load_rows('train',[index],use_cache=False)[0]
g,_=collate([row]*32,'cuda')
m=GeometryModel(width=96,layers=12,rectangle=True,ffn_dropout=.05).cuda().train()
state=m(inputs_only(g)).detach().requires_grad_(True)
parameters=[state] if mode=='state' else list(m.parameters())
def step():
 for p in parameters:
  if p.grad is not None:p.grad.zero_()
 if mode=='state':v=losses(state,g,None,'main',0.)['total']
 elif mode=='model':v=m(inputs_only(g)).square().mean()
 else:v=losses(m(inputs_only(g)),g,None,'main',0.)['total']
 v.backward()
s=torch.cuda.Stream();s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
 for i in range(3):step()
torch.cuda.current_stream().wait_stream(s);torch.cuda.synchronize()
cg=torch.cuda.CUDAGraph()
with torch.cuda.graph(cg,stream=s):step()
cg.replay();torch.cuda.synchronize()
print('CAPTURE_OK',mode,flush=True)
