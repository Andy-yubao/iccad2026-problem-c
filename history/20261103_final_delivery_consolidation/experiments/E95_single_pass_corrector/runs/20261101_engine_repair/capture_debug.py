import traceback
from pathlib import Path
import torch
from experiments.E95_single_pass_corrector.runtime import seed,CONFIG
from experiments.E95_single_pass_corrector.io import HERE
from experiments.E95_single_pass_corrector.data import load_rows,collate,inputs_only
from experiments.E95_single_pass_corrector.model import GeometryModel
from experiments.E95_single_pass_corrector.packed import PublicPacker
from experiments.E95_single_pass_corrector.loss import per_case
from experiments.E95_single_pass_corrector.graph_step import CapturedStep
seed();s=torch.load(HERE/'runs/20261101_direct_bbox846_seed0/checkpoints/latest.pt',map_location='cpu',weights_only=False)
m=GeometryModel(**dict(CONFIG,attention_backend='efficient')).cuda();m.load_state_dict(s['model']);rows=load_rows('train',[1683,10027,5474,10140,15001]);m.eval()
with torch.no_grad():d=m.draft(inputs_only(collate(rows,'cuda'))).cpu()
for i,r in enumerate(rows):r['draft']=d[i,:r['graph']['n']]
p=PublicPacker(rows[0],p=480);h,d,_=p.pack([rows[i%5] for i in range(32)]);g={k:v.cuda() for k,v in h.items()};d=d.cuda()

o=torch.optim.AdamW(m.corrector.parameters());o.load_state_dict(s['optimizer'])
original_backward=torch.Tensor.backward
last=[]
def traced_backward(value,*args,**kwargs):
 if torch.cuda.is_current_stream_capturing():
  seen=set();pending=[value.grad_fn]
  while pending:
   node=pending.pop()
   if node is None or node in seen:continue
   seen.add(node);name=node.name()
   def hook(grads,name=name):
    last.append(name)
    return grads
   node.register_prehook(hook)
   pending.extend(x[0] for x in node.next_functions)
 try:return original_backward(value,*args,**kwargs)
 except BaseException:
  print('LAST_BACKWARD_NODES',last[-12:],flush=True)
  raise
torch.Tensor.backward=traced_backward
engine=CapturedStep(m,o,g,d)
print('Capture passed',flush=True)
