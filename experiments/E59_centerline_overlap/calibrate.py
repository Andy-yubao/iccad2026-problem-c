"""Single pretraining gradient-scale measurement on five fixed train members."""
import json
import math
import torch
from .io import ROOT,HERE,sha,save
from .cache import load_rows
from .data import collate
from .batches import inputs_only
from .model import GeometryModel
from .losses import components


def main():
    if (HERE/'loss_scale.json').exists():
        print('Fixed scale already exists; no recalibration');return
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    ids=json.loads((HERE/'runs/20261027_wide_seed1_audit/results.json').read_text(encoding='utf-8'))['train_indices']
    rows=load_rows('train',ids)
    checkpoint=ROOT/'experiments/E42_joint_regularization/runs/20261023_rectangle_w96d12_ab_seed0/checkpoints/best.pt'
    digest=sha(checkpoint)
    if digest!='12878108e9a58b8c195b314ba6586686c2a6e2796c1f4b6443e102b4db4224a6':raise ValueError('E42 changed')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=GeometryModel(**saved['model_config']).eval();model.load_state_dict(saved['model'])
    records=[]
    for row in rows:
        g,_=collate([row],'cpu')
        with torch.no_grad():s=model(inputs_only(g))
        s=s.detach().requires_grad_();c=components(s,g)
        norms={k:float(torch.autograd.grad(c[k].mean(),s,retain_graph=True)[0].norm()) for k in ('overlap_original','overlap_centerline')}
        records.append({'member':list(row['member']),'gradient_norms':norms,'raw_values':{k:float(c[k].mean().detach()) for k in norms}})
    original=sum(x['gradient_norms']['overlap_original'] for x in records)/len(records)
    radial=sum(x['gradient_norms']['overlap_centerline'] for x in records)/len(records)
    if not (math.isfinite(original) and math.isfinite(radial) and original>0 and radial>0):raise ValueError('Invalid gradient-scale calibration')
    report={'revision':'E59.gradient_scale.v1','kappa':original/radial,'train_indices':ids,'checkpoint_sha256':digest,
            'records':records,'train_cases':5,'dev_cases':0,'test_read':False,'selection':'one preregistered gradient-ratio measurement; no tuning'}
    save(HERE/'loss_scale.json',report)
    print(json.dumps({'kappa':report['kappa'],'train_cases':5,'dev_cases':0}))


if __name__=='__main__':main()
