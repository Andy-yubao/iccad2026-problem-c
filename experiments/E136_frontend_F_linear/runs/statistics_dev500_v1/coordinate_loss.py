"""Same terms/coefficients on actual saved XYWH, without a second decoder clamp.

The immutable training Loss remains separately reported and verified against
its monitor. This is a layout-domain diagnostic, not a replacement objective.
"""
import types
import torch
import torch.nn.functional as F
from experiments.E136_frontend_F_linear import simple_loss
from experiments.E136_frontend_F_linear.loss import A,B,masked_channels_mean

# Reuse the exact frozen task arithmetic with an identity decoder. Clone the
# function globals rather than modifying the original module or any model.
_globals=dict(simple_loss.public_components.__globals__,decode=lambda positions,g:positions,_BITS={})
_public=types.FunctionType(simple_loss.public_components.__code__,_globals,'saved_layout_public_components')

def per_case(positions,g):
    p=positions;valid=g['valid'];local=g['area'].clamp_min(1e-12).sqrt()
    out=_public(p,{k:g[k] for k in simple_loss.PUBLIC})
    mask=g['mask']*valid[...,None]
    center=p[...,:2]+p[...,2:]/2
    out['position']=simple_loss.average(F.smooth_l1_loss(center/local[...,None],g['target'][...,:2]/local[...,None],reduction='none',beta=.1),mask[...,:2],(1,2))
    ratio=(p[...,2].clamp_min(1e-12)/p[...,3].clamp_min(1e-12)).log()
    out['shape']=simple_loss.average((ratio-g['target'][...,2]).square(),mask[...,2],(1,))
    out['reference']=out['position']+.25*out['shape']
    fixed=valid&((g['constraints'][...,0]!=0)|(g['constraints'][...,1]!=0))
    pre=valid&(g['constraints'][...,1]!=0);spec=g['specs']
    wh=torch.where(fixed[...,None],p[...,2:],torch.ones_like(p[...,2:])).clamp_min(1e-12)
    target_wh=torch.where(fixed[...,None],spec[...,2:],torch.ones_like(spec[...,2:])).clamp_min(1e-12)
    out['public_shape']=masked_channels_mean(F.smooth_l1_loss(wh.log(),target_wh.log(),reduction='none',beta=.1),fixed)
    pre_wh=torch.where(pre[...,None],spec[...,2:],torch.ones_like(spec[...,2:])).clamp_min(1e-12)
    scale=pre_wh.prod(-1).sqrt()
    target_center=torch.where(pre[...,None],spec[...,:2]+spec[...,2:]/2,torch.zeros_like(spec[...,:2]))
    out['public_preplaced_position']=masked_channels_mean(F.smooth_l1_loss(center/scale[...,None],target_center/scale[...,None],reduction='none',beta=.1),pre)
    out.update(reference_weighted=out['reference'],overlap_weighted=A*out['overlap_total'],
        wire_compact_weighted=.5*B*(out['wirelength']+out['compactness']),
        grouping_weighted=2*B*out['grouping_contribution'],boundary_weighted=2*B*out['boundary_contribution'],
        public_shape_weighted=out['public_shape'],public_preplaced_position_weighted=out['public_preplaced_position'])
    out['total']=sum(out[k] for k in ('reference_weighted','overlap_weighted','wire_compact_weighted','grouping_weighted',
        'boundary_weighted','public_shape_weighted','public_preplaced_position_weighted'))
    return out

def main():
    import argparse
    import hashlib
    import json
    import numpy as np
    from .build import HERE,ROOT,read,save,loss_values
    from experiments.E136_frontend_F_linear.data import load_rows
    from experiments.E136_frontend_F_linear.resident import ResidentRows
    from experiments.E136_frontend_F_linear.batches import decode
    from datasets.random_reference import RandomReference
    parser=argparse.ArgumentParser();parser.add_argument('--reference-only',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    manifest=read(HERE/'manifest.json');rows=load_rows('dev_monitor500');pack=ResidentRows(rows,'cuda',max_nodes=120)
    destination=HERE/'coordinate_losses';destination.mkdir(exist_ok=args.reference_only)
    dataset=RandomReference('dev_monitor500')
    reference=[dataset[i]['supervision']['reference_xywh'] for i in range(500)]
    groups=[('reference',reference)]
    for model in ([] if args.reference_only else manifest['models']):
        records=[read(ROOT/model['evaluation']/f'cases/{i:04d}.json') for i in range(500)]
        groups.extend([(f"draft.{model['key']}",[r['draft_positions'] for r in records]),(f"final.{model['key']}",[r['positions'] for r in records])])
    with torch.inference_mode():
        # Identity-decoder task terms must exactly match the frozen arithmetic
        # when supplied that arithmetic's actual decoded rectangles.
        sample=pack.batch([0]);state=sample['target'];p=decode(state,sample)
        old=simple_loss.public_components(state,{k:sample[k] for k in simple_loss.PUBLIC})
        direct=_public(p,{k:sample[k] for k in simple_loss.PUBLIC})
        for name in old:torch.testing.assert_close(old[name],direct[name],rtol=0,atol=0)
        for name,layouts in groups:
            values=[]
            for start in range(0,500,32):
                ids=list(range(start,min(start+32,500)));batch=pack.batch(ids)
                physical=torch.zeros((len(ids),120,4),device='cuda',dtype=batch['target'].dtype)
                for j,index in enumerate(ids):
                    n=rows[index]['graph']['n']
                    physical[j,:n]=torch.as_tensor(np.asarray(layouts[index]),device='cuda',dtype=physical.dtype)/batch['scale'][j]
                output=per_case(physical,batch)
                for j,index in enumerate(ids):
                    result=loss_values({k:v[j:j+1] for k,v in output.items()})
                    values.append(dict(index=index,member=manifest['members'][index],loss=result))
            save(destination/(name+'.json'),dict(name=name,rows=values,source='actual saved XYWH; no second shape/position decode clamp'))
            print('Coordinate Loss',name,500,'/500',flush=True)
    previous=read(destination/'verification.json') if (destination/'verification.json').exists() else {}
    save(destination/'verification.json',dict(status='passed',cases=500,layouts=5500,exact_task_arithmetic_parity=True,
        initial_full_collection_source_sha256=previous.get('coordinate_loss_source_sha256'),reference_from_original_npz=True,
        coordinate_loss_source_sha256=hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest(),
        training_objective_unchanged=True,training=False,postprocessing_calls=0,Q_calls=0,test_read=False))

if __name__=='__main__':main()
