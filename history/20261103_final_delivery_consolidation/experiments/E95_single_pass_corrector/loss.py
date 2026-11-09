"""Single-call task objective; input intent comes only from the current draft."""
import torch
import torch.nn.functional as F
import json
from .io import HERE
from .base_model import decode
from .data import PUBLIC
from .public_task import public_components

BOX_WEIGHT=json.loads((HERE/'bbox_objective.json').read_text(encoding='utf-8'))['box_weight']
WEIGHTS={'overlap':1.,'bbox':BOX_WEIGHT,'wirelength':.05,'soft':.2,'intent':.1}

def configuration():
    return dict(revision='E95.single_call.direct_bbox.v3',weights=WEIGHTS,
        overlap='penetration + mean pair load + maximum excess above10%',
        bbox='unclipped B/A, no baseline subtraction or expansion allowance',
        task='legacy diagnostic only; total has O + box_weight*B/A + .05W + .2soft + .1intent',
        intent='reliable_pair_direction + reliable_near_pair_distance + .01*movement_shape',
        reference=False,selection='one call, fixed dev500 arithmetic mean of same total',
        threshold=.10,exact_union_coverage='report only, distinguish pair-load upper bound',
        coefficient_basis='median five initial train-case free-state overlap/bbox gradient norm ratio, three significant figures; frozen before dev; no sweep')

def task_inputs(g):
    n=g['valid'].shape[1]
    return {k:g[k] for k in ('area','mask','valid','specs','constraints','group','mib')} | dict(
        pin_coordinates=g['port_xy'][:,None].expand(-1,n,-1,-1),
        pin_weights=g['p2b_weights'],pin_valid=g['port_valid'][:,None].expand(-1,n,-1),
        # b2b is symmetric; count each original undirected edge only once.
        wire_b2b=torch.triu(g['b2b_weights'],diagonal=1)+.5*torch.diag_embed(g['b2b_weights'].diagonal(dim1=1,dim2=2)))

def intent_components(state,draft,g):
    p=decode(state,g);src=decode(draft.detach(),g)
    center=p[...,:2]+p[...,2:]/2;initial=src[...,:2]+src[...,2:]/2
    local=g['area'].clamp_min(1e-12).sqrt();scale=(local[:,:,None]+local[:,None,:])/2
    valid=g['valid'];n=valid.shape[1];ix=torch.arange(n,device=state.device)
    pairs=valid[:,:,None]&valid[:,None,:]&(ix[:,None]!=ix[None,:])[None]
    d0=initial[:,:,None]-initial[:,None,:];d=center[:,:,None]-center[:,None,:]
    gap=(d0.abs()-(src[:,:,None,2:]+src[:,None,:,2:])/2)/scale[...,None].clamp_min(1e-6)
    # Reliability here means separated in the input. It is a hypothesis, not a
    # claim that the inferred intent is correct. Overlapping source pairs do not
    # supply protected directions or target distances.
    reliable=gap.amax(-1)>=.05
    distance=d0.norm(dim=-1)/scale.clamp_min(1e-6)
    near=pairs&(distance<=2.)
    related=(g['b2b_weights']>0)|(g['group']>0)|(g['mib']>0)
    chosen=pairs&reliable&(near|related)
    axes=(gap>=.05)&chosen[...,None]
    margin=(d0.abs()/scale[...,None].clamp_min(1e-6)).clamp(max=.25)
    reversal=(margin-d0.sign()*d/scale[...,None].clamp_min(1e-6)).clamp_min(0).square()
    direction=(reversal*axes).sum((1,2,3))/axes.sum((1,2,3)).clamp_min(1)
    # Permit neighbours to move closer; penalize pulling a reliable nearby pair
    # farther apart. No exact reference displacement/contact tree is preserved.
    neighbours=chosen&near
    expansion=(d.norm(dim=-1)/scale.clamp_min(1e-6)-distance).clamp_min(0).square()
    adjacency=(expansion*neighbours).sum((1,2))/neighbours.sum((1,2)).clamp_min(1)
    mask=g['mask']*valid[...,None]
    movement=F.smooth_l1_loss((center-initial)/local[...,None],torch.zeros_like(center),reduction='none',beta=.1)
    displacement=(movement*mask[...,:2]).sum((1,2))/mask[...,:2].sum((1,2)).clamp_min(1)
    shape=((state[...,2]-draft.detach()[...,2]).square()*mask[...,2]).sum(1)/mask[...,2].sum(1).clamp_min(1)
    return dict(intent_direction=direction,intent_neighbour=adjacency,input_movement=displacement,
        input_shape_change=shape,intent=direction+adjacency+.01*(displacement+.25*shape))

def per_case(state,g,draft):
    if set(g)-set(PUBLIC):raise ValueError('Corrector objective rejects reference labels and extra fields')
    values=public_components(state,task_inputs(g))
    values.update(intent_components(state,draft,g))
    values['v2_total_diagnostic']=values['overlap_total']+.1*values['task']+.1*values['intent']
    values['bbox_cost']=BOX_WEIGHT*values['bbox_over_A']
    values['total']=values['overlap_total']+values['bbox_cost']+.05*values['wirelength']+.2*values['soft']+.1*values['intent']
    return values

def losses(state,g,draft):return {k:v.mean() for k,v in per_case(state,g,draft).items()}
