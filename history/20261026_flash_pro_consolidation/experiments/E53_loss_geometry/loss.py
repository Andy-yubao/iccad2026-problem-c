"""One fixed Loss for optimization, train/dev monitoring and authorized tests."""
import torch
import torch.nn.functional as F
from .losses import components, average
from .module_loss import components as module_components
from .batches import decode
from .loss_config import configuration
from .ops import product2


def weights(arm, epoch, start_epoch=None):
    return configuration(arm)['weights']


def losses(state, graph, targets, arm, epoch, *, include_quality=True, start_epoch=None):
    # Compatibility arguments never change terms/weights by split, epoch or mode.
    config=configuration(arm);w=config['weights']
    values=components(state,graph)
    values.update(module_components(state, graph))
    values['module_overlap']=values['module_overlap_counted' if config['count_factor'] else 'module_overlap_base']
    valid=graph['valid'];mask=graph['mask'];area=graph['area'];local=area.clamp_min(1e-12).sqrt()
    p=decode(state,graph);r=graph['reference'];pc=p[...,:2]+p[...,2:]/2;rc=r[...,:2]+r[...,2:]/2
    center_error=((state[...,:2]-graph['target'][...,:2])/local[...,None]).abs()
    center_band=(center_error-config['center_band_local']).clamp_min(0)
    values['center_objective']=average(F.smooth_l1_loss(center_band,torch.zeros_like(center_band),reduction='none',beta=.1),mask[...,:2],(1,2))
    shape_error=(state[...,2]-graph['target'][...,2]).abs()
    values['shape_objective']=average((shape_error-config['shape_band_logratio']).clamp_min(0).square(),mask[...,2],(1,))
    pd=pc[:,:,None]-pc[:,None,:];rd=rc[:,:,None]-rc[:,None,:]
    scale=(local[:,:,None]+local[:,None,:])/2
    n=state.shape[1]
    pairs=valid[:,:,None]&valid[:,None,:]&~torch.eye(n,device=state.device,dtype=torch.bool)[None]
    pairs=pairs&(mask[...,0].bool()[:,:,None]|mask[...,0].bool()[:,None,:])
    rm=rd.abs()-(r[:,:,None,2:]+r[:,None,:,2:])/2
    near=(rm.clamp_min(0).square().sum(-1).sqrt()<=.25*scale)&pairs
    rel_error=((pd-rd)/scale.clamp_min(1e-8)[...,None]).abs()
    rel_band=(rel_error-config['relative_band_local']).clamp_min(0)
    values['relative_objective']=average(F.smooth_l1_loss(rel_band,torch.zeros_like(rel_band),reduction='none',beta=.1).mean(-1),near,(1,2))
    lo=p[...,:2].masked_fill(~valid[...,None],1e4).amin(1)
    hi=(p[...,:2]+p[...,2:]).masked_fill(~valid[...,None],-1e4).amax(1)
    bbox=product2(hi-lo).clamp_min(1e-8);total_area=(area*valid).sum(1).clamp_min(1e-8)
    values['compactness_objective']=(bbox/total_area).clamp_min(1).log().square()
    # Relative separation in each module's own narrow axis, robust to outliers.
    margin=pd.abs()-(p[:,:,None,2:]+p[:,None,:,2:])/2
    minimum_wh=torch.minimum(p[:,:,None,2:],p[:,None,:,2:]).clamp_min(1e-8)
    depth=(-margin).clamp_min(0)/minimum_wh
    clear=torch.where(depth[...,0]<=depth[...,1],depth[...,0],depth[...,1])
    values['clearance_objective']=(F.smooth_l1_loss(clear,torch.zeros_like(clear),beta=.1,reduction='none')*pairs).sum((1,2))/valid.sum(1).clamp_min(1)
    # Recover edge weights already encoded in public features/normalized adjacency.
    degree=graph['features'][...,13].expm1().clamp_min(0)
    adjacency=graph['adjacency']*degree[:,:,None]*pairs
    wire=(pd.abs().sum(-1)*adjacency).sum((1,2))
    refwire=(rd.abs().sum(-1)*adjacency).sum((1,2)).clamp_min(1e-8)
    values['wire_guard_objective']=(wire/refwire-1).clamp_min(0).square()
    total=sum(w[k]*values[v].mean() for k,v in {
        'center':'center_objective','shape':'shape_objective','relative':'relative_objective',
        'overlap':'overlap','contact':'contact','boundary':'boundary','mib':'mib','bbox':'bbox',
        'module':'module_overlap','compactness':'compactness_objective','clearance':'clearance_objective',
        'wire_guard':'wire_guard_objective'}.items() if w[k]!=0.)
    return {'total':total, **{key:value.mean() for key,value in values.items()},
             'weighted_module_objective':w['module']*values['module_overlap'].mean(),
            'supervision':values['local'].mean()}
