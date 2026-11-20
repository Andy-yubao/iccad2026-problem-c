"""Joint objective and mutable-hidden interventions for preparation only."""
import torch
from .loss import per_case,TOTAL_COMPONENTS
from .loss_config import ARMS,WEIGHTS

def objective(states,batch):
    from experiments.E148_shared_functional_shape.loss import base_per_case as oracle
    x=states.detach().clone().requires_grad_(True);new=per_case(x,batch,ARMS[0]);expected=0.
    for k,w in WEIGHTS.items():
        old=oracle(x[k],batch)
        value=sum(old[name] for name in TOTAL_COMPONENTS if k not in (0,4) or name!='overlap_weighted')
        torch.testing.assert_close(new[f'layer{k:02d}_total'],value,rtol=0,atol=0)
        expected=expected+w*value
    torch.testing.assert_close(new['total'],expected,rtol=0,atol=0)
    a=torch.autograd.grad(new['total'].sum(),x,retain_graph=True)[0]
    b=torch.autograd.grad(expected.sum(),x)[0];torch.testing.assert_close(a,b,rtol=0,atol=0)
    for k in range(13):assert bool(a[k].abs().sum()>0)==(k in WEIGHTS)
    assert torch.count_nonzero(new['layer00_overlap_weighted'])==0 and torch.count_nonzero(new['layer04_overlap_weighted'])==0
    return dict(actual_readouts=13,supervised=[0,4,8,12],weights=list(WEIGHTS.values()),no_overlap=[0,4],base_and_joint_values_and_gradients_exact=True)

def interfaces(model,g,states,batch):
    with torch.no_grad():
        initial=model.initial_generator(g)[0]
    torch.testing.assert_close(initial,states[0],rtol=0,atol=0)
    counts=[0,0,0];sizes=[];hooks=[]
    for i,unit in enumerate(model.units):
        def count(module,args,value,index=i):counts[index]+=1
        hooks.append(unit.register_forward_hook(count))
        for k,encoder in enumerate(unit.demand_encoder):hooks.append(encoder.register_forward_pre_hook(lambda m,a,k=k:sizes.append((k,a[0].shape[-1]))))
    with torch.no_grad():model(g)
    for h in hooks:h.remove()
    assert counts==[2,2,2] and all(size==(81 if k==0 else 65) for k,size in sizes)
    for boundary in range(5):
        unit=model.units[model.unit_order[boundary]];occurrence=boundary%2;calls=[0];heads=[0]
        def alter(module,args,value):
            k=calls[0];calls[0]+=1
            return value+.37*g['valid'][...,None] if k==occurrence else value
        def hold(module,args,value):
            k=heads[0];heads[0]+=1
            return states[2*boundary+2].clone() if k==occurrence else value
        h=unit.layers[1].register_forward_hook(alter);f=unit.heads[1].register_forward_hook(hold)
        try:
            with torch.no_grad():changed=model(g)
        finally:h.remove();f.remove()
        torch.testing.assert_close(changed[2*boundary+3:],states[2*boundary+3:],rtol=0,atol=0)
    # Initial dynamic hidden is excluded from the pure-P backend condition.
    h=model.initial_generator.initial_blocks[-1].register_forward_hook(lambda m,a,v:v+.37*g['valid'][...,None])
    f=model.initial_generator.head.register_forward_hook(lambda m,a,v:states[0].clone())
    try:
        with torch.no_grad():changed=model(g)
    finally:h.remove();f.remove()
    torch.testing.assert_close(changed,states,rtol=0,atol=0)
    layout=[];hooks=[model.initial_generator.head.register_forward_hook(lambda m,a,v:layout.append(v))]
    for unit in model.units:hooks.append(unit.heads[1].register_forward_hook(lambda m,a,v:layout.append(v)))
    try:
        output=model(g);final=per_case(output,batch,ARMS[0])['final_total'].mean()
        # Direct initial auxiliary supervision is absent from this derivative.
        targets=[*layout[:-1],model.initial_generator.role_input[0].weight,model.initial_generator.port_input[0].weight]
        gradients=torch.autograd.grad(final,targets)
        assert all(bool(torch.isfinite(v).all()) and bool(v.abs().sum()>0) for v in gradients)
    finally:
        for h in hooks:h.remove()
    return dict(initial_output_real=True,shared_calls=counts,entry_only_static_injection=True,five_hidden_cuts_bitwise=True,
        initial_hidden_isolated=True,final_only_loss_reaches_initial_role_and_ports=True,layout_gradients_across_all_cuts=True)
