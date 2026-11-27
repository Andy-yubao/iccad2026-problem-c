"""Repeat only corrected independent optimizer controls and branch interventions."""
from cpu_audit import *
import copy


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    previous=json.loads((HERE/'cpu_results.json').read_text(encoding='utf-8'))
    rows=load_rows('train',previous['global_train_indices'],use_cache=False)
    g=ResidentRows([rows[i%5] for i in range(32)],'cpu',120).batch(list(range(32)))
    best=torch.load(RUN/'checkpoints/best.pt',map_location='cpu',weights_only=False)
    last=torch.load(RUN/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
    controls=[]
    for lr in (.001,.0001):
        model=GeometryModel(**best['model_config']).eval();model.load_state_dict(best['model'])
        optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.0003)
        optimizer.load_state_dict(copy.deepcopy(best['optimizer']))
        for group in optimizer.param_groups:group['lr']=lr
        before_step=sorted(set(int(x['step']) for x in optimizer.state.values()))
        optimizer.zero_grad(set_to_none=True)
        value=losses(model(inputs_only(g)),g,None,'geometry160',0)['total'];value.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        clipped=torch.stack([p.grad.norm().square() for p in model.parameters() if p.grad is not None]).sum().sqrt()
        optimizer.step()
        with torch.no_grad():after=losses(model(inputs_only(g)),g,None,'geometry160',0)['total']
        delta=sum((p.detach()-best['model'][n]).square().sum() for n,p in model.named_parameters()).sqrt()
        controls.append({'lr':lr,'optimizer_step_before':before_step,'loss_before':float(value.detach()),
            'loss_after':float(after),'grad_norm_before_clip':float(norm),'grad_norm_after_clip':float(clipped),
            'parameter_delta_norm':float(delta)})
        del model,optimizer,value,after
    assert all(int(x['step'])==6000 for x in best['optimizer']['state'].values())
    interventions={}
    for label,checkpoint in [('best',best),('latest',last)]:
        for variant in ('baseline','no_graph_aggregate','no_state_feedback','neither'):
            model=GeometryModel(**checkpoint['model_config']).eval();model.load_state_dict(checkpoint['model'])
            hooks=[]
            if variant in ('no_graph_aggregate','neither'):
                hooks += [layer.local.register_forward_hook(lambda module,args,out:torch.zeros_like(out)) for layer in model.layers]
            if variant in ('no_state_feedback','neither'):
                hooks.append(model.state_embed.register_forward_hook(lambda module,args,out:torch.zeros_like(out)))
            result=trace(model,g)
            interventions[label+'/'+variant]={'final_layer':result['layers'][-1],
                'peak_hidden_abs':max(x['hidden_abs_max'] for x in result['layers']),
                'peak_raw_center_abs':max(x['raw_center_abs_max'] for x in result['layers']),
                'layer_trace':result['layers'],'losses':result['losses']}
            for hook in hooks:hook.remove()
            del model
    result={'optimizer_controls':controls,'independent_optimizer_state_copies_verified':True,
        'supersedes':'cpu_results.json one_step_CPU_controls only; the initial diagnostic shared in-memory Adam state across controls. Saved training checkpoints were never changed.',
        'interventions':interventions,'unique_train_cases':5,'GPU_used':False,'test_read':False,
        'limits':'Post-hoc CPU eval-mode branch interventions identify an amplification pathway, not the first causal update of the historical CUDA/Dropout trajectory. Branch removal is not a validated model fix.'}
    save(HERE/'controls_and_branches.json',result)
    print(json.dumps({'controls':controls,'branches':{k:{q:v[q] for q in ('final_layer','peak_hidden_abs','peak_raw_center_abs')} for k,v in interventions.items()}}),flush=True)


if __name__=='__main__':main()
