"""Inspect saved final readout saturation on the same five train cases."""
from cpu_audit import *


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    ids=json.loads((HERE/'cpu_results.json').read_text(encoding='utf-8'))['global_train_indices']
    rows=load_rows('train',ids,use_cache=False)
    g=ResidentRows([rows[i%5] for i in range(32)],'cpu',120).batch(list(range(32)))
    free=g['valid']&(g['mask'][...,0]>0);result={}
    for label,file in [('best','best.pt'),('latest','latest.pt')]:
        state=torch.load(RUN/'checkpoints'/file,map_location='cpu',weights_only=False)
        model=GeometryModel(**state['model_config']).eval();model.load_state_dict(state['model'])
        pre=[];act=[]
        def before(module,args,out):
            x=out.detach()[free]
            pre.append({'fraction_below_minus20':float((x< -20).float().mean()),
                'min':float(x.min()),'max':float(x.max())})
        def after(module,args,out):
            x=out.detach()[free]
            act.append({'fraction_exact_zero':float((x==0).float().mean()),'max_abs':float(x.abs().max())})
        h1=model.head[0].register_forward_hook(before);h2=model.head[1].register_forward_hook(after)
        output=model(inputs_only(g));value=losses(output,g,None,'geometry160',0)['total'];value.backward()
        result[label]={'final_head_preactivation':pre[-1],'final_head_activation':act[-1],
            'free_output_max_difference_from_readout_bias':float((output.detach()[free]-model.head[2].bias.detach()).abs().max()),
            'gradient_norms':{name:float(p.grad.norm()) if p.grad is not None else None for name,p in model.named_parameters()
                if name in ('input.0.weight','state_embed.weight','head.0.weight','head.2.weight','head.2.bias')},
            'loss':float(value.detach())}
        h1.remove();h2.remove()
        del model,output,value
    save(HERE/'readout_probe.json',{'unique_train_cases':5,'same_cases_as_prior_checks':True,'GPU_used':False,'test_read':False,'results':result})
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
