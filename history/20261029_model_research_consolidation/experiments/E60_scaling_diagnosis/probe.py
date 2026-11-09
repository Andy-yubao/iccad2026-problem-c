"""Five fixed train cases, CPU only; no network update or solver call."""
import copy
import json
import time
import torch
import numpy as np
from .io import ROOT, OUT, save, sha
from .cache import load_rows
from .batches import collate, inputs_only, decode
from .model import GeometryModel
from .losses import components, WEIGHTS
from .diagnostics import geometry_checks

def cosine(a,b):
    denom=float(a.norm()*b.norm())
    return float(a.dot(b)/denom) if denom>1e-20 else None

def state_metrics(state,g,rows):
    with torch.no_grad():
        values=components(state,g)
        p=(decode(state,g)*g['scale'][:,None,None]).double().numpy()
        records=[]
        for row,pos in zip(rows,p):
            _,checks,diag=geometry_checks(pos,row['case'].solver_inputs)
            records.append({'member':list(row['member']),'hard_legal':bool(checks['is_feasible']),**diag})
        free=g['valid']&g['mask'][...,0].bool()
        shape_free=g['valid']&g['mask'][...,2].bool()
        wh=decode(state,g)[...,2:]
        return {'loss':{k:float(v.mean()) for k,v in values.items() if k in WEIGHTS},
            'total':sum(w*float(values[k].mean()) for k,w in WEIGHTS.items()),
            'raw_overlap_mean':float(np.mean([r['overlap_area_fraction'] for r in records])),
            'records':records,'shape_clamp_fraction':float((state[...,2][shape_free].abs()>=5).float().mean()),
            'center_decode_clamp_fraction':float(((state[...,:2]-wh/2)[free]<0).float().mean())}

def gradients(model,state,g):
    vv=components(state,g)
    terms={k:w*vv[k].mean() for k,w in WEIGHTS.items()}
    ref=terms['center_local']+terms['shape']+terms['relative_local']
    residual=sum(terms[k] for k in ('overlap','contact','boundary','mib','bbox'))
    grads={k:torch.autograd.grad(v,state,retain_graph=True)[0]*g['mask'] for k,v in terms.items()}
    a=sum(grads[k] for k in ('center_local','shape','relative_local'));b=sum(grads[k] for k in ('overlap','contact','boundary','mib','bbox'))
    params=[p for p in model.parameters() if p.requires_grad]
    def pg(value):
        gg=torch.autograd.grad(value,params,retain_graph=True,allow_unused=True)
        return torch.cat([(torch.zeros_like(p) if grad is None else grad).reshape(-1) for p,grad in zip(params,gg)])
    pa=pg(ref);pb=pg(residual)
    return {'weighted_output_gradient_norm':{k:float(v.norm()) for k,v in grads.items()},
        'output_cosine':{k:{q:cosine(v.flatten(),gg.flatten()) for q,gg in grads.items()} for k,v in grads.items()},
        'reference_vs_residual_output_cosine':cosine(a.flatten(),b.flatten()),
        'reference_vs_residual_output_cosine_per_case':[cosine(x.flatten(),y.flatten()) for x,y in zip(a,b)],
        'reference_vs_residual_parameter_cosine':cosine(pa,pb),
        'parameter_gradient_norm_reference':float(pa.norm()),'parameter_gradient_norm_residual':float(pb.norm()),
        'parameter_gradient_norm_total':float((pa+pb).norm()),
        'fraction_total_output_gradient_opposes_overlap':float(((a+b)*grads['overlap']).sum((1,2)).lt(0).float().mean())}

def forward_health(model,g):
    hidden=[];zeros=[];local=[];inputs=[]
    free=g['valid']&g['mask'][...,0].bool()
    handles=[model.head.register_forward_pre_hook(lambda m,a:hidden.append(float(a[0].detach()[g['valid']].abs().max()))),
        model.head[1].register_forward_hook(lambda m,a,o:zeros.append(float((o.detach()[free]==0).float().mean())))]
    for layer in model.layers:
        handles.append(layer.register_forward_pre_hook(lambda m,a:inputs.append(float(a[0].detach()[g['valid']].square().mean().sqrt()))))
        handles.append(layer.local.register_forward_hook(lambda m,a,o:local.append(float(o.detach()[g['valid']].square().mean().sqrt()))))
    state=model(inputs_only(g))
    for handle in handles:handle.remove()
    return state,{'hidden_max_per_readout':hidden,'final_silu_zero_fraction':zeros[-1],
        'local_output_rms':local,'block_input_rms':inputs,'local_to_input_rms':[a/b if b else None for a,b in zip(local,inputs)]}

def main():
    started=time.perf_counter();torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.manual_seed(20261027)
    ids=json.loads((ROOT/'experiments/E57_capacity_data_diagnosis/runs/20261026_wide_instability_audit/cpu_results.json').read_text())['global_train_indices']
    assert len(ids)==5 and len(set(ids))==5
    rows=load_rows('train',ids);g=collate([r['graph'] for r in rows],list(range(5)),'cpu',max_nodes=120)
    runs=[('96_16k_s0','E57_capacity_data_diagnosis/runs/20261026_data16000_geometry96_seed0'),
          ('96_16k_s1','E57_capacity_data_diagnosis/runs/20261026_data16000_geometry96_seed1'),
          ('160_norm_16k_s1','E59_centerline_overlap/runs/20261027_wide_graph_prenorm_seed1'),
          ('160_norm_64k_s0','E59_centerline_overlap/runs/20261027_large64000_original_seed0')]
    report={'train_indices':ids,'members':[list(r['member']) for r in rows],'unique_cases':5,
        'sampling':'same historical five fixed train probes; not chosen using new model quality',
        'device':'cpu','network_training_steps':0,'postprocessing_calls':0,'test_read':False,'models':{}}
    # Reference target decoded under derived public areas can contain collisions.
    report['reference_target_decoded']=state_metrics(g['target'],g,rows)
    for label,rel in runs:
        path=ROOT/'experiments'/rel/'checkpoints/best.pt';digest=sha(path)
        cp=torch.load(path,map_location='cpu',weights_only=False)
        config=dict(cp['model_config']);model=GeometryModel(**config).eval();model.load_state_dict(cp['model'])
        state,health=forward_health(model,g)
        entry={'path':str(path.relative_to(ROOT)),'sha256':digest,'config':config,'step':cp['progress']['step'],
            'original':state_metrics(state.detach(),g,rows),'health':health,'gradients':gradients(model,state,g)}
        if label=='160_norm_64k_s0':
            # A bounded output-space diagnostic. This is not model training:
            # parameters are frozen and only five independent output tensors vary.
            for p in model.parameters():p.requires_grad_(False)
            variable=torch.nn.Parameter(state.detach().clone());optimizer=torch.optim.Adam([variable],lr=.002)
            curve=[]
            for step in range(81):
                if step in (0,10,40,80):curve.append({'step':step,**state_metrics(variable.detach(),g,rows)})
                if step==80:break
                optimizer.zero_grad(set_to_none=True);v=components(variable,g)
                loss=sum(w*v[k].mean() for k,w in WEIGHTS.items());loss.backward()
                variable.grad.mul_(g['mask']);optimizer.step()
            entry['output_space_relaxation']={'steps':80,'lr':.002,'curve':curve,
                'limitation':'No model parameters updated; direct free outputs have more freedom than a shared network. No generalization claim.'}
        # Frozen-weight distribution intervention; cannot rank retrained architectures.
        was=config.get('graph_prenorm',False)
        for layer in model.layers:layer.graph_prenorm=not was
        with torch.no_grad():
            toggled,toggled_health=forward_health(model,g)
            entry['frozen_weight_normalization_toggle']={'graph_prenorm':not was,'metrics':state_metrics(toggled,g,rows),
                'health':toggled_health,'limitation':'Changes trained branch input distribution; not a training ablation.'}
        # Verify the in-memory model and on-disk checkpoint both remained unchanged.
        assert all(torch.equal(v,cp['model'][k]) for k,v in model.state_dict().items())
        assert sha(path)==digest
        entry['checkpoint_unchanged']=True;report['models'][label]=entry
        save(OUT/'five_case_probe.json',report)
        print(json.dumps({'completed':label,'G':entry['original']['total'],'toggle_G':entry['frozen_weight_normalization_toggle']['metrics']['total'],
            'reference_residual_parameter_cosine':entry['gradients']['reference_vs_residual_parameter_cosine']},ensure_ascii=False),flush=True)
    report['elapsed_seconds']=time.perf_counter()-started
    save(OUT/'five_case_probe.json',report)

if __name__=='__main__':main()
