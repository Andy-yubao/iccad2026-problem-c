"""Step3: five frozen-development cases, layer states and true-training gradients."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import time
import numpy as np
import torch
from .orderliness import OUT,HERE,ROOT,read,save,sha,pileup
from .feedback_geometry import geometry
from .data import load_rows,collate
from .batches import inputs_only,decode
from .model import GeometryModel
from .losses import components,WEIGHTS
from .occupancy import components as occupancy
from .inference import graph as inference_graph


def cosines(vectors):
    names=list(vectors);result={}
    for a in names:
        result[a]={}
        for b in names:
            x,y=vectors[a],vectors[b];denom=float(x.norm()*y.norm())
            result[a][b]=float(torch.dot(x,y)/denom) if denom>1e-14 else None
    norms={k:float(v.norm()) for k,v in vectors.items()}
    total=sum(vectors.values());ratio=float(total.norm())/max(sum(norms.values()),1e-15)
    return {'cosines':result,'norms':norms,'sum_to_sum_of_norms':ratio}


def main():
    if not (OUT/'stage2_matching.json').exists():raise RuntimeError('Finish steps 1 and 2 first')
    torch.manual_seed(0);torch.set_num_threads(4);torch.set_num_interop_threads(1)
    started=time.perf_counter();(OUT/'layers_local').mkdir(exist_ok=True)
    registry=read(ROOT/'protocol/reference_workset_v5.json');records=registry['sets']['dev_monitor500']['records']
    ids=[]
    for target in (25,45,70,95,115):
        ids.append(min((i for i in range(len(records)) if i not in ids),key=lambda i:(abs(records[i]['n']-target),i)))
    checkpoint=HERE/'runs/20261023_rectangle_w96d12_ab_seed0/checkpoints/best.pt';identity=sha(checkpoint)
    if identity!='12878108e9a58b8c195b314ba6586686c2a6e2796c1f4b6443e102b4db4224a6':raise ValueError('Frozen checkpoint changed')
    config={'role':'dev_monitor500','indices':ids,'members':[records[i] for i in ids],
            'selection':'nearest public n to25/45/70/95/115, index tie-break, no score conditioning',
            'checkpoint_sha256':identity,'source_sha256':sha(__file__),'threads':4,'interop':1,
            'budget_seconds':480,'training_steps':0,'optimizer_loaded':False,
            'gradient_space':'state and parameters separately; real geometry training objective, occupancy proxy only diagnosed'}
    save(OUT/'stage3_config.json',config)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False);model=GeometryModel(**state['model_config']).eval();model.load_state_dict(state['model'],strict=True)
    params=[p for p in model.parameters() if p.requires_grad];results=[]
    for ordinal,index in enumerate(ids):
        row=load_rows('dev_monitor500',indices=[index],use_cache=False)[0]
        g,_=collate([row],'cpu');captured=[]
        hook=model.head.register_forward_hook(lambda module,args,output:captured.append(output.detach().clone()))
        final=model(inputs_only(g));hook.remove()
        assert len(captured)==13 and torch.equal(captured[-1],final.detach())
        values=components(final,g);out_vectors={};param_vectors={}
        for key,weight in WEIGHTS.items():
            term=values[key].mean()*weight
            a=torch.autograd.grad(term,final,retain_graph=True)[0]
            p=torch.autograd.grad(term,params,retain_graph=True,allow_unused=True)
            out_vectors[key]=a.detach().flatten()
            param_vectors[key]=torch.cat([(v if v is not None else torch.zeros_like(parameter)).detach().flatten() for parameter,v in zip(params,p)])
        total=sum(WEIGHTS[k]*values[k].mean() for k in WEIGHTS)
        direct=torch.autograd.grad(total,final,retain_graph=True)[0].detach().flatten()
        summed=sum(out_vectors.values());sum_error=float((direct-summed).norm()/direct.norm().clamp_min(1e-12));assert sum_error<1e-5
        out_summary=cosines(out_vectors);param_summary=cosines(param_vectors)
        with torch.no_grad():
            independent=model(inference_graph(row['case'].solver_inputs))
            parity=float((independent-final.detach()).abs().max());assert parity<1e-5
            trajectories=[];positions=[]
            previous=None;scale=float(g['scale'][0]);free=np.asarray(row['case'].solver_inputs[5])[:,1]==0
            for layer,tensor in enumerate(captured):
                p=(decode(tensor,g)[0]*scale).numpy().astype(float);positions.append(p.tolist());metrics=pileup(p)
                losses=components(tensor,g);occ=occupancy(tensor,g)
                record={'layer':layer,'pileup':metrics,'losses':{k:float(v.mean()) for k,v in losses.items()},
                        'occupancy_proxy':{k:float(v.mean()) for k,v in occ.items()},
                        'center_clamped_coordinate_fraction':float(((tensor[0,:,:2]-decode(tensor,g)[0,:,2:]/2)<0).float().mean())}
                if previous is not None:
                    a=(p[:,:2]+p[:,2:]/2)-(previous[:,:2]+previous[:,2:]/2)
                    normalized=np.abs(a).sum(1)/np.sqrt(p[:,2:].prod(1))
                    record['center_step_mean_free']=float(normalized[free].mean())
                    record['center_step_max']=float(normalized.max())
                trajectories.append(record);previous=p
            target_positions=(decode(g['target'],g)[0]*scale).numpy().astype(float)
            target_diagnostics={'pileup':pileup(target_positions),
                    'losses':{k:float(v.mean()) for k,v in components(g['target'],g).items()},
                    'scope':'original supervision state decoded under derived public areas; a conflict witness, not an attainable loss lower bound or legal target'}
        result={'index':index,'member':list(row['member']),'n':records[index]['n'],
                'output_state_gradients':out_summary,'parameter_gradients':param_summary,
                'gradient_sum_relative_error':sum_error,'inference_route_state_max_error':parity,
                'trajectory':trajectories,'layer_positions':positions,'constraints':np.asarray(row['case'].solver_inputs[5]).tolist(),
                'original_target_state_diagnostic':target_diagnostics,
                'final_training_terms':{k:float(v.detach().mean()) for k,v in values.items()},
                'weights':WEIGHTS,'model_eval':not model.training,'dropout_active':False}
        save(OUT/'layers_local'/f'dev_{index:04d}.json',result)
        results.append({k:v for k,v in result.items() if k not in ('layer_positions','constraints')})
        print(f"Step3 {ordinal+1}/5 dev index{index} n{records[index]['n']} output cancellation {out_summary['sum_to_sum_of_norms']:.3f}, parameter {param_summary['sum_to_sum_of_norms']:.3f}",flush=True)
        if time.perf_counter()-started>480:raise TimeoutError('Step3 diagnostic budget reached; saved evidence retained')
    assert sha(checkpoint)==identity
    aggregate={'count':5,'seconds':time.perf_counter()-started,'checkpoint_unchanged':True,
        'training_steps':0,'test_labels_read':False,'cases':results,
        'interpretation':'five exploratory development cases; eval gradients do not measure dropout/AdamW update trajectories; intermediate heads were not independently supervised'}
    save(OUT/'stage3_summary.json',aggregate)


if __name__=='__main__':main()
