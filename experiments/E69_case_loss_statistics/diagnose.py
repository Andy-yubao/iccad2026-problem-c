"""Bounded, read-only follow-up: frozen sensitivities, gradients, and saved controls."""
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
import copy
import multiprocessing as mp
import os
import time
import numpy as np
import torch
from .io import ROOT,HERE,read,save,sha
from .collect import RUNS,key,raw_overlap
from .data import load_rows,collate
from .batches import inputs_only,sample,collate as graph_collate
from .base_model import GeometryModel as E42Model,decode
from .model import GeometryModel as E65Model
from .losses import components,WEIGHTS
from .diagnostic_metrics import coverage_label,spatial_metrics,graph_node_metrics,cosine,mean_rows

OUT=HERE/'runs/20261028_sparse_anchor_diagnosis'
OLD=HERE/'runs/20261028_dev2000'


def model_load(name):
    path=RUNS[name]/'checkpoints/best.pt';d=torch.load(path,map_location='cpu',weights_only=False)
    model=(E42Model if name=='E42' else E65Model)(**d['model_config']).cuda().eval()
    model.load_state_dict(d['model'],strict=True)
    return model,sha(path)


def original_inputs(row):
    return tuple(int(v) if i==0 else np.asarray(v).copy() for i,v in enumerate(row['case'].solver_inputs))


def summarize(rows):
    output={}
    for label in ('low','middle','high','all'):
        selected=[r for r in rows if label=='all' or coverage_label(r['pin_coverage'])==label]
        output[label]={'count':len(selected)}
        for field in ('G','overlap','center_gradient_ratio_to_rest','center_overlap_cosine','total_overlap_cosine'):
            output[label][field]=mean_rows(selected,field)
        for step in (0,3,6,9,12):
            output[label]['layer'+str(step)]={field:mean_rows([r['layers'][str(step)] for r in selected],field)
                for field in ('spread_ratio','fit_reference_scale','centered_mse')}
        for group in ('pin','no_pin','hop1','hop2plus','unreachable'):
            nodes=[r['nodes']['node_groups'][group] for r in selected]
            output[label][group]={'module_count':sum(r['count'] for r in nodes),
                  'case_mean_local_error':mean_rows(nodes,'mean_local_error'),
                  'case_mean_spread_ratio':mean_rows(nodes,'spread_ratio')}
    return output


def forward_diagnosis(model,rows,features,deadline):
    result=[]; fmap={key(r['member']):r for r in features}; selected=[]
    for start in range(0,len(rows),16):
        deadline(); chosen=rows[start:start+16];g,_=collate(chosen,'cuda'); states=[]
        def capture(module,args,output):states.append(output.detach())
        hook=model.head.register_forward_hook(capture)
        with torch.no_grad():prediction=model(inputs_only(g))
        hook.remove()
        assert len(states)==13
        state=prediction.detach().requires_grad_(True);values=components(state,g)
        gradients={term:torch.autograd.grad((WEIGHTS[term]*values[term]).sum(),state,retain_graph=True)[0].detach()
                   for term in WEIGHTS}
        total=sum(gradients.values());rest=total-gradients['center_local']
        p=decode(state,g).detach().cpu().numpy();ref=g['reference'].cpu().numpy();mask=g['mask'][...,0].cpu().numpy().astype(bool)
        for j,row in enumerate(chosen):
            n=row['graph']['n'];f=fmap[key(row['member'])];mutable=mask[j,:n]
            cg=gradients['center_local'][j,:n].cpu().numpy().ravel();og=gradients['overlap'][j,:n].cpu().numpy().ravel()
            rg=rest[j,:n].cpu().numpy().ravel();tg=total[j,:n].cpu().numpy().ravel()
            layers={}
            for step in (0,3,6,9,12):
                pp=decode(states[step][j:j+1],{k:v[j:j+1] for k,v in g.items()}).cpu().numpy()[0,:n]
                layers[str(step)]=spatial_metrics(pp[:,:2]+pp[:,2:]/2,ref[j,:n,:2]+ref[j,:n,2:]/2,mutable)
            record={'member':list(row['member']),'pin_coverage':f['pin_connected_fraction'],
                    'density':f['connection_density'],'n':n,
                    'G':float(values['geometry'][j]),'overlap':raw_overlap(p[j,:n]*float(g['scale'][j]),row['case'].solver_inputs),
                    'center_gradient_ratio_to_rest':float(np.linalg.norm(cg)/max(np.linalg.norm(rg),1e-12)),
                    'center_overlap_cosine':cosine(cg,og),'total_overlap_cosine':cosine(tg,og),
                    'center_coefficient_vs_overlap_cosine':{str(w):cosine(w*cg+rg,og) for w in (1.,.75,.5,.25,0.)},
                    'weighted_terms':{k:float(WEIGHTS[k]*values[k][j]) for k in WEIGHTS},
                    'weighted_gradient_norms':{k:float(v[j,:n].norm()) for k,v in gradients.items()},'layers':layers}
            result.append(record)
            scale=float(g['scale'][j]);selected.append((list(row['member']),original_inputs(row),ref[j,:n]*scale,p[j,:n]*scale))
        del state,values,gradients,total,rest,states,prediction,g
        if start%80==0:print('Frozen layer/output gradients',start+len(chosen),'/',len(rows),flush=True)
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
        nodes=list(pool.map(graph_node_metrics,selected,chunksize=8))
    for r,node in zip(result,nodes):r['nodes']=node
    return result


def thin_inputs(model,rows,features,deadline):
    fmap={key(r['member']):r for r in features};pool=[r for r in rows if fmap[key(r['member'])]['pin_connected_fraction']>=.8]
    chosen=[]
    for target_n in (30,50,70,90,110):
        row=min((r for r in pool if key(r['member']) not in {key(x['member']) for x in chosen}),
                key=lambda r:(abs(r['graph']['n']-target_n),key(r['member'])))
        chosen.append(row)
    output=[]
    for row in chosen:
        deadline();inputs=original_inputs(row);reference=row['graph']['reference']*row['graph']['scale']
        variants=[];labels=[]
        for label,pin_keep,edge_keep in (('full',1.,1.),('pins_half',.5,1.),('pins_none',0.,1.),
                                         ('edges_half',1.,.5),('edges_none',1.,0.),('pins_edges_none',0.,0.)):
            x=list(copy.deepcopy(inputs))
            for slot,keep in ((2,edge_keep),(3,pin_keep)):
                a=x[slot];a=a[~(a==-1).all(1)]
                ids=np.arange(len(a));rng=np.random.default_rng(6970+slot);rng.shuffle(ids)
                x[slot]=a[np.sort(ids[:int(np.ceil(keep*len(a)))])]
            variants.append(sample(tuple(x),reference));labels.append(label)
        g=graph_collate(variants,list(range(len(variants))),'cuda')
        with torch.no_grad():
            state=model(inputs_only(g));values=components(state,g);p=decode(state,g)
            constant=components(state[:1].expand_as(state),g)['geometry']
        assert (constant-constant[0]).abs().max()<2e-6
        out={'member':list(row['member']),'n':inputs[0],'variants':{}}
        for j,label in enumerate(labels):
            pos=(p[j,:inputs[0]]*g['scale'][j]).cpu().numpy()
            out['variants'][label]={'G_fixed_reference':float(values['geometry'][j]),
                    'overlap':raw_overlap(pos,inputs),
                    'draft_center_displacement_from_full':float(((p[j,:,:2]+p[j,:,2:]/2)-(p[0,:,:2]+p[0,:,2:]/2)).norm(dim=-1).mean()),
                    'spread':spatial_metrics((p[j,:,:2]+p[j,:,2:]/2).cpu().numpy(),
                                            (g['reference'][j,:,:2]+g['reference'][j,:,2:]/2).cpu().numpy(),
                                            g['mask'][j,:,0].cpu().numpy().astype(bool))}
        output.append(out)
    return output


def parameter_probes(model,deadline):
    indices=[1683,10027,5474,9793,1814]
    rows=load_rows('train',indices=indices,use_cache=False);results=[]
    params=tuple(model.parameters())
    for row,index in zip(rows,indices):
        deadline();g,_=collate([row],'cuda');state=model(inputs_only(g));v=components(state,g);grads={}
        for term in WEIGHTS:
            tensors=torch.autograd.grad(WEIGHTS[term]*v[term].sum(),params,retain_graph=True,allow_unused=True)
            grads[term]=torch.cat([(torch.zeros_like(p) if x is None else x).ravel() for x,p in zip(tensors,params)]).detach()
        cg=grads['center_local'];og=grads['overlap'];rest=sum(t for k,t in grads.items() if k!='center_local')
        def cos(a,b):return float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(1e-15))
        rowout={'train_index':index,'member':list(row['member']),
                'pin_coverage':float((g['features'][0,:,14]>0).float().mean()),'G':float(v['geometry'][0]),
                'weighted_gradient_norms':{k:float(x.norm()) for k,x in grads.items()},
                'center_to_rest_gradient_norm_ratio':float(cg.norm()/rest.norm().clamp_min(1e-12)),
                'center_overlap_cosine':cos(cg,og),
                'center_coefficient_vs_overlap_cosine':{str(w):cos(w*cg+rest,og) for w in (1.,.75,.5,.25,0.)}}
        results.append(rowout);del g,state,v,grads,cg,og,rest
    return results


def saved_center_controls(features):
    fmap={key(r['member']):r for r in features};data={};sources={}
    for name in ('control','weight075','softband005'):
        path=ROOT/f'experiments/E63_e42_center_relaxation/runs/20261027_{name}_seed0/full_dev.json'
        d=read(path);data[name]={key(r['member']):r for r in d['records']};sources[str(path.relative_to(ROOT))]=sha(path)
        assert set(data[name])==set(fmap)
    results={}
    for label in ('low','middle','high','all'):
        keys=[k for k,f in fmap.items() if label=='all' or coverage_label(f['pin_connected_fraction'])==label]
        base=np.array([data['control'][k]['overlap_area_fraction'] for k in keys]);r={'count':len(keys),'control_overlap':float(base.mean())}
        for name in ('weight075','softband005'):
            values=np.array([data[name][k]['overlap_area_fraction'] for k in keys]);diff=values-base
            r[name]={'mean_overlap':float(values.mean()),'delta_percentage_points':float(diff.mean()*100),
                     'relative_change':float(values.mean()/base.mean()-1),'improved_fraction':float((diff<0).mean())}
        results[label]=r
    return {'strata':results,'input_sha256':sources,'scope':'Existing single-seed paired runs with branch-specific fixed selection; exploratory subgroup analysis'}


def pin_encoding_collision():
    from .features import graph_features
    # Total area4 makes the normalization scale exactly2, avoiding a one-ULP
    # rounding difference between mathematically identical weighted centroids.
    common=(2,np.array([2.,2.]),np.array([[0,1,1.]]),None,np.array([[1.,1.],[2.,1.],[4.,1.]]),
            np.zeros((2,5)),np.full((2,4),-1.))
    a=list(common);b=list(common);a[3]=np.array([[0,0,2.],[2,0,1.]])
    b[3]=np.array([[1,0,3.]])
    ga=graph_features(a);gb=graph_features(b)
    assert all(torch.equal(ga[k],gb[k]) for k in ('features','adjacency'))
    return {'identical_encoded_features':True,'pin_coordinates':common[4].tolist(),
            'A_pin_weights':[2,0,1],'B_pin_weights':[0,3,0],
            'both_total_pin_weight':3,'both_weighted_pin_x_centroid':2,
            'A_optimal_block_center_x':1,'B_optimal_block_center_x':2,
            'A_pin_wire_cost_at_x1_x2':[3,4],'B_pin_wire_cost_at_x1_x2':[3,0],
            'scope':'Unconstrained one-dimensional weighted two-terminal pin wirelength subproblem. Exact encoding collision; not an observed dataset duplicate or complete-layout optimum.'}


def main():
    if (OUT/'results.json').exists():raise ValueError('Completed diagnosis must not be overwritten')
    OUT.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    def deadline():
        if time.perf_counter()-started>900:raise RuntimeError('15-minute diagnostic budget exhausted')
    features=read(OLD/'features.json')['records'];model,digest=model_load('E65_geometry')
    config={'plan_sha256':sha(HERE/'diagnosis_plan.json'),'sources':{p.name:sha(p) for p in HERE.glob('*.py')},
            'source_analysis':'runs/20261028_dev2000','E65_checkpoint_sha256':digest,
            'utc':datetime.now(timezone.utc).isoformat(),'max_seconds':900,'cpu_workers':4,
            'model_training_mode':False,'optimizer_created':False,'test_read':False}
    save(OUT/'config.json',config)
    rows=load_rows('dev_monitor500')
    diagnostics=forward_diagnosis(model,rows,features,deadline);save(OUT/'case_diagnostics.json',diagnostics)
    stored={key(r['member']):r for r in read(OLD/'forwards.json')['models']['E65_geometry']}
    check_error=max(abs(r['G']-stored[key(r['member'])]['loss_total']) for r in diagnostics)
    assert check_error<2e-5,check_error
    sensitivity={'E65':thin_inputs(model,rows,features,deadline)}
    print('Output diagnostics complete; five-case parameter gradients',flush=True)
    probes=parameter_probes(model,deadline);save(OUT/'parameter_gradients.json',probes)
    assert sha(RUNS['E65_geometry']/'checkpoints/best.pt')==digest
    del model;torch.cuda.empty_cache();base,base_digest=model_load('E42')
    sensitivity['E42']=thin_inputs(base,rows,features,deadline)
    assert sha(RUNS['E42']/'checkpoints/best.pt')==base_digest
    save(OUT/'input_sensitivity.json',sensitivity)
    result={'status':'complete','case_count':len(diagnostics),'summary':summarize(diagnostics),
            'saved_E63_controls':saved_center_controls(features),'pin_encoding_collision':pin_encoding_collision(),
            'E42_checkpoint_sha256':base_digest,'elapsed_seconds':time.perf_counter()-started,
            'parameter_probe_count':len(probes),'input_sensitivity_cases_per_model':5,
            'training_started':False,'optimizer_created':False,'postprocessing_started':False,'test_read':False}
    density_median={n:float(np.median([f['connection_density'] for f in features if f['n']==n])) for n in {f['n'] for f in features}}
    result['pin_density_cross_strata']={}
    for p in ('low','middle','high'):
        for sparse in (True,False):
            subset=[r for r in diagnostics if coverage_label(r['pin_coverage'])==p and (r['density']<=density_median[r['n']])==sparse]
            result['pin_density_cross_strata'][p+('_sparse' if sparse else '_dense')]=summarize(subset)['all']
    result['stored_E65_G_max_abs_difference']=check_error
    save(OUT/'results.json',result)
    print('Sparse-anchor diagnosis complete',result['elapsed_seconds'],flush=True)
    print('strata',result['summary'],flush=True)


if __name__=='__main__':main()
