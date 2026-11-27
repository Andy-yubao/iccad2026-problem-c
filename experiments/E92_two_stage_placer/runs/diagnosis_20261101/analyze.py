"""Read-only E92 diagnosis: saved evidence plus at most five frozen-model probes."""
import argparse,json,sys,time,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='4'
import numpy as np
from experiments.E92_two_stage_placer.io import HERE,ROOT,save,sha

OUT=Path(__file__).resolve().parent
RUN=HERE/'runs/20261031_two_stage32k_seed0'
Q=RUN/'qualification'
E90=ROOT/'experiments/E90_auxiliary_weight_placer/runs/20261031_aux3_seed0'

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def offline():
    study=read(HERE/'study.json');assert study['qualification_registration_sha256']==sha(HERE/'runs/qualification_registration.json')
    reg=read(HERE/'runs/qualification_registration.json');frozen=read(Q/'frozen.json')
    for name,digest in reg['baseline']['source_files'].items():assert sha(ROOT/name)==digest
    assert sha(RUN/'checkpoints/F_best.pt')==frozen['checkpoint_sha256']
    features=read(ROOT/'experiments/E69_case_loss_statistics/runs/20261028_dev2000/features.json')['records']
    raw=read(Q/'full_dev.json')['records'];old=read(E90/'full_dev.json')['records']
    assert len(features)==len(raw)==len(old)==2000 and all(a['member']==b['member']==f['member'] for a,b,f in zip(raw,old,features))
    n=np.asarray([f['n'] for f in features]);pin=np.asarray([f['pin_connected_fraction'] for f in features]);density=np.asarray([f['connection_density'] for f in features])
    medians={int(k):float(np.median(density[n==k])) for k in np.unique(n)}
    sparse=np.array([p<=.3 and d<=medians[int(k)] for p,d,k in zip(pin,density,n)])
    assert sparse.sum()==157
    groups={}
    masks={'all':np.ones(2000,dtype=bool),'sparse157':sparse,'others':~sparse,'N100plus':n>=100,
        'low_pin_sparse':(pin<=.3)&(density<=np.array([medians[int(k)] for k in n])),
        'low_pin_dense':(pin<=.3)&(density>np.array([medians[int(k)] for k in n])),
        'other_pin_sparse':(pin>.3)&(density<=np.array([medians[int(k)] for k in n])),
        'other_pin_dense':(pin>.3)&(density>np.array([medians[int(k)] for k in n]))}
    for name,mask in masks.items():
        values={'count':int(mask.sum()),'mean_n':float(n[mask].mean())}
        for key in ('overlap_area_fraction','max_covered_fraction','module_pass10_fraction'):
            a=np.asarray([r[key] for r in raw]);b=np.asarray([r[key] for r in old])
            values[key]={'E90':float(b[mask].mean()),'E92':float(a[mask].mean()),'delta':float((a-b)[mask].mean())}
        groups[name]=values
    # Exact-N standardisation removes the change in module-count mix from this
    # subgroup comparison. It does not establish a causal role for sparse edges.
    by_n=[]
    delta=np.array([a['overlap_area_fraction']-b['overlap_area_fraction'] for a,b in zip(raw,old)])
    for k in np.unique(n):
        s=(n==k)&sparse;other=(n==k)&~sparse
        if s.any() and other.any():by_n.append({'n':int(k),'sparse_count':int(s.sum()),'sparse_delta':float(delta[s].mean()),'other_delta':float(delta[other].mean())})
    standardised=float(np.average([r['sparse_delta']-r['other_delta'] for r in by_n],weights=[r['sparse_count'] for r in by_n]))
    cpu=read(Q/'terminal/results.json')['records'];old_cpu=read(E90/'terminal/results.json')['records']
    raw_lookup={tuple(r['member']):r for r in raw};feature_lookup={tuple(f['member']):f for f in features}
    sparse_members={tuple(f['member']) for f,s in zip(features,sparse) if s};drift=[]
    for row in cpu:
        a=np.asarray(row['draft_positions']);b=np.asarray(raw_lookup[tuple(row['member'])]['positions'])
        lengths=np.sqrt(a[:,2:].prod(1));v=np.linalg.norm(a[:,:2]+a[:,2:]/2-b[:,:2]-b[:,2:]/2,axis=1)/lengths
        drift.append({'index':row['index'],'n':row['n'],'local_center_mean':float(v.mean()),'local_center_max':float(v.max())})
    tn=np.array([r['n'] for r in cpu]);w=np.exp((tn-tn.max())/12);qd=np.array([a['metrics']['cost']-b['metrics']['cost'] for a,b in zip(cpu,old_cpu)])
    strata=[]
    for low,high in ((20,40),(40,60),(60,80),(80,100),(100,121)):
        mask=(tn>=low)&(tn<high)
        if mask.any():strata.append({'N':[low,high-1],'count':int(mask.sum()),'Q_weight_fraction':float(w[mask].sum()/w.sum()),'Q_delta':float(np.average(qd[mask],weights=w[mask]))})
    baseloss=read(E90/'progress.json')['history'];baseloss=min(baseloss,key=lambda r:r['dev_monitor']['total'])['dev_monitor']
    cur=read(Q/'dev500_loss.json')['losses'];coef={'reference':1.,'overlap_total':.08975162861857286,'task':.3336832108335229}
    losses={key:{'E90':baseloss[key],'E92':cur[key],'weighted_delta':(cur[key]-baseloss[key])*c,'E92_fraction_of_total':cur[key]*c/cur['total']} for key,c in coef.items()}
    history=read(RUN/'progress.json')['history'];f_history=[r for r in history if r['phase']=='F'];curves=[]
    for r in f_history:
        v=np.asarray(r['dev_monitor']['per_case_total']);assert len(v)==500
        curves.append({'epoch':r['phase_epoch'],'lr':r['lr'],'next_lr':r['lr_next'],'total':r['dev_monitor']['total'],
            **{k:r['dev_monitor'][k] for k in coef},'N100plus_loss':float(v[tn>=100].mean()),'N20to59_loss':float(v[tn<60].mean()),'Q_weighted_loss_diagnostic':float(np.average(v,weights=w))})
    # Deliberately selected failure cases plus a small-layout positive control;
    # this is not a representative five-case estimate of population gradients.
    selected=[]
    def add(index,reason):
        if index not in {r['index'] for r in selected}:
            selected.append({'index':int(index),'member':cpu[index]['member'],'n':cpu[index]['n'],'reason':reason})
    add(int(np.argmax(qd)),'largest Q regression')
    add(15,'known large-layout regression')
    candidates=[i for i,r in enumerate(cpu) if tuple(r['member']) in sparse_members and i not in {s['index'] for s in selected}]
    add(max(candidates,key=lambda i:qd[i]),'largest Q regression within frozen sparse157')
    candidates=[i for i,r in enumerate(cpu) if r['n']==int(tn.max()) and i not in {s['index'] for s in selected}]
    add(max(candidates,key=lambda i:qd[i]),'largest-N regression')
    candidates=[i for i,r in enumerate(cpu) if r['n']<60 and i not in {s['index'] for s in selected}]
    add(min(candidates,key=lambda i:qd[i]),'small-layout positive control')
    assert len(selected)==5
    pairs={tuple(a['member']):(a,b) for a,b in zip(old,raw)}
    paired=[pairs[tuple(r['member'])] for r in cpu]
    old_totals=np.asarray([a['loss_total'] for a,b in paired])
    new_totals=np.asarray(cur['per_case_total'])
    common_loss={'old_reconstructed_monitor_mean':float(old_totals.mean()),
        'old_Q_weighted_F_loss':float(np.average(old_totals,weights=w)),
        'new_Q_weighted_F_loss':float(np.average(new_totals,weights=w)),
        'strata':[]}
    for low,high in ((20,40),(40,60),(60,80),(80,100),(100,121)):
        mask=(tn>=low)&(tn<high)
        common_loss['strata'].append({'N':[low,high-1],'E90_mean':float(old_totals[mask].mean()),
            'E92_mean':float(new_totals[mask].mean()),'E90_weighted':float(np.average(old_totals[mask],weights=w[mask])),
            'E92_weighted':float(np.average(new_totals[mask],weights=w[mask]))})
    dref=np.asarray([b['losses']['reference']-a['loss_reference'] for a,b in paired])
    dov=np.asarray([b['losses']['overlap_total']-a['loss_overlap_total'] for a,b in paired])
    dtask=np.asarray([b['losses']['task']-a['loss_task'] for a,b in paired]);non_tie=abs(qd)>1e-12;sensitivity=[]
    for multiplier in (1,1.5,2,3):
        difference=dref+.08975162861857286*multiplier*dov+.3336832108335229*dtask
        sensitivity.append({'overlap_multiplier':multiplier,'arithmetic_loss_delta':float(difference.mean()),
            'Q_weighted_loss_delta':float(np.average(difference,weights=w)),
            'paired_sign_agreement_fraction':float((np.sign(difference[non_tie])==np.sign(qd[non_tie])).mean()),
            'case15_loss_delta':float(difference[15])})
    pool=read(ROOT/'experiments/E88_sparse_training/runs/20261031_sparse_pool/sparse_members.json')
    registry=read(ROOT/'protocol/reference_workset_v6.json')
    base_members={tuple(r['member']) for r in registry['sets']['train']['records'][:32000]}
    new_count=sum(tuple(r['member']) not in base_members for r in pool['new_members'])
    result={'status':'saved_evidence_analysed','sparse_definition':'pin<=.30 and density<=frozen exact-N dev2000 median; identical E69 sparse157',
        'groups':groups,'exactN_sparse_extra_overlap_delta':standardised,'exactN_sparse_comparison':by_n,
        'Q_strata':strata,'loss_components':losses,'F_history':curves,'probe_cases':selected,
        'saved_cpu_gpu_drift':{'mean':float(np.mean([r['local_center_mean'] for r in drift])),
            'case_mean_p95':float(np.percentile([r['local_center_mean'] for r in drift],95)),
            'max_single_module':max(r['local_center_max'] for r in drift)},
        'audits':{'same2000_members':True,'E90_frozen_evidence_hashes_match':True,'E92_F_best_hash_matches':True,
                  'new_Q_calls':0,'new_training_updates':0,'test_read':False},
        'reference_notes':'Current adjacency/MIB/group features are already row-normalised; no raw degree-sum explanation assumed.'}
    result.update(common_F_loss_by_N=common_loss,posthoc_weight_sensitivity=sensitivity,
        sparse_pool_reuse={'existing_count':len(pool['new_members']),'genuinely_outside_E92_32k':new_count,
                           'already_inside_32k':len(pool['new_members'])-new_count,
                           'selection_public_only':pool.get('no_free_reference_fields_analyzed')})
    save(OUT/'offline.json',result);return result

def probes(data):
    import torch,gc
    from experiments.E92_two_stage_placer.model import GeometryModel
    from experiments.E92_two_stage_placer.arms import model_config
    from experiments.E92_two_stage_placer.data import load_rows
    from experiments.E92_two_stage_placer.resident import ResidentRows
    from experiments.E92_two_stage_placer.batches import inputs_only
    from experiments.E92_two_stage_placer.loss import per_case
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    begin=time.perf_counter();ids=[c['index'] for c in data['probe_cases']]
    rows=load_rows('dev_monitor500',ids);pack=ResidentRows(rows,'cuda',max_nodes=120)
    paths={'E90':E90/'checkpoints/best.pt','E92':RUN/'checkpoints/F_best.pt'}
    before={key:sha(path) for key,path in paths.items()};records=[]
    for name,path in paths.items():
        state=torch.load(path,map_location='cpu',weights_only=False)
        model=GeometryModel(**model_config('simple')).cuda().eval();model.load_state_dict(state['model']);del state
        model.requires_grad_(False)
        params=list(model.head[2].parameters())
        for p in params:p.requires_grad_(True)
        for i,row in enumerate(rows):
            graph=pack.batch([i]);valid=graph['valid'];trace=[];handles=[]
            def hook(label):
                def capture(module,args,value):
                    v=value.detach()[valid]
                    trace.append({'layer':label,'rms':float(v.square().mean().sqrt()),'max_abs':float(v.abs().max())})
                return capture
            for j,layer in enumerate(model.layers):
                handles += [layer.register_forward_hook(hook(f'hidden{j}')),
                            layer.local.register_forward_hook(hook(f'local{j}')),
                            layer.output.register_forward_hook(hook(f'attention{j}'))]
            prediction=model(inputs_only(graph))
            for handle in handles:handle.remove()
            values=per_case(prediction,graph,'F');keys=('reference','overlap_total','task');grads={}
            for j,key in enumerate(keys):
                pieces=torch.autograd.grad(values[key].mean(),params,retain_graph=j<2,allow_unused=True)
                grads[key]=torch.cat([(torch.zeros_like(p) if v is None else v).flatten() for p,v in zip(params,pieces)])
            cos={}
            for a,b in (('reference','overlap_total'),('reference','task'),('overlap_total','task')):
                cos[a+'__'+b]=float((grads[a]@grads[b])/(grads[a].norm()*grads[b].norm()).clamp_min(1e-20))
            weights={'reference':1.,'overlap_total':.08975162861857286,'task':.3336832108335229}
            weighted_norm={k:float((v*weights[k]).norm()) for k,v in grads.items()}
            with torch.no_grad():
                target_losses=per_case(graph['target'],graph,'F')
                graph_sums={key:graph[key].sum(-1)[valid].cpu().tolist() for key in ('adjacency','mib','group')}
            records.append({'model':name,**data['probe_cases'][i],
                'losses':{key:float(values[key].mean()) for key in ('total',*keys)},
                'readout_weighted_gradient_norms':weighted_norm,'readout_gradient_cosines':cos,
                'target_decoded_losses':{key:float(target_losses[key].mean()) for key in ('total',*keys)},
                'output_max_abs':float(prediction.detach()[valid].abs().max()),'activation_trace':trace,
                'graph_row_sum_max':{key:max(value) for key,value in graph_sums.items()}})
            del prediction,values,grads
        del model;gc.collect();torch.cuda.empty_cache()
    after={key:sha(path) for key,path in paths.items()};assert before==after
    result={'status':'completed','unique_cases':5,'records':records,'checkpoint_sha256_before':before,
        'checkpoint_sha256_after':after,'optimizer_created':False,'training_updates':0,'new_Q_calls':0,'test_read':False,
        'wall_seconds':time.perf_counter()-begin,
        'interpretation':'Gradients are of the shared last readout, not full-parameter Adam displacement. Selected counterexamples are not a population estimate. Only scalar derivative/activation diagnostics; no parameters updated.'}
    save(OUT/'probes.json',result)
    print(json.dumps({'probe_status':result['status'],'cases':ids,'wall_seconds':result['wall_seconds'],'training_updates':0},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--probe',action='store_true');args=p.parse_args()
    data=offline()
    print(json.dumps({key:data[key] for key in ('Q_strata','loss_components','saved_cpu_gpu_drift','probe_cases','exactN_sparse_extra_overlap_delta')},ensure_ascii=False,indent=2))
    if args.probe:probes(data)
