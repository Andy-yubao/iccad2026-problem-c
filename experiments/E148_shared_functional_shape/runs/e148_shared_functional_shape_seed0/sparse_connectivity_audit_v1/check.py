"""Small fixed-dev500 connectivity check, including frozen true-initial-draft inference."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import json
import time
import numpy as np
import torch
from scipy.stats import rankdata
from scipy.sparse.csgraph import connected_components
from experiments.E148_shared_functional_shape.io import ROOT,HERE,sha,save
from experiments.E148_shared_functional_shape.train import seed_all
from experiments.E148_shared_functional_shape.model import GeometryModel
from experiments.E148_shared_functional_shape.data import load_rows
from experiments.E148_shared_functional_shape.resident import ResidentRows
from experiments.E148_shared_functional_shape.batches import inputs_only
from experiments.E148_shared_functional_shape.loss import base_per_case

OUT=Path(__file__).resolve().parent
RUN=OUT.parent

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def corr(x,y):
    return None if np.std(x)<1e-12 or np.std(y)<1e-12 else float(np.corrcoef(x,y)[0,1])
def rho(x,y):return corr(rankdata(x),rankdata(y))
def partial_rank(x,y,controls):
    design=np.column_stack((np.ones(len(x)),*[rankdata(c) for c in controls]))
    a=rankdata(x);b=rankdata(y)
    return corr(a-design@np.linalg.lstsq(design,a,rcond=None)[0],b-design@np.linalg.lstsq(design,b,rcond=None)[0])

def main():
    started=time.perf_counter();assert not (OUT/'results.json').exists()
    frozen=read(HERE/'evaluation/frozen_e148_shared_functional_shape.json');weight=ROOT/frozen['checkpoint']
    assert sha(weight)==frozen['checkpoint_sha256']
    seed_all(0);weights=torch.load(weight,map_location='cpu',weights_only=False)
    for p,d in weights['identity']['sources'].items():assert sha(ROOT/p)==d,p
    old=read(RUN/'insert_s0_s1_dev500_v1/per_case.json');extended=read(RUN/'add_s1_s2_dev500_v1/per_case.json')
    members=[r['member'] for r in old];assert members==[r['member'] for r in extended]
    rows=load_rows('dev_monitor500');assert members==[list(r['member']) for r in rows]
    save(OUT/'plan.json',dict(authorization='User asked for a simple sparse-connectivity weakness check and potential solutions',
        checkpoint_sha256=frozen['checkpoint_sha256'],panel='Original fixed dev500,500 members; cache reused',
        scope='Connectivity summaries from public cached inputs; existing L4/L12/L16 Loss; only true initial generator receives new frozen inference',
        primary_connectivity='Unique positive undirected B2B edges excluding self; density=2E/[N(N-1)], mean degree=2E/N',
        controls=['module count','fixed/preplaced fraction','preplaced fraction','pin-connected fraction','group-member fraction','boundary-required fraction','area coefficient of variation'],
        bootstrap='1000 source-shard cluster resamples; seed20261118',
        training_updates=0,postprocessing_calls=0,Q_calls=0,test_read=False,diagnostic_sha256=sha(Path(__file__))))
    features=[];duplicate_groups=[]
    for row in rows:
        g=row['graph'];n=g['n'];adj=g['adjacency'].numpy()>0;np.fill_diagonal(adj,False);adj=adj|adj.T
        degree=adj.sum(1);edges=int(degree.sum()/2);num,labels=connected_components(adj,directed=False)
        group=g['group'].numpy()>0;grouped=group.any(1)
        pin=((g['pin_weights']>0)&g['pin_valid']).any(1).numpy()
        c=g['constraints'].numpy();known=(c[:,0]!=0)|(c[:,1]!=0);pre=c[:,1]!=0;boundary=c[:,4]!=0
        cg,cl=connected_components(adj|group,directed=False)
        free=(degree==0)&~pin&~grouped&~known&~boundary
        buckets={}
        for i in np.flatnonzero(free):buckets.setdefault(g['features'][i].numpy().tobytes(),[]).append(int(i))
        duplicates=[ids for ids in buckets.values() if len(ids)>1];duplicate_groups.append(duplicates)
        area=g['area'].numpy()
        features.append(dict(n=n,edges=edges,density=2*edges/(n*(n-1)),mean_degree=2*edges/n,
            isolated_fraction=float((degree==0).mean()),degree_le1_fraction=float((degree<=1).mean()),
            components=int(num),components_per_node=num/n,largest_component_fraction=float(np.bincount(labels).max()/n),
            wire_group_components=int(cg),wire_group_largest_fraction=float(np.bincount(cl).max()/n),
            pin_fraction=float(pin.mean()),group_fraction=float(grouped.mean()),known_fraction=float(known.mean()),
            preplaced_fraction=float(pre.mean()),boundary_fraction=float(boundary.mean()),area_cv=float(area.std()/area.mean()),
            no_wire_no_pin_no_group_no_anchor_fraction=float(free.mean()),
            indistinguishable_free_nodes=sum(len(ids) for ids in duplicates)))
    resources=read(RUN/'resources.json');pack=ResidentRows(rows,'cuda',max_nodes=resources['max_nodes'],max_pins=resources['max_loss_ports_per_block'])
    model=GeometryModel(**weights['model_config']).cuda().eval();model.load_state_dict(weights['model']);model.requires_grad_(False)
    initial_values=[];symmetry_max=[]
    with torch.inference_mode():
        for start in range(0,500,64):
            batch=pack.batch(range(start,min(start+64,500)));public=inputs_only(batch)
            h=model.input(public['features'])*public['valid'][...,None]
            for block in model.initial_blocks:h=block(h,public)
            state=model.head(h)*public['valid'][...,None]
            measured=base_per_case(state,batch);keys=list(measured)
            array=torch.stack([measured[k] for k in keys],-1).cpu().double().numpy();assert np.isfinite(array).all()
            initial_values.extend([dict(zip(keys,r.tolist())) for r in array])
            state=state.cpu().numpy()
            for local,groups in enumerate(duplicate_groups[start:start+len(state)]):
                symmetry_max.extend(float(np.abs(state[local,ids]-state[local,ids[0]]).max()) for ids in groups)
            print('initial generator checked',min(start+64,500),'/500',flush=True)
    assert sha(weight)==frozen['checkpoint_sha256']
    x={k:np.array([r[k] for r in features]) for k in features[0]}
    y={
        'initial_full':np.array([r['total'] for r in initial_values]),
        'initial_reference':np.array([r['reference_weighted'] for r in initial_values]),
        'initial_overlap':np.array([r['overlap_weighted'] for r in initial_values]),
        'L4_full':np.array([r['baseline']['S0']['total'] for r in old]),
        'L4_reference':np.array([r['baseline']['S0']['reference_weighted'] for r in old]),
        'L4_overlap':np.array([r['baseline']['S0']['overlap_weighted'] for r in old]),
        'L12_full':np.array([r['baseline']['S2']['total'] for r in old]),
        'L16_full':np.array([r['layer16']['total'] for r in extended]),
        'L16_reference':np.array([r['layer16']['reference_weighted'] for r in extended]),
        'L16_overlap':np.array([r['layer16']['overlap_weighted'] for r in extended])}
    y['L16_without_reference']=y['L16_full']-y['L16_reference']
    controls=[x[k] for k in ('n','known_fraction','preplaced_fraction','pin_fraction','group_fraction','boundary_fraction','area_cv')]
    correlations={key:{out:dict(pearson=corr(v,target),spearman=rho(v,target),partial_rank_N=partial_rank(v,target,[x['n']]),
        partial_rank_public=partial_rank(v,target,controls)) for out,target in y.items()} for key,v in x.items() if np.std(v)>1e-12}
    order=np.argsort(x['density'],kind='stable');low=order[:125];high=order[-125:]
    strata={}
    for name,ids in (('sparsest125',low),('densest125',high)):
        strata[name]=dict(count=len(ids),mean_features={k:float(v[ids].mean()) for k,v in x.items()},mean_losses={k:float(v[ids].mean()) for k,v in y.items()})
    n_bands=[]
    for lo,hi in ((0,60),(60,80),(80,100),(100,121)):
        ids=np.flatnonzero((x['n']>=lo)&(x['n']<hi))
        if len(ids)<10:continue
        ids=ids[np.argsort(x['density'][ids],kind='stable')];a=ids[:len(ids)//2];b=ids[len(ids)//2:]
        n_bands.append(dict(n_range=[lo,hi],counts=[len(a),len(b)],density_means=[float(x['density'][a].mean()),float(x['density'][b].mean())],
            loss_means={k:[float(v[a].mean()),float(v[b].mean())] for k,v in y.items()}))
    groups=[np.flatnonzero(np.array([r[0] for r in members])==s) for s in sorted({r[0] for r in members})]
    boot={key:[] for key in ('initial_full','L4_full','L12_full','L16_full')};rng=np.random.default_rng(20261118)
    for _ in range(1000):
        ids=np.concatenate([groups[i] for i in rng.integers(0,len(groups),size=len(groups))])
        for key in boot:boot[key].append(partial_rank(x['density'][ids],y[key][ids],[v[ids] for v in controls]))
    result=dict(status='complete',count=500,source_shards=len(groups),checkpoint_sha256=frozen['checkpoint_sha256'],
        correlations=correlations,density_public_adjusted_cluster95={k:np.quantile(v,[.025,.975]).tolist() for k,v in boot.items()},
        density_quartile_comparison=strata,within_N_bands=n_bands,
        density_L16_adjusted_for_public_and_L4=partial_rank(x['density'],y['L16_full'],controls+[y['L4_full']]),
        symmetry=dict(cases_with_exactly_indistinguishable_free_peers=sum(bool(g) for g in duplicate_groups),
            groups=len(symmetry_max),max_initial_state_difference=max(symmetry_max) if symmetry_max else None),
        definitions='Density uses B2B positive unique undirected edges; public controls do not contain reference geometry. Partial rank is Pearson of residualized rank variables.',
        limitations='One frozen model and reused dev500; observational correlation/conditioning does not establish causality. Initial=actual initial generator, L4=two S0 calls.',
        training_updates=0,postprocessing_calls=0,Q_calls=0,test_read=False,elapsed_seconds=time.perf_counter()-started)
    save(OUT/'results.json',result)
    save(OUT/'per_case.json',[dict(index=i,member=members[i],connectivity=features[i],initial_losses=initial_values[i],existing_losses={k:float(v[i]) for k,v in y.items()},indistinguishable_groups=duplicate_groups[i]) for i in range(500)])
    lines=['# E148 sparse connectivity check','',
        'Frozen best46, original fixed dev500. True initial generator checked once without updating weights; L4/L12/L16 losses reused. No Q/postprocessing/test.','',
        '|Outcome|Density Spearman|Partial rank: N|Partial rank: public controls|Cluster95% public-adjusted|','|---|---:|---:|---:|---|']
    for key in y:
        v=correlations['density'][key];ci=result['density_public_adjusted_cluster95'].get(key)
        lines.append(f'|{key}|{v["spearman"]:.6f}|{v["partial_rank_N"]:.6f}|{v["partial_rank_public"]:.6f}|{ci or "-"}|')
    lines+=['','Negative coefficient means denser graph is associated with lower Loss.','',
        '|Group|Count|Mean N|Mean degree|Mean density|Isolated fraction|Initial full|L4 full|L16 full|L16 reference|L16 without reference|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for label,r in strata.items():
        f=r['mean_features'];v=r['mean_losses']
        lines.append(f'|{label}|{r["count"]}|{f["n"]:.2f}|{f["mean_degree"]:.3f}|{f["density"]:.5f}|{f["isolated_fraction"]:.3f}|{v["initial_full"]:.6f}|{v["L4_full"]:.6f}|{v["L16_full"]:.6f}|{v["L16_reference"]:.6f}|{v["L16_without_reference"]:.6f}|')
    lines+=['','Public controls: N, F-or-P fraction, P fraction, pin-connected fraction, group-member fraction, boundary fraction, area CV. Cluster bootstrap respects299 source shards;1000 repeats.','',
        'Exact indistinguishable isolated/free peers: '+str(result['symmetry']),
        'Density vs final, additionally adjusted for L4: '+str(result['density_L16_adjusted_for_public_and_L4']),
        '',result['limitations']]
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(quartiles=strata,density=correlations['density'],confidence=result['density_public_adjusted_cluster95'],symmetry=result['symmetry']),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
