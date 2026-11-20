"""Paired dev500 L4-vs-final Loss association from existing frozen records only."""
from pathlib import Path
import csv
import hashlib
import json
import time
import numpy as np
from scipy.stats import pearsonr,spearmanr

OUT=Path(__file__).resolve().parent
RUN=OUT.parent
EXPERIMENT=RUN.parents[1]
KEYS=('reference_weighted','overlap_weighted','wire_compact_weighted','grouping_weighted','boundary_weighted','public_shape_weighted','public_preplaced_position_weighted')

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def stats(x,y):
    return dict(pearson_r=float(pearsonr(x,y).statistic),spearman_rho=float(spearmanr(x,y).statistic))

def main():
    started=time.perf_counter()
    if (OUT/'results.json').exists():raise RuntimeError('Analysis already exists; preserve original')
    paths={
        'stage_cases':RUN/'insert_s0_s1_dev500_v1/per_case.json',
        'stage_result':RUN/'insert_s0_s1_dev500_v1/results.json',
        'extended_cases':RUN/'add_s1_s2_dev500_v1/per_case.json',
        'extended_result':RUN/'add_s1_s2_dev500_v1/results.json',
        'frozen':EXPERIMENT/'evaluation/frozen_e148_shared_functional_shape.json'}
    identity={key:dict(path=str(p),sha256=sha(p)) for key,p in paths.items()}
    frozen=read(paths['frozen']);a=read(paths['stage_cases']);b=read(paths['extended_cases'])
    assert len(a)==len(b)==500
    members=[r['member'] for r in a];assert members==[r['member'] for r in b]
    for name in ('stage_result','extended_result'):
        result=read(paths[name]);assert result['status']=='complete' and result['count']==500
        assert result['checkpoint_sha256']==frozen['checkpoint_sha256']
    x_full=np.array([r['baseline']['S0']['total'] for r in a])
    x_stage=np.array([r['baseline']['S0']['stage_objective'] for r in a])
    y12=np.array([r['baseline']['S2']['total'] for r in a])
    y16=np.array([r['layer16']['total'] for r in b])
    assert abs(y12.mean()-frozen['own_new_final_dev_loss'])<1e-12
    assert abs(y16.mean()-read(paths['extended_result'])['final_full_loss'])<1e-12
    assert np.isfinite(np.stack((x_full,x_stage,y12,y16))).all()
    shards=[r[0] for r in members];unique=sorted(set(shards))
    groups=[np.flatnonzero(np.array(shards)==shard) for shard in unique]
    replicates=2000;seed=20261117
    save(OUT/'plan.json',dict(authorization='User requested correlation between quality at L4 after two S0 calls and final output Loss on same500; no Q analysis',
        checkpoint_sha256=frozen['checkpoint_sha256'],best_epoch=46,source_records=identity,
        primary='Original12-layer L12 full Loss vs original L4; secondary current extended16-layer final Loss',
        L4_measures=['complete full Loss','registered stage objective excluding direct overlap'],
        correlation=['Pearson linear association','Spearman rank association'],
        uncertainty=dict(method='Source-shard cluster percentile bootstrap, resample299 shards with replacement retaining all their cases',replicates=replicates,seed=seed),
        tails='Top50 and top100 highest losses; stable ordering by original case index for exact ties',
        inference_calls=0,training_updates=0,postprocessing_calls=0,Q_calls=0,test_read=False,
        diagnostic_source_sha256=sha(Path(__file__))))
    pairs={f'{xn}_vs_{yn}':(x,y) for xn,x in (('L4_full',x_full),('L4_stage_no_overlap',x_stage)) for yn,y in (('L12_full',y12),('L16_full',y16))}
    associations={key:stats(x,y) for key,(x,y) in pairs.items()}
    intervals={key:np.zeros((replicates,2)) for key in pairs}
    rng=np.random.default_rng(seed)
    for repeat in range(replicates):
        ids=np.concatenate([groups[i] for i in rng.integers(0,len(groups),size=len(groups))])
        for key,(x,y) in pairs.items():
            values=stats(x[ids],y[ids]);intervals[key][repeat]=[values['pearson_r'],values['spearman_rho']]
        if (repeat+1)%500==0:print('cluster bootstrap',repeat+1,'/',replicates,flush=True)
    for key,value in associations.items():
        limits=np.quantile(intervals[key],[.025,.975],axis=0)
        value.update(pearson_cluster95=limits[:,0].tolist(),spearman_cluster95=limits[:,1].tolist(),
            descriptive_linear_R_squared=value['pearson_r']**2)
        x,y=pairs[key];order_x=np.argsort(-x,kind='stable');order_y=np.argsort(-y,kind='stable')
        value['tail_overlap']={}
        for k in (50,100):
            final=set(order_y[:k]);early=set(order_x[:k]);early100=set(order_x[:100])
            value['tail_overlap'][str(k)]=dict(final_worst_count=k,also_L4_worst_same_count=len(final&early),
                fraction=len(final&early)/k,expected_overlap_if_independent=k*k/500,
                also_L4_worst100_count=len(final&early100),
                final_worst_L4_loss_mean=float(x[order_y[:k]].mean()),others_L4_loss_mean=float(x[order_y[k:]].mean()))
        # Compare final Loss over ascending L4 quintiles; descriptive grouping only.
        value['L4_quintiles']=[]
        for i,ids in enumerate(np.array_split(order_x[::-1],5)):
            value['L4_quintiles'].append(dict(quintile=i+1,count=len(ids),L4_mean=float(x[ids].mean()),final_mean=float(y[ids].mean())))
    components={}
    for key in KEYS:
        x=np.array([r['baseline']['S0'][key] for r in a])
        components[key]={name:stats(x,y) for name,y in (('L12_full',y12),('L16_full',y16))}
    # Total Loss shares a reference-fitting component; separate it for interpretation.
    x_ref=np.array([r['baseline']['S0']['reference_weighted'] for r in a])
    y12_ref=np.array([r['baseline']['S2']['reference_weighted'] for r in a])
    y16_ref=np.array([r['layer16']['reference_weighted'] for r in b])
    nonreference={name:stats(x_full-x_ref,y-yref) for name,y,yref in (('L12',y12,y12_ref),('L16',y16,y16_ref))}
    for key,p in paths.items():assert sha(p)==identity[key]['sha256'],'Saved source changed during analysis'
    result=dict(status='complete',count=500,source_shards=len(unique),best_epoch=46,checkpoint_sha256=frozen['checkpoint_sha256'],
        associations=associations,L4_component_vs_final=components,full_loss_without_reference_component=nonreference,
        means=dict(L4_full=float(x_full.mean()),L4_stage_no_overlap=float(x_stage.mean()),L12_full=float(y12.mean()),L16_full=float(y16.mean())),
        notes=['All measurements are paired frozen best46 dev500 records; no new model inference or data sampling.',
            'L4 stage objective excludes direct overlap exactly as training; full L4 includes it. Neither is multiplied by auxiliary weight. Final Loss is single output full Loss.',
            'Correlation and descriptive R-squared are in-sample association, not causal attribution or held-out prediction accuracy.',
            'Reference fitting appears in both early and final totals; correlation alone does not establish that early topology causes final errors.',
            'Bootstrap respects repeated cases sharing299 source shards; single checkpoint and repeatedly used dev500 still limit generalization.'],
        training_updates=0,inference_calls=0,postprocessing_calls=0,Q_calls=0,test_read=False,elapsed_seconds=time.perf_counter()-started)
    save(OUT/'results.json',result)
    with (OUT/'paired_loss.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['index','shard','row','L4_full','L4_stage_no_overlap','L12_full','L16_full'])
        for i in range(500):writer.writerow([i,*members[i],x_full[i],x_stage[i],y12[i],y16[i]])
    lines=['# E148 best46: L4 versus final Loss association','',
        'Fixed original dev500,500 paired cases from299 source shards. Frozen saved per-case records only; no inference, training, postprocessing, Q or test.','',
        '|Early measure|Final output|Pearson r|Cluster95% interval|Spearman rho|Cluster95% interval|Final worst50 also early worst50|Final worst50 also early worst100|',
        '|---|---|---:|---|---:|---|---:|---:|']
    for key,v in associations.items():
        early,final=key.split('_vs_');p=v['pearson_cluster95'];s=v['spearman_cluster95'];tail=v['tail_overlap']['50']
        lines.append(f'|{early}|{final}|{v["pearson_r"]:.6f}|[{p[0]:.6f}, {p[1]:.6f}]|{v["spearman_rho"]:.6f}|[{s[0]:.6f}, {s[1]:.6f}]|{tail["also_L4_worst_same_count"]}/50|{tail["also_L4_worst100_count"]}/50|')
    lines+=['','L4 full includes direct overlap; L4 stage objective excludes it. Both use original coefficients, no auxiliary layer weight. Final uses complete single-output Loss.','',
        'Source-shard cluster bootstrap:2000 repeats, seed20261117. For independent random rankings, worst50 overlap would average5 cases.','',
        '|L4 component|Pearson vs L12|Spearman vs L12|Pearson vs L16|Spearman vs L16|','|---|---:|---:|---:|---:|']
    for key,v in components.items():lines.append(f'|{key}|{v["L12_full"]["pearson_r"]:.6f}|{v["L12_full"]["spearman_rho"]:.6f}|{v["L16_full"]["pearson_r"]:.6f}|{v["L16_full"]["spearman_rho"]:.6f}|')
    lines+=['','After removing the weighted reference component from both full Loss values:']
    for key,v in nonreference.items():lines.append(f'- L4 vs {key}: Pearson {v["pearson_r"]:.6f}, Spearman {v["spearman_rho"]:.6f}.')
    lines+=['','These are associations on the observed panel, not proof of a causal early-layout bottleneck or of insufficient downstream correction. Shared reference fitting and common problem difficulty can contribute.','']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10.5,8),constrained_layout=True)
    for ax,(key,(x,y)) in zip(axes.flat,pairs.items()):
        final_bad=np.argsort(-y,kind='stable')[:50];mask=np.zeros(500,dtype=bool);mask[final_bad]=True
        ax.scatter(x[~mask],y[~mask],s=12,alpha=.5,color='#3974a6',edgecolors='none')
        ax.scatter(x[mask],y[mask],s=18,alpha=.8,color='#c34b39',edgecolors='none')
        v=associations[key];early,final=key.split('_vs_')
        ax.set_xlabel('L4 full Loss' if early=='L4_full' else 'L4 stage Loss (without overlap)')
        ax.set_ylabel(final.replace('_',' ')+' Loss');ax.grid(alpha=.18)
        ax.set_title(f'r={v["pearson_r"]:.3f}, rank rho={v["spearman_rho"]:.3f}')
    fig.suptitle('E148 best46, fixed dev500; red = final worst50 cases',fontsize=13)
    fig.savefig(OUT/'scatter.png',dpi=170);plt.close(fig)
    print(json.dumps(dict(associations=associations,full_loss_without_reference_component=nonreference),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
