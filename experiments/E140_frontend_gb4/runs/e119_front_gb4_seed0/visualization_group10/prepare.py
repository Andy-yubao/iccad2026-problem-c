"""Prepare exactly ten saved final-grouping-failure examples; no model or repair."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[4];sys.path.insert(0,str(ROOT))
from visualization.style import normalize
from visualization.layouts import render


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def groups(p,flags):
    p=np.asarray(p);items=[]
    for g in sorted(set(flags[:,3])-{0}):
        ids=np.flatnonzero(flags[:,3]==g)
        union=unary_union([box(x,y,x+w,y+h) for x,y,w,h in p[ids]])
        count=len(union.geoms) if union.geom_type=='MultiPolygon' else 1
        items.append(dict(group=int(g),components=count,violations=count-1,modules=ids.tolist()))
    return sum(v['violations'] for v in items),items


def main():
    run=HERE.parent;aud=run/'comparative_audit_v1';aux=ROOT/'experiments/E134_supervision_interval/runs/e119_stride3_seed0/layer_analysis_v1'
    summary=read(run/'evaluation_no_mib_dev500/results.json');manifest=read(aud/'capture_manifest.json')
    assert sha(run/'checkpoints/best.pt')==summary['checkpoint_sha256']==manifest['checkpoint_sha256']
    records=[read(run/f'evaluation_no_mib_dev500/cases/{i:04d}.json') for i in range(500)]
    pool=[i for i,r in enumerate(records) if r['metrics']['grouping_violations']>0]
    chosen=sorted(np.random.default_rng(20261115).choice(pool,10,replace=False).tolist())
    pos=np.load(aud/'positions.npy',mmap_mode='r');loss=np.load(aud/'losses.npy',mmap_mode='r')
    ref=np.load(aux/'reference.npy',mmap_mode='r');flags=np.load(aux/'flags.npy',mmap_mode='r')
    names=manifest['loss_names'];cases=[];notes=[]
    layer_ids=list(range(2,13));weights={2:.05,6:.05,8:.05,10:.05,12:.8}
    for i in chosen:
        r=records[i];n=r['n'];f=flags[i,:n];assert r['member']==manifest['members'][i]
        np.testing.assert_array_equal(pos[i,12,:n],r['draft_positions'])
        final_count,final_groups=groups(r['positions'],f);reference_count,_=groups(ref[i,:n],f)
        assert final_count==r['metrics']['grouping_violations']>0
        details=[]
        for layer in layer_ids:
            values={key:float(loss[i,layer,j]) for j,key in enumerate(names)}
            count,_=groups(pos[i,layer,:n],f)
            values.update(grouping_violations=count,supervision_weight=weights.get(layer,0),directly_supervised=layer in weights)
            if layer in (2,6):
                values.update(objective_name='无碰撞，G/B各×4',objective_total=values['total']-values['overlap_weighted']+3*(values['grouping_weighted']+values['boundary_weighted']))
            elif layer in weights:values.update(objective_name='完整Loss',objective_total=values['total'])
            details.append(values)
        assert details[-1]['grouping_violations']==r['raw_metrics']['grouping_violations']
        bad=[v for v in final_groups if v['violations']]
        c=dict(label=f'dev {i:04d} · 最终Group违规 {final_count}',dev_index=i,member=r['member'],n=n,
            constraints=f.tolist(),reference=ref[i,:n].tolist(),layers=pos[i,layer_ids,:n].tolist(),layer_ids=layer_ids,
            layer_losses=details,final=r['positions'],
            grouping_summary=dict(reference=reference_count,final=final_count,final_bad_groups=bad),
            objective_note='分项列为共同原系数完整Loss，未乘层权重。S2/S6实际监督去碰撞且G/B各×4；草稿断连数可受重叠影响。此视图展示真实S2–S12，S0/S1原件仍保留。',
            metrics=dict(Q=r['metrics']['cost'],hard_legal=r['metrics']['is_feasible'],complete_ms=r['complete_seconds']*1000))
        normalize(c);cases.append(c)
        notes.append(dict(index=i,n=n,final_grouping=final_count,bad_groups=bad,
            layer_grouping_counts={str(k):details[j]['grouping_violations'] for j,k in enumerate(layer_ids)},
            layer_grouping_loss={str(k):details[j]['grouping_weighted'] for j,k in enumerate(layer_ids)}))
    data=dict(title='E140 best53 · 10例最终分组违规布局',
        note='从最终分组违规>0的86题中，以固定seed20261115无放回随机抽10题。左Reference，中栏真实S2–S12滑块，右原后处理。只读已有坐标。',
        identity=dict(experiment='E140',checkpoint_sha256=summary['checkpoint_sha256'],seed=20261115,eligible_cases=len(pool),indices=chosen,
            source_positions_sha256=sha(aud/'positions.npy'),postprocessing_reused=True,training=False,inference_calls=0,postprocessing_calls=0,test_read=False),cases=cases)
    (HERE/'data.json').write_text(json.dumps(data,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    (HERE/'selection.json').write_text(json.dumps(dict(pool=pool,selected=chosen,seed=20261115,case_details=notes),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for i,c in enumerate(cases):
        render(c,HERE/f'dev{c["dev_index"]:04d}_S12.png',layer=12,title=f'E140 best53 | dev{c["dev_index"]:04d} | final Group violations: {c["grouping_summary"]["final"]}')
    # Raw grouping/boundary means recovered from existing weighted components only.
    B=.3336832108335229;raw={};ng=[];nb=[]
    for i,r in enumerate(records):
        f=flags[i,:r['n']];ng.append(sum(max(0,int((f[:,3]==g).sum())-1) for g in set(f[:,3])-{0}));nb.append(int((f[:,4]!=0).sum()))
    ng=np.array(ng);nb=np.array(nb)
    for layer in (2,6,12):
        g=loss[:,layer,names.index('grouping_weighted')]/(2*B)*(ng+nb)/np.maximum(ng,1)
        b=loss[:,layer,names.index('boundary_weighted')]/(2*B)*(ng+nb)/np.maximum(nb,1)
        raw[str(layer)]={k:dict(mean=float(v.mean()),median=float(np.median(v)),p95=float(np.quantile(v,.95)),
            median_gradient_factor_if_whole_term_squared=float(np.median(2*v))) for k,v in [('grouping',g),('boundary',b)]}
    g=loss[:,:,names.index('grouping_weighted')]
    preservation=dict(S2_lower_grouping_loss_than_S12=int((g[:,2]<g[:,12]).sum()),S6_lower_grouping_loss_than_S12=int((g[:,6]<g[:,12]).sum()),
        some_S2_to_S11_lower_than_S12=int((g[:,2:12].min(1)<g[:,12]-1e-10).sum()),
        among_final_grouping_failures_some_earlier_lower=int(sum(g[i,2:12].min()<g[i,12]-1e-10 for i in pool)))
    (HERE/'loss_square_assessment.json').write_text(json.dumps(dict(raw_terms=raw,preservation=preservation),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(selected=chosen,eligible=len(pool),raw_terms=raw,preservation=preservation),ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
