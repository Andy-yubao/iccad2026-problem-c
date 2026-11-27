"""Saved-panel comparison and public-frame audit; never calls a postprocessor."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.E80_modular_transfer.io import HERE,save,sha
OUT=HERE/'runs/20261029_e42_frozen_e79_transfer'
FOLDERS={
 'E42':ROOT/'experiments/E42_joint_regularization/runs/20261023_rectangle_w96d12_ab_seed0',
 'E74':ROOT/'experiments/E74_collision_32k_posttrain/runs/20261029_e42_topology_seed0',
 'E79':ROOT/'experiments/E79_center_refinement/runs/20261029_e78_center_seed0',
 'E80':OUT}
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def paired(a,b,key=lambda r:r['metrics']['cost']):
    assert [r['member'] for r in a]==[r['member'] for r in b]
    x=np.asarray([key(r) for r in a]);y=np.asarray([key(r) for r in b]);d=x-y
    n=np.asarray([r['n'] for r in a]);w=np.exp((n-n.max())/12)
    workers=np.asarray([r['member'][0].split('/')[0] for r in a]);groups=np.unique(workers)
    numerator=np.asarray([np.sum(w[workers==g]*d[workers==g]) for g in groups])
    denominator=np.asarray([w[workers==g].sum() for g in groups])
    picks=np.random.default_rng(20261029).integers(0,len(groups),(5000,len(groups)))
    samples=numerator[picks].sum(1)/denominator[picks].sum(1)
    return {'count':len(a),'current_mean':float(x.mean()),'baseline_mean':float(y.mean()),
       'arithmetic_delta':float(d.mean()),'weighted_delta':float(np.average(d,weights=w)),
       'worker_resampling95':np.quantile(samples,[.025,.975]).tolist(),
       'better_equal_worse':[int((d< -1e-8).sum()),int((abs(d)<=1e-8).sum()),int((d>1e-8).sum())]}

def geometry(p):
    p=np.asarray(p);lo=p[:,:2];hi=lo+p[:,2:];area=p[:,2:].prod(-1).sum();bbox=(hi.max(0)-lo.min(0)).prod()
    xs=np.unique(np.r_[lo[:,0],hi[:,0]]);union=0.
    for left,right in zip(xs[:-1],xs[1:]):
        active=(lo[:,0]<right)&(hi[:,0]>left);end=-np.inf;length=0.
        for bottom,top in sorted(zip(lo[active,1],hi[active,1])):
            length+=max(0.,top-max(bottom,end));end=max(end,top)
        union+=(right-left)*length
    return {'bbox_over_A':float(bbox/area),'void_over_bbox':float((bbox-union)/bbox)}

def main():
    records={k:read(p/'terminal/results.json')['records'] for k,p in FOLDERS.items()}
    summaries={k:read(p/'terminal/results.json')['summary'] for k,p in FOLDERS.items()}
    protocol=read(OUT/'frozen_protocol.json');assert protocol['checkpoint_sha256']==sha(OUT/'composition.pt')
    from experiments.E80_modular_transfer.freeze import frozen_checkpoint
    state,identity=frozen_checkpoint(OUT)
    import torch
    from experiments.E80_modular_transfer.model import GeometryModel,decode
    from experiments.E80_modular_transfer.inference import graph
    from experiments.E80_modular_transfer.data import load_rows
    from experiments.E80_modular_transfer.train import seed_all
    seed_all(0,2);model=GeometryModel(**state['model_config']);model.load_state_dict(state['model']);model.eval()
    rows=load_rows('dev_monitor500');excesses=[];invariants=[]
    with torch.no_grad():
        for row,saved in zip(rows,records['E80']):
            assert list(row['member'])==saved['member']
            g=graph(row['case'].solver_inputs)
            base=(decode(model.placer(g),g)[0]*g['scale'][0]).numpy().astype(float)
            constraints=np.asarray(row['case'].solver_inputs[5]);specs=np.asarray(row['case'].solver_inputs[6])
            locked=constraints[:,:2].any(1);base[locked,2:]=specs[locked,2:]
            base[constraints[:,1]!=0]=specs[constraints[:,1]!=0]
            p=np.asarray(saved['draft_positions'])
            scale=np.sqrt(base[:,2:].prod(-1).sum())
            excess=max(0.,float((base[:,:2].min(0)-p[:,:2].min(0)).max()),
                float(((p[:,:2]+p[:,2:]).max(0)-(base[:,:2]+base[:,2:]).max(0)).max()))/scale
            excesses.append(float(excess))
            g80=geometry(p);g42=geometry(base)
            invariants.append({'index':saved['index'],'member':saved['member'],'n':saved['n'],
                'E80':g80,'E42_public_base':g42})
    if max(excesses)>3e-6:raise ValueError('Transferred drafts violate supplied E42 frame')
    raw=read(OUT/'full_dev.json')
    output={'status':'audited_frozen_transfer','training_updates':0,'parameters':1537022,
        'correction_parameters':65483,'matrix':summaries,'test_read':False,'baseline_promoted':False,
        'source_payload_sha256':{k:sha(p/'terminal/results.json') for k,p in FOLDERS.items()},
        'paired_Q':{'E80_vs_E42':paired(records['E80'],records['E42']),
                    'E79_vs_E74':paired(records['E79'],records['E74']),
                    'E80_vs_E79':paired(records['E80'],records['E79'])},
        'raw_dev2000':raw['losses'],
        'paired_overlap500_vs_E42':paired(records['E80'],records['E42'],lambda r:r['draft_diagnostics']['overlap_area_fraction']),
        'frame500':{'violations':sum(v>3e-6 for v in excesses),'max_normalized_excess':max(excesses),'frame_source':'supplied E42 public prediction'},
        'geometry500_vs_actual_supplied_E42':{k:{'current_mean':float(np.mean([r['E80'][k] for r in invariants])),
             'baseline_mean':float(np.mean([r['E42_public_base'][k] for r in invariants])),
             'comparison_tolerance':3e-6,
             'better_equal_worse':[sum(r['E80'][k]<r['E42_public_base'][k]-3e-6 for r in invariants),
                                  sum(abs(r['E80'][k]-r['E42_public_base'][k])<=3e-6 for r in invariants),
                                  sum(r['E80'][k]>r['E42_public_base'][k]+3e-6 for r in invariants)]}
             for k in ('bbox_over_A','void_over_bbox')},
        'limits':'Adaptive dev; related placer family only. F includes containment and MIB coordination, so whole-package effect is not an isolated neural-weight effect. No training or Q-based checkpoint selection.',
        'worst10':[{k:r[k] for k in ('index','member','n','metrics','rescue_used')} for r in sorted(records['E80'],key=lambda r:r['metrics']['cost'],reverse=True)[:10]]}
    save(HERE/'runs/inspection.json',output)
    table='\n'.join(f"| {k} | {v['weighted_Q']:.6f} | {v['max_Q']:.6f} | {v['hard_legal']}/500 |" for k,v in summaries.items())
    d=output['paired_Q']['E80_vs_E42'];overlap=raw['losses']['draft_diagnostics']['overlap_area_fraction']['mean']
    text=f'''# E80冻结模块迁移结果

全部修正参数不变：E79的65,483参数完整修正包，从E74布局模型迁移到E42best51。E42仅1,471,539参数，总计1,537,022；训练更新0次。框从E42预测生成，500例外伸检查通过。源码、来源哈希与五例原E79重建已核对。

| 组合 | 加权Q（低更好） | 最大Q | 后处理硬合法 |
|---|---:|---:|---:|
{table}

相对E42，weighted差值{d['weighted_delta']:+.6f}，worker重抽样区间{d['worker_resampling95']}；胜/平/负{d['better_equal_worse']}。完整dev2000草稿交面积{100*overlap:.3f}%。原始草稿硬合法{raw['losses']['raw_hard_legal']}/2000。完整外框、精确空白和尾部见[机器审计](inspection.json)。

前三个已有组合只读复用，没有重新终末评价；唯一新组合只评价一次flash500。修正包也含MIB协调与外框投影，不能把收益全归因某个头。本组只验证同族布局模型的零训练迁移，不能直接证明任意主网络都可替换或大小分工已经最优。测试封存，E42/E50未升级。

[固定100例reference／迁移draft／E42图](20261029_e42_frozen_e79_transfer/draft_comparison100/index.html)。待人工查看3至5图后补最终解释。
'''
    (HERE/'runs/inspection.md').write_text(text,encoding='utf-8')
    print(json.dumps({'weighted_Q':summaries['E80']['weighted_Q'],'overlap2000':overlap,'paired_vs_E42':d,'frame':output['frame500']},ensure_ascii=False))

if __name__=='__main__':main()
