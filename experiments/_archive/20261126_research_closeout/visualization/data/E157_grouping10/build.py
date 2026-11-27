"""Select ten saved final grouping failures; no model inference or repair."""
import json
import random
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from experiments.E157_sequential_s1_s2.data import load_rows
from visualization.layouts import render
from visualization.style import normalize
import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union

HERE=Path(__file__).resolve().parent
EVAL=ROOT/'experiments/E157_sequential_s1_s2/runs/e157_s2_seed0/evaluation_mib_dev500'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(name,value):(HERE/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def grouping(p,flags):
    polygons=[box(x,y,x+w,y+h) for x,y,w,h in p];bad=[]
    for group in sorted(set(int(v) for v in flags[:,3] if v>0)):
        ids=np.flatnonzero(flags[:,3]==group).tolist()
        union=unary_union([polygons[i] for i in ids])
        components=len(union.geoms) if union.geom_type=='MultiPolygon' else 1
        if components>1:bad.append(dict(group=group,components=components,modules=ids))
    return sum(v['components']-1 for v in bad),bad

result=read(EVAL/'results.json')
records=[read(EVAL/f'cases/{i:04d}.json') for i in range(500)]
eligible=[r['index'] for r in records if r['metrics']['grouping_violations']>0]
assert len(eligible)==39
indices=sorted(random.Random(20261120).sample(eligible,10))
rows=load_rows('dev_monitor500',indices=indices)
print('Selected',indices,flush=True)
cases=[]
for index,row in zip(indices,rows):
    rec=records[index];m=rec['metrics'];assert list(row['member'])==rec['member']
    flags=np.asarray(row['inputs'][5],dtype=int)
    reference=row['reference_xywh'].tolist()
    count,bad=grouping(rec['positions'],flags)
    assert count==m['grouping_violations']
    ref_count,_=grouping(reference,flags)
    label=f'dev{index:03d} · grouping违规 {count} · '+', '.join('G'+str(v['group']) for v in bad)
    case=dict(label=label,member=rec['member'],n=rec['n'],constraints=flags.tolist(),
              reference=reference,reference_label='原始 Reference（评分基准）',
              draft=rec['draft_positions'],draft_label='E157 · L12 已保存草稿',final=rec['positions'],
              grouping_summary=dict(reference=ref_count,final=count,final_bad_groups=bad),
              metrics=dict(Q=m['cost'],hard_legal=m['is_feasible'],complete_ms=rec['complete_seconds']*1000),
              objective_note='此处仅展示原评价保存的 L12 草稿和后处理结果；未保存中间层，不补造滑块。')
    normalize(case)
    render(case,HERE/f'png/dev{index:03d}.png',title=label)
    cases.append(case)
    print('Rendered',index,flush=True)
data=dict(title='E157 · 最终 grouping 违规 10 例 · best34',
          note='从最终仍违反 grouping 的 39 题中固定随机抽取 10 题（seed20261120）。同色为同组；点击模块可跨栏定位。参考为原始评分基准，草稿和后处理均读取已存坐标。未训练、未推理、未重跑后处理或评分。',
          identity=dict(experiment='E157_sequential_s1_s2',best_epoch=34,
              checkpoint_sha256=result['checkpoint_sha256'],source=str(EVAL.relative_to(ROOT)),
              indices=indices,eligible_count=len(eligible),test_read=False),cases=cases)
write('data.json',data)
write('selection.json',dict(seed=20261120,eligible_indices=eligible,selected_indices=indices,
    grouping_counts=[c['grouping_summary']['final'] for c in cases],
    grouping_groups=[c['grouping_summary']['final_bad_groups'] for c in cases],
    saved_coordinates_only=True,model_forward_calls=0,postprocessing_calls=0,test_read=False))
print('Complete',flush=True)
