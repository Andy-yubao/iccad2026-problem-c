"""Capture actual L0-L12 for the existing ten selected dev cases, without updates."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from experiments.E157_sequential_s1_s2.data import load_rows,collate
from experiments.E157_sequential_s1_s2.model import GeometryModel
from experiments.E157_sequential_s1_s2.evaluation.collect import public_batch
from experiments.E157_sequential_s1_s2.batches import decode
from experiments.E157_sequential_s1_s2.loss import layer_loss,drift_terms
from visualization.style import normalize
from visualization.layouts import render
from shapely.geometry import box
from shapely.ops import unary_union

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(name,data):(HERE/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def grouping(p,flags):
    polygons=[box(x,y,x+w,y+h) for x,y,w,h in p];bad=[]
    for group in sorted(set(int(v) for v in flags[:,3] if v>0)):
        ids=np.flatnonzero(flags[:,3]==group).tolist()
        union=unary_union([polygons[i] for i in ids])
        components=len(union.geoms) if union.geom_type=='MultiPolygon' else 1
        if components>1:bad.append(dict(group=group,components=components,modules=ids))
    return sum(v['components']-1 for v in bad),bad

torch.set_num_threads(1);torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
data=read(HERE/'data.json');selection=read(HERE/'selection.json')
indices=selection['selected_indices']
frozen=read(ROOT/'experiments/E157_sequential_s1_s2/frozen_s2.json')
checkpoint=ROOT/frozen['checkpoint'];digest=sha(checkpoint)
assert digest==frozen['checkpoint_sha256']==data['identity']['checkpoint_sha256']
state=torch.load(checkpoint,map_location='cpu',weights_only=False)
model=GeometryModel(**state['model_config']).cuda().eval();model.load_state_dict(state['model']);del state
rows=load_rows('dev_monitor500',indices=indices)
print('Loaded frozen checkpoint and selected dev10',flush=True)
labels=['L0 · 初稿']+[f'L{l} · '+('S0' if l<=4 else 'S1' if l<=8 else 'S2')+(' 输出' if l in (4,8,12) else '') for l in range(1,13)]
verification=[]
with torch.inference_mode():
    for index,row,case in zip(indices,rows,data['cases']):
        assert list(row['member'])==case['member']
        g=public_batch(row['inputs']);loss_g,_=collate([row],'cuda')
        states=model(g);assert states.shape[0]==13
        n=case['n'];layouts=[(decode(s,g)[0,:n]*g['scale'][0]).cpu().numpy().astype(float).tolist() for s in states]
        saved=np.asarray(case['draft']);terminal=np.asarray(layouts[-1]);difference=float(np.max(np.abs(saved-terminal)))
        assert difference<=1e-6,('L12 differs from saved original evaluation',index,difference)
        flags=np.asarray(case['constraints']);losses=[]
        for l in range(13):
            components=layer_loss(states[l],loss_g)
            values={k:float(v[0]) for k,v in components.items()}
            group_count,bad=grouping(layouts[l],flags)
            direct=l in (6,8,10,12);objective=values['total']
            drift=0.
            if l==12:
                center,ratio=drift_terms(states[8],states[12],loss_g)
                drift=float((.2*(center+.25*ratio))[0]);objective+=drift
            values.update(grouping_violations=group_count,grouping_bad_groups=bad,
                          directly_supervised=direct,supervision_weight=(.2 if l in (6,10) else .8) if direct else 0.,
                          objective_total=objective,drift_weighted=drift,
                          objective_name='P + S2移动容差' if l==12 else 'P')
            losses.append(values)
        case.update(layers=layouts,layer_ids=list(range(13)),layer_labels=labels,layer_losses=losses,
                    draft=layouts[-1],draft_label='L12 · S2 输出',loss_label='共同公开 Loss P',
                    loss_columns=[['total','公开Loss P'],['objective_total','阶段监督Loss'],['overlap_weighted','重叠'],
                        ['grouping_weighted','分组'],['boundary_weighted','贴边'],['mib_weighted','MIB Loss'],
                        ['drift_weighted','移动容差'],['grouping_violations','分组断连数'],['supervision_weight','监督权重']],
                    loss_caption='各层共同 P 不含 reference、瓶颈项或移动项。E157 的 S1 阶段监督 L6/L8，S2 阶段监督 L10/L12，权重均 .2/.8；L12 阶段目标另含移动容差。分组断连为当前原始草稿几何诊断，重叠可能掩盖断连。L0–L4 是冻结前缀。',
                    objective_note='L0–L12 均来自 E157 best34 的实际逐层输出；L12 已与原评价草稿核对。参考和后处理坐标保持原件。')
        normalize(case)
        render(case,HERE/f'png/dev{index:03d}.png',layer=12,title=case['label']+' · 真实 L0–L12 已采集')
        for l in (0,4,8):render(case,HERE/f'png/dev{index:03d}_L{l}.png',layer=l,title=case['label']+f' · L{l}')
        verification.append(dict(index=index,member=case['member'],layer_ids=list(range(13)),
            terminal_max_abs_difference=difference,terminal_bitwise_equal=bool(np.array_equal(saved,terminal)),
            reference_unchanged=True,postprocessed_unchanged=True))
        print('Captured L0-L12',index,'L12 difference',difference,flush=True)
if not (HERE/'data_final_only.json').exists():(HERE/'data_final_only.json').write_bytes((HERE/'data.json').read_bytes())
data['title']='E157 · grouping违规10例 · 真实 L0–L12 · best34'
data['note']='同一批最终 grouping 违规10例。拖动滑块查看实际 L0→L4(S0)→L8(S1)→L12(S2) 的全部13层；点击模块可跨栏同步定位。左右参考/后处理保持不变。'
data['identity'].update(layer_ids=list(range(13)),capture='singleton evaluation preprocessing; frozen best34; eval/inference_mode')
write('data.json',data)
write('layer_verification.json',dict(checkpoint_sha256=digest,checkpoint_unchanged=sha(checkpoint)==digest,
    cases=verification,model_forward_calls=10,optimizer_steps=0,postprocessing_calls=0,test_read=False))
print('Complete',flush=True)
