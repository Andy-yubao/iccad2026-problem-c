"""Paired fixed-dev500 stage objectives for single-call and half/full-repeat inference."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import json
import time
from pathlib import Path
import numpy as np
import torch
from .io import HERE,ROOT,sha,save
from .model import GeometryModel
from .public import inputs_only
from .data import load_rows,records
from .resident import ResidentRows
from .loss import per_case
from .loss_config import ARMS,REFERENCE,OVERLAP,BLOCK

KEYS=('total','reference_weighted','overlap_weighted','wire_compact_weighted','grouping_weighted',
      'boundary_weighted','public_shape_weighted','public_preplaced_position_weighted','group_block_weighted')
EXITS=(('L0','e161_l0',(0,)),('S0.1_half','e161_s0',(0,1)),('S0.2_full','e161_s0',(0,2)),
       ('S1.1_half','e161_s1',(0,2,3)),('S1.2_full','e161_s1',(0,2,4)),
       ('S2.1_half','e161_s2',(0,2,4,5)),('S2.2_full','e161_s2',(0,2,4,6)))

def main():
    out=HERE/'loss_audit_dev500'
    if out.exists():raise FileExistsError('Preserve the registered loss audit; no overwrite')
    out.mkdir()
    started=time.perf_counter()
    frozen=json.loads((HERE/'frozen.json').read_text(encoding='utf-8'))
    checkpoint=ROOT/frozen['checkpoint']
    if sha(checkpoint)!=frozen['checkpoint_sha256']:raise ValueError('Frozen weight changed')
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    def beat(**kw):save(out/'state.json',dict(status='running',**kw))
    rows=load_rows('dev_monitor500',beat=beat)
    panel=records('dev_monitor500')
    if len(rows)!=500 or [list(r['member']) for r in rows]!=[r['member'] for r in panel]:
        raise ValueError('Original fixed dev500 member/order mismatch')
    pack=ResidentRows(rows,'cuda',max_nodes=120,max_pins=max(1,max(r['graph']['pin_coordinates'].shape[1] for r in rows)))
    weights=torch.load(checkpoint,map_location='cpu',weights_only=False)
    configuration=dict(weights['model_config'],architecture='e162_repeat_half')
    model=GeometryModel(**configuration).cuda().eval();model.load_state_dict(weights['model'],strict=True)
    baseline={arm:[] for arm in ARMS};repeat={name:[] for name,_,_ in EXITS}
    def collect(states,batch,phase):
        values=per_case(states,batch,phase)
        matrix=torch.stack([values[key] for key in KEYS],-1)
        if not bool(torch.isfinite(matrix).all()):raise RuntimeError('Nonfinite diagnostic loss')
        return matrix.cpu().double().numpy()
    with torch.inference_mode():
        for start in range(0,500,64):
            batch=pack.batch(range(start,min(start+64,500)));public=inputs_only(batch)
            single=model.forward_single(public);repeated=model(public)
            torch.testing.assert_close(single[0],repeated[0],rtol=0,atol=0)
            for index,arm in enumerate(ARMS):baseline[arm].append(collect(single[:index+1],batch,arm))
            for name,arm,indices in EXITS:
                repeat[name].append(collect(torch.stack([repeated[index] for index in indices]),batch,arm))
            beat(stage='stage_loss_comparison',completed=min(start+64,500),total=500)
            print(f'fixed dev500 stage Loss {min(start+64,500)}/500',flush=True)
    baseline={arm:np.concatenate(parts) for arm,parts in baseline.items()}
    repeat={name:np.concatenate(parts) for name,parts in repeat.items()}
    mean=lambda values:{key:float(values[:,index].mean()) for index,key in enumerate(KEYS)}
    baseline_mean={arm:mean(value) for arm,value in baseline.items()}
    repeat_mean={name:mean(value) for name,value in repeat.items()}
    comparison=[]
    for index,label in enumerate(('L0','S0','S1','S2')):
        arm=ARMS[index];old=baseline_mean[arm]['total']
        half=repeat_mean[label if not index else label+'.1_half']['total']
        last=label if not index else label+'.2_full';new=repeat_mean[last]['total']
        comparison.append(dict(stage=label,objective=arm,baseline=old,first_half=half,second_full=new,
            change=new-old,change_percent=100*(new-old)/old,
            improved_cases=int((repeat[last][:,0]<baseline[arm][:,0]).sum()),
            worsened_cases=int((repeat[last][:,0]>baseline[arm][:,0]).sum())))
    previous=json.loads((ROOT/'experiments/E161_kaggle_transfer/local_return_audit.json').read_text(encoding='utf-8'))
    consistency={row['arm']:dict(stored=row['own_selection_loss'],recomputed=baseline_mean[row['arm']]['total'],
        absolute_difference=abs(row['own_selection_loss']-baseline_mean[row['arm']]['total'])) for row in previous['stages']}
    report=dict(status='complete',count=500,panel='original fixed dev_monitor500',checkpoint_sha256=frozen['checkpoint_sha256'],
        baseline_mean=baseline_mean,repeat_mean=repeat_mean,comparison=comparison,baseline_consistency=consistency,
        reference_coefficients=list(REFERENCE),overlap_multipliers=list(OVERLAP),block_multiplier=BLOCK,
        interpretation='Compare within each stage objective. S1/S2 repeated-path results include upstream repeated-stage effects. No Q/postprocessing/training performed.',
        training_updates=0,test_read=False,elapsed_seconds=time.perf_counter()-started)
    save(out/'results.json',report)
    save(out/'per_case_totals.json',dict(members=[list(r['member']) for r in rows],
        baseline={key:values[:,0].tolist() for key,values in baseline.items()},
        repeat={key:values[:,0].tolist() for key,values in repeat.items()}))
    lines=['# 固定dev500逐阶段Loss对照','',
        '同一E161 S2 best21权重；不训练。原单次调用与L0→S0半/全→S1半/全→S2半/全路径逐例对照。',
        '50%仅作用于中心x/y和logratio的末端布局残差；各阶段第二遍从半步布局重新入口编码。不同阶段Loss系数不同，只做同行比较。','',
        '| 阶段 | 原单次 | 第一遍半步 | 第二遍全步后 | 相对原单次变化 | 改善/500 |',
        '|---|---:|---:|---:|---:|---:|']
    for row in comparison:
        lines.append(f"| {row['stage']} | {row['baseline']:.9f} | {row['first_half']:.9f} | {row['second_full']:.9f} | {row['change_percent']:+.2f}% | {row['improved_cases']} |")
    lines+=['','S1/S2新结果包含前面阶段重复带来的输入变化；不能把总变化全部归因于该阶段自身。Loss不是最终后处理Q。','',
        '## 加权分项（原单次 → 两遍后）','',
        '| 阶段 | reference | overlap | wire/compact | grouping | boundary | F/P shape | P position | MST block |',
        '|---|---|---|---|---|---|---|---|---|']
    for index,label in enumerate(('L0','S0','S1','S2')):
        old=baseline_mean[ARMS[index]];new=repeat_mean[label if not index else label+'.2_full']
        lines.append('| '+label+' | '+' | '.join(f'{old[key]:.6f} → {new[key]:.6f}' for key in KEYS[1:])+' |')
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(out/'state.json',dict(status='complete',completed=500,total=500,training_updates=0,test_read=False))
    print(json.dumps(dict(comparison=comparison,baseline_consistency=consistency)),flush=True)

if __name__=='__main__':main()
