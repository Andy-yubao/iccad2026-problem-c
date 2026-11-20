"""Capture ten size-stratified training cases from the paused latest model."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import json
import pathlib
import time
import zipfile
import numpy as np
import torch
from experiments.E156_two_s0_stage import train
from experiments.E156_two_s0_stage.io import ROOT,HERE as EXP,sha,save,sources
from experiments.E156_two_s0_stage.model import GeometryModel
from experiments.E156_two_s0_stage.data import records,load_rows
from experiments.E156_two_s0_stage.resident import ResidentRows
from experiments.E156_two_s0_stage.public import inputs_only
from experiments.E156_two_s0_stage.batches import decode
from experiments.E156_two_s0_stage.loss import layer_loss,per_case
from experiments.E156_two_s0_stage.monitor_audit import audit
from visualization.style import normalize
from visualization.layouts import plot_case

HERE=pathlib.Path(__file__).resolve().parent


@torch.inference_mode()
def main():
    started=time.perf_counter()
    assert not (HERE/'data.json').exists(),'Preserve completed visualization'
    train.seed_all(0)
    path=EXP/'runs/e156_s0_backend_seed0/checkpoints/latest.pt'
    checksum=sha(path);state=torch.load(path,map_location='cpu',weights_only=False)
    assert state['status']=='user_stop' and not state['finished']
    assert state['identity']['sources']==sources()
    registry=records('train');rng=np.random.default_rng(20261119);selected=[]
    for lower in range(21,121,10):
        candidates=[i for i,r in enumerate(registry) if lower<=r['n']<=lower+9 and state['progress']['seen'][i]]
        assert candidates
        selected.append(int(rng.choice(candidates)))
    rows=load_rows('train',selected)
    pack=ResidentRows(rows,'cuda',max_nodes=120,max_pins=143,max_ports=363)
    g=pack.batch(range(10));model=GeometryModel(**state['model_config']).cuda().eval()
    model.load_state_dict(state['model']);before={k:v.detach().clone() for k,v in model.state_dict().items()}
    output=model(inputs_only(g));again=model(inputs_only(g))
    assert torch.equal(output,again),'Eval must not perturb incoming layouts'
    values=per_case(output,g,'e156_s0_backend');diagnostics=audit(output,g,values)
    losses={k:layer_loss(output[k],g) for k in (0,2,4)}
    positions={k:(decode(output[k],g)*g['scale'][:,None,None]).cpu().numpy() for k in (0,2,4)}
    cases=[];summaries=[]
    labels=['L0 · 冻结初始布局','L2 · 第一次 S0 完成','L4 · 第二次 S0 完成']
    columns=[['total','S0基础Loss'],['reference_weighted','Reference'],['grouping_weighted','原分组项'],
        ['group_bottleneck_weighted','分组瓶颈'],['boundary_weighted','贴边'],
        ['overlap_total','重叠诊断'],['wire_compact_weighted','线长/紧凑'],['supervision_weight','监督权重']]
    for j,(index,row) in enumerate(zip(selected,rows)):
        n=registry[index]['n'];layers=[positions[k][j,:n].astype(float).tolist() for k in (0,2,4)]
        lr=[]
        for k in (0,2,4):
            entry={key:float(v[j]) for key,v in losses[k].items()}
            entry.update(directly_supervised=k>0,supervision_weight={0:0.,2:.2,4:.8}[k],
                         objective_name='S0基础Loss',objective_total=entry['total'])
            lr.append(entry)
        total=float(values['total'][j]);guard=float(values['relative_overlap_weighted'][j])
        case=dict(label=f'{j+1:02d} · train {index:05d}',group='训练集分规模随机10例',member=list(row['member']),
            n=n,constraints=row['inputs'][5].int().tolist(),reference=row['supervision_xywh'].double().tolist(),
            reference_label='形状修正 Reference（监督标签）',hide_missing_final=True,
            layer_ids=[0,2,4],layer_labels=labels,layers=layers,layer_losses=lr,loss_columns=columns,
            loss_label='S0基础Loss',
            loss_caption='仅显示实际L0、L2、L4。基础Loss含0.75 reference及0.2分组瓶颈，不含绝对重叠惩罚。重叠列是原始几何诊断，未直接乘入基础Loss。',
            objective_note=f'本题训练目标 = 0.2×L2基础Loss + 0.8×L4基础Loss + L4相对重叠惩罚 {guard:.6f} = {total:.6f}。展示关闭扰动。Reference形状已修正，可能仍重叠，不能视为合法解。')
        c=normalize(case);cases.append(case)
        stages=(('reference','Shape-corrected reference'),('L0','Frozen L0'),('L2','S0 first call / L2'),('L4','S0 second call / L4'))
        variants=dict(reference=dict(positions=case['reference']),**{f'L{k}':dict(positions=p) for k,p in zip((0,2,4),layers)})
        plot_case(dict(title=f'E156 paused latest | train {index:05d} | N={n}',constraints=c['constraints'],
            axis_limits=c['bounds'],variants=variants),HERE/f'png/case_{j+1:02d}_N{n}.png',stages=stages)
        summaries.append(dict(case=j+1,train_index=index,n=n,
            **{f'L{k}_base_loss':float(losses[k]['total'][j]) for k in (0,2,4)},
            **{f'L{k}_overlap':float(losses[k]['overlap_total'][j]) for k in (0,2,4)},
            **{f'L{k}_bottleneck_weighted':float(losses[k]['group_bottleneck_weighted'][j]) for k in (0,2,4)},
            **{f'L{k}_blocked_edges':int(diagnostics[f'L{k}_blocked_tree_edges'][j]) for k in (0,2,4)},
            L4_side_reversals=int(diagnostics['L4_side_reversals'][j]),
            L4_beneficial_side_reversal=bool(diagnostics['L4_beneficial_side_reversal'][j]),total=total,relative_overlap_guard=guard))
        print(f'[PNG] {j+1}/10 train={index} N={n}',flush=True)
    for k,v in model.state_dict().items():assert torch.equal(v,before[k])
    assert sha(path)==checksum
    identity=dict(experiment='E156_two_s0_stage',checkpoint='latest.pt',checkpoint_sha256=checksum,
        completed_epochs=state['progress']['epoch'],step=state['progress']['step'],cursor=state['progress']['cursor'],
        parent_L0='E155 best51',selection_seed=20261119,train_indices=selected,training_updates=0,
        postprocessing_calls=0,Q_calls=0,test_read=False)
    save(HERE/'data.json',dict(title='E156 · 暂停时最新权重 · 训练集10例 · L0→两次S0',
        note='每个模块数区间随机1题，未按Loss筛选；所有样例均已参与训练。仅真实L0/L2/L4，无未训练后层、无后处理。Reference为形状修正监督标签，未必合法。',identity=identity,cases=cases))
    save(HERE/'summary.json',dict(identity=identity,cases=summaries))
    save(HERE/'selection.json',dict(seed=20261119,rule='One uniform random previously-seen train member per N band21-30,...111-120; no Loss/Q selection',indices=selected,members=[list(r['member']) for r in rows]))
    np.savez_compressed(HERE/'actual_states.npz',states=output.cpu().numpy(),scale=g['scale'].cpu().numpy(),train_indices=np.array(selected))
    with zipfile.ZipFile(HERE/'train10_layouts.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in sorted((HERE/'png').glob('*.png')):archive.write(p,'png/'+p.name)
    lines=['# E156 暂停最新权重：训练10例','',f'Latest step={identity["step"]}；固定E155 best51；每N区间随机1题。未训练、未后处理；只展示实际L0/L2/L4。','',
        '| 图 | N | L0基础Loss | L2基础Loss | L4基础Loss | L0重叠 | L4重叠 | 完整换侧对数 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        lines.append(f'| [图{s["case"]:02d}](png/case_{s["case"]:02d}_N{s["n"]}.png) | {s["n"]} | {s["L0_base_loss"]:.5f} | {s["L2_base_loss"]:.5f} | {s["L4_base_loss"]:.5f} | {s["L0_overlap"]:.4f} | {s["L4_overlap"]:.4f} | {s["L4_side_reversals"]} |')
    lines+=['','基础Loss使用同一S0公式比较，L0未被该公式训练。10例仅供观察，不代表完整训练集或验证集统计。']
    (HERE/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(HERE/'verification.json',dict(status='complete',count=10,latest_checkpoint_sha256=checksum,
        training_still_paused=(EXP/'PAUSE_REQUEST').exists(),weights_unchanged=True,eval_deterministic=True,
        actual_layer_ids=[0,2,4],PNG_count=10,no_postprocessing=True,no_test=True,elapsed_seconds=time.perf_counter()-started))


if __name__=='__main__':main()
