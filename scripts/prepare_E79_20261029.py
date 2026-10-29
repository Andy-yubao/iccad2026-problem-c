"""Second phase candidate2: preserve E78 coherent shapes, learn only centers."""
import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
import torch
ROOT=Path(__file__).resolve().parents[1];old=ROOT/'experiments/E78_connectivity_posttrain';new=ROOT/'experiments/E79_center_refinement'
assert not new.exists();new.mkdir()
skip={'model.py','initialization.py','preflight.py','loss.py','loss_config.py'}
for p in old.rglob('*'):
    rel=p.relative_to(old)
    if not p.is_file() or 'runs' in rel.parts or '__pycache__' in rel.parts or p.suffix not in ('.py','.txt') or p.name in skip:continue
    out=new/rel;out.parent.mkdir(parents=True,exist_ok=True)
    if p.name in ('features.py','batches.py','base_model.py','source_model.py','corrector.py','pretrained_model.py','frame.py','mib_projection.py'):
        out.write_bytes(p.read_bytes())
    else:
        s=p.read_text(encoding='utf-8').replace('E78_connectivity_posttrain','E79_center_refinement').replace('E78','E79')
        s=s.replace('20261029_e77_connectivity_seed0','20261029_e78_center_seed0');out.write_text(s,encoding='utf-8')
(new/'warm_model.py').write_bytes((old/'model.py').read_bytes())
s=(old/'corrector.py').read_text(encoding='utf-8').replace('class Corrector(nn.Module):','class CenterCorrector(nn.Module):').replace('nn.Linear(width,3)','nn.Linear(width,2)')
(new/'center_corrector.py').write_text(s,encoding='utf-8')
p=json.loads((old/'provenance.json').read_text(encoding='utf-8'))
source='experiments/E78_connectivity_posttrain/runs/20261029_e77_connectivity_seed0/checkpoints/best.pt'
state=torch.load(ROOT/source,map_location='cpu',weights_only=False)
for name in ('best','latest'):
    f=old/f'runs/20261029_e77_connectivity_seed0/checkpoints/{name}.pt'
    p['protected_checkpoints_sha256'][f.relative_to(ROOT).as_posix()]=hashlib.sha256(f.read_bytes()).hexdigest()
p.update(experiment=new.name,source_experiment='experiments/E78_connectivity_posttrain',
    initialization_checkpoint=source,initialization_sha256='fad342b53a7360829c8315f6e05675cf2115fce20faa359b2c0c48d45f72dd07',
    source_best_epoch=21,source_best_step=21000,source_completed_epochs=29,source_completed_elapsed_seconds=1677.2296909000142,
    pretraining_elapsed_seconds=p['pretraining_elapsed_seconds']+state['elapsed_seconds'],
    copy_note='Independent E78 runtime frozen entirely in eval. Add zero-init center-only local corrector, retain source dimensions bit-exact and original E74 frame. Output6=(predicted3, detached public source3) for one-sided source-topology protection; no new labels/input channels.')
p['pretraining_history'].append({'experiment':'E78','selected_checkpoint_seconds':state['elapsed_seconds'],'epoch':21,'step':21000})
(new/'provenance.json').write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
plan={'experiment':new.name,'phase':2,'phase_candidate':2,'seed':0,'train':32000,'dev_monitor':500,'batch_size':32,
    'max_hours':2,'max_epochs':200,'test_read':False,'source':'Unique own-Loss E78 best21; entire source frozen/eval',
    'architecture':'New96-node/32-message center-only corrector21763 parameters, zero output initialization. Source5873659 parameters fixed; all widths/heights and MIB coherence kept. Centers clipped into same E74 public predicted frame.',
    'initial_function':'Physical output exactly E78 within FP32 tolerance; no shape projection change',
    'output_contract':'6 channels: predicted canonical state3, detached public source canonical state3. Decoder reads first3, Loss uses latter3 only as public source-regression anchor. No label enters model inputs.',
    'fixed_loss':'G+4I+5E+.1R, R=mean+.25max squared one-sided source native-axis/order margin shortfall, margin capped.1. A fromE78 removed: dimensions fixed so its selected-path derivative cannot train center motion.',
    'R_goal':'Zero at source. Penalize deterioration only; do not reward moving already healthy nonintersecting pairs farther apart or matching arbitrary reference topology.',
    'hypothesis':'E78 nearly eliminates MIB preparation jumps but worsens1964/2000 raw overlaps vsE77. Preserve coherent shape/repair behavior, add positional capacity and stronger real intersection signal with soft source-order guard.',
    'limits':'Source topology may be wrong; soft guard can hinder needed rearrangement. Local center motion may not suffice. Frozen shape removes adaptation. No hard legality guarantee; combined experiment not isolated attribution.',
    'optimizer':'Fresh new-head AdamW, clip1, decay.0003, original4 LR/early-stop; source optimizer/budgets untouched',
    'selection':'Same fixed own Loss train/dev500/full-dev; epoch0 eligible; one unique best terminal500, no Q selection',
    'data':'Same nested32k and v5 dev/monitor members; train500 remains old16k only; FP32'}
(new/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
path=new/'train.py';s=path.read_text(encoding='utf-8').replace('E79.e77_MIB_anchored_posttrain.v1','E79.frozen_e78_center_source_guard.v1')
s=s.replace("'common_original_G', 'anchored_corridor',", "'common_original_G', 'topology_regression', 'topology_flip_fraction', 'source_true_intersection',")
s=s.replace('E76 source frozen/eval; existing E77 last corrector dropout.05/decay.0003; equal-area public MIB tying and E74 frame','Entire E78 frozen/eval; new center-only head dropout.05/decay.0003; source dimensions exact and E74 frame')
s=s.replace('G+2I+5E+.25 anchored corridor mean-plus-max squared pressure','G+4I+5E+.1 one-sided source topology regression')
path.write_text(s,encoding='utf-8')
path=new/'report.py';s=path.read_text(encoding='utf-8')
needle="    save(HERE/'runs/review.json',study);"
s=s.replace(needle,"""    e78_path=ROOT/'experiments/E78_connectivity_posttrain/runs/20261029_e77_connectivity_seed0'
    e78=read(e78_path/'terminal/results.json')['records']
    assert [r['member'] for r in records]==[r['member'] for r in e78]
    q78=np.array([r['metrics']['cost'] for r in e78]);d78=q-q78
    study['E78_baseline_flash']=read(e78_path/'terminal/summary.json')
    study['paired_Q_vs_E78']={'better_equal_worse':[int((d78< -1e-8).sum()),int((abs(d78)<=1e-8).sum()),int((d78>1e-8).sum())],'mean_delta':float(d78.mean())}
    study['paired_geometry_vs_E78']=compare(current,read(ROOT/'experiments/E78_connectivity_posttrain/runs/geometry500.json')['current'])
    width_height_deltas=[float(np.max(np.abs(np.asarray(r['draft_positions'])[:,2:]-np.asarray(b['draft_positions'])[:,2:]))) for r,b in zip(records,e78)]
    study['saved_source_shape_deltas500']={'max_absolute':max(width_height_deltas),'note':'CPU inference roundoff across batch/library calls may remain; no learned shape channel'}
    save(HERE/'runs/review.json',study);""")
start=s.index('从E77 own-Loss best38沿用完整权重');end=s.index('\n\n完成',start)
s=s[:start]+'从E78 own-Loss best21冻结全部模型，只学习零初始化中心位置修正器，源尺寸/MIB协调保持，中心限制在原E74公开预测框。固定L=G+4I+5E+.1R，R只惩罚相对源分离轴/次序的退化，初始为零；不奖励健康模块无限分开。初始物理输出等于E78。新增位置容量、交面积权重与保护共同改变，不能单独归因。'+s[end:]
s=s.replace('# E79 MIB一致输出＋锚点通道后训练结果','# E79固定同形尺寸＋中心修正结果');path.write_text(s,encoding='utf-8')
path=new/'render_drafts.py';s=path.read_text(encoding='utf-8').replace('MIB-coherent E77 posttrain in public E74 frame','frozen coherent E78 plus center correction in public E74 frame');path.write_text(s,encoding='utf-8')
print('Prepared',new)
