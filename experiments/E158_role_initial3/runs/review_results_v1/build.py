"""Summarize saved training and capture frozen best L0 for twenty dev examples."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import csv,json,hashlib,sys,random
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT))
from experiments.E158_role_initial3.model import GeometryModel
from experiments.E158_role_initial3.data import load_rows,collate
from experiments.E158_role_initial3.public import inputs_only
from experiments.E158_role_initial3.loss import per_case
from experiments.E158_role_initial3.batches import decode
from visualization.style import normalize
from visualization.layouts import render
from shapely.geometry import box
from shapely.ops import unary_union

EXP=HERE.parents[1]
RUN=EXP/'runs/e158_initial_seed0'
GALLERY=EXP/'runs/visualization_dev20_best58_v1'
GALLERY.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

COMPONENTS=[('reference_weighted','Reference'),('overlap_weighted','重叠 ×0.25'),
 ('wire_compact_weighted','线长与紧凑'),('grouping_weighted','Grouping'),('boundary_weighted','Boundary'),
 ('public_shape_weighted','F/P尺寸'),('public_preplaced_position_weighted','P位置'),('mib_weighted','MIB')]

def details(m):
    a=.25*.08975162861857286;b=.5*.3336832108335229
    return dict(reference_center=m['position'],reference_shape=.25*m['shape'],
        overlap_penetration=a*m['penetration'],overlap_mean_load=a*m['module_load_mean'],
        overlap_max_excess=a*m['module_max_excess10'],wirelength=b*m['wirelength'],
        compactness=b*m['compactness'],grouping=m['grouping_weighted'],boundary=m['boundary_weighted'],
        public_shape=m['public_shape_weighted'],preplaced_position=m['public_preplaced_position_weighted'],mib=m['mib_weighted'])

def grouping(positions,flags):
    polygons=[box(x,y,x+w,y+h) for x,y,w,h in positions];bad=[]
    for group in sorted(set(int(v) for v in flags[:,3] if v>0)):
        ids=np.flatnonzero(flags[:,3]==group).tolist();union=unary_union([polygons[i] for i in ids])
        count=len(union.geoms) if union.geom_type=='MultiPolygon' else 1
        if count>1:bad.append(dict(group=group,components=count,modules=ids))
    return sum(v['components']-1 for v in bad)

def main():
    result=read(RUN/'results.json');progress=read(RUN/'progress.json');frozen=read(EXP/'frozen_initial.json')
    assert result['status']=='early_stop' and result['best_epoch']==frozen['best_epoch']==58
    history=progress['history'];best=next(h for h in history if h['epoch']==58)
    old_run=ROOT/'experiments/E155_fresh_mib_l0/runs/e155_initial_seed0'
    old=read(old_run/'progress.json');old_best=next(h for h in old['history'] if h['epoch']==old['best_epoch'])
    panels=read(RUN/'panels.json');old_panels=read(old_run/'panels.json')
    for name in ['train_monitor_members','dev_monitor_members']:assert panels[name]==old_panels[name]
    comparison={}
    for panel in ['train_monitor','dev_monitor']:
        m=old_best[panel];n=best[panel]
        comparison[panel]=dict(E155_epoch=old['best_epoch'],E158_epoch=58,
            E155_original_no_overlap=m['total'],E158_same_no_overlap=n['original_L0_no_overlap_total'],
            E155_reconstructed_quarter_overlap=m['total']+.25*.08975162861857286*m['overlap_total'],E158_quarter_overlap=n['total'],
            E155_overlap_total=m['overlap_total'],E158_overlap_total=n['overlap_total'])
    summary=dict(training=result,frozen=frozen,best_train=best['train_monitor'],best_dev=best['dev_monitor'],
        weighted_details_train=details(best['train_monitor']),weighted_details_dev=details(best['dev_monitor']),
        comparison_E155=comparison,mathematical_min_epoch=min(history,key=lambda h:h['dev_monitor']['total'])['epoch'],
        last_dev=history[-1]['dev_monitor']['total'],selection_tolerance=1e-5,
        test_read=False,backend_calls=0,postprocessing_calls=0)
    # Per-case arrays stay in original progress; use concise mean summary here.
    for name in ['best_train','best_dev']:summary[name]={k:v for k,v in summary[name].items() if k!='per_case_total'}
    write(HERE/'summary.json',summary)
    with (HERE/'epoch_loss.csv').open('w',encoding='utf-8-sig',newline='') as f:
        columns=['epoch','panel','total']+[k for k,_ in COMPONENTS]+['position','shape','overlap_total','original_L0_no_overlap_total']
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for h in history:
            for panel in ['train_monitor','dev_monitor']:
                writer.writerow(dict(epoch=h['epoch'],panel=panel,**{k:h[panel][k] for k in columns[2:]}))
    with (HERE/'best_loss.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['component','train500','dev500'])
        for k,label in COMPONENTS:writer.writerow([label,best['train_monitor'][k],best['dev_monitor'][k]])
        writer.writerow(['total',best['train_monitor']['total'],best['dev_monitor']['total']])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10})
    fig,axs=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for ax,zoom in [(axs[0,0],False),(axs[0,1],True)]:
        subset=[h for h in history if h['epoch']>= (10 if zoom else 1)]
        for panel,label in [('train_monitor','固定 train500'),('dev_monitor','固定 dev500')]:
            ax.plot([h['epoch'] for h in subset],[h[panel]['total'] for h in subset],label=label)
        ax.axvline(58,color='gray',ls='--',label='已冻结 best58');ax.legend();ax.grid(alpha=.2)
        ax.set(title='总 Loss'+('（后期放大）' if zoom else '（省略随机初始化 epoch0）'),xlabel='epoch',ylabel='Loss')
    for k,label in COMPONENTS:
        axs[1,0].plot([h['epoch'] for h in history[1:]],[h['dev_monitor'][k] for h in history[1:]],label=label)
    axs[1,0].set(title='dev500 各项加权贡献',xlabel='epoch',ylabel='Loss贡献',yscale='log')
    axs[1,0].legend(ncol=2,fontsize=8);axs[1,0].grid(alpha=.2)
    labels=[label for _,label in COMPONENTS];x=np.arange(len(labels));w=.38
    axs[1,1].bar(x-w/2,[best['train_monitor'][k] for k,_ in COMPONENTS],w,label='train500')
    axs[1,1].bar(x+w/2,[best['dev_monitor'][k] for k,_ in COMPONENTS],w,label='dev500')
    axs[1,1].set(title='best58 的 Loss 构成',xticks=x,xticklabels=labels,ylabel='Loss贡献')
    axs[1,1].tick_params(axis='x',rotation=40);axs[1,1].legend()
    fig.savefig(HERE/'training_loss.png',dpi=160);plt.close(fig)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    checkpoint=ROOT/frozen['checkpoint'];assert sha(checkpoint)==frozen['checkpoint_sha256']
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=GeometryModel(**saved['model_config']).cuda().eval();model.load_state_dict(saved['model']);del saved
    indices=sorted(random.Random(20261120).sample(range(500),20))
    rows=load_rows('dev_monitor500',indices=indices);cases=[];differences=[];checks=[]
    print('Loaded best58 and selected dev20',indices,flush=True)
    with torch.inference_mode():
        for index,row in zip(indices,rows):
            batch,_=collate([row],'cuda');states=model(inputs_only(batch));assert states.shape[0]==1
            loss=per_case(states,batch,'e158_initial');values={k:float(v[0]) for k,v in loss.items()}
            n=int(row['inputs'][0]);positions=(decode(states[0],batch)[0,:n]*batch['scale'][0]).cpu().numpy().astype(float).tolist()
            reference=row['supervision_xywh'].tolist();flags=np.asarray(row['inputs'][5],dtype=int)
            difference=abs(values['total']-best['dev_monitor']['per_case_total'][index]);differences.append(difference)
            assert difference<2e-4,('fixed monitor parity',index,difference)
            assert list(row['member'])==panels['dev_monitor_members'][index]
            values.update(objective_total=values['total'],directly_supervised=True,supervision_weight=1.,
                objective_name='R + 0.25重叠 + 原任务项',grouping_violations=grouping(positions,flags))
            label=f'dev{index:03d} · Loss {values["total"]:.6f}'
            case=dict(label=label,member=list(row['member']),n=n,constraints=flags.tolist(),
                reference=reference,reference_label='监督参考（MIB形状修正）',
                layers=[positions],layer_ids=[0],layer_labels=['L0 · 三层 attention 后唯一输出'],layer_losses=[values],
                hide_missing_final=True,loss_label='E158 L0 Loss',
                loss_columns=[['total','总Loss'],['reference_weighted','Reference'],['overlap_weighted','重叠×0.25'],
                    ['wire_compact_weighted','线长/紧凑'],['grouping_weighted','Grouping'],['boundary_weighted','Boundary'],
                    ['public_shape_weighted','F/P尺寸'],['public_preplaced_position_weighted','P位置'],['mib_weighted','MIB Loss']],
                loss_caption='均为实际训练系数后的单题贡献；Reference=中心误差+0.25×形状误差，重叠使用原完整组的0.25倍。',
                objective_note='三个 attention/FFN 块在隐藏空间协调，最后仅产生一个真实 L0；不存在三份中间布局。监督参考经过MIB形状修正，不保证无重叠。未执行后段或后处理。')
            normalize(case);render(case,GALLERY/f'png/dev{index:03d}.png',layer=0,title=label)
            cases.append(case);checks.append(dict(index=index,member=case['member'],loss=values['total'],monitor_loss_difference=difference))
            print('Rendered',index,flush=True)
    gallery=dict(title='E158 · best58 · 验证集随机20例 · L0',note='固定随机抽取20个dev500成员（seed20261120），对比形状修正监督参考与真实L0。可勾选草稿重叠并点击模块定位；单题Loss在图下显示。',identity=dict(checkpoint_sha256=frozen['checkpoint_sha256'],best_epoch=58,selection='uniform random dev500 sample; seed20261120',indices=indices),cases=cases)
    write(GALLERY/'data.json',gallery)
    write(GALLERY/'selection.json',dict(seed=20261120,role='dev_monitor500',indices=indices,checks=checks,
        max_monitor_loss_difference=max(differences),checkpoint_unchanged=sha(checkpoint)==frozen['checkpoint_sha256'],
        model_forward_calls=20,optimizer_steps=0,postprocessing_calls=0,test_read=False))
    lines=['# E158 训练结果','',f'正常平台早停：61轮，34343次更新，耗时{result["elapsed_seconds"]/60:.2f}分钟。冻结best58，dev Loss0.3809590292。','',
        '| 加权Loss贡献 | train500 | dev500 |','|---|---:|---:|']
    for k,label in COMPONENTS:lines.append(f'| {label} | {best["train_monitor"][k]:.9f} | {best["dev_monitor"][k]:.9f} |')
    lines.append(f'| 总Loss | {best["train_monitor"]["total"]:.9f} | {best["dev_monitor"]["total"]:.9f} |')
    lines+=['','Reference、重叠与线长/紧凑的详细分项：','', '| 细分贡献 | train500 | dev500 |','|---|---:|---:|']
    td,dd=details(best['train_monitor']),details(best['dev_monitor'])
    for k in td:lines.append(f'| {k} | {td[k]:.9f} | {dd[k]:.9f} |')
    lines+=['', '所有数值取同一冻结best58。第61轮dev为0.3809535674，比best58低约0.00000546，小于登记best更新容差1e-5，因此未替换权重。正常dev早停不等于训练集已充分拟合。',
        '', '与E155同成员监测日志的共同口径比较，固定使用各自原own-best，不重新选模：', '', '| 口径 dev500 | E155 best51 | E158 best58 |','|---|---:|---:|']
    c=comparison['dev_monitor']
    lines += [f'| 原无重叠L0目标 | {c["E155_original_no_overlap"]:.9f} | {c["E158_same_no_overlap"]:.9f} |',
        f'| 加入0.25原重叠的共同诊断目标 | {c["E155_reconstructed_quarter_overlap"]:.9f} | {c["E158_quarter_overlap"]:.9f} |',
        f'| 原始重叠原语O | {c["E155_overlap_total"]:.9f} | {c["E158_overlap_total"]:.9f} |','',
        '共同目标由已存Loss分项重组，无历史权重重推理。E158同时改变深度和重叠监督，单seed不能单独证明深度是收益来源。',
        '', '20例来自dev500均匀无放回随机抽样，seed20261120；每例均为best58实际L0，不把隐藏块伪装成中间布局。监督参考为MIB形状修正标签。未训练、未运行后段、后处理、Q评价或test。',
        '', '训练曲线：training_loss.png；完整逐轮分项：epoch_loss.csv；结果摘要：summary.json；展示：../visualization_dev20_best58_v1/data.json。']
    (HERE/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Review and dev20 gallery complete; max monitor difference',max(differences),flush=True)

if __name__=='__main__':main()
