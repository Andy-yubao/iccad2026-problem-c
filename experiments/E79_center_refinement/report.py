"""Saved-results comparison; never reruns inference or selects checkpoints."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import json,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from .io import HERE,ROOT,save
from .pipeline import OUTPUT

CONTROL=ROOT/'history/20261029_model_research_consolidation/experiments/E63_e42_center_relaxation/runs/20261027_control_seed0'
E42_DRAFTS=ROOT/'history/20261026_flash_pro_consolidation/experiments/E51_module_overlap/runs/20261025_reference_E51_E42_draft100/e42_drafts_local'

def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def geometry(positions):
    p=np.asarray(positions,dtype=float);lo=p[:,:2];hi=lo+p[:,2:];area=p[:,2:].prod(1).sum()
    bbox=(hi.max(0)-lo.min(0)).prod()
    inter=np.maximum(np.minimum(hi[:,None],hi[None,:])-np.maximum(lo[:,None],lo[None,:]),0).prod(-1)
    overlap=np.triu(inter,1).sum()/area
    xs=np.unique(np.r_[lo[:,0],hi[:,0]]);union=0.
    for left,right in zip(xs[:-1],xs[1:]):
        active=(lo[:,0]<right)&(hi[:,0]>left);ys=sorted(zip(lo[active,1],hi[active,1]));length=0.;end=-np.inf
        for bottom,top in ys:
            length+=max(0.,top-max(bottom,end));end=max(end,top)
        union+=(right-left)*length
    return {'bbox_over_A':float(bbox/area),'overlap_area_fraction':float(overlap),'void_over_bbox':float((bbox-union)/bbox),'duplicate_over_A':float((area-union)/area)}

def calculate(payload):
    index,member,p=payload
    return {'index':index,'member':member,**geometry(p)}

def compare(a,b):
    bm={tuple(r['member']):r for r in b};pairs=[(r,bm[tuple(r['member'])]) for r in a if tuple(r['member']) in bm]
    out={'count':len(pairs)}
    for key in ('bbox_over_A','overlap_area_fraction','void_over_bbox','duplicate_over_A'):
        x=np.array([x[key] for x,y in pairs]);y=np.array([y[key] for x,y in pairs]);d=x-y
        out[key]={'current_mean':float(x.mean()),'baseline_mean':float(y.mean()),'mean_delta':float(d.mean()),'better_equal_worse':[int((d< -1e-9).sum()),int((abs(d)<=1e-9).sum()),int((d>1e-9).sum())]}
    return out

def main():
    result=read(OUTPUT/'results.json');progress=read(OUTPUT/'progress.json');raw=read(OUTPUT/'full_dev.json')
    flash=read(OUTPUT/'terminal/summary.json');records=read(OUTPUT/'terminal/results.json')['records']
    original=read(CONTROL/'terminal/results.json')['records'];history=progress['history']
    best=next(r for r in history if r['epoch']==result['best_epoch']);initial=history[0]
    payload=[(r['index'],r['member'],r['draft_positions']) for r in records]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:current=list(pool.map(calculate,payload))
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:control=list(pool.map(calculate,[(r['index'],r['member'],r['draft_positions']) for r in original]))
    e42=[]
    for row in records:
        path=E42_DRAFTS/f"case_{row['index']:04d}.json"
        if path.exists():
            old=read(path)
            if old['member']!=row['member']:raise ValueError('E42 gallery member mismatch')
            e42.append({'index':row['index'],'member':row['member'],**geometry(old['positions'])})
    for a,b in zip(records,original):
        if a['member']!=b['member']:raise ValueError('Control panel mismatch')
    q=np.array([r['metrics']['cost'] for r in records]);cq=np.array([r['metrics']['cost'] for r in original]);delta=q-cq
    pairedQ={'better_equal_worse':[int((delta< -1e-8).sum()),int((abs(delta)<=1e-8).sum()),int((delta>1e-8).sum())],'mean_delta':float(delta.mean())}
    train=best['train_monitor']['total'];dev=best['dev_monitor']['total']
    study={'status':'one_candidate_complete','training':result,'best_monitor':best,'initial_monitor':initial,
        'full_dev':raw['losses'],'flash':flash,'control_flash':read(CONTROL/'terminal/summary.json'),
        'paired_geometry_vs_control500':compare(current,control),'paired_geometry_vs_E42_gallery':compare(current,e42),
        'paired_Q_vs_control500':pairedQ,'train_dev_relative_gap':float(dev/train-1),
        'train_improvement_fraction':float(1-train/initial['train_monitor']['total']),
        'dev_improvement_fraction':float(1-dev/initial['dev_monitor']['total']),
        'worst10':sorted([{'index':r['index'],'member':r['member'],'n':r['n'],'Q':r['metrics']['cost'],'raw_diagnostics':r['draft_diagnostics'],'metrics':r['metrics']} for r in records],key=lambda r:r['Q'],reverse=True)[:10],
        'test_read':False,'baseline_promoted':False,
        'limits':'Frozen E76 plus hard E74 public-predicted frame and new local corrector. Combined mechanism; not isolated causal attribution. Adaptive single-seed development data, own fixed Loss selects best.'}
    previous=ROOT/'experiments/E74_collision_32k_posttrain/runs/20261029_e42_topology_seed0'
    e74=read(previous/'terminal/results.json')['records']
    assert [r['member'] for r in records]==[r['member'] for r in e74]
    eq=np.array([r['metrics']['cost'] for r in e74]);ed=q-eq
    study['E74_baseline_flash']=read(previous/'terminal/summary.json')
    study['paired_Q_vs_E74']={'better_equal_worse':[int((ed< -1e-8).sum()),int((abs(ed)<=1e-8).sum()),int((ed>1e-8).sum())],'mean_delta':float(ed.mean())}
    study['paired_geometry_vs_E74']=compare(current,read(ROOT/'experiments/E74_collision_32k_posttrain/runs/geometry500.json')['current'])
    e76_path=ROOT/'experiments/E76_bounded_refinement/runs/20261029_frozen_e74_corrector_seed0'
    e76=read(e76_path/'terminal/results.json')['records']
    assert [r['member'] for r in records]==[r['member'] for r in e76]
    q76=np.array([r['metrics']['cost'] for r in e76]);d76=q-q76
    study['paired_Q_vs_E76']={'better_equal_worse':[int((d76< -1e-8).sum()),int((abs(d76)<=1e-8).sum()),int((d76>1e-8).sum())],'mean_delta':float(d76.mean())}
    study['paired_geometry_vs_E76']=compare(current,read(ROOT/'experiments/E76_bounded_refinement/runs/geometry500.json')['current'])
    protrusions=[]
    for row,source in zip(records,e74):
        p=np.asarray(row['draft_positions']);s=np.asarray(source['draft_positions'])
        scale=np.sqrt(s[:,2:].prod(1).sum())
        excess=max(0.,float((s[:,:2].min(0)-p[:,:2].min(0)).max()),float(((p[:,:2]+p[:,2:]).max(0)-(s[:,:2]+s[:,2:]).max(0)).max()))/scale
        protrusions.append(excess)
    study['saved_public_frame_containment500']={'count':len(protrusions),'max_protrusion_over_sqrt_public_area':max(protrusions),'fp32_tolerance':3e-6,'violations':int(sum(v>3e-6 for v in protrusions))}
    if study['saved_public_frame_containment500']['violations']:raise ValueError('Saved drafts exceed declared public frame')
    e77_path=ROOT/'experiments/E77_compact_refinement/runs/20261029_e76_compact_corrector_seed0'
    e77=read(e77_path/'terminal/results.json')['records']
    assert [r['member'] for r in records]==[r['member'] for r in e77]
    q77=np.array([r['metrics']['cost'] for r in e77]);d77=q-q77
    study['E77_baseline_flash']=read(e77_path/'terminal/summary.json')
    study['paired_Q_vs_E77']={'better_equal_worse':[int((d77< -1e-8).sum()),int((abs(d77)<=1e-8).sum()),int((d77>1e-8).sum())],'mean_delta':float(d77.mean())}
    study['paired_geometry_vs_E77']=compare(current,[{'index':r['index'],'member':r['member'],**geometry(r['draft_positions'])} for r in e77])
    e78_path=ROOT/'experiments/E78_connectivity_posttrain/runs/20261029_e77_connectivity_seed0'
    e78=read(e78_path/'terminal/results.json')['records']
    assert [r['member'] for r in records]==[r['member'] for r in e78]
    q78=np.array([r['metrics']['cost'] for r in e78]);d78=q-q78
    study['E78_baseline_flash']=read(e78_path/'terminal/summary.json')
    study['paired_Q_vs_E78']={'better_equal_worse':[int((d78< -1e-8).sum()),int((abs(d78)<=1e-8).sum()),int((d78>1e-8).sum())],'mean_delta':float(d78.mean())}
    study['paired_geometry_vs_E78']=compare(current,read(ROOT/'experiments/E78_connectivity_posttrain/runs/geometry500.json')['current'])
    width_height_deltas=[float(np.max(np.abs(np.asarray(r['draft_positions'])[:,2:]-np.asarray(b['draft_positions'])[:,2:]))) for r,b in zip(records,e78)]
    study['saved_source_shape_deltas500']={'max_absolute':max(width_height_deltas),'note':'CPU inference roundoff across batch/library calls may remain; no learned shape channel'}
    save(HERE/'runs/review.json',study);save(HERE/'runs/geometry500.json',{'current':current,'control':control,'E42':e42})
    text=f'''# E79固定同形尺寸＋中心修正结果

从E78 own-Loss best21冻结全部模型，只学习零初始化中心位置修正器，源尺寸/MIB协调保持，中心限制在原E74公开预测框。固定L=G+4I+5E+.1R，R只惩罚相对源分离轴/次序的退化，初始为零；不奖励健康模块无限分开。初始物理输出等于E78。新增位置容量、交面积权重与保护共同改变，不能单独归因。

完成{result['epochs']}轮，best{result['best_epoch']}；训练阶段{result['elapsed_seconds']/60:.2f}分钟。best固定train/dev500 Loss为{train:.6f}/{dev:.6f}，相对差距{100*(dev/train-1):.2f}%。初始相同Loss见review.json，可用于判断是否训练/开发共同改善；不能与不同定义的旧总Loss直接排名。

完整dev2000 G={raw['losses']['common_original_G']:.6f}；交面积均值{100*raw['losses']['draft_diagnostics']['overlap_area_fraction']['mean']:.3f}%。flash500硬合法{flash['hard_legal']}/500，weighted Q={flash['weighted_Q']:.6f}，P95={flash['p95_Q']:.6f}，最差5%均值={flash['worst5pct_mean_Q']:.6f}，最大={flash['max_Q']:.6f}。E42冻结flash参考weighted Q=1.025860，E74候选=1.022511。Q下降更好，耗时与正式比赛成绩分开。

同500例E74/E76及历史control、同100例E42的外框、精确空白、交面积与重复覆盖配对结果见[review.json](review.json)。同500例原始草稿均须包含于已存E74框内，允许FP32容差3e-6个公开总面积等效边长；flash修复不受该框限制。草稿仍须实际检查，不能以Loss下降替代视觉与硬约束判断，也不能仅靠Q下降升级基线。

[同100例reference／本轮draft／E42，无评分控件](20261029_e78_center_seed0/draft_comparison100/index.html)。本报告为自动汇总，深度失败分析和图像核验由后续唤醒完成；不自动升级基线，测试封存。
'''
    (HERE/'runs/review.md').write_text(text,encoding='utf-8');print('E79 saved-evidence report complete',flush=True)
if __name__=='__main__':main()
