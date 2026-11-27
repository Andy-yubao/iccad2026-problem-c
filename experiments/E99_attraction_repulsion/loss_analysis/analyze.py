"""Paired offline Loss/Q statistics on existing100; never calls a solver."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import hashlib
import json
from pathlib import Path
import subprocess
import time
import zipfile
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from threadpoolctl import threadpool_limits
from datasets.paths import ROOT
from datasets.reference_layouts import ReferenceLayouts
from scoring.evaluation import evaluate_case
from .batches import sample,collate
from .c3_loss import per_case,decode

HERE=Path(__file__).resolve().parent
EXPERIMENT=HERE.parent
RUN=EXPERIMENT/'runs/20261102_area_mass_v2_dev100'
TEACHER=ROOT/'experiments/E97_gpu_hybrid_solver/runs/20261101_public_teacher'


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def counts(before,after,tolerance):
    d=np.asarray(after)-np.asarray(before);t=np.broadcast_to(tolerance,d.shape)
    return dict(decreased=int((d < -t).sum()),unchanged=int((np.abs(d)<=t).sum()),increased=int((d > t).sum()))


def directions(before,after,tolerance):
    d=np.asarray(after)-np.asarray(before)
    return np.where(d < -tolerance,0,np.where(d > tolerance,2,1))


def correlation(x,y,weights=None):
    x=np.asarray(x);y=np.asarray(y)
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('nonfinite correlation input')
    if np.ptp(x)==0 or np.ptp(y)==0:
        return dict(n=len(x),pearson=None,spearman=None,reason='constant input')
    p=pearsonr(x,y);s=spearmanr(x,y)
    fisher=np.arctanh(np.clip(float(p.statistic),-.999999999999,.999999999999))
    se=1/np.sqrt(len(x)-3)
    result=dict(n=len(x),pearson=float(p.statistic),pearson_p=float(p.pvalue),
        pearson_fisher95=[float(np.tanh(fisher-1.95996398454*se)),float(np.tanh(fisher+1.95996398454*se))],
        spearman=float(s.statistic),spearman_p=float(s.pvalue))
    if weights is not None:
        w=np.asarray(weights)/np.sum(weights);a=x-np.sum(w*x);b=y-np.sum(w*y)
        result['weighted_pearson']=float(np.sum(w*a*b)/np.sqrt(np.sum(w*a*a)*np.sum(w*b*b)))
    loo=[]
    for i in range(len(x)):
        keep=np.arange(len(x))!=i
        loo.append(float(pearsonr(x[keep],y[keep]).statistic))
    result['leave_one_out_pearson_range']=[min(loo),max(loo)]
    result['most_influential_removed_index']=int(np.argmax(np.abs(np.asarray(loo)-p.statistic)))
    return result


@torch.no_grad()
def evaluate_loss(p,g,teacher,usable,dtype):
    typed={k:v.to(dtype) if v.is_floating_point() else v for k,v in g.items()}
    physical=torch.as_tensor(p,dtype=dtype)
    state=torch.cat(((physical[:,:2]+physical[:,2:]/2)/typed['scale'][0],
                     (physical[:,2]/physical[:,3]).log()[:,None]),-1)[None]
    values=per_case(state,typed,teacher.to(dtype)[None],torch.tensor([usable],dtype=dtype))
    restored=decode(state,typed)[0]*typed['scale'][0]
    local=np.asarray(p)[:,2:].min(1)
    error=float((torch.abs(restored-physical).numpy()/local[:,None]).max())
    return {k:float(v[0]) for k,v in values.items()},error


def main():
    plan=read(HERE/'plan.json');out=EXPERIMENT/plan['output'];out.mkdir(exist_ok=False)
    start=time.perf_counter();existing=read(RUN/'results.json');assert existing['finished'] and existing['count']==100
    manifest=read(TEACHER/'dev_monitor500/manifest.json');tc=read(TEACHER/'config.json')
    chosen=[];labels={}
    for part in manifest['parts']:
        wanted=set(part['indices'])&set(range(100))
        if not wanted:continue
        path=TEACHER/part['path'];assert sha(path)==part['sha256'];chosen.append(path)
        packed=torch.load(path,map_location='cpu',weights_only=False)
        assert packed['indices']==part['indices']
        for i,row in zip(packed['indices'],packed['rows']):
            if i in wanted:
                assert row['index']==i and row['member']==tc['members']['dev_monitor500'][i]
                labels[i]=row
    assert sorted(labels)==list(range(100))
    source_paths=[*HERE.glob('*.py'),HERE/'plan.json',HERE/'provenance.json',RUN/'results.json',RUN/'config.json',
        TEACHER/'dev_monitor500/manifest.json',TEACHER/'config.json',*chosen,
        *[ROOT/p for p in ('datasets/reference_layouts.py','datasets/paths.py','scoring/evaluation.py','scoring/checks.py','scoring/official.py')]]
    source_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(source_paths))}
    coordinate_hashes={f'{i:04d}.npz':sha(RUN/'coordinates'/f'{i:04d}.npz') for i in range(100)}
    save(out/'config.json',dict(plan=plan,source_sha256=source_hashes,coordinate_sha256=coordinate_hashes,
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        all_geometry_reused=True,no_network_or_solver=True,test_read=False))
    with zipfile.ZipFile(out/'source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
        for p in [*HERE.glob('*.py'),HERE/'plan.json',HERE/'provenance.json']:archive.write(p,p.relative_to(ROOT))
    data=ReferenceLayouts('dev_monitor500');records=[]
    with threadpool_limits(limits=1):
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        for index in range(100):
            row=data[index];label=labels[index];old=existing['records'][index]
            assert list(row['member'])==label['member']==old['member']
            with np.load(RUN/'coordinates'/f'{index:04d}.npz',allow_pickle=False) as saved:
                before=saved['positions'].copy();after=saved['relaxed'].copy()
                control=saved['control_final'].copy();final=saved['relaxed_final'].copy()
            preparation=sample(row['case'].solver_inputs,torch.as_tensor(row['positions']).float())
            g=collate([preparation],[0],'cpu')
            required=('mask','valid','area','specs','scale')
            g={k:g[k] for k in required}
            fp64_before,e0=evaluate_loss(before,g,label['target'],float(label['usable']),torch.float64)
            fp64_after,e1=evaluate_loss(after,g,label['target'],float(label['usable']),torch.float64)
            fp32_before,e2=evaluate_loss(before,g,label['target'],float(label['usable']),torch.float32)
            fp32_after,e3=evaluate_loss(after,g,label['target'],float(label['usable']),torch.float32)
            raw_before=evaluate_case(row['case'],before,0.)['metrics']
            raw_after=evaluate_case(row['case'],after,0.)['metrics']
            assert old['control']['metrics']['is_feasible'] and old['mass']['metrics']['is_feasible']
            record=dict(index=index,member=list(row['member']),n=row['case'].block_count,teacher_usable=bool(label['usable']),
                loss_before=fp64_before,loss_after=fp64_after,loss_change={k:fp64_after[k]-v for k,v in fp64_before.items()},
                loss32_before=fp32_before,loss32_after=fp32_after,
                direct_Q_before=raw_before['cost'],direct_Q_after=raw_after['cost'],
                direct_raw_feasible_before=raw_before['is_feasible'],direct_raw_feasible_after=raw_after['is_feasible'],
                final_Q_before=old['control']['metrics']['cost'],final_Q_after=old['mass']['metrics']['cost'],
                decode_roundtrip_max_local_error=max(e0,e1,e2,e3),
                overlap_change=old['raw_after']['pair_intersection_over_area_sum']-old['raw_before']['pair_intersection_over_area_sum'],
                coverage_change=old['raw_after']['mean_module_coverage']-old['raw_before']['mean_module_coverage'])
            records.append(record)
            if index%25==0:print('LOSS',index+1,'/100',flush=True)
    before=np.array([r['loss_before']['total'] for r in records]);after=np.array([r['loss_after']['total'] for r in records]);dl=after-before
    qb=np.array([r['final_Q_before'] for r in records]);qa=np.array([r['final_Q_after'] for r in records]);dq=qa-qb
    rb=np.array([r['direct_Q_before'] for r in records]);ra=np.array([r['direct_Q_after'] for r in records])
    weights=np.exp((np.array([r['n'] for r in records])-120)/12);weights/=weights.sum()
    tolerance=1e-10*(1+np.abs(before));ldir=directions(before,after,tolerance);qdir=directions(qb,qa,1e-12)
    cross=np.zeros((3,3),int)
    np.add.at(cross,(ldir,qdir),1)
    components={}
    for term in records[0]['loss_before']:
        b=np.array([r['loss_before'][term] for r in records]);a=np.array([r['loss_after'][term] for r in records])
        components[term]=dict(mean_before=float(b.mean()),mean_after=float(a.mean()),mean_change=float((a-b).mean()),
            median_change=float(np.median(a-b)),directions=counts(b,a,1e-10*(1+np.abs(b))),correlation_with_final_Q_change=correlation(a-b,dq))
    b32=np.array([r['loss32_before']['total'] for r in records]);a32=np.array([r['loss32_after']['total'] for r in records])
    wcontribution=weights*dq
    result=dict(status='complete',count=100,loss_definition=plan['loss'],stage_definition=plan['stages'],
        loss_counts=counts(before,after,tolerance),strict_loss_counts=counts(before,after,0),
        final_Q_counts=counts(qb,qa,1e-12),direct_raw_Q_counts=counts(rb,ra,1e-12),
        loss_mean_before=float(before.mean()),loss_mean_after=float(after.mean()),
        loss_mean_relative_change=float(after.mean()/before.mean()-1),loss_median_case_relative_change=float(np.median(dl/np.maximum(np.abs(before),1e-12))),
        final_Q_mean_before=float(qb.mean()),final_Q_mean_after=float(qa.mean()),
        final_Q_weighted_before=float(weights@qb),final_Q_weighted_after=float(weights@qa),
        direct_raw_Q_unique_before=np.unique(rb).tolist(),direct_raw_Q_unique_after=np.unique(ra).tolist(),
        cross_table_rows_loss_columns_Q=cross.tolist(),cross_order=['decreased','unchanged','increased'],
        correlation_loss_change_final_Q_change=correlation(dl,dq,weights),
        correlation_relative_loss_change_final_Q_change=correlation(dl/np.maximum(np.abs(before),1e-12),dq),
        correlation_loss_change_direct_raw_Q_change=correlation(dl,ra-rb),
        correlation_overlap_change_final_Q_change=correlation([r['overlap_change'] for r in records],dq),
        correlation_coverage_change_final_Q_change=correlation([r['coverage_change'] for r in records],dq),
        fp32_sensitivity=dict(loss_counts=counts(b32,a32,1e-7*(1+np.abs(b32))),strict_counts=counts(b32,a32,0),
            sign_disagreement_vs_fp64=int((directions(b32,a32,1e-7*(1+np.abs(b32)))!=ldir).sum()),
            correlation=correlation(a32-b32,dq),max_loss_change_abs_difference=float(np.abs(a32-b32-dl).max())),
        usable_teacher_count=sum(r['teacher_usable'] for r in records),
        max_decode_roundtrip_local_error=max(r['decode_roundtrip_max_local_error'] for r in records),components=components,
        largest_weighted_Q_improvements=[dict(index=int(i),weighted_Q_change=float(wcontribution[i]),loss_change=float(dl[i]),Q_change=float(dq[i])) for i in np.argsort(wcontribution)[:5]],
        largest_weighted_Q_regressions=[dict(index=int(i),weighted_Q_change=float(wcontribution[i]),loss_change=float(dl[i]),Q_change=float(dq[i])) for i in np.argsort(-wcontribution)[:5]],
        records=records,test_read=False,no_solve_or_training=True,elapsed_seconds=time.perf_counter()-start)
    assert all(sha(ROOT/p)==digest for p,digest in source_hashes.items())
    save(out/'results.json',result)
    # A standalone scientific figure, all signed deltas; zero axes show quadrants.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=np.where(dq < -1e-12,'#16836b',np.where(dq > 1e-12,'#c64f47','#9299a3'))
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,x,title in ((axes[0],dl,'C3 fixed Loss change'),(axes[1],np.array([r['overlap_change'] for r in records]),'Intersection fraction change')):
        ax.scatter(x,dq,c=colors,s=24,alpha=.8,edgecolors='none');ax.axhline(0,color='#888',lw=.7);ax.axvline(0,color='#888',lw=.7)
        ax.set_xlabel(title+' (after - before)');ax.set_ylabel('Final Q change (after - before)');ax.grid(alpha=.15)
        for i in np.argsort(-np.abs(dq))[:4]:ax.annotate(str(i),(x[i],dq[i]),fontsize=8,xytext=(4,3),textcoords='offset points')
    fig.suptitle('E99 area mass: fixed100 paired changes / exposed development subset')
    fig.savefig(out/'loss_Q_scatter.png',dpi=180);plt.close(fig)
    corr=result['correlation_loss_change_final_Q_change']
    lines=['# E99 固定100例：Loss与Q变化', '',
        'Loss比较吸引—排斥之前/之后的草稿；最终Q比较各自经过原配置规划器后的结果。所有坐标、最终Q与教师标签均只读复用，没有重新求解、训练或选系数。', '',
        '## Loss口径', '',
        '本批草稿来源是E97 C3，使用它原登记的教师模仿Loss：usable × (center + 0.25 shape + 0.25 relative + clearance_tail)。并非E98十项Loss。主统计按同一公式从保存坐标重建规范化状态后以FP64重算，FP32作敏感性核对；不是历史GPU监测日志的逐位重放。', '',
        '所有变化均定义为处理后减处理前，负数表示改善。Loss持平阈值为1e-10×(1+原Loss)，Q为1e-12，严格符号结果也保存在results。', '',
        '| 指标 | 下降 | 持平 | 上升 |','|---|---:|---:|---:|']
    for label,key in (('草稿固定C3 Loss','loss_counts'),('最终Q','final_Q_counts'),('草稿直接Q','direct_raw_Q_counts')):
        v=result[key];lines.append(f"| {label} | {v['decreased']} | {v['unchanged']} | {v['increased']} |")
    lines += ['',f"平均Loss：{result['loss_mean_before']:.9f} → {result['loss_mean_after']:.9f}（{result['loss_mean_relative_change']:+.4%}）。",
        f"最终算术平均Q：{result['final_Q_mean_before']:.9f} → {result['final_Q_mean_after']:.9f}；按模块数加权Q：{result['final_Q_weighted_before']:.9f} → {result['final_Q_weighted_after']:.9f}。", '',
        '## 同例交叉统计', '', '| Loss变化 | Q下降 | Q持平 | Q上升 |','|---|---:|---:|---:|']
    for label,row in zip(('下降','持平','上升'),cross):lines.append(f'| {label} | {row[0]} | {row[1]} | {row[2]} |')
    lines += ['', '## 相关性', '',
        f"ΔLoss与Δ最终Q：Pearson r={corr['pearson']:.6f}，Spearman ρ={corr['spearman']:.6f}；Pearson描述性p={corr['pearson_p']:.6g}，Spearman描述性p={corr['spearman_p']:.6g}。正相关表示两者倾向同向改善/恶化，负相关表示方向相反。", '',
        f"Pearson Fisher95区间：{corr['pearson_fisher95']}；按Q权重加权的Pearson={corr['weighted_pearson']:.6f}；逐一去掉任一例后的Pearson范围={corr['leave_one_out_pearson_range']}。", '',
        f"相对ΔLoss与Δ最终Q的Pearson={result['correlation_relative_loss_change_final_Q_change']['pearson']:.6f}；Δ交面积占比与Δ最终Q的Pearson={result['correlation_overlap_change_final_Q_change']['pearson']:.6f}。", '',
        f"草稿直接Q的处理前取值={result['direct_raw_Q_unique_before']}，处理后取值={result['direct_raw_Q_unique_after']}。若变化全为零，其与Loss变化的相关系数没有定义，不能写成零相关。", '',
        '## 分项', '', '| Loss分项 | 平均前→后 | 下降/持平/上升 | Δ分项与Δ最终Q Pearson |','|---|---:|---:|---:|']
    for key,v in components.items():
        n=v['directions'];cr=v['correlation_with_final_Q_change']['pearson']
        lines.append(f"| {key} | {v['mean_before']:.9f} → {v['mean_after']:.9f} | {n['decreased']}/{n['unchanged']}/{n['increased']} | {'未定义' if cr is None else f'{cr:.6f}'} |")
    lines += ['', '## 数值与证据边界', '',
        f"教师usable={result['usable_teacher_count']}/100；重建/解码的最大局部尺度误差={result['max_decode_roundtrip_local_error']:.3g}。FP32方向统计={result['fp32_sensitivity']['loss_counts']}，与主统计方向不一致={result['fp32_sensitivity']['sign_disagreement_vs_fp64']}例；FP32的Pearson={result['fp32_sensitivity']['correlation']['pearson']:.6f}。", '',
        '这里是同一个已反复曝光的100例开发子集。相关、p值和区间仅描述当前配对样本，不消除自适应开发、确定性单版本和后续规划器运行时预算波动，也不能证明因果或独立泛化。没有按结果改变阈值、Loss权重或筛选案例。', '',
        '[散点图](loss_Q_scatter.png) · [完整逐例数据](results.json) · [分析身份](config.json)', '',
        '| ID | 原Loss | 新Loss | ΔLoss | 原最终Q | 新最终Q | ΔQ |','|---|---:|---:|---:|---:|---:|---:|']
    for r in records:lines.append(f"| {r['index']} | {r['loss_before']['total']:.9f} | {r['loss_after']['total']:.9f} | {r['loss_change']['total']:+.9f} | {r['final_Q_before']:.9f} | {r['final_Q_after']:.9f} | {r['final_Q_after']-r['final_Q_before']:+.9f} |")
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','components')},indent=2))


if __name__=='__main__':main()
