"""One-off saved-result report/figures; no optimization or model selection calls."""
from pathlib import Path
import os
import sys
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT));os.environ['MPLCONFIGDIR']=str(ROOT/'.cache/matplotlib')
from experiments.E32_learned_relation_feedback.resources import configure
configure()
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.E32_learned_relation_feedback.cases import prepare_cache
from visualization.layouts import plot_case,geometry_details

HERE=ROOT/'experiments/E32_learned_relation_feedback';OUT=Path(__file__).resolve().parent

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    data={};table=[];training=[];sources={};eligible=[]
    for arm in ('replay','draft_only'):
        folder=HERE/'runs'/f'20261022_{arm}_seed0';terminal=HERE/'runs'/f'20261022_{arm}_seed0_terminal500/results.json'
        trained=read(folder/'results.json');result=read(terminal)
        if result['status']!='complete':raise ValueError('Report requires complete endpoints')
        data[arm]=result;sources[str(terminal.relative_to(ROOT))]=sha(terminal);training.append({'arm':arm,**trained})
        baseline=result['summaries']['e25']
        for method,stats in result['summaries'].items():
            table.append({'arm':arm,'method':method,**stats})
            if method not in ('e25','implementation_control'):
                passed=trained['best_epoch']>0 and stats['hard_legal']==500 and stats['weighted_Q']<baseline['weighted_Q']-1e-4 and stats['seconds_mean']<baseline['seconds_mean'] and stats['cvar95_Q']<=baseline['cvar95_Q']*1.005 and stats['max_Q']<=baseline['max_Q']+.05
                if passed:eligible.append({'experiment':'E32','arm':arm,'method':method,'weighted_Q':stats['weighted_Q'],'seconds_mean':stats['seconds_mean']})
    sparse_path=ROOT/'experiments/E34_sparse_contact_feedback/runs/20261022_sparse_terminal500/results.json'
    if sparse_path.exists():
        result=read(sparse_path);baseline=data['replay']['summaries']['e25'];sources[str(sparse_path.relative_to(ROOT))]=sha(sparse_path)
        for method,stats in result['summaries'].items():
            table.append({'arm':'replay_sparse','method':method,**stats})
            if stats['hard_legal']==500 and stats['weighted_Q']<baseline['weighted_Q']-1e-4 and stats['seconds_mean']<baseline['seconds_mean'] and stats['cvar95_Q']<=baseline['cvar95_Q']*1.005 and stats['max_Q']<=baseline['max_Q']+.05:
                eligible.append({'experiment':'E34','arm':'replay','method':method,'weighted_Q':stats['weighted_Q'],'seconds_mean':stats['seconds_mean']})
    eligible.sort(key=lambda r:(r['weighted_Q'],r['seconds_mean']))
    diagnostics={}
    for arm in ('replay','draft_only'):
        path=HERE/'runs'/f'20261022_{arm}_seed0/diagnostics.json'
        if path.exists():diagnostics[arm]=read(path)['summary'];sources[str(path.relative_to(ROOT))]=sha(path)
    signed_path=OUT/'signed_order_diagnostic.json'
    if signed_path.exists():diagnostics['signed_order']=read(signed_path);sources[str(signed_path.relative_to(ROOT))]=sha(signed_path)
    write(OUT/'summary.json',{'training':training,'methods':table,'eligible':eligible,'diagnostics':diagnostics,'sources':sources,'test_read':False,'unique_monitor_members':500,'timing_sparse_controls':'historical, if present; confirm contemporaneously before claiming stability'})
    result=data['replay'];records=result['records'];n=np.asarray([r['n'] for r in records]);w=np.exp((n-n.max())/12)
    fig,axes=plt.subplots(1,3,figsize=(15,4.4))
    for method,label,color in [('e25','E25','#d97706'),('contact_static','Static contacts','#2563eb'),('contact_feedback','Contact feedback','#16a34a'),('full_feedback','Full feedback','#dc2626')]:
        q=np.asarray([r['methods'][method]['metrics']['cost'] for r in records]);t=np.asarray([r['methods'][method]['seconds'] for r in records]);order=np.argsort(q);y=np.arange(1,len(q)+1)/len(q)
        axes[0].step(q[order],y,where='post',label=label,color=color);axes[1].step(q[order],np.cumsum(w[order])/w.sum(),where='post',label=label,color=color);axes[2].step(np.sort(t),y,where='post',label=label,color=color)
    axes[0].set(title='Per-case quality tail',xlabel='Neutral-runtime Q',ylabel='Cumulative probability',ylim=(.9,1.002))
    axes[1].set(title='Module-weighted quality tail',xlabel='Neutral-runtime Q',ylabel='Cumulative exp(n/12) weight',ylim=(.9,1.002))
    axes[2].set(title='Complete solve runtime',xlabel='Seconds',ylabel='Cumulative probability')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'quality_runtime.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.2))
    for arm in ('replay','draft_only'):
        progress=read(HERE/'runs'/f'20261022_{arm}_seed0/progress.json');history=progress['history'];epochs=[r['epoch'] for r in history]
        axes[0].plot(epochs,[r['dev_monitor']['selection_proxy'] for r in history],label=arm)
        axes[1].plot(epochs,[r['dev_monitor']['contact_bce'] for r in history],label=arm)
    for ax in axes:ax.set_xlabel('Epoch');ax.grid(alpha=.2);ax.legend()
    axes[0].set_title('Fixed three-state structure selection proxy');axes[1].set_title('Balanced contact BCE')
    fig.tight_layout();fig.savefig(OUT/'training_curves.png',dpi=180);plt.close(fig)
    q0=np.asarray([r['methods']['e25']['metrics']['cost'] for r in records]);q1=np.asarray([r['methods']['full_feedback']['metrics']['cost'] for r in records]);chosen=[]
    for i in (int(np.argmax(w*(q1-q0))),int(np.argmax(q1)),int(np.argmin(q1-q0))):
        if i not in chosen:chosen.append(i)
    cached=prepare_cache()['rows'];figures=[]
    for rank,index in enumerate(chosen,1):
        r=records[index];case=cached[r['index']]['case'];c=np.asarray(case.solver_inputs[5]).tolist();variants={}
        for method in ('e25','full_feedback'):
            value=r['methods'][method];variants[method]={'metrics':value['metrics']}
            if 'positions' in value:variants[method].update(positions=value['positions'],soft_details=geometry_details(value['positions'],c))
            else:variants[method]['unavailable']='No returned layout'
        name=f'layout_{rank:02d}.png';plot_case({'panel':'E32 monitor diagnostic','case_id':str(r['member']),'constraints':c,'variants':variants},OUT/name,stages=(('e25',f'E25 Q={q0[index]:.4f}'),('full_feedback',f'Feedback Q={q1[index]:.4f}')))
        figures.append({'index':r['index'],'member':r['member'],'file':name,'baseline_Q':float(q0[index]),'feedback_Q':float(q1[index]),'n':r['n']})
    write(OUT/'figures.json',{'selected_by':'largest weighted regression, worst feedback, best absolute improvement; duplicates removed','layouts':figures,'source_script_sha256':sha(Path(__file__)),'test_read':False})
    lines=['# E32与稀疏实现的阶段结论','','两组唯一best均按固定三状态结构代理选择，终末500不参与选checkpoint。所有结果为v5自适应开发/MIB派生任务的中性时间Q，非比赛成绩。两组共5000次求解仍只有500个不同监测成员。CPU一进程、两个执行槽，测试封存。','', '| 训练组/方法 | 加权Q | 平均秒 | CVaR95 | 最大Q | 硬合法 |','|---|---:|---:|---:|---:|---:|']
    for r in table:lines.append(f"| {r['arm']}/{r['method']} | {r['weighted_Q']:.6f} | {r['seconds_mean']:.6f} | {r['cvar95_Q']:.6f} | {r['max_Q']:.6f} | {r['hard_legal']}/500 |")
    lines+=['','## 取舍','',('筛选得到候选，仍需冻结后同期确认；暂不替换E25。' if eligible else '没有方法通过已登记联合筛选，不替换E25；不读取额外确认来寻找收益。'),'', '原始关系代理改善与最终Q/时间收益分开判断。完整轴反馈可能降低不加权的极端Q，却把更大代价转移到高模块权重实例；加权CDF与逐例布局用于定位这一现象，不能靠无权重尾部单项升级。稀疏结果若存在，使用此前同资源控制，时间证据须披露并非同期交替。','', '![质量与时间分布](quality_runtime.png)','','![结构代理曲线](training_curves.png)','', '[精确汇总与来源SHA256](summary.json) · [布局索引](figures.json)','']
    if 'replay' in diagnostics:
        d=diagnostics['replay'];lines+=['## 已保存轨迹的事后诊断','',f"replay近邻轴准确率{d['near_axis_accuracy_heuristic']:.4%}→{d['near_axis_accuracy_learned']:.4%}；top-k接触精度{d['top_k_contact_precision_heuristic']:.4%}→{d['top_k_contact_precision_learned']:.4%}。结构检索确有改善，但没有转化为最终联合收益。这只是保存的500例三状态检索诊断，无求解器调用。",'']
    if 'signed_order' in diagnostics:
        counts=diagnostics['signed_order']['arms']['replay']['counts'];lines+=[f"候选改轴近邻对（双向计数）{counts['candidate_override_pairs']}中，参考允许轴且当前顺序与参考相反者仅{counts['candidate_override_axis_correct_order_differs']}；该事后诊断不支持把方向反转视为主要失败机制。不同顺序也可能是有效替代布局，且筛选mask不是实际完整callback回放，不能作因果证明。",'']
    lines+=['120模块反例：E25 Q1.243915→完整反馈2.094497，官方分组违反0→9；57模块正例：2.525082→1.100938，边界违反9→0、分组违反5→1。两者均硬合法；高模块权重反例说明无权重极端Q改善不代表加权稳定性改善。对应布局已统一样式渲染并人工查看。','']
    (OUT/'analysis.md').write_text('\n'.join(lines),encoding='utf-8');print(json.dumps({'eligible':eligible,'figures':figures},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
