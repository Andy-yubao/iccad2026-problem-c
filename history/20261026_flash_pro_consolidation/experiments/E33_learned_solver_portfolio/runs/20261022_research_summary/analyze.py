"""Report saved evidence only. Does not select, train, solve or open tests."""
from pathlib import Path
import os
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT));os.environ['MPLCONFIGDIR']=str(ROOT/'.cache/matplotlib')
from experiments.E33_learned_solver_portfolio.resources import configure
configure()
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=ROOT/'experiments/E33_learned_solver_portfolio'
OUT=Path(__file__).resolve().parent


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    evidence={};sources={};lines=['# E33：学习求解器选择的运行结论','',
        '仅汇总已保存证据。原train中的反事实理想选择器消费参考Q，不可部署，不算模型收益；内部校准与外层开发评价分开。全部Q为v5 MIB派生任务的中性时间开发指标，非正式比赛分数。CPU最多两个执行槽，数值任务顺序，测试封存。','']
    def record(key,path):
        if not path.exists():return None
        value=read(path);evidence[key]=value;sources[str(path.relative_to(ROOT))]=digest(path);return value
    labels=HERE/'runs/prepared_expert_outcomes_v1'
    headroom=record('train_pilot',labels/'headroom_1000.json')
    record('label_preparation',labels/'results_16000.json')
    record('label_manifest',labels/'manifest.json')
    record('label_identity',labels/'identity.json')
    if headroom:
        baseline=headroom['experts']['0'];oracle=headroom['optimistic_oracle_with_prefix_cost']
        lines+=['## 训练反事实先导','',f"1000例原train四专家配对：E25加权Q {baseline['weighted_Q']:.6f}、均时{baseline['seconds_mean']:.6f}秒；不可部署理想选择器含前缀开销的Q {oracle['weighted_Q']:.6f}、均时{oracle['seconds_mean']:.6f}秒。扩大条件通过：{headroom['continue_to_full_labels']}。这只判断条件互补空间。",'']
    for seed in (0,1,2):
        folder=HERE/'runs'/f'20261022_gate_seed{seed}'
        trained=record(f'seed{seed}_training',folder/'results.json')
        calibration=record(f'seed{seed}_calibration',folder/'calibration.json')
        if trained:
            lines+=[f"seed{seed}：{trained['status']}，训练{trained['epochs']}轮，唯一best{trained['best_epoch']}，代理{trained['best_proxy']:.6f}，累计{trained['elapsed_seconds']:.1f}秒；fit {trained['fit_count']}、内部监测500、校准{trained['calibration_count']}。",'']
        if calibration:lines+=[f"seed{seed}内部校准：{calibration['status']}，冻结部署方法：{calibration['deployment_method']}。",'']
    for path in sorted((HERE/'runs').glob('*/frozen.json')):
        record(path.parent.name+'_frozen',path)
    endpoints=[]
    for path in sorted((HERE/'runs').glob('*/summary.json')):
        if path.parent==OUT:continue
        value=read(path)
        if value.get('status')!='complete' or 'summaries' not in value:continue
        endpoints.append((path.parent.name,value));record(path.parent.name,path)
    for tag,value in endpoints:
        lines+=['## '+tag,'','| 方法 | 加权Q | 平均秒 | CVaR95 | 最大Q | 硬合法 |','|---|---:|---:|---:|---:|---:|']
        for method,r in value['summaries'].items():
            lines.append(f"| {method} | {r['weighted_Q']:.6f} | {r['seconds_mean']:.6f} | {r['cvar95_Q']:.6f} | {r['max_Q']:.6f} | {r['hard_legal']}/{r['count']} |")
        lines+=['','配对差、完整分片bootstrap区间、逐例退化与救援详见该运行summary.json。六方法的结果不能用来在终末后重新挑选部署方法；只检验先前内层冻结的方案。','']
        full=read(HERE/'runs'/tag/'results.json');records=full['records'];methods=list(value['summaries'])
        fig,axes=plt.subplots(1,3,figsize=(15,4.2));n=np.asarray([r['n'] for r in records]);w=np.exp((n-n.max())/12)
        for method in methods:
            q=np.asarray([r['methods'][method]['metrics']['cost'] for r in records]);t=np.asarray([r['methods'][method]['seconds'] for r in records]);order=np.argsort(q)
            axes[0].step(q[order],np.arange(1,len(q)+1)/len(q),where='post',label=method)
            axes[1].step(q[order],np.cumsum(w[order])/w.sum(),where='post',label=method)
            axes[2].step(np.sort(t),np.arange(1,len(t)+1)/len(t),where='post',label=method)
        axes[0].set(title='Per-case quality tail',xlabel='Neutral-runtime Q',ylabel='CDF',ylim=(.9,1.002))
        axes[1].set(title='Module-weighted quality tail',xlabel='Neutral-runtime Q',ylabel='Weighted CDF',ylim=(.9,1.002))
        axes[2].set(title='Complete solve runtime',xlabel='Seconds',ylabel='CDF')
        for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=7)
        fig.tight_layout();name=tag+'_distributions.png';fig.savefig(OUT/name,dpi=170);plt.close(fig);lines+=[f'![质量与完整时间分布]({name})','']
    seed0=HERE/'runs/20261022_gate_seed0/progress.json'
    if seed0.exists():
        history=read(seed0)['history'];fig,axes=plt.subplots(1,2,figsize=(10,4.2))
        for key,label in [('train_monitor','Fit monitor'),('internal_monitor','Held-out shard monitor')]:
            epochs=[r['epoch'] for r in history]
            axes[0].plot(epochs,[r[key]['selection_proxy'] for r in history],label=label)
            axes[1].plot(epochs,[r[key]['accuracy'] for r in history],label=label)
        for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8);ax.set_xlabel('Epoch')
        axes[0].set_title('Fixed supervised selection proxy');axes[1].set_title('Oracle action accuracy')
        fig.tight_layout();fig.savefig(OUT/'gate_training.png',dpi=170);plt.close(fig);lines+=['![门控训练曲线](gate_training.png)','']
    lines+=['## 解释边界','',
        '完整分片隔离的是新门控的梯度、监测与校准角色；原专家已经见过原train，不代表专家未见泛化。开发monitor和预留确认都属于自适应开发数据，不是封存测试。实际部署只执行所选路径，前缀、特征、门控和额外几何推理全部计时。内部估计不能代替在线完整测量。','',
        '未通过的路线不因平均单项改善升级；规模规则若获益也不算额外MLP贡献。所有独特布局、真实最佳权重、标签数据库和源快照留在本机，Git只保存源码与精简证据。','']
    evidence['sources']=sources;evidence['test_read']=False
    (OUT/'summary.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'analysis.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'endpoints':[tag for tag,_ in endpoints],'report':str(OUT/'analysis.md')}),flush=True)


if __name__=='__main__':main()
