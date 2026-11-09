"""Report only completed saved stages; evidence limits are explicit."""
import json
import numpy as np
from .io import HERE, save
from .protection import verify


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def main():
    plan = read(HERE/'study_plan.json');rows=[]
    for task in plan['tasks']:
        folder = HERE/'runs'/task['name']
        if not (folder/'results.json').exists():continue
        result = read(folder/'results.json');p=read(folder/'progress.json')
        if not p['history']:continue
        best = next((x for x in p['history'] if x['epoch']==result['best_epoch']),None)
        if best is None:continue
        row = {**task,'training':result,'best_dev_monitor':best['dev_monitor'],
            'best_train_monitor':best['train_monitor'],'best_step':best['step'],
            'step_curve':[{'step':h['step'],'dev':h['dev_monitor']['total'],
                           'fit':h['fit_train']['total']} for h in p['history']],
            'parameters':read(folder/'config.json')['identity']['parameters']}
        if task['kind']=='memory':
            minimum = min(p['history'],key=lambda h:h['fit_train']['total'])
            row['minimum_observed_fit_loss']=minimum['fit_train']['total']
            row['minimum_fit_step']=minimum['step']
            row['near_fit_mean_loss_le_1e_3']=minimum['fit_train']['total']<=.001
            row['capacity_limit_proven']=False
        if (folder/'full_dev.json').exists():row['full_dev2000']=read(folder/'full_dev.json')['losses']
        if (folder/'terminal/summary.json').exists():row['flash500']=read(folder/'terminal/summary.json')
        rows.append(row)
    protected=verify()
    result={'rows':rows,'planned_runs':len(plan['tasks']),'test_read':False,
        'protected_checkpoints_unchanged':protected,
        'interpretation_limits':plan['interpretation_limits']}
    save(HERE/'runs/study_report/results.json',result)
    lines=['# E57：容量与训练规模诊断（已有结果）','',
           f'预设{len(plan["tasks"])}个训练，目前保存{len(rows)}组训练记录；未完成项不作结论。', '',
           '## 32例纯参考拟合','',
           '| 配置 | seed | 最低观测拟合Loss | 对应step | 开发best Loss | 状态 |',
           '|---|---:|---:|---:|---:|---|']
    for x in rows:
        if x['kind']=='memory':
            lines.append(f'| {x["arm"]} | {x["seed"]} | {x["minimum_observed_fit_loss"]:.8f} | {x["minimum_fit_step"]} | {x["best_dev_monitor"]["total"]:.6f} | {x["training"]["status"]} |')
    lines+=['','拟合Loss最小值只证明模型曾达到该经验拟合程度，不用于best/LR/早停；仍按固定dev500同一Loss选模。去掉Dropout/weight decay是单独标注的诊断对照。小样本不能拟合不能单独证明表达能力不足，须同时考虑优化、对称性和预算；拟合成功也不证明全分布容量充足。','',
            '## 数据规模与完整容量对照','',
            '| 配置 | 训练例数 | seed | dev500 G | dev2000 G | best step | flash500加权Q | 状态 |',
            '|---|---:|---:|---:|---:|---:|---:|---|']
    for x in rows:
        if x['kind']=='memory':continue
        full=x.get('full_dev2000',{}).get('total');q=x.get('flash500',{}).get('weighted_Q')
        f='待完成' if full is None else f'{full:.6f}';qs='待完成' if q is None else f'{q:.6f}'
        lines.append(f'| {x["arm"]} | {x["dataset_size"]} | {x["seed"]} | {x["best_dev_monitor"]["total"]:.6f} | {f} | {x["best_step"]} | {qs} | {x["training"]["status"]} |')
    lines+=['','每个监测块固定500次更新／16000次样本暴露；这里的“epoch”字段指监测块，不是一次遍历。必须同时查看各曲线的共同更新步数与各自停止点，不能把训练次数差异误归因为数据收益。两个seed对应两个预先冻结的嵌套子集家族，16k成员相同；结果含抽样与初始化差异。','',
            '16k附近若仍稳定改善，说明本分布上数据收益未饱和；若平坦，只能说明本范围未见收益，不能证明全覆盖或训练集足够。宽模型改善训练但不改善开发支持泛化限制；二者同时改善支持当前容量可能仍有限，但不证明唯一原因。','',
            '完整dev2000和已暴露dev500均是开发证据，不是独立盲测。最终Q仅辅助解释，不能重新选择checkpoint。构造对称反例是普适表达限制，不代表真实数据中此类情况的频率。','',
            '[预设方案](../../study_plan.json) · [对称反例](../symmetry_counterexample/results.json)']
    (HERE/'runs/study_report/analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if any(x['kind']!='memory' for x in rows):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        for seed,color in [(0,'#267b9a'),(1,'#d65e37')]:
            points=sorted([x for x in rows if x['kind']=='data' and x['seed']==seed],key=lambda x:x['dataset_size'])
            if points:axes[0].plot([x['dataset_size'] for x in points],[x['best_dev_monitor']['total'] for x in points],marker='o',label=f'seed/family {seed}',color=color)
            for x in points:
                c=x['step_curve'];axes[1].plot([v['step'] for v in c],[v['dev'] for v in c],label=f'N={x["dataset_size"]},s{seed}',alpha=.8)
        axes[0].set_xscale('log',base=2);axes[0].set_xlabel('Training layouts');axes[0].set_ylabel('Best dev500 fixed G')
        axes[1].set_xlabel('Optimizer updates');axes[1].set_ylabel('Dev500 fixed G')
        for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.savefig(HERE/'runs/study_report/learning_curves.png',dpi=160);plt.close(fig)
    print(json.dumps({'reported_training_records':len(rows),'planned':len(plan['tasks'])}),flush=True)


if __name__=='__main__':
    main()
