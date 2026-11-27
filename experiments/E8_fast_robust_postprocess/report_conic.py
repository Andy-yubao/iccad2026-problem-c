"""Saved-output comparison only: no tensor reads, model calls or solver calls."""
import argparse
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def rows(path):return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default='20261018_conic_diagonal_dev100_seed0')
    parser.add_argument('--variant',default='conic_diagonal')
    args=parser.parse_args();folder=(HERE/'runs'/args.run).resolve()
    if not folder.is_relative_to(HERE/'runs'):raise ValueError('Run outside E8')
    result=read(folder/'results.json');config=read(folder/'config.json')
    if result['status']!='complete':raise ValueError('Wait for complete comparison')
    records=rows(folder/'cases.jsonl');name=args.variant
    base='safe_plain120' if 'safe_plain120' in result['variants'] else 'control120'
    old_summary=result['variants'][base].copy();new_summary=result['variants'][name]
    old_metrics=[r['variants'][base]['final']['metrics'] for r in records]
    replay=False
    if base=='control120':
        replay_folder=HERE/'runs/20261018_dev100_guard_replay_seed0'
        override={r['case_id']:r['after']['metrics'] for r in rows(replay_folder/'changed_cases.jsonl')}
        old_metrics=[override.get(r['case_id'],m) for r,m in zip(records,old_metrics)]
        quality=read(replay_folder/'results.json')['after_cached_guard_replay']
        for k in ('Q_mean','Q_weighted','Q_p95','Q_max'):old_summary[k]=quality[k]
        old_summary['worst10percent_mean']=quality['worst10_mean'];replay=True
    new_metrics=[r['variants'][name]['final']['metrics'] for r in records]
    a=np.array([r['cost'] for r in old_metrics]);b=np.array([r['cost'] for r in new_metrics]);d=b-a
    if replay:
        old_summary['Q_above1_5']=int((a>1.5).sum());old_summary['Q_above2']=int((a>2).sum())
        old_summary['online_guard_fallback_count_before_replay']=old_summary.pop('guard_fallback_count')
    rng=np.random.default_rng(20261018)
    interval=np.quantile(d[rng.integers(0,len(d),(20000,len(d)))].mean(1),[.025,.975])
    pairkey='paired_vs_safe_plain120' if base=='safe_plain120' else 'paired_vs_control120'
    speed=result[pairkey][name]
    detail={'panel':config['panel'],'n':len(records),'model_frozen':True,'test_payload_reads':0,
            'baseline':old_summary,'selected':new_summary,'baseline_quality_cached_guard_replay':replay,
            'paired_Q_mean_difference':float(d.mean()),'paired_mean_difference_bootstrap95_descriptive':interval.tolist(),
            'wins':int((d < -1e-8).sum()),'ties':int((abs(d)<=1e-8).sum()),'losses':int((d>1e-8).sum()),
            'postprocess_speedup':speed['postprocess_speedup'],'inference_postprocess_speedup':speed['end_to_end_speedup'],
            'inference_worker_wall_speedup':speed.get('inference_and_worker_wall_speedup'),
            'kernel_sha256':config['source_sha256']['experiments/E8_fast_robust_postprocess/conic_geometry.py'],
            'rescue_used_count':sum(bool(r['variants'][name]['stats'].get('rescue_used')) for r in records),
            'reference_reused_count':sum('reused_from' in r['variants'][base] for r in records),
            'scope':('Adaptive development validation after confirmation failure feedback' if name=='conic_robust' else 'Additional development confirmation') if config['panel']=='confirm100' else 'Adaptive development selection panel',
            'policy':{'fast_target_seconds':1.3,'total_target_seconds':5.,'rescue_trigger':'no_public_proxy_improvement_over_initial_legal_row','seed_pack_target_max_seconds':1.2} if name=='conic_robust' else {'target_seconds':config['short_seconds']},
            'metric_means':{k:{'baseline':float(np.mean([m[k] for m in old_metrics])),
                               'selected':float(np.mean([m[k] for m in new_metrics]))}
                            for k in ('hpwl_gap','area_gap','boundary_violations','grouping_violations','mib_violations')}}
    (folder/'comparison.json').write_text(json.dumps(detail,indent=2,ensure_ascii=False),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
    fig,axes=plt.subplots(1,3,figsize=(14,4.1))
    axes[0].plot(np.arange(1,len(a)+1),np.sort(a),label='原算法（数值修复后）')
    axes[0].plot(np.arange(1,len(b)+1),np.sort(b),label='E8快速后处理')
    axes[0].set(xlabel='全部案例按Q排序',ylabel='Q（越低越好）',title='质量与尾部');axes[0].legend(fontsize=8)
    axes[1].scatter(a,b,s=15,alpha=.7)
    lower=min(a.min(),b.min())-.01;upper=max(a.max(),b.max())+.01
    axes[1].plot([lower,upper],[lower,upper],color='gray',ls='--',lw=.8)
    axes[1].set(xlabel='原算法Q',ylabel='E8 Q',title=f'逐例：{detail["wins"]}胜 / {detail["ties"]}平 / {detail["losses"]}负')
    oldtime=np.array([r['variants'][base]['solve_seconds'] for r in records])
    newtime=np.array([r['variants'][name]['solve_seconds'] for r in records])
    axes[2].plot(np.sort(oldtime),label='原算法');axes[2].plot(np.sort(newtime),label='E8')
    axes[2].set(xlabel='全部案例按耗时排序',ylabel='核心后处理秒',title=f'平均后处理加速{speed["postprocess_speedup"]:.2f}倍')
    axes[2].legend(fontsize=8);fig.tight_layout();fig.savefig(folder/'quality_time.png',dpi=170);plt.close(fig)
    text=['# E8快速后处理完整对照','',
          f'面板：{config["panel"]}，{len(records)}例。E7 attention_geometry冻结；Q为官方中性时间质量分，越低越好。参考只用于事后计分。','',
          '| 指标 | 原算法（已修复数值回退） | E8选定配置 |','|---|---:|---:|']
    for label,key in [('平均Q','Q_mean'),('块数加权Q','Q_weighted'),('Q P95','Q_p95'),('最大Q','Q_max'),
                      ('最差10%平均Q','worst10percent_mean'),('平均后处理秒','postprocess_seconds_mean'),
                      ('P95后处理秒','postprocess_seconds_p95'),('最长后处理秒','postprocess_seconds_max')]:
        text.append(f'| {label} | {old_summary[key]:.6f} | {new_summary[key]:.6f} |')
    text+=['',f'新配置硬合法{new_summary["feasible"]}/{len(records)}，执行异常{new_summary["errors"]}。',
           f'后处理加速{speed["postprocess_speedup"]:.2f}倍，模型推理＋后处理加速{speed["end_to_end_speedup"]:.2f}倍。',
           f'相对修复后参照，逐例{detail["wins"]}胜、{detail["ties"]}平、{detail["losses"]}负；平均Q差{d.mean():.6f}。',
           f'配对均值差的描述性bootstrap 95%区间为[{interval[0]:.6f}, {interval[1]:.6f}]；开发选择/数据生成相关性使其不能当作最终泛化保证。','']
    if replay:
        text+=['原算法质量列复用既有末端数值修复的缓存重放（2例重评、98例原结果）；原算法时间列仍是原在线运行，未冒充修复后的新在线计时。原始未修复参照平均Q=1.276534、最大Q=5.379855。',
               f'本轮研究命令实际墙钟{result["wall_seconds_this_invocation"]:.2f}秒。复用草稿时本轮不重新执行其模型前向；端到端速度使用每例已测模型时间＋本轮实际求解时间之比，不拿研究脚本墙钟比冒充完整应用加速。','']
    else:
        text+= [('原算法参照复用此前完整在线运行（无观测器、含微小坐标修复）的结果及实际时间；本轮新配置逐例在线运行。两次均使用常驻公开输入子进程与末端评分。' if detail['reference_reused_count'] else '两种算法均在同一常驻公开输入子进程在线运行，取消原算法观测器，只评分末端。'),
               f'含模型推理、文件I/O与worker墙钟的配对加速为{speed["inference_and_worker_wall_speedup"]:.2f}倍；另行排除数据加载和事后评分开销。本轮研究命令墙钟{result["wall_seconds_this_invocation"]:.2f}秒，不冒充两种算法独立应用的完整墙钟比。','']
    text+=['![质量与耗时](quality_time.png)','',
           '## 机制与质量来源','',
           '一次锥求解精确约束凸面积下界，保留原保守上界与尺寸调整范围，避免反复补面积切面。密集线长先用IRLS构造二次上界，再分解成单模块项，降低稠密矩阵填充；官方线长与评分仍计算全部连线。公开边界、接触树、兼容MIB、固定尺寸和预放置均保留。候选用公开输入代理选择，终末独立硬检查。',
           '质量不是逐例不降；软违反、线长与面积的变化见comparison.json。原几何变量/连续惩罚、曼哈顿锥模型、完整二次模型和分解模型的受控筛选见[五例机制对照](../20261018_conic_diagonal_diagnostic5_seed0/analysis.md)。五例是回顾挑选，不是额外泛化证据。','',
           '## 复现与证据','',
           '[配置与实际命令](config.json)、[完整统计](results.json)、[逐例坐标/分解](cases.jsonl)、[可恢复源码](source_snapshot.zip)、[比较数据](comparison.json)。',
           ('选定入口为../../optimizer.py，等同conic_robust：普通例1.3秒；没有候选改善初始合法行排时才更换可行种子，整例最多5秒目标。救援只看公开输入/代理改善状态。' if name=='conic_robust' else '历史conic_diagonal：1.3秒目标、保留上界切面、约束约简、单模块二次线长、最多24轮。'),
           '目标预算不是严格实时上界，表内实际最大时间完整报告。已复用的原算法结果/耗时在逐例reused_from中标明，不冒充本次在线重测。',
           '没有启动训练、读取封存local_test或改变冻结split/workset/官方评分。单一开发面板的最大Q是经验最差值，不是未知输入的数学保证。','']
    if name=='conic_robust':
        normal=a<=1.5
        text+=['## 救援为什么必要','',
               '原1.3秒快路径在额外100例出现一例Q≈10行排保底，完整面板平均Q和最大Q均输给原算法，未通过联合目标。该反证完整保留，没有删除失败例或改换平均方式。',
               '[3秒/10秒同轨迹及放松面积上界](../20261018_rescue_mechanism2_seed0/results.json)均未救回该例；[更换可行种子的五例验证](../20261018_feasible_seed_rescue5_seed0/results.json)才使Q降到2.047516。不是无条件增加普通案例的预算。',
               f'本面板救援触发{detail["rescue_used_count"]}/{len(records)}次，全部耗时计入表内。修订后两组共200例重新在线运行，同一源码与同一调度政策；第二组已参与修订，不再作为独立未见确认。',
               f'仅作事后分解：原算法Q≤1.5的{int(normal.sum())}例，原/新平均Q为{a[normal].mean():.6f}/{b[normal].mean():.6f}。主指标仍保留全部样本；整体均值改善不能写成普通例逐例都更好。',
               '更高绝对目标（平均Q≤1.20、最大Q≤1.8）尚未全部达到。当前完成的是两组开发面板上的相对质量、尾部与至少8倍速度联合目标，尚无封存测试泛化或未知输入Q上界保证。','']
    (folder/'research.md').write_text('\n'.join(text),encoding='utf-8')
    print(json.dumps({k:detail[k] for k in ('panel','n','paired_Q_mean_difference','wins','ties','losses','postprocess_speedup','inference_postprocess_speedup')},ensure_ascii=False))


if __name__=='__main__':main()
