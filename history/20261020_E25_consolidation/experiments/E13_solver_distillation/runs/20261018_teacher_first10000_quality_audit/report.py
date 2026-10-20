"""Summarize immutable paired audit; add official size-weighted quality and plots."""
import gzip,json,time
from pathlib import Path
from audit import ROOT,OUT,np,save,sha
from scoring.official import official_evaluator

def main():
    begin=time.perf_counter();result=json.loads((OUT/'results.json').read_text(encoding='utf-8'))
    assert result['status']=='completed' and result['count']==10000
    rows=[json.loads(line) for line in (OUT/result['comparison_file']).read_text(encoding='utf-8').splitlines()]
    state=json.loads(gzip.decompress((OUT/'teacher_results_snapshot.json.gz').read_bytes()))
    bank=ROOT/'experiments/E12_distillation_readiness/runs/teacher_train20000_v1'
    counts=[]
    for row,record in zip(rows,state['records']):
        assert row['case_id']==record['case_id']
        with np.load(bank/record['label_file'],allow_pickle=False) as z:counts.append(int(z['geometry'].shape[0]))
    official=official_evaluator();r=np.array([x['reference']['cost'] for x in rows]);t=np.array([x['teacher']['cost'] for x in rows])
    wr=official.compute_total_score(r.tolist(),counts);wt=official.compute_total_score(t.tolist(),counts)
    normalized={}
    for key in ('bbox_area','hpwl_total'):
        a=np.array([row['reference'][key] for row in rows]);b=np.array([row['teacher'][key] for row in rows]);valid=a>0
        normalized[key]={'mean_casewise_relative_change_teacher_minus_reference':float(np.mean((b[valid]-a[valid])/a[valid])),
          'median_casewise_relative_change_teacher_minus_reference':float(np.median((b[valid]-a[valid])/a[valid])),
          'improved_count':int((b<a-1e-6).sum()),'worse_count':int((b>a+1e-6).sum())}
    both=np.array([x['teacher']['is_feasible'] and x['reference']['is_feasible'] for x in rows])
    cfg=json.loads((OUT/'config.json').read_text(encoding='utf-8'));resume=json.loads((OUT/'resume_config.json').read_text(encoding='utf-8'))
    gain=r-t;workers=[x['worker'] for x in rows];ids=sorted(set(workers),key=lambda s:int(s.split('_')[1]));weights=np.exp((np.array(counts)-max(counts))/12.)
    sums=np.array([[sum(weights[j] for j,x in enumerate(workers) if x==w),
                    sum(weights[j]*r[j] for j,x in enumerate(workers) if x==w),
                    sum(weights[j]*t[j] for j,x in enumerate(workers) if x==w)] for w in ids])
    rng=np.random.default_rng(20261018);sample=rng.integers(0,len(ids),(10000,len(ids)));boot=sums[sample].sum(1)
    weighted_CI=np.quantile(1-boot[:,2]/boot[:,1],[.025,.975]).tolist()
    worst=sorted(rows,key=lambda x:x['Q_delta_teacher_minus_reference'],reverse=True)[:10]
    extra={'official_block_weighted_reference_Q':wr,'official_block_weighted_teacher_Q':wt,
       'official_block_weighted_Q_reduction_fraction':1-wt/wr,'official_block_weighted_reduction_worker_bootstrap_95CI':weighted_CI,
       'block_count_min':min(counts),'block_count_max':max(counts),'normalized_geometry':normalized,
       'both_hard_legal_count':int(both.sum()),'both_legal_reference_Q_mean':float(r[both].mean()),
       'both_legal_teacher_Q_mean':float(t[both].mean()),'elapsed_seconds':time.perf_counter()-begin,
       'source_sha256':sha(__file__),'scoring_scope':'Official exp(n/12) aggregation with neutral runtime; not official contest score.'}
    save('supplement.json',extra)
    ci=result['worker_cluster_bootstrap']['absolute_improvement_95CI'];rc=result['worker_cluster_bootstrap']['relative_mean_Q_reduction_95CI']
    lines=['# 已完成前10,000例：教师答案与官方原reference的Q核对','',
      f"快照时间：{cfg['timestamp_Asia_Shanghai']}。当时教师库已保存{cfg['bank_completed_at_snapshot']}例，本报告固定其中前10,000例（索引0..9999），不按成绩选择。覆盖worker_0..49，每个worker 200例。",'',
      '**当前阶段是教师标签生成，学生蒸馏尚未开始。** 本报告比较的是新教师答案与原reference，不是训练后学生模型成绩。没有修改/停止用户的标签生成任务，没有启动求解器或新训练，没有读取封存测试。','',
      'Q越低越好。两种布局调用同一份未修改的官方评测器，runtime=median_runtime=1，排除时间倍率；原reference按其实际布局重新计分，不假设其Q=1。','',
      '| 指标 | 原reference | 教师答案 |','|---|---:|---:|',
      f"| 平均Q | {result['Q_reference_mean']:.6f} | {result['Q_teacher_mean']:.6f} |",
      f"| 中位Q | {result['Q_reference_median']:.6f} | {result['Q_teacher_median']:.6f} |",
      f"| P95 Q | {result['Q_reference_p95']:.6f} | {result['Q_teacher_p95']:.6f} |",
      f"| 最大Q | {result['Q_reference_max']:.6f} | {result['Q_teacher_max']:.6f} |",
      f"| 硬合法 | {result['reference_hard_legal']}/10000 | {result['teacher_hard_legal']}/10000 |",
      f"| 官方模块数加权Q（中性时间） | {wr:.6f} | {wt:.6f} |",'',
      f"平均Q降低 **{result['mean_Q_reduction_fraction']*100:.2f}%**，平均配对改善（reference−teacher）为{result['mean_absolute_improvement']:.6f}。逐例相对改善的中位数为{result['paired_relative_reduction_median']*100:.2f}%。官方exp(n/12)模块数加权口径下降 **{(1-wt/wr)*100:.2f}%**；这里仍然是质量对照，不是正式比赛总分。",'',
      f"改善{result['improved_count']}例（{result['improved_count']/100:.2f}%），退化{result['worse_count']}例（{result['worse_count']/100:.2f}%），持平{result['tied_count']}例；差值绝对值≤1e-6记持平。至少改善5%：{result['improved_at_least_5pct_count']}例；至少改善10%：{result['improved_at_least_10pct_count']}例；至少退化5%：{result['worse_at_least_5pct_count']}例。",'',
      '## 显著性和覆盖边界','',
      f"按worker整组重采样10,000次（seed20261018），平均Q绝对改善的95%区间为[{ci[0]:.6f}, {ci[1]:.6f}]；平均Q相对下降的95%区间为[{rc[0]*100:.2f}%, {rc[1]*100:.2f}%]。{result['improved_worker_count']}/{len(ids)}个worker的平均Q改善。",'',
      '聚类区间用于检验这批数据中改善的稳定性，避免将同一worker的200个布局简单当成独立证据。跨worker是否完全独立未被证明；前半工作集不覆盖worker50..99，且这些是训练任务，不能据此声称学生泛化或隐藏测试成绩。','',
      '## 改善来源与退化尾部','', '| 平均违规数 | 原reference | 教师 |','|---|---:|---:|']
    for key in ('boundary_violations','grouping_violations','mib_violations','total_soft_violations'):
        values=result[key];lines.append(f"| {key} | {values['reference_mean']:.4f} | {values['teacher_mean']:.4f} |")
    lines+=['', f"每例bbox面积相对变化平均{normalized['bbox_area']['mean_casewise_relative_change_teacher_minus_reference']*100:+.2f}%；HPWL相对变化平均{normalized['hpwl_total']['mean_casewise_relative_change_teacher_minus_reference']*100:+.2f}%。这些是配对几何指标，不把不同规模布局的物理面积直接混为改善百分比。",'',
      f"教师中Q>2有{result['teacher_Q_above_2']}例，Q>3有{result['teacher_Q_above_3']}例；救援{result['rescue_count']}例。所有尾部均计入均值和分位数，没有删除退化案例。",'',
      '| 最明显退化案例 | 原Q | 教师Q | ΔQ（正数退化） | 救援 |','|---|---:|---:|---:|---|']
    for row in worst:
        lines.append(f"| {row['case_id']} | {row['reference']['cost']:.6f} | {row['teacher']['cost']:.6f} | {row['Q_delta_teacher_minus_reference']:+.6f} | {row['rescue']} |")
    lines+=['','结论用于支持制作版本化答案标签层；不把每一条教师答案无条件称为更优答案。当前E13既定蒸馏仍使用全部教师标签，本次只审计，不修改训练数据或运行配置。','',
      '## 完整性与计分核验','',
      '全部10,000个npz标签的哈希与固定快照匹配，教师缓存的官方与独立硬合法标记逐例检查；固定索引0/2499/4999/7499/9999教师布局重新计分，与缓存Q一致至1e-10。',
      '原始reference全部重新调用官方评测及独立硬约束复核。审核途中为减少重复标量Tensor处理，改为直接读取原始数据中已有的官方基准、以NumPy连接数据输入同一官方函数；5个已完成样例的全部指标与原慢路径相同至1e-10。原始部分结果、源码与压缩快照全部保留，累计计算预算不重置。',
      '', f"审核累计保守计费{result['elapsed_seconds']:.2f}秒；补充统计{extra['elapsed_seconds']:.2f}秒。活动E12/E13源码未修改。",'',
      '关键证据：[汇总](results.json)、[加权及几何统计](supplement.json)、[逐例比较](comparisons_continued.jsonl)、[固定快照配置](config.json)、[续审计分等价核验](resume_config.json)。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    xs=np.arange(2);width=.32
    axs[0].bar(xs-width/2,[result['Q_reference_mean'],wr],width,label='Official reference',color='#8496ad')
    axs[0].bar(xs+width/2,[result['Q_teacher_mean'],wt],width,label='E10 teacher',color='#278a75')
    axs[0].set_xticks(xs,['Arithmetic mean','Official size weighting']);axs[0].set(ylabel='Neutral-runtime Q (lower is better)',title='Same 10,000 training tasks');axs[0].legend(fontsize=8)
    vals=np.array([w['mean_improvement'] for w in result['workers']]);axs[1].bar(np.arange(len(ids)),vals,color=np.where(vals>=0,'#278a75','#c25353'))
    axs[1].axhline(0,color='black',lw=.8);axs[1].set(xlabel='Worker index (0..49)',ylabel='Mean Q(reference) - Q(teacher)',title='Paired improvement within each worker')
    fig.tight_layout();fig.savefig(OUT/'quality_comparison.png',dpi=180);plt.close(fig)
    print(json.dumps({k:v for k,v in extra.items() if k not in ('normalized_geometry',)},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
