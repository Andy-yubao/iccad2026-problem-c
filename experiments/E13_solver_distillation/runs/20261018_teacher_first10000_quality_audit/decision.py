"""Decision from completed audit; prospective best-of quality, no label mutations."""
import json,collections
from audit import ROOT,OUT,np,save
from scoring.official import official_evaluator

rows=[json.loads(x) for x in (OUT/'comparisons_continued.jsonl').read_text(encoding='utf-8').splitlines()]
result=json.loads((OUT/'results.json').read_text(encoding='utf-8'));supp=json.loads((OUT/'supplement.json').read_text(encoding='utf-8'))
import gzip
records=json.loads(gzip.decompress((OUT/'teacher_results_snapshot.json.gz').read_bytes()))['records'][:10000]
bank=ROOT/'experiments/E12_distillation_readiness/runs/teacher_train20000_v1'
counts=[]
for r in records:
    with np.load(bank/r['label_file'],allow_pickle=False) as z:counts.append(len(z['geometry']))
reference=np.array([r['reference']['cost'] for r in rows]);teacher=np.array([r['teacher']['cost'] for r in rows]);best=np.minimum(reference,teacher)
best_weighted=official_evaluator().compute_total_score(best.tolist(),counts)
regressions=collections.Counter(r['worker'] for r in rows if r['Q_delta_teacher_minus_reference']>1e-6)
rescues=collections.Counter(r['worker'] for r in rows if r['rescue'])
near_cap=collections.Counter(r['worker'] for r in rows if r['teacher']['cost']>9.9)
counter={'actual_labels_modified':False,'existing_answers_only':True,'best_of_reference_teacher_mean_Q':float(best.mean()),
  'best_of_reference_teacher_weighted_Q':float(best_weighted),
  'best_of_mean_improvement_fraction':float(1-best.mean()/reference.mean()),
  'best_of_weighted_improvement_fraction':float(1-best_weighted/supp['official_block_weighted_reference_Q']),
  'regression_counts_by_worker':dict(regressions),'rescue_counts_by_worker':dict(rescues),'near_cap_counts_by_worker':dict(near_cap)}
save('selection_counterfactual.json',counter)
ci=result['worker_cluster_bootstrap']['relative_mean_Q_reduction_95CI']
text=f'''# 结论：多数样例改善，但现有教师库不能整体视作更优答案库

当前运行仍是教师标签生成，学生训练尚未开始。只读审计固定前10,000例（worker0..49），原reference逐例使用同一官方评测器重新评分，两边均中性时间；任务未被停止或修改。

| 口径 | 原reference | 教师答案 | 变化 |
|---|---:|---:|---:|
| 算术平均Q | {result['Q_reference_mean']:.6f} | {result['Q_teacher_mean']:.6f} | 改善{result['mean_Q_reduction_fraction']*100:.2f}% |
| 官方exp(n/12)加权Q | {supp['official_block_weighted_reference_Q']:.6f} | {supp['official_block_weighted_teacher_Q']:.6f} | 退化{-supp['official_block_weighted_Q_reduction_fraction']*100:.2f}% |
| 中位Q | {result['Q_reference_median']:.6f} | {result['Q_teacher_median']:.6f} | 改善 |
| P95 Q | {result['Q_reference_p95']:.6f} | {result['Q_teacher_p95']:.6f} | 改善 |
| 最大Q | {result['Q_reference_max']:.6f} | {result['Q_teacher_max']:.6f} | 明显退化 |
| 硬合法 | 10000/10000 | 10000/10000 | 一致 |

7830例改善（78.30%）、1515例退化（15.15%）、655例持平（6.55%）。至少5%退化有547例。230例触发救援，全部比reference差；救援组平均Q从1.238148升到2.912106。正常组9770例平均Q从1.303374降到1.215818。

worker_49的200例平均Q从1.235901升到2.274414。它不能从汇总中删掉，不能用正常组结果代替全体表现。按worker整组bootstrap的平均相对改善95%区间为[{ci[0]*100:.2f}%, {ci[1]*100:.2f}%]，跨零；官方模块数加权口径实际更差。因此不宜宣称这批数据整体稳定、统计显著地优于官方原答案。

收益主要体现在边界违规平均数2.0579→0.3514；分组违规0.0811→0.2410、MIB违规3.9022→3.9124。每例bbox面积相对变化平均+1.02%，HPWL平均+1.51%。不能把综合Q的多数改善解释为所有几何指标都更优。

## 对数据更新的含义

新的教师答案具有价值，但不能无条件替换原reference。合理的下一步是制作独立版本的择优答案层：对同一任务，保留两种现有答案中合法且Q更低的完整布局；原始reference与官方评分基准继续保留。这个选择只用于训练标签，不能把reference送入部署求解器。

对已完成这10,000例做只读的“两者择优”计算，算术平均Q会是{best.mean():.6f}，模块数加权Q会是{best_weighted:.6f}；相对原reference分别改善{counter['best_of_mean_improvement_fraction']*100:.2f}%和{counter['best_of_weighted_improvement_fraction']*100:.2f}%。这是从两份已有完整布局中逐例选择后可构造的标签质量，不是新的学生模型效果；还需复核所选reference标签与当前低面积解码接口的兼容性。本次只计算该对照，没有生成/替换择优标签，也没有改变E13既定全教师蒸馏的配置。

鉴于上述尾部，建议正式启动全库蒸馏前先审议标签择优或救援样例处理；不能继续把所有教师输出称为质量升级。

完整数据：[详细报告](report.md)、[汇总](results.json)、[模块数加权](supplement.json)、[择优对照及尾部分布](selection_counterfactual.json)、[逐例比较](comparisons_continued.jsonl)。这些是训练任务的标签审计，不代表未见数据泛化或正式比赛总分。
'''
(OUT/'decision.md').write_text(text,encoding='utf-8')
print(json.dumps(counter,ensure_ascii=False,indent=2))
