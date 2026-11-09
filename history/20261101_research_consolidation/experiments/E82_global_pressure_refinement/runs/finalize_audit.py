"""Close the completed E82 configuration without promoting or repeating it."""
import json
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
a=read(HERE/'runs/completion/inspection.json');review=read(HERE/'runs/review.json')
a['decision']='Retain E81 as primary compact candidate. E82 slightly reduces own Loss/raw overlap, but weightedQ and maximum regress; mean paired interval crosses0. No baseline promotion, no repeat terminal or Q-based reselect.'
a['case164_saved_failure']={'E81_Q':1.8257730769575755,'E82_Q':2.056040033465631,
 'E81_primary_failures':'29/29','E82_primary_failures':'28/28','both_packed_rescue':True,
 'E81_E82_grouping_violations':[6,9],'E81_E82_hpwl_gap':[.6419076761848734,.6580705200957004],
 'E81_E82_bbox_gap':[.08586120398099698,.052779734662747164],
 'interpretation':'Lower raw overlap and smaller repaired bbox did not prevent grouping/wirelength regression. Context or scalar path pressure is not sufficient; do not infer a single cause.'}
save(HERE/'runs/completion/inspection.json',a)
text='''# E82完成复核

28轮正常早停，唯一own-Loss best12，训练3843.31秒（64.06分钟）。源码/唯一best/full-dev/terminal身份核对通过，best/latest全部497个E81源张量逐字保持，部署/独立F导出均来自同一best。已实际查看217/399/194三图并核对哈希，所有500例包含于自身公开E42预测框。测试封存，未重跑终末或按Q重选。

固定train/dev Loss .421133/.482077→.419237/.477922，改善约.45%/.86%。完整dev2000交面积2.491%→2.445%，是小幅几何改善；全套同例统计、弱端口/稀疏子群和框检查见inspection.json。相对E42仍有明显重叠收益，原始草稿仍0/2000硬合法，flash500/500硬合法。

加权Q1.018035，相对E81的1.016242退化+.001793，worker区间[-.001226,+.005969]跨零；175胜/135平/190负。最大Q2.056040劣于E81的1.825773，也劣于E42的1.927279。故本配置没有充分升级证据；保留E81作为小布局＋修正路线主要候选，不更换冻结E42/E50。

最差164仍28/28主形状不合法并依赖packed救援（E81为29/29）。其原始交面积又略降，修复后外框相对参考差距8.586%→5.278%也下降，但分组违反6→9、线长差距64.19%→65.81%，Q最终恶化。不能以A/重叠下降代替实际修复质量，也不能仅凭此例单因果归咎某一输入或Loss项。

三张图整体与E81接近，局部边界和小块仍冲突；没有替用户打视觉分。后续优先分析目标与修复的错位，研究预算还有一个候选名额，须有证据才训练，不为用满次数自动启动。

[完整审计](inspection.json) · [既有报告](../review.md) · [100例图](../20261029_e81_pressure_seed0/draft_comparison100/index.html)。
'''
(HERE/'runs/completion/inspection.md').write_text(text,encoding='utf-8')
state=read(HERE/'state.json');state.update(status='complete_audited',controller_pid=None,child_pid=None,report='runs/completion/inspection.md');save(HERE/'state.json',state)
ledger=ROOT/'experiments/E81_multisource_refinement/study.json';s=read(ledger);entry=next(r for r in s['run_ledger'] if r['experiment']==HERE.name)
entry.update(status='early_stop_complete_audited',controller_pid=None,child_pid=None,
 elapsed_seconds=review['training']['elapsed_seconds'],steps=28000,epochs=28,best_epoch=12,
 terminal_runs=1,weighted_Q=review['flash']['weighted_Q'],max_Q=review['flash']['max_Q'],
 raw_overlap2000=a['paired_overlap2000_vs_E42']['current_mean'],report='runs/completion/inspection.md')
s.update(status='E82_complete_failure_analysis_next',current_stage='complete_audited')
s['phase_training_seconds']=sum(r['elapsed_seconds'] for r in s['run_ledger']);s['global_charged_seconds']=s['previous_global_charged_seconds']+s['phase_training_seconds']
s['last_check']={'utc':datetime.now(timezone.utc).isoformat(),'state':'E82 complete audited, no active training'};save(ledger,s)
print('E82 audit closed, E81 retained, no active training, no promotion.')
