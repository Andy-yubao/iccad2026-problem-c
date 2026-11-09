"""Record completed frozen transfer, image inspection and unchanged budgets."""
from datetime import datetime,timezone
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from experiments.E80_modular_transfer.io import HERE,save,sha
def read(p):return json.loads(p.read_text(encoding='utf-8'))
out=HERE/'runs/20261029_e42_frozen_e79_transfer';inspection=read(HERE/'runs/inspection.json')
images=out/'draft_comparison100';manifest=read(images/'manifest.json')
inspection['images_reviewed']=[{'index':i,'sha256':sha(images/f'images/case_{i:04d}.png')} for i in (217,399,194)]
inspection['visual_observations']='Macro placement retained. Compared with E42, large visible intrusions at middle/lower interfaces shrink; small-block clusters and group boundaries still conflict. No external-frame expansion is used. Visual review does not establish hard legality or user preference.'
inspection['gallery_rendering']={'caption_correction':'Inherited E74 frame title corrected to supplied E42 frame using saved coordinates only, no inference or scoring repeated',
 'renderer':'experiments/E80_modular_transfer/runs/render_transfer_gallery.py',
 'renderer_sha256':sha(HERE/'runs/render_transfer_gallery.py'),'manifest_sha256':sha(images/'manifest.json')}
assert len(manifest['cases'])==100
for r in manifest['cases']:assert sha(images/r['image'])==r['image_sha256']
inspection['decision']='Evidence supports same frozen correction package transferring to the smaller related placer. Retain modular research direction, no baseline promotion: worst case164 worsens to2.200946 and no measured speed advantage. Enlarging F remains an untested hypothesis.'
save(HERE/'runs/inspection.json',inspection)
report=HERE/'runs/inspection.md';text=report.read_text(encoding='utf-8')
text=text.replace('待人工查看3至5图后补最终解释。','已实际查看217/399/194三图并核对更新后的图像哈希。大体布局保持，中央和下部的大面积侵入减少；小块和组交界仍有冲突，不能声称草稿硬合法。初始继承图题误写E74框，已仅用保存坐标重新出图更正为E42框，未重做模型或评分。')
text+='''

本次支持“布局模型＋修正模型”的接口和同族迁移，但不证明最优规格分工：总参数较原E79减少约73.93%，weighted Q仅高0.000946，差值区间跨零也不等于统计等效。最差例164经packed救援后Q2.200946，劣于E42的1.927279和E79的1.738220。313胜/79平/108负，仍有明确反例。原E42小主网络与E74大主网络有共同来源，普遍迁移仍待检验。

同500例原始草稿交面积和精确空白全部改善；外框在3e-6归一化容差下17缩小/483不变/0扩大，500均包含在自身E42预测框内。平均求解0.276秒，旧E79为0.271秒；这不是严格同时性能基准，且没有观察到参数缩小带来的速度收益。下一步优先查尾部与来源依赖，再研究扩大修正器，不把幅度或容量不足当作已证实原因。
'''
report.write_text(text,encoding='utf-8')
state=read(HERE/'state.json');state.update(status='complete_audited',pid=None,checked_utc=datetime.now(timezone.utc).isoformat(),
 report='runs/inspection.md',training_updates=0,test_read=False,baseline_promoted=False)
save(HERE/'state.json',state)
ledger_path=ROOT/'experiments/E78_connectivity_posttrain/study.json';ledger=read(ledger_path)
ledger.update(status='modular_transfer_complete_analyze_tail_next',current_stage='complete_audited',current_experiment='E80_modular_transfer')
diag=ledger['inference_diagnostics'][-1];diag.update(status='complete_audited',tool_session=None,
 evaluation_seconds=state['evaluation_seconds'],weighted_Q=1.0173292491225494,raw_overlap_dev2000=.02784877391673398,
 parameters=1537022,correction_parameters=65483,training_updates=0,terminal_runs=1,
 report='experiments/E80_modular_transfer/runs/inspection.md',baseline_promoted=False)
ledger['last_check']={'utc':state['checked_utc'],'state':'E80 frozen transfer complete audited, no training active'}
save(ledger_path,ledger)
with (HERE/'README.md').open('a',encoding='utf-8') as f:
 f.write('\n已完成并复核：[结果](runs/inspection.md)。E42＋冻结F加权Q1.017329、dev交面积2.785%，500例框内；总参数153.7万。均值改善但最差Q2.200946退化，保留模块化研究方向，不升级基线；本组没有训练。\n')
print('Transfer evidence finalized; training ledger costs and caps unchanged.')
