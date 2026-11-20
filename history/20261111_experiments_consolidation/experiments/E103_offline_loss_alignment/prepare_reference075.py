from pathlib import Path
import json,hashlib,shutil
root=Path.cwd();exp=root/'experiments/E103_offline_loss_alignment';src=exp/'finetune_learned_v1';dst=exp/'reference075_posttrain';dst.mkdir(exist_ok=False)
def write(n,s):(dst/n).write_text(s,encoding='utf-8')
def replace(s,a,b):
 assert a in s,a[:120]
 return s.replace(a,b)
for name in ['__init__.py','io.py','runtime.py','data.py','resident.py','augmentation.py','step_control.py','captured_projection.py','engine.py','training.py']:
 shutil.copy2(src/name,dst/name)
training=(dst/'training.py').read_text(encoding='utf-8').replace('lr=1e-3,weight_decay=.0003','lr=1e-4,weight_decay=.0003')
write('training.py',training)
schedule=(src/'schedule.py').read_text(encoding='utf-8').replace('TIERS=(1e-3,2.5e-4,1e-4,5e-5)','TIERS=(1e-4,5e-5)')
write('schedule.py',schedule)
common=(src/'common.py').read_text(encoding='utf-8').replace('20261104_learned_v1_lr1e3_seed0','20261104_reference075_seed0')
common=replace(common,"files=list(OWNER.glob('*.py'))+[OWNER/n for n in ('plan.json','copy_provenance.json','README.md','launch.ps1')]","files=list(OWNER.rglob('*.py'))+[OWNER/n for n in ('plan.json','copy_provenance.json','README.md','launch.ps1','evaluation/plan.json','evaluation/copy_provenance.json')]")
write('common.py',common)
run=(src/'run.py').read_text(encoding='utf-8').replace('20261104_learned_v1_lr1e3_seed0','20261104_reference075_seed0').replace('lr=.001','lr=.0001')
run=replace(run,"assert sha(ROOT/plan['coefficient_source'])==plan['coefficient_source_sha256']","assert plan['coefficients']['position']==.75 and plan['coefficients']['shape']==.1875")
run=replace(run,"coefficient_source_sha256=plan['coefficient_source_sha256']","coefficient_policy='original E98 position/shape multiplied by.75, other8 unchanged'")
run=replace(run,"'learned_v1',","'reference075',")
run=run.replace('E103第一版权重的增强基线续训','E98参考监督0.75倍后训练').replace('初始LR1e-3','初始LR1e-4')
run=run.replace('没有根据Q重选权重，未自动后处理或读取测试。','没有根据Q重选权重；由单次流水线随后执行固定dev500，测试封存。')
write('train.py',run)
# Fixed initial coefficients, deliberately no sum normalization.
from experiments.E103_offline_loss_alignment.loss_config import COEFFICIENTS
base=dict(COEFFICIENTS);co=dict(base);co['position']*=.75;co['shape']*=.75
parent=exp/'assets/e98_generalization/best.pt';digest=hashlib.sha256(parent.read_bytes()).hexdigest()
assert digest=='3f7ca0086af8a13bd52eedf0635117312e5c0a72e80e423e9cee3db5ca9b87ea'
plan=json.loads((src/'plan.json').read_text(encoding='utf-8'))
plan.update(revision='E103.E98_enhanced.reference075.posttrain.dev500.v1',authorization='User requested one small ablation: original E98 reference position/shape weights times.75, posttrain from penultimate LR, no15epoch forced stop, automatically evaluate500 after training; background launch when ready.',
 coefficients=co,initial_coefficients=base,coefficient_policy='position1→.75, shape.25→.1875; other8 exact original E98 values; no sum normalization; no learned/canonical/global fitted coefficients.',
 lr_tiers=[.0001,.00005],terminal='Train one candidate to own-devLoss normal earlystop with no cap; freeze own-Loss best; automatically evaluate originaldev500 with unchanged E98 fixed-depth repair; reuse parent evaluation; report then stop. Test sealed.',
 evidence_boundary='One ablation from original augmented E98 best71; parent frozen. Effects include extra training; no claim of pure coefficient causality without trained control.',
 validation=dict(panel='original frozen dev_monitor500',cases=500,postprocessor='E98 fixed-depth v1 unmodified',selection='unique own-Loss best frozen beforeQ',training_source='original E98 augmented best71',test_read=False))
plan.pop('coefficient_source',None);plan.pop('coefficient_source_sha256',None)
write('plan.json',json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
provenance=[dict(source=p.relative_to(root).as_posix(),copy=(dst/p.name).relative_to(root).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in [src/n for n in ['io.py','runtime.py','data.py','resident.py','augmentation.py','step_control.py','captured_projection.py','engine.py']]]
write('copy_provenance.json',json.dumps(provenance,indent=2)+'\n')
# Copy the validated terminal evaluator, changing only run/epoch metadata and orchestration.
evalsrc=exp/'validate_learned_v1';ed=dst/'evaluation';ed.mkdir()
for name in ['__init__.py','metrics.py','predictor.py','plots.py']:
 text=(evalsrc/name).read_text(encoding='utf-8')
 if name=='predictor.py':
  text=text.replace('from ..sources','from ...sources').replace('from ..features','from ...features').replace('from ..batches','from ...batches')
 if name=='plots.py':text=text.replace('E103 learned-v1 best29','E98 reference075 posttrain')
 (ed/name).write_text(text,encoding='utf-8')
shutil.copy2(evalsrc/'copy_provenance.json',ed/'copy_provenance.json')
valplan=json.loads((evalsrc/'plan.json').read_text(encoding='utf-8'))
valplan.update(authorization='User explicitly requested automatic500 evaluation after this small reference075 experiment; use originaldev500, tests sealed.',
 train_run='experiments/E103_offline_loss_alignment/runs/20261104_reference075_seed0',
 candidate_checkpoint='experiments/E103_offline_loss_alignment/runs/20261104_reference075_seed0/checkpoints/best.pt',
 candidate_sha256=None,selection='Unique reference075 own-devLoss best, whatever epoch, frozen beforeQ',
 parent_bridge_reuse='experiments/E103_offline_loss_alignment/runs/20261104_learned_v1_lr1e3_seed0/evaluation_depth_dev500',
 parent_reuse='Reuse previously passed five-case exact bridge after verifying all relevant scientific source hashes and copied inference bodies; no repeated baseline solves.',
 identical_model_policy='If selected epoch0 model tensors equal parent, reuse all completed parent500 Q and original measured timings instead of rerunning identical model.')
(ed/'plan.json').write_text(json.dumps(valplan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
s=(evalsrc/'run.py').read_text(encoding='utf-8')
s=s.replace('from ..common import ROOT,OWNER as EXPERIMENT','from ...common import ROOT,OWNER as EXPERIMENT').replace('from ..finetune_learned_v1.data','from ..data').replace('from ..postprocessor','from ...postprocessor')
s=s.replace('20261104_learned_v1_lr1e3_seed0','20261104_reference075_seed0')
s=s.replace("'finetune_learned_v1/data.py'","'reference075_posttrain/data.py'")
s=replace(s,"assert training['status']=='early_stop' and progress['finished'] and training['best_epoch']==29","assert training['status']=='early_stop' and progress['finished']\n    best_epoch=training['best_epoch'];plan['candidate_sha256']=training['best_sha256']")
s=replace(s,"assert saved['progress']['epoch']==saved['progress']['best_epoch']==29","assert saved['progress']['epoch']==saved['progress']['best_epoch']==best_epoch")
start=s.index("                sizes=[r['case'].block_count for r in rows]")
end=s.index("                predictor.model.load_state_dict",start)
s=s[:start]+'''                # Reuse the exact five-case parent bridge already established with this inference/repair path.
                proof=ROOT/plan['parent_bridge_reuse']
                bridge=read(proof/'parent_reproduction.json');assert bridge['status']=='passed'
                frozen_proof=read(proof/'frozen.json')
                assert frozen_proof['plan']['parent_sha256']==plan['parent_sha256']
                relevant=[name for name in frozen_proof['sources'] if name.startswith(('datasets/','scoring/','third_party/')) or
                    name.startswith('experiments/E103_offline_loss_alignment/postprocessor/') or
                    name in ('experiments/E103_offline_loss_alignment/common.py','experiments/E103_offline_loss_alignment/features.py',
                        'experiments/E103_offline_loss_alignment/batches.py','experiments/E103_offline_loss_alignment/ports.py',
                        'experiments/E103_offline_loss_alignment/cache.py','experiments/E103_offline_loss_alignment/sources/e98_generalization/model.py')]
                for name in relevant:assert sha(ROOT/name)==frozen_proof['sources'][name]
                assert sha(OWNER.parent/'data.py')==frozen_proof['sources']['experiments/E103_offline_loss_alignment/finetune_learned_v1/data.py']
                # Relative import paths differ; inference function/class bodies must remain byte-identical.
                import ast
                old_predictor=ROOT/'experiments/E103_offline_loss_alignment/validate_learned_v1/predictor.py'
                def bodies(path):
                    text=path.read_text(encoding='utf-8')
                    return [ast.get_source_segment(text,node) for node in ast.parse(text).body if isinstance(node,(ast.FunctionDef,ast.ClassDef))]
                assert bodies(OWNER/'predictor.py')==bodies(old_predictor)
                save(OUT/'parent_reproduction.json',dict(status='passed',reused_bridge=plan['parent_bridge_reuse'],
                    checks=bridge['checks'],scientific_dependencies_unchanged=True,parent500_reused=True))
                identical=(mh==baseline_frozen['model_hash'])
''' +s[end:]
s=replace(s,"                    if path.exists():","                    if identical:\n                        record=dict(parent_records[i]);record.update(checkpoint_sha256=plan['candidate_sha256'],\n                            model_hash=mh,cached_original_checkpoint_sha256=plan['parent_sha256'],timings_reused=True)\n                        record['raw_overlap']=geometry(record['draft_positions'])\n                    elif path.exists():")
s=s.replace('evaluating_frozen_best29','evaluating_frozen_reference075_best')
s=s.replace("training=dict(epochs=32,best_epoch=29,","training=dict(epochs=training['epochs'],best_epoch=best_epoch,")
s=s.replace("if h['epoch']==29","if h['epoch']==best_epoch")
s=s.replace("frozen_checkpoint_sha256=plan['candidate_sha256'],parent500_reused=True,test_read=False,","frozen_checkpoint_sha256=plan['candidate_sha256'],parent500_reused=True,candidate_model_identical_to_parent=identical,\n                actual_candidate_solver_calls=0 if identical else500,test_read=False,")
s=s.replace('else500','else 500')
s=s.replace("timing_boundary='Sequential complete candidate timing; parent historical timing, not same-session speed ablation'","timing_boundary='Candidate timings reused from identical parent; no new latency measurement' if identical else 'Sequential complete candidate timing; parent historical timing, not same-session speed ablation'")
s=s.replace("'# E103新Loss续训best29：固定dev500验证'","'# E98参考监督0.75倍后训练：固定dev500验证'")
s=s.replace("('新Loss续训best29',new)","(f'参考0.75倍best{best_epoch}',new)")
s=s.replace('训练32轮正常早停，best29；',"训练{training['epochs']}轮正常早停，best{best_epoch}；")
s=s.replace('基线五题逐位复现后复用旧500结果。','基线复用先前已通过的五题逐位复现及旧500结果；科学依赖已重新核对。')
(ed/'run.py').write_text(s,encoding='utf-8')
print('Prepared one reference075 training plus automatic fixeddev500 evaluation.')
