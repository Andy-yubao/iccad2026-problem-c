from pathlib import Path
import json,hashlib,shutil
root=Path.cwd();e103=root/'experiments/E103_offline_loss_alignment';old=root/'experiments/E98_loss_weight_learning';dst=e103/'finetune_learned_v1'
dst.mkdir(exist_ok=False)
def write(name,s):(dst/name).write_text(s,encoding='utf-8')
provenance=[]
for name,origin in [('resident.py',old/'resident.py'),('step_control.py',old/'step_control.py'),('captured_projection.py',old/'captured_projection.py'),('augmentation.py',old/'generalization_baseline/augmentation.py')]:
 shutil.copy2(origin,dst/name);provenance.append(dict(source=origin.relative_to(root).as_posix(),copy=(dst/name).relative_to(root).as_posix(),sha256=hashlib.sha256(origin.read_bytes()).hexdigest()))
data=(old/'data.py').read_text(encoding='utf-8').replace('from .dataset_view import ReferenceLayouts','from datasets.reference_layouts import ReferenceLayouts').replace('from .ports import attach','from ..ports import attach').replace('from .batches import sample, collate as batch_collate','from ..batches import sample, collate as batch_collate').replace('from .cache import load_rows as cached','from ..cache import load_rows as cached')
write('data.py',data)
engine=(old/'dynamic_weight_learning_v3/engine.py').read_text(encoding='utf-8')
engine=engine.replace('from ..loss import per_case as base_components','from ..simple_loss import components\nfrom ..loss_config import A,B').replace('from ..captured_projection import CapturedProjection','from .captured_projection import CapturedProjection').replace('from ..generalization_baseline.augmentation import axis_swap','from .augmentation import axis_swap')
engine=engine.replace('values=base_components(state,graph)',"values=components(state,graph,dict(overlap=A,task=B))\n    values['legacy_total']=values['total']")
write('engine.py',engine)
training=(old/'dynamic_weight_learning_v3/run.py').read_text(encoding='utf-8');training=training[:training.index('from .evaluation import evaluate_panel')]
training=training.replace('from ..io import ROOT, HERE, sha, save, checkpoint, exclusive','from .io import ROOT, HERE, sha, save, checkpoint')
training=training.replace('from ..train import rng_state, restore_rng','from .runtime import rng_state, restore_rng').replace('from ..generalization_baseline.model import GeometryModel','from ..sources.e98_generalization.model import GeometryModel').replace('from ..generalization_baseline.augmentation import flags_for','from .augmentation import flags_for').replace('from ..data import load_rows, collate, register_resident','from .data import load_rows, collate, register_resident').replace('from ..dataset_view import ReferenceLayouts','from datasets.reference_layouts import ReferenceLayouts').replace('from ..resident import ResidentRows','from .resident import ResidentRows')
training=training.replace('from ..qualification import evaluate as evaluator\n','').replace('from ..depth_evaluation.repair.canonical_native import warmup\n','')
training=training.replace('from datasets.reference_layouts import public_hard_reference\n','').replace('from scoring.evaluation import evaluate_case\n','')
training=training.replace('from .common import OWNER, RUN, PLAN_PATH, MODEL, read, sources, coefficients, q_summary, PhaseSchedule','from .common import OWNER, RUN, PLAN_PATH, MODEL, read, sources\nfrom .schedule import PhaseSchedule')
training=training.replace('lr=1e-4,weight_decay=.0003','lr=1e-3,weight_decay=.0003')
training=training.replace("optimizer='fresh AdamW for plus/minus only; frozen control has no optimizer'","optimizer='fresh AdamW m/v/age for new objective; parent model/RNG/projection retained'")
training=training.replace("if early or p['epoch']>=15:","if early:").replace("reason = 'early_stop' if early else 'maximum_epochs'","reason = 'early_stop'")
training=training.replace('round_spec','phase_spec').replace("phase_spec['round']","phase_spec['phase']").replace("round=phase_spec['phase']","phase=phase_spec['phase']")
training=training.replace("phase_spec['sampler_seed']+p['epoch']","phase_spec['sampler_seed']+phase_spec['epoch_offset']+p['epoch']")
training=training.replace("flags_for(phase_spec['sampler_seed'],p['epoch'],ids)","flags_for(phase_spec['sampler_seed'],phase_spec['epoch_offset']+p['epoch'],ids)")
write('training.py',training)
write('__init__.py','"""User-authorized E103 learned-v1 continuation from enhanced E98 best71."""\n')
proposal=e103/'runs/20261104_bank1000x5_seed0/analysis/proposal.json';p=json.loads(proposal.read_text(encoding='utf-8'))
parent=e103/'assets/e98_generalization/best.pt';digest=hashlib.sha256(parent.read_bytes()).hexdigest()
assert digest=='3f7ca0086af8a13bd52eedf0635117312e5c0a72e80e423e9cee3db5ca9b87ea'
plan=dict(revision='E103.learned_v1.generalization_best71.lr1e3.no_protection.unlimited.v1',authorization='User explicitly requested using first fitted new Loss to continue training the E98 augmented baseline, initialLR1e-3, no LR protection and no15-epoch limit; prior background authorization remains.',
 source_checkpoint=parent.relative_to(root).as_posix(),source_sha256=digest,parent_best_epoch=71,coefficient_source=proposal.relative_to(root).as_posix(),coefficient_source_sha256=hashlib.sha256(proposal.read_bytes()).hexdigest(),coefficients=p['physical_coefficients'],coefficient_policy='Exact first-v1 fitted physical coefficients; no rounding, no changing near-zero wire coefficient, no canonical or quadratic objective; sum3.8550373616903784',
 model=dict(width=96,layers=12,rectangle=True,ffn_dropout=.10,residual_dropout=.05),
 data='Original v5 train16000; fixed train/dev monitoring500 each; public MIB mean-area task; test sealed',
 batch_size=32,precision='FP32 deterministic CUDA, no AMP or TF32',
 augmentation='Original .5 probability stateless XY transposition; fixed/preplaced positions, fixed shapes, boundary bits, pins, masks and reference targets transformed together; unchanged copied implementation',
 optimizer=dict(name='AdamW',fresh_momentum=True,fresh_age=True,weight_decay=.0003,clip_norm=1.,max_module_relative_update=.005),
 rng='Inherit frozen parent RNG; independent new phase, continue parent epoch71 sample/augmentation seed stream; within-phase complete state restoration',
 phase=dict(phase=1,sampler_seed=0,epoch_offset=71),
 lr_tiers=[.001,.00025,.0001,.00005],protection_epochs=0,scheduler_baseline_epoch=0,plateau_patience=3,minimum_meaningful_absolute=1e-5,minimum_meaningful_relative=.001,best_minimum_absolute=1e-5,
 early_stop='After reaching5e-5, three consecutive validation-Loss stalls stop. No protected epochs or extra minimum hold. On each drop reset per-tier plateau reference to current validation Loss.',
 maximum_epochs=None,maximum_seconds=None,
 objective='Exact ten unweighted E98 components at original network state, multiplied by the registered learned-v1 coefficients; arithmetic mean per sample; identical for backprop/train/dev/best/LR/earlystop',
 selection='Only own fixed dev500 Loss; epoch0 eligible; Q never selects epochs or changes weights',
 preservation='New run only; parent and all historical best/latest/STOPs intact; own best/latest retain optimizer, RNG, sampler order/cursor, projection and scheduler',
 terminal='Freeze own-Loss best and report; no automatic postprocessor Q, test or extra training branches',
 numerical_guards='Original finite/Loss<1000/gradient norm<10000 before AdamW; no batch skipping, guard relaxation or automatic retry',
 evidence_boundary='Prior offline descent gate failed; user expressly authorized this direct training trial. High startingLR does not guarantee escaping local optima. No claim of pure coefficient causal benefit without a trained control.',
 stop='Create this package STOP or newrun STOP; accepted batch boundary saves latest and stops; no E98 STOP removal',
 implementation_indices=[1683,5474,10027,10140,15001],test_read=False)
write('plan.json',json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
write('copy_provenance.json',json.dumps(provenance,indent=2)+'\n')
print('Prepared exact learned-v1 continuation, no epoch/time cap; original E98 STOPs untouched.')
