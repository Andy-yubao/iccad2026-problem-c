"""Declare the new authorized modular phase, retaining every old charge/cap."""
import json
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=ROOT/'experiments/E81_multisource_refinement'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
old=read(ROOT/'experiments/E79_center_refinement/provenance.json')
parents=read(ROOT/'experiments/E80_modular_transfer/provenance.json')['parents']
protected=dict(old['protected_checkpoints_sha256'])
import hashlib
for name in ('best','latest'):
 p=ROOT/f'experiments/E79_center_refinement/runs/20261029_e78_center_seed0/checkpoints/{name}.pt'
 protected[p.relative_to(ROOT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
composition=ROOT/'experiments/E80_modular_transfer/runs/20261029_e42_frozen_e79_transfer/composition.pt'
protected[composition.relative_to(ROOT).as_posix()]=hashlib.sha256(composition.read_bytes()).hexdigest()
save(HERE/'provenance.json',{'experiment':'E81_multisource_refinement','source_experiment':'experiments/E79_center_refinement',
 'copy_note':'Independent runtime. P42 frozen deployment. P74 frozen extra TRAIN/LOSS source only. F three heads uniformly widened96/32 to192/64 and all jointly post-trained; eval initialization preserves E80/E79.',
 'parents':parents,'protected_checkpoints_sha256':protected,'test_read':False,
 'optimizer_transition':'Fresh AdamW and same four-tier rule for all widened F; no source optimizer reset or source budget deletion',
 'previous_global_charged_seconds':22680.532898700098})
plan={'experiment':'E81_multisource_refinement','phase':3,'phase_candidate':1,
 'authorization':'User: 你可以继续尝试这条路线，这依旧是一个长任务，你可以自行启动训练',
 'seed':0,'train':32000,'dev_monitor':500,'batch_size':32,'draft_views_per_case':2,
 'max_hours':2,'max_epochs':200,'test_read':False,
 'architecture':'Deployment=frozen P42 96x12 + independent F192-node/64-message. F contains three old learned stages, uniformly warm widened, no new random stage. Training also uses fixed P74, excluded from deployment.',
 'deployment_parameters':1713086,'trainable_F_parameters':241547,'old_F_parameters':65483,'training_model_parameters_including_auxiliary_P74':7543025,
 'initial_function':'E80 P42+oldF and E79 P74+oldF within checked FP32 tolerance, not a new random correction function',
 'output_contract':'Training/full-dev output6=(P42 predicted3,P74 predicted3); physical decode uses first3. DeploymentModel strips P74 weights and computation, outputs3.',
 'fixed_loss':'For each source L=G+4I+5E+.25A. Per-case fixed total=mean_source(L)+.25*(max_source(L)-mean_source(L)); train/dev/full-dev use same function. G retains all original eight terms. A same16-step public selected mode0 anchor corridor term from E78. No ramp.',
 'why':'E80 transfers in mean but case164 has29/29 illegal shape candidates and packed rescue grows frame49%, with preplaced path14-41-50-20 overfill. Train shape-capable F jointly, expose both source distributions, and add per-case worst-source protection.',
 'counterevidence':'Anchor certificate is not sufficient: E80 case219 has worse certificate but betterQ; case21 loose fixed-order bound fails while actual repair succeeds. A is an auxiliary shape-pressure hypothesis, not a full feasibility proof or Q surrogate guarantee. No evidence yet that capacity is the main bottleneck.',
 'limits':'Related source family only; main shape/order changes can remain blocked. No straight-through estimator. All-stage final projection/MIB can exceed any individual.15 pre-projection residual cap. Mixed-source total can improve while one deployment component degrades; components reported separately.',
 'optimizer':'Fresh AdamW over all F, clip1, decay.0003, dropout.05, original four LR/early-stop; frozen placers always eval',
 'selection':'Unique own fixed dev500 mean selects best, epoch0 eligible; one P42 deployment terminal500 only; P74 not terminal-scored in this candidate',
 'data':'Frozen v6 train32000 prefix, same old dev2000/train-dev500; train500 remains old16k source, two sources for every monitored case; FP32',
 'evaluation':'full dev2000 with identical two-source Loss, physical P42 raw geometry; one flash500 and fixed100 reference/current/E42 gallery; reuse saved E42/E74/E79/E80 results; tests sealed'}
save(HERE/'plan.json',plan)
previous=read(ROOT/'experiments/E78_connectivity_posttrain/study.json')
save(HERE/'study.json',{'phase':3,'status':'preparing_warm_multisource_candidate','authorization':plan['authorization'],
 'authorization_utc':datetime.now(timezone.utc).isoformat(),'previous_phase':'experiments/E78_connectivity_posttrain/study.json',
 'previous_global_charged_seconds':previous['global_charged_seconds'],'max_new_formal_candidates':3,
 'max_phase_training_seconds':21600,'max_seconds_per_candidate':7200,'global_training_cap_seconds':64800,
 'phase_training_seconds':0,'global_charged_seconds':previous['global_charged_seconds'],
 'accounting':'Latest explicit user instruction opens modular phase; phase1/2 actual charges and historical self-limits are preserved. No prior failed/stopped run is restarted.',
 'current_experiment':'E81_multisource_refinement','current_stage':'preflight_preparation','run_ledger':[],
 'automation_id':'problem-c','wake_minutes':30,'test_read':False,'baseline_promoted':False})

# Explicit runtime migrations, all before preflight and source freezing.
p=HERE/'train.py';t=p.read_text(encoding='utf-8')
t=t.replace("REVISION = 'E81.frozen_e78_center_source_guard.v1'","REVISION = 'E81.smallP_warmwideF_two_source.v1'")
t=t.replace("'topology_regression', 'topology_flip_fraction', 'source_true_intersection',", "'anchored_corridor', 'P42_total', 'P74_total', 'P74_true_intersection', 'P74_anchored_corridor', 'P74_geometry', 'source_loss_gap',")
t=t.replace('Entire E78 frozen/eval; new center-only head dropout.05/decay.0003; source dimensions exact and E74 frame','P42/P74 frozen/eval; all warm widened F trainable with dropout.05/decay.0003; source-owned public frames')
t=t.replace('G+4I+5E+.1 one-sided source topology regression','Two-source robust fixed G+4I+5E+.25A')
t=t.replace('new center-only Adam; entireE78 frozen, all source shapes exact, center bound.15 and E74 frame','all widened F Adam; P42/P74 frozen, own-source frame and per-stage.15 residual cap')
t=t.replace("'public_source_channels':[3,4,5]","'auxiliary_P74_prediction_channels':[3,4,5], 'deployment_parameters':1713086, 'draft_views_per_case':2")
p.write_text(t,encoding='utf-8')
p=HERE/'step_control.py';t=p.read_text(encoding='utf-8').replace('Plain AdamW only new center head; entire E78 frozen/eval; dimensions fixed, center cap.15 and E74 frame','Plain AdamW all warm-widened F, P42/P74 frozen/eval, fixed source-owned frames and per-stage residual caps');p.write_text(t,encoding='utf-8')
p=HERE/'final.py';t=p.read_text(encoding='utf-8').replace('from .model import GeometryModel','from .model import GeometryModel,DeploymentModel,deployment_weights')
t=t.replace("worker_model = GeometryModel(**state['model_config']).eval()","worker_model = DeploymentModel(**state['model_config']).eval()")
t=t.replace("worker_model.load_state_dict(state['model'])","worker_model.load_state_dict(deployment_weights(state['model']),strict=True)")
p.write_text(t,encoding='utf-8')
p=HERE/'pipeline.py';t=p.read_text(encoding='utf-8').replace('20261029_e78_center_seed0','20261029_e42_multisource_seed0');p.write_text(t,encoding='utf-8')
p=HERE/'render_drafts.py';t=p.read_text(encoding='utf-8').replace('frozen coherent E78 plus center correction in public E74 frame','small E42 placer plus warm-wide multisource F in public E42 frame');p.write_text(t,encoding='utf-8')
print('E81 phase3 registered; inherited costs and frozen protocols retained.')
