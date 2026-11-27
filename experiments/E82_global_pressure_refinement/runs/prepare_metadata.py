"""Declare an information hypothesis, preserving E81 and the same fixed Loss."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=ROOT/'experiments/E82_global_pressure_refinement';OLD=ROOT/'experiments/E81_multisource_refinement'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
protected=read(OLD/'provenance.json')['protected_checkpoints_sha256'].copy()
folder=OLD/'runs/20261029_e42_multisource_seed0'
for p in [folder/'checkpoints/best.pt',folder/'checkpoints/latest.pt',folder/'deployment.pt',folder/'correction_F.pt']:
 protected[p.relative_to(ROOT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
parent=folder/'checkpoints/best.pt'
save(HERE/'provenance.json',{'experiment':'E82_global_pressure_refinement','source_experiment':'experiments/E81_multisource_refinement',
 'initialization_checkpoint':parent.relative_to(ROOT).as_posix(),'initialization_sha256':hashlib.sha256(parent.read_bytes()).hexdigest(),
 'protected_checkpoints_sha256':protected,'copy_note':'All E81 placers and F tensors preserved, only3 zero-initialized public-context embeddings4-to192 learned. Same two-source fixed Loss, source/context algorithms copied independently.',
 'pretraining_completed_seconds':3886.089079700003,'source_best_epoch':26,'test_read':False})
plan={'experiment':'E82_global_pressure_refinement','phase':3,'phase_candidate':2,'authorization':'Existing latest explicit user authorization for autonomous modular long research',
 'seed':0,'train':32000,'dev_monitor':500,'batch_size':32,'draft_views_per_case':2,'max_hours':2,'max_epochs':200,
 'architecture':'All E81 P42/P74 and F weights frozen. Inject a learned zero4-to192 pressure embedding before each existing F node SiLU/LayerNorm. Same four public features shared across three stages, computed once from each frozen placer draft.',
 'new_trainable_parameters':2304,'F_total_parameters':243851,'deployment_parameters':1715390,'training_model_parameters':7545329,
 'source_weight_mode':'Old F dropout disabled/eval. New pressure embedding dropout.05 active in training; old source NN parameters and placers bit-exact frozen.',
 'context':'Selected mode0 source topology, frame lower/upper bounds and public preplaced anchors,16 max/min relaxations. Four node features=(needed_lower-xy,xy-needed_upper)/sqrt(public area), clamp[-8,8], padding0, detached. No reference, solver or labels. No gradients or STE through source order.',
 'initial_function':'E81 within checked FP32 tolerance on P42/P74 due zero embeddings; inherited full F reused, no random new correction stage.',
 'hypothesis':'E81 improved raw overlap, mean and worstQ vsE42, but432/500 fixed-shape anchor certificates still fail and six rescues remain;164 still29/29 primary failures and329 newly27/27 failures. Explicit global path pressure may help a locally informed F without another width increase.',
 'counterevidence':'Better fixed-shape certificates do not imply betterQ,329 and219 counterexamples. Source topology/dimensions can be wrong; context is truncated and clamped and does not certify SOCP feasibility. This experiment tests input information, not proof of a unique bottleneck.',
 'fixed_loss':'EXACT E81 per-source G+4I+5E+.25A, same case mean+.25*(max-mean); same train/dev/full-dev/best, no ramp or coefficient change.',
 'optimizer':'Fresh AdamW only2304 new weights, clip1/decay.0003, original four LR and early-stop; all E81 states/costs preserved, not an exact optimizer continuation.',
 'limits':'Context computed on raw placer before later shape coordination, may be noisy for MIB; old F frozen may limit adaptability. Extra interval passes may increase latency. Per-stage residual bounds precede projection/MIB, not a global total bound.',
 'evaluation':'Same frozen32k/dev panels, one own-best P42 deployment terminal500 only, full dev2000 same two-source Loss, fixed100 gallery; read-only E81/E80/E42 comparisons, no duplicate terminals, tests sealed.'}
save(HERE/'plan.json',plan)
# Correct factual identity/report fields before preflight.
p=HERE/'train.py';t=p.read_text(encoding='utf-8')
t=t.replace("E82.smallP_warmwideF_two_source.v1","E82.public_interval_input_adapters.v1")
t=t.replace('P42/P74 frozen/eval; all warm widened F trainable with dropout.05/decay.0003; source-owned public frames','All E81 P42/P74 and F frozen/eval; only3 zero public-pressure embeddings trainable, their dropout.05/decay.0003; source-owned frames')
t=t.replace('all widened F Adam; P42/P74 frozen, own-source frame and per-stage.15 residual cap','Only2304 new pressure weights Adam; all E81 weights fixed, own-source frame and per-stage.15 cap')
p.write_text(t,encoding='utf-8')
p=HERE/'step_control.py';t=p.read_text(encoding='utf-8').replace('Plain AdamW all warm-widened F, P42/P74 frozen/eval, fixed source-owned frames and per-stage residual caps','Plain AdamW only2304 zero pressure embeddings, all E81 weights frozen/eval, fixed source-owned frames');p.write_text(t,encoding='utf-8')
p=HERE/'report.py';t=p.read_text(encoding='utf-8').replace("'trainable_refiner_parameters':243851","'correction_model_parameters':243851,'trainable_pressure_parameters':2304")
t=t.replace("baseline_paths={'E42':","baseline_paths={'E81':'E81_multisource_refinement/runs/20261029_e42_multisource_seed0','E42':")
t=t.replace('扩大F、两来源训练、联合形状优化和通道压力同时改变，只能先报告组合效果。','保持E81全部权重与同一Loss，仅学习全局公共通道信息的输入适配器。输入特征是否充分仍须反例检验。')
p.write_text(t,encoding='utf-8')
p=HERE/'render_drafts.py';t=p.read_text(encoding='utf-8').replace('small E42 placer plus warm-wide multisource F in public E42 frame','frozen E81 plus global-pressure input adapters in public E42 frame');p.write_text(t,encoding='utf-8')

# Five-case preflight independent copy, with correct parent/model mapping.
t=(OLD/'preflight.py').read_text(encoding='utf-8').replace('E81_multisource_refinement','E82_global_pressure_refinement').replace('E81','E82')
t=t.replace('from .old_model import GeometryModel as OldModel,extract','from .old_model import GeometryModel as OldModel')
start=t.index("    p=provenance()['parents']");end=t.index('    try:model(g)',start)
new='''    old=OldModel(**source['model_config']).cuda().eval();old.load_state_dict(source['model'],strict=True)
    assert sum(p.numel() for p in model.parameters())==7545329
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==2304
    with torch.no_grad():
        state=model(pub);previous=old(pub);first=previous[...,:3];second=previous[...,3:]
        torch.testing.assert_close(state,previous,rtol=1e-5,atol=2e-6)
        geometry(state[...,:3],g,model.placer(pub));geometry(state[...,3:],g,model.alternative(pub))
        initial_error=[float((state[...,:3]-first).abs().max()),float((state[...,3:]-second).abs().max())]
'''
t=t[:start]+new+t[end:]
t=t.replace('==1713086','==1715390')
t=t.replace('old_small,old_large,placer,state','old,state').replace('mean,duplicated,direct;clear()','mean,duplicated,direct,previous;clear()')
t=t.replace("if key.startswith(('placer.','alternative.')):","if '.pressure.' not in key:")
t=t.replace("'initial_F_all_stages_finite_and_trainable':True","'all_E81_F_weights_frozen_only_pressure_trainable':True")
t=t.replace("'deployment_parameters':1713086,'trainable_F_parameters':241547","'deployment_parameters':1715390,'trainable_pressure_parameters':2304,'F_total_parameters':243851")
t=t.replace('warm_widened_initial_max_state_error_P42_P74','zero_context_adapter_initial_max_state_error_P42_P74')
t=t.replace('E82 five-case warm-wide multisource preflight PASSED','E82 five-case global-pressure adapter preflight PASSED')
t=t.replace("    return {'penalty':float(a),'ordinary_shape_gradient':True,'ordinary_center_gradient_zero':True}","""    from .global_context import interval_context
    frame=(torch.zeros(1,2,device='cuda'),torch.tensor([[3.,1.]],device='cuda'))
    context=interval_context(s,g,frame)
    assert context.shape==(1,3,4) and not context.requires_grad and torch.isfinite(context).all()
    assert float(context[0,2,0])>.49 and float(context[0,1,2])>.49
    return {'penalty':float(a),'ordinary_shape_gradient':True,'ordinary_center_gradient_zero':True,'global_context_lower_upper_signs':True}
""")
t=t.replace('model.train();assert not model.placer.training and not model.alternative.training and model.refiner.training',"""model.train();assert not model.placer.training and not model.alternative.training and model.refiner.training
    for name in ('first','second','third'):
        head=getattr(model.refiner,name);assert not head.node.training and not head.output.training and head.pressure.training
""")
(HERE/'preflight.py').write_text(t,encoding='utf-8')
study=read(OLD/'study.json');study.update(status='E82_preflight_preparation',current_experiment=HERE.name,current_stage='preflight_preparation');save(OLD/'study.json',study)
print('E82 registered as phase3 candidate2; prior charges retained, fixed Loss unchanged.')
