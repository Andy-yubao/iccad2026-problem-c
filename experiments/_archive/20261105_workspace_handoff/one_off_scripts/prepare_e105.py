from pathlib import Path
import json,shutil,hashlib
root=Path.cwd();src=root/'experiments/E104_scaling32k';dst=root/'experiments/E105_gated_prenorm_placer';dst.mkdir(exist_ok=False)
copies=[]
for f in [*src.glob('*.py'),src/'launch.ps1']+list((src/'depth_evaluation').rglob('*.py'))+[src/'depth_evaluation/budget.json']:
    q=dst/f.relative_to(src);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,q)
    copies.append(dict(source=f.relative_to(root).as_posix(),copy=q.relative_to(root).as_posix(),sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
for name in ['train.py','pipeline.py','launch.ps1']:
    f=dst/name;s=f.read_text(encoding='utf-8').replace('E104_scaling32k','E105_gated_prenorm_placer').replace('20261104_scaling128x16_seed0','20261105_gated128x16_seed0').replace('E104.scaling32k128x16.local_prenorm.v2','E105.gated_prenorm_swiglu.scratch.v1').replace('Started E104','Started E105')
    if name=='train.py':s=s.replace('normalize_local=True','gate_init=.1')
    f.write_text(s,encoding='utf-8')
# Preserve data/augmentation/loss/schedule/engine exactly. Only architecture and reporting vary.
f=dst/'evaluate.py';s=f.read_text(encoding='utf-8').replace("BASE=ROOT/'experiments/E98_loss_weight_learning/runs/20261104_generalization96_seed0'", "BASE=ROOT/'experiments/E104_scaling32k/runs/20261104_scaling128x16_seed0'")
s=s.replace('E98 16k / 96x12','E104 32k / 128x16').replace("'E104 32k / 128x16',32000", "'E105 32k / 128x16',32000").replace("'E104 32k / 128x16',16000", "'E104 32k / 128x16',32000")
s=s.replace('E104 32k / 128×16 终末评价','E105方案一终末评价').replace('E98增强best71已完成500结果只读复用','E104修复后best81已完成500结果只读复用').replace('E98增强16k / 96×12|E104扩大32k / 128×16','E104修复版32k / 128×16|E105方案一32k / 128×16')
lines=s.splitlines()
for i,line in enumerate(lines):
    if "text+='正常dev早停" in line:
        lines[i]="    text+='正常dev早停表示登记调度的平台，训练拟合另看曲线。E105为全程固定的方案一、随机初始化32k训练，唯一best由原固定Loss选择；E104历史best81包含step20350的local-prenorm迁移，且这里只保留其修复后曲线，因此该历史比较不能单独归因于结构，未额外启动从头控制组。本次未实施方案二/三，不拟合scaling指数。dev为反复使用的开发面板，单seed，不称独立未见泛化或正式比赛成绩；测试封存。\\n\\n'"
s='\n'.join(lines)+'\n'
s=s.replace('E104 32k 128x16 / frozen Loss-best / depth v1','E105 scheme1 / frozen Loss-best / depth v1').replace("('raw','E104 raw')","('raw','E105 raw')").replace('<title>E104 dev500</title>','<title>E105 dev500</title>')
s=s.replace("baseline=json.loads(baseline_path.read_text(encoding='utf-8'))", "if sha(BASE/'progress.json')!=plan['baseline_progress_sha256']:raise ValueError('Baseline Loss history drift')\n    baseline=json.loads(baseline_path.read_text(encoding='utf-8'))")
f.write_text(s,encoding='utf-8')
# Exactly five fixtures include the actual E104 failing member and a v6 expansion member.
fail=json.loads((src/'runs/20261104_scaling128x16_seed0/repair_20261105/case_diagnosis.json').read_text(encoding='utf-8'))['member']
v6=json.loads((root/'protocol/reference_workset_v6.json').read_text(encoding='utf-8'))
fi=next(i for i,r in enumerate(v6['sets']['train']['records'][:32000]) if r['member']==fail)
plan=json.loads((src/'plan.json').read_text(encoding='utf-8'))
plan.update(experiment=dst.name,run='20261105_gated128x16_seed0',revision='E105.gated_prenorm_swiglu.scratch.v1',architecture=dict(width=128,layers=16,rectangle=True,ffn_dropout=.10,residual_dropout=.05,gate_init=.1),authorization='User explicitly authorized implementing scheme1, matching E104 augmentation and all non-architecture settings, and background-starting once ready.',scope='one scratch scheme1 candidate; normal early-stop, freeze unique fixed Loss-best, automatic same fixed dev500 and depth solver once; report then stop',initialization='all parameters random seed0; fresh AdamW; no E104 warm-start and no extra control candidate',execution_status='preparing',latest_authorization='Implement scheme1 and directly background start; non-architecture settings match E104',training_hold='none; direct start explicitly authorized',numerical_path='new registered gated full pre-norm and SwiGLU architecture, same FP32 backward/AdamW/projection engine and numerical guards',preflight_train_indices=[1683,5474,10140,31001,fi])
for key in ['repair_migration','best_and_scheduler_migration']:plan.pop(key,None)
base=src/'runs/20261104_scaling128x16_seed0'
plan['baseline_evaluation_sha256']=hashlib.sha256((base/'evaluation_depth_dev500/results.json').read_bytes()).hexdigest();plan['baseline_progress_sha256']=hashlib.sha256((base/'progress.json').read_bytes()).hexdigest()
plan['comparison']['baseline']=base.relative_to(root).as_posix();plan['comparison']['interpretation']='E104 historical repaired best81 reused, original run changed forward path after step20350; E105 scratch scheme1 only. No clean extra scratch control authorized/launched; do not isolate causal structure effect.'
plan['comparison']['loss']='same original fixed ten E104 coefficients, not E103 learned coefficients'
plan['gates']='one sigmoid scalar per layer per attention/local/FFN/geometry branch, initialized .1, no scan'
plan['ffn']='SwiGLU middle width256=2*width; ffn_dropout .10 after gated product, residual_dropout .05 on each attention/local/FFN message; geometry feedback has no additional Dropout'
plan['geometry_feedback']='shared Linear3->128, SiLU, Linear128->128, then per-layer geometry LayerNorm and gate; state masked as E104'
plan['head']='shared head preceded by dedicated shared LayerNorm; same unconstrained3 state and public decoder'
(dst/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(dst/'provenance.json').write_text(json.dumps(dict(copies=copies,proposal='output/model_architecture_proposals_20261105.md:scheme1',private_cross_experiment_imports=False),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
f=dst/'preflight.py';s=f.read_text(encoding='utf-8').replace("rows=load_rows('train',[1683,10027,5474,10140,31001])", "rows=load_rows('train',json.loads((OWNER/'plan.json').read_text(encoding='utf-8'))['preflight_train_indices'])")
f.write_text(s,encoding='utf-8')
print(json.dumps(dict(directory=str(dst),fixtures=plan['preflight_train_indices'],e104_failure_index=fi),ensure_ascii=False))
