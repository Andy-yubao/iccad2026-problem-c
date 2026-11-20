from pathlib import Path
import json,datetime,hashlib
import torch
root=Path.cwd();owner=root/'experiments/E103_offline_loss_alignment'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
run=root/'experiments/E98_loss_weight_learning/runs/20261104_dynamic_train500_v3_seed0'
state=read(run/'state.json');assert state['status']=='user_stop'
branch=run/'rounds/r03_worst_overlap_vs_mean_overlap/branches/plus'
latest=branch/'checkpoints/latest.pt';best=branch/'checkpoints/best.pt'
s=torch.load(latest,map_location='cpu',weights_only=False)
assert s['status']=='user_stop' and not s['finished']
assert s['progress']['step']==1149 and s['progress']['cursor']==4768
assert (run/'STOP').exists() and best.exists()
assert not (branch.parent/'minus').exists()
assert not any((run/'rounds').glob('r04_*'))
(root/'experiments/E98_loss_weight_learning/dynamic_weight_learning_v3/STOP').write_text('User abandoned this search; no automatic resumption.\n',encoding='utf-8')
verification=dict(status='stopped_and_abandoned',verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),run=run.relative_to(root).as_posix(),round=3,arm='plus',completed_epochs=s['progress']['epoch'],updates=s['progress']['step'],cursor=s['progress']['cursor'],checkpoint_status=s['status'],latest=latest.relative_to(root).as_posix(),latest_sha256=sha(latest),best=best.relative_to(root).as_posix(),best_sha256=sha(best),latest_and_best_preserved=True,supervisor_session=64386,supervisor_exit_code=0,process_exit_evidence='Supervised launch session returned exit0 after STOP; terminal state user_stop; GPU compute process list empty',next_minus_started=False,next_round_started=False,automatic_resume=False,test_read=False)
save(run/'stop_verification_20261104.json',verification)
save(owner/'e98_stop_verification.json',verification)
plan=dict(experiment='E103_offline_loss_alignment',status='models_staged_awaiting_user_review',authorization='User requested abandoning E98, opening independent research and selecting/staging suitable fully trained models first; report model choices before database generation.',
 objective='Learn and validate differentiable draft Loss aligned with final quality after one frozen fixed-depth postprocessor; offline within-problem ranking, local direction checks, then limited matched-start training only after objective registration.',
 database=dict(planned_cases=1000,drafts_per_case=5,planned_core_drafts=5000,population='original fixed v5 train16000; keep original split membership; sample same1000 without replacement after selection review',source_model_ids=['e42','e68','e70','e71','e98_generalization'],draft_rule='Exactly one unperturbed neural draft from each of five distinct completed model weights for every case; do not use five checkpoints from one run or perturbations to fill quota',perturbations='Optional later explicitly registered auxiliary checks; not used to replace any of the five core model drafts',all_models_completed_normal_early_stop=True,convergence_scope='Operational validation-based stopping; does not claim global convergence or flat training Loss',sampling_started=False),
 scope=dict(current_stage='model selection, independent weight/source/evidence copies and CPU strict-load verification only',database_generation_started=False,inference_calls=0,postprocessing_calls=0,loss_fitting_started=False,formal_training_started=False,test_read=False,dev_postprocessing_Q_for_weight_fitting=False,automatic_wakeups=False),
 public_task='Public MIB mean-area transform from protocol, same IDs and scale for all sources; reference used only for Loss/supervision, never network inputs',
 archived_port_representation='For historical E68/E70/E71 frozen inference only, retain their original public full-port readers and14-dimensional input; compressing their inputs would change the selected function. New trainable architecture is not introduced here.',
 evaluation='Later stage will freeze an independent local fixed-depth evaluator copied with provenance; regenerate all5000 Q labels with that one evaluator rather than mixing historical clock/depth Q',
 analysis='Store actual independent Loss terms; fit nonnegative scale-constrained regularized within-case ranking weights. Split by case and examine leave-one-model-source-out behavior; mixed-Q objective includes equal and official size weighting.',
 decision='Check unseen-case ranking/regret and local Loss-descent/Q direction before actual training; fixed own-Loss controls within-phase monitoring/best/LR/earlystop; Q is the outer alignment criterion, not hidden checkpoint selection.',
 protection='All E98 STOPs and historical evidence remain; no E98 restarts; original model paths remain unchanged; E101/E102 untouched.',
 next_stage='Report five model choices to user first; no database or training automatically launched in this turn.')
save(owner/'plan.json',plan)
(owner/'.gitignore').write_text('__pycache__/\nassets/\nruns/\ndatabase/\n*.log\n',encoding='utf-8')
(owner/'README.md').write_text('''# E103：从后处理数据学习训练Loss

当前只完成模型选择和独立副本整理，先向用户汇报；未读训练/验证/测试载荷，未生成草稿、后处理Q或拟合权重，不启动新训练。E98调权搜索已按用户要求安全停止并放弃，不自动恢复。

研究目标：使训练Loss能够引导模型输出在固定后处理器下具有更低混合Q的草稿。路线为同题多模型草稿数据库 → 同题排序/受约束权重拟合 → 局部下降方向检查 → 少量同父模型真实训练验证。离线排序好不自动等于训练有效。

## 数据库规格

原固定train16k中统一抽1000题，每题分别由5个充分训练且权重不同的模型各生成1份原始草稿，共5000份。此阶段不抽样、不生成。禁止用同一模型5个早期checkpoint或4份扰动凑数量。小幅扰动仅可作为后续另登记的辅助检查，保留固定位置、固定形状、面积与公开约束。

所有草稿在同一公开MIB平均面积任务、ID与尺度下比较；后处理Q标签将由同一独立冻结计算深度评价器重新生成，不拼接历史不同后处理器/时钟预算的Q。各Loss项按同一函数重新计算，保留reference监督但不将reference喂给网络。拟合与检查按题目分隔，并检查留出模型来源，测试封存。

## 五个拟选来源

|模型|正常早停轮数|所取best|选择理由|
|---|---:|---:|---|
|E42|54|51|成熟历史reference/几何基线，作为经典草稿来源|
|E68|57|54|完整公开端口表示＋原G目标，提供输入表示差异|
|E70|74|71|独立从头训练的任务Loss，补充避碰/线长/紧凑取舍|
|E71|70|67|密度项与reference课程，补充空间组织和紧凑取舍|
|E98泛化增强基线|71|71|轴交换及增强Dropout，提供当前成熟泛化基线|

五者均完成其原始验证规则的正常早停；这说明不是未训完的中间候选，不证明全局最优或训练Loss完全平台。E68/E70/E71共享架构但为不同训练目标的独立从头训练权重；总体有两类端口表示及不同目标/泛化方式。没有使用已放弃的E98调权分支或单步候选。

`assets/`保存逐字节相同的best完整副本；`sources/`保存匹配checkpoint训练身份的模型、特征、解码和必要端口源码；`evidence/`保存原results/config副本；`model_manifest.json`登记来源和哈希。原文件未移动、未修改。五个模型已通过CPU严格权重加载和有限值检查，尚未进行任何样本推理。历史完整端口模型必须保留其原输入才能忠实复现；这仅是冻结来源推理，不引入新的训练架构。

计划见[plan.json](plan.json)，模型清单见[model_manifest.json](model_manifest.json)，停止证据见[e98_stop_verification.json](e98_stop_verification.json)。本阶段到模型汇报为止，无自动后台任务。
''',encoding='utf-8')
# Update current boundaries, preserving past source/run evidence and unrelated user work.
p=root/'AGENTS.md';s=p.read_text(encoding='utf-8')
a=s.index('用户最新授权E98从泛化增强best71');b=s.index('除此不得擅自新增候选',a)
s=s[:a]+'用户已放弃并要求停止E98调权搜索，不自动恢复；另授权E103离线Loss对齐研究，当前仅挑选并复制5个已充分训练的不同模型，先报告选择理由；后续数据库为原train16k同1000题×每题5模型各1草稿，共5000份，扰动不替代模型数量，模型汇报前不生成数据库或启动新训练。'+s[b:]
a=s.index('E98旧搜索结果不采用');b=s.index('\n- GPU工程',a)
s=s[:a]+'E98所有旧搜索及单步调权结果不采用，原STOP保留；新train500调权运行亦已安全停止并放弃，不自动恢复。E103当前阶段仅模型整理与汇报，不自动启动数据库、拟合或长训练。'+s[b:]
p.write_text(s,encoding='utf-8')
for path,relative in [('README.md',''),('docs/protocol.md','../')]:
 p=root/path;s=p.read_text(encoding='utf-8')
 marker='用户最新授权[E98 train500调权重启]'
 a=s.index(marker);b=s.index('\n',a)
 replacement=f'用户最新已要求停止并放弃E98调权方向；train500运行在第3轮plus第1149步安全保存，进程退出，后续分支未启动，STOP保留，不自动恢复。[停止核验]({relative}experiments/E98_loss_weight_learning/runs/20261104_dynamic_train500_v3_seed0/stop_verification_20261104.json)。另授权[E103离线Loss对齐研究]({relative}experiments/E103_offline_loss_alignment/README.md)：当前仅选定并复制E42/E68/E70/E71/E98泛化增强基线的已完成best及推理源码，先报告模型理由；后续按同1000训练题×5模型建立5000草稿数据库。未启动数据库、后处理或训练，不影响E101/E102。'
 s=s[:a]+replacement+s[b:];p.write_text(s,encoding='utf-8')
p=root/'docs/workspace_structure.md';s=p.read_text(encoding='utf-8');a=s.index('E98最新授权入口为dynamic_weight_learning_v3');s=s[:a]+'E98调权研究已停止并放弃，所有STOP与权重证据保留，不自动恢复。新增E103_offline_loss_alignment独立研究区，assets保存五个完成训练模型的best副本，sources保存对应推理源码，evidence保存原训练结果。当前仅模型整理与汇报，不启动1000×5数据库或训练。\n';p.write_text(s,encoding='utf-8')
p=root/'experiments/README.md';s=p.read_text(encoding='utf-8');s+='\nE98调权方向已按用户要求停止并放弃，不自动恢复。新增[E103离线Loss对齐](E103_offline_loss_alignment/README.md)：先整理五个充分训练模型并汇报，后续同1000训练题×5模型；目前无新后台任务。\n';p.write_text(s,encoding='utf-8')
print('Stopped E98 verified at1149; staged E103 plan and five-model evidence; no database generation.')
