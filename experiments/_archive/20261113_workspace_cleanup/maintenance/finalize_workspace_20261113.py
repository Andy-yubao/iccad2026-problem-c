import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT/'history/20261113_workspace_cleanup'
documents={
'experiments/E119_global_initial_generator/README.md':'''# E119 global2

当前主要研究基线为scratch seed0、36k、712275参数的global2 best55：两层初始全局交互＋12层几何主干/共享头，监督.2L4+.8L12，FP32/batch32。原面积、无MIB、全模块预测，固定规格由后处理满足。

训练57轮正常早停，原dev500末层Loss0.333128；原E98.depth_budget.v1后处理混合Q1.021568，最终500/500硬合法，草稿0/500。来源与身份见 `plan.json`、`frozen_model.json` 和 `protocol/research_baseline_v3.json`；原件在 `runs/global2_seed0/`，队列complete。

- 原dev500：`runs/global2_seed0/evaluation_no_mib_dev500/`。
- 历史三倍工作预算：`evaluation_budget500ms/`，不替换默认预算。
- 已授权独立test500：`evaluation_test500_20261109/recovery1_20261109/output/`，已完成且曝光，其余1500例继续封存。
- 独立新Loss变体：`variants/public_priority_v1/`，不是原基线权重。

浏览已存结果统一调用根目录 `python -B -m visualization --e119 --port 8779`，说明见 `visualization/README.md`。原评价仅保存末层草稿，没有12层缓存。旧E118串行启动命令属于已完成历史，不能作为新启动授权；不自动恢复、重训、Q选模或追加测试。
''',
'experiments/E120_node12_collision/README.md':'''# E120 Node12-Collision

E119结构对照：保留两层初始全局交互、12层几何主干/共享头，以每层Node12动态几何投影注入Q/K，V保持。tau=.25、每头缩放初值.1可学习；公式见 `formula_spec.md`。725619参数，仍O(N²)。

单组scratch seed0/36k/.2L4+.8L12、FP32/batch32，原面积、无MIB、全部模块预测。共享完整CPU静态缓存，动态几何逐层计算。具体冻结配置、源码和门禁见 `plan.json` 与run登记。

训练和原固定dev500已完成，队列complete；混合Q1.039004，最终500/500硬合法、草稿0/500，未改善原E119。结果在 `runs/node12_seed0/evaluation_no_mib_dev500/`，历史三倍工作预算另存 `evaluation_budget500ms/`。旧启动命令不构成重训/恢复/测试授权。

新布局展示统一使用 `visualization/README.md` 中的已存JSON接口，不新增专用网页。
''',
'experiments/E131_relation_input/README.md':'''# E131 关系输入64×4（已停止）

两组full/pure共用400→128→64编码、64×4/4头/SwiGLU128及128→64→3读出；full保留关系偏置和连接/G聚合，pure块仅接收h/valid。参数268387/235331。固定16k、FP32/batch32、原面积/无MIB/全模块预测，公开地址按登记随机流重命名，不保证任意重命名不变性。

2026-11-12用户先授权启动，后明确停止并改做独立E13296×6。当前E131状态为user_stop：full完成93轮/46929接受步，pure未启动；完整model/Adam/RNG/排列/游标/调度及累计2108.6928秒保留。STOP/QUEUE_STOP和禁用授权保持，不能因旧启动授权恢复。

配置见 `plan.json`，最新授权/停止记录见 `authorization.json`、`queue_state.json`，训练证据在runs与launch_logs。此前“仅准备”与“已启动”是历史阶段。本实验没有后处理/Q/test；E132是全新scratch，未继承本组权重。
''',
'experiments/E132_relation_input_96x6/README.md':'''# E132 关系输入96×6

E131停止后的独立scratch对照。full/pure共用400→192→96编码、96×6/4头/SwiGLU192及192→96→3标准读出；参数786195/674979。固定16k、原train500/dev500、FP32/batch32、原增强/LR/数值门禁；原始面积、无MIB、F/P全预测不覆盖。共享完整CPU缓存，未加载E131权重/Adam。

队列已complete，两组均正常dev早停：

| arm | 完成轮数 | best epoch | own dev Loss |
|---|---:|---:|---:|
| full | 153 | 145 | 0.403183 |
| pure | 144 | 136 | 0.775958 |

原件在 `runs/full_seed0/results.json` 与 `runs/pure_seed0/results.json`，最新状态见 `queue_state.json`。两组均未观察到train平台，不称充分拟合。只有训练授权，未做后处理/Q/test；Loss差异不等于最终质量差异。错误/用户停止不自动重试，不追加seed或恢复E131。
''',
'experiments/E133_flow_matching/README.md':'''# E133 Flow Matching

单组scratch seed0、920643参数：400→192→96公开编码＋96×6纯Attention，时间/状态条件与AdaLN，共享速度读出。固定16k、原train500/dev500、FP32/batch32；原始面积、无MIB、F/P全预测。训练纯速度MSE，独立高斯到reference直线路径。具体公式/门禁/增强与总8h预算见 `plan.json`、`pipeline_budget.json` 及run配置。

## 已完成结果

训练CUDA捕获在零正式更新边界失败，登记改为eager FP32；原模型/目标/Adam/门禁及预算起点保留。正式训练247轮/123500步正常早停，唯一FM Loss-best239，dev Loss0.415453，权重SHA256：

`1510d786df358ad3e4390596b948dd2c64289945b7f4639ef1ed9acef80d352f`

原自动评价因重复dict字段在0例失败，`queue_state.json` 保留needs_attention现场。用户后续明确请求后，以独立适配器和独立1800秒只读预算完成原dev500：Euler8、每例唯一固定噪声、原E98.depth_budget.v1；最终500/500硬合法、0回退，草稿0/500合法，混合Q1.157321，完整均时206.4ms/P95499.3ms。弱于E119历史基线，但16k对36k/单seed/目标不同不支持FM普遍无效的因果结论。4/16步仅五训练例诊断，未做Q选步数。

## 证据位置

- 训练：`runs/fm_seed0/`；正式日志：`launch_logs/20261112_seed0_prepared/`。
- 独立成功评价与完整报告：`evaluation_quality_20261113/`，逐例数据在其 `dev500/cases/`。
- 启动登记：`launch_registration_20261112/`；初次捕获失败原件：`preparation_diagnostics/startup_capture_failure1/`。

上述两处记录和独立评价于2026-11-13从根output归回本实验，字节/哈希核对见 `history/20261113_workspace_cleanup/manifest.json`。冻结源码里的旧output路径保留历史原样，不能直接用于新运行；本次未重写原监督器/失败队列。新运行需另行登记入口与授权，不自动重试、重训、追加候选或读取test。
'''
}
for name,body in documents.items():
    path=ROOT/name
    before=ARCHIVE/'documents_before'/name
    before.parent.mkdir(parents=True,exist_ok=True)
    if not before.exists():shutil.copy2(path,before)
    path.write_text(body,encoding='utf-8')

# Keep the directory index navigable without repeating every deep path three times.
catalog='''# 实验位置索引

2026-11-13核对。目录存在不代表可以启动；状态和权重身份看实验自身记录。

## 现有实验

| 实验 | 实际目录 |
|---|---|
'''
for p in sorted((ROOT/'experiments').iterdir()):
    if p.is_dir() and p.name.startswith('E'):
        catalog+=f'| {p.name.split("_")[0]} | [{p.name}](../experiments/{p.name}/) |\n'
catalog+='''
## 近期归档

下列29个实验统一位于 [history/20261111_experiments_consolidation/experiments/](../history/20261111_experiments_consolidation/experiments/)，完整迁移/哈希见其 [归档说明](../history/20261111_experiments_consolidation/README.md)。

| 实验 | 该归档内目录 |
|---|---|
'''
base=ROOT/'history/20261111_experiments_consolidation/experiments'
for p in sorted(base.iterdir(),key=lambda p:int(p.name.split('_')[0][1:]) if p.name.startswith('E') else -1):
    if p.is_dir() and p.name.startswith('E'):
        catalog+=f'| {p.name.split("_")[0]} | `{p.name}/` |\n'
catalog+='''
竞赛交付是E101原C5＋完整后处理，E119是研究基线。E101部分隔离运行根超过MAX_PATH，读取使用Windows长路径。更早E1–E97按归档事件导航见 [history/README](../history/README.md)，不把旧路径/授权当现行入口。

2026-11-13的旧文档与全部旧可视化见 [本次清理归档](../history/20261113_workspace_cleanup/README.md)。现行展示只调用 `visualization`；历史实验私有显示脚本不作为第二套默认。
'''
(ROOT/'docs/experiment_catalog.md').write_text(catalog,encoding='utf-8')

old=ARCHIVE/'documents_before'
counts={}
for name in ['AGENTS.md','README.md','docs']:
    before=old/name;after=ROOT/name
    getsize=lambda p:sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) if p.is_dir() else p.stat().st_size
    counts[name]=dict(before_bytes=getsize(before),after_bytes=getsize(after))
report=json.loads((ARCHIVE/'validation.json').read_text())
report.update(document_sizes=counts,browser_verified_layers=[1,6],browser_errors=0,
              screenshot='visualization/data/slider_preview.jpg',e119_service='http://127.0.0.1:8779/',
              example_service='http://127.0.0.1:8780/')
(ARCHIVE/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
maintenance=ARCHIVE/'maintenance'
maintenance.mkdir(exist_ok=True)
for name in ['cleanup_workspace_20261113.py','refresh_docs_20261113.py','finalize_workspace_20261113.py']:
    src=ROOT/'tmp'/name;dst=maintenance/name
    assert src.resolve().is_relative_to(ROOT) and dst.resolve().is_relative_to(ROOT)
    assert not dst.exists()
    src.rename(dst)
print(json.dumps(counts,ensure_ascii=False))
