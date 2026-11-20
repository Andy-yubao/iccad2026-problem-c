# 布局可视化

只维护这一套：原 E6 的 Set3 分组色、无分组灰色、模块黑框。模块内标编号和 G/F/P，L/R/T/B 徽记表示当前布局下的边界满足/违反，底部统一图例。无 MIB 编号、纹理、图例或计数。网页与 PNG 采用相同配色和约束语义。

## 直接调用

在项目根目录运行，使用现有环境：

```powershell
# E119 best55：10普通例＋5最坏例，reference／真实12层滑块／原后处理
& .\.conda\envs\iccad2026\python.exe -B -m visualization --e119 --port 8779

# E120 best79：同规则10普通例＋5最坏例、真实12层滑块，同一套界面
& .\.conda\envs\iccad2026\python.exe -B -m visualization --e120 --port 8781

# E134已完成的E119两组：同一界面，显示每层新Loss与分项
& .\.conda\envs\iccad2026\python.exe -B -m visualization --data experiments/E134_supervision_interval/runs/e119_stride3_seed0/visualization_dev15_loss/data.json --port 8782
& .\.conda\envs\iccad2026\python.exe -B -m visualization --data experiments/E134_supervision_interval/runs/e119_stride2_seed0/visualization_dev15_loss/data.json --port 8783

# E135 A/B/C：每组相同普通10例＋各自最坏5例，S0–S12与逐层Loss
& .\.conda\envs\iccad2026\python.exe -B -m visualization --data experiments/E135_initial_supervision/runs/e119_A_seed0/visualization_dev15_loss/data.json --port 8784
& .\.conda\envs\iccad2026\python.exe -B -m visualization --data experiments/E135_initial_supervision/runs/e119_B_seed0/visualization_dev15_loss/data.json --port 8785
& .\.conda\envs\iccad2026\python.exe -B -m visualization --data experiments/E135_initial_supervision/runs/e119_C_seed0/visualization_dev15_loss/data.json --port 8786

# 带12层滑块的真实历史样例，同一套界面
& .\.conda\envs\iccad2026\python.exe -B -m visualization --data visualization/data/E114_saved12.json --port 8780

# 使用同一样式导出 PNG，不启动服务
& .\.conda\envs\iccad2026\python.exe -B -m visualization --e119 --case 0 --png visualization/data/E119_preview.png
& .\.conda\envs\iccad2026\python.exe -B -m visualization --data visualization/data/E114_saved12.json --case 0 --layer 6 --png visualization/data/E114_layer6_preview.png
```

网址为 `http://127.0.0.1:<port>/`，E119/E120默认端口分别为8779/8781。只监听本机，不打开浏览器。服务只开放页面及所选结果的读取接口，无训练、推理、修复、Q重评或评分写入功能。`--e119`/`--e120` 读取各自实验的 `visualization/20261113_dev15/data.json`；E119若没有这份展示数据，才浏览原dev500末层结果。后者按需读取已有dev NPZ并核对成员/权重身份，不重建训练张量。

后台运行（日志写入可重建的 `.cache/visualization/`）：

```powershell
$workspacePath = (Get-Location).Path
$viewerLogs = Join-Path $workspacePath '.cache\visualization'
New-Item -ItemType Directory -Path $viewerLogs -Force | Out-Null
$viewerProcess = Start-Process -FilePath "$workspacePath\.conda\envs\iccad2026\pythonw.exe" -ArgumentList @('-B','-m','visualization','--e119','--port','8779') -WorkingDirectory $workspacePath -WindowStyle Hidden -RedirectStandardOutput "$viewerLogs\e119.stdout.log" -RedirectStandardError "$viewerLogs\e119.stderr.log" -PassThru
$viewerProcess.Id
```

停止服务时使用上一步返回的进程号：`Stop-Process -Id <进程号>`。

## 中间层与数据接口

2026-11-16用户要求：后续含S0读出的模型展示应同时保存并显示真实S0，滑块从S0开始。S0须由模型实际head调用捕获，不能复制S1或补造缺失状态。既有只保存S1–S12的展示在后续重新采集时补充，本轮设计讨论不自动重跑采集。

E135 使用真实 S0–S12 共13个状态，显式保存 `layer_ids: [0,1,...,12]`，共享界面按实际层号显示。三组使用 seed20261114 抽取相同普通10例，排除三组最坏5例的并集；最坏例各按自己的已存最终Q排序。每层都显示共同完整新Loss；“监督目标Loss”显示实际目标，C 的 S0 为 F，并列出四项组成。均未乘层权重，未直接监督的层在目标列显示 —。末层与原评价草稿逐位一致，后处理/Q复用原件；选择和核验记录均在各 run 的 `visualization_dev15_loss/`。原始结果对比见 `experiments/E135_initial_supervision/runs/results_review.md`。

其他数据不提供 `layer_ids` 时仍按1开始编号。PNG 的 `--layer` 使用实际层号，E135 的 S0 可用 `--layer 0`；`layers` 中不得补造未保存状态。

滑块只在真实保存了多个中间层时出现，选择1–12层只改变中栏，reference 和后处理保持固定。所有层共用坐标范围与模块编号。E119原评价未保存中间层；2026-11-13用户请求展示后，仅对固定15例补存真实12层，末层与原评价逐位一致，后处理坐标原样复用。选择记录在该实验 `visualization/20261113_dev15/selection.json`：普通10例用seed20261113从排除最坏5的剩余成员中固定随机无放回抽样，最坏5按原dev500最终Q降序、index升序破同分。不训练、不重跑后处理/Q、不读test。

`E114_saved12.json` 是原 E114 best48 的固定训练5例＋原 dev 最差5例，仍只是同一界面的历史数据示例。

E134两组E119使用相同随机普通10例，各自按已完成最终Q选最坏5例，排除两组最坏例的并集后再抽普通例。每例保存12层真实草稿及 `layer_losses`，为相同public_priority_v1本层完整Loss及已计系数的分项，未乘监督层权重；这是当前单例数值，不是dev500均值。滑块同步切换Loss，表格列出12层并标示监督权重，未直接监督的层也按同一公式诊断。既有未保存Loss的数据明确显示未保存，不伪造为0。

E120 best79按相同抽样、排序和采集规则独立生成15例；最坏例按E120自己的原评价Q确定，不冒称与E119是同一组配对成员。数据/选择/校验记录在E120的同名展示目录，末层与E120原草稿逐位一致，后处理复用原坐标。

新结果使用一个 JSON，传给 `--data`，不再新增专用网页或服务器：

```json
{
  "title": "模型与checkpoint身份",
  "note": "成员选择和来源说明",
  "identity": {"checkpoint_sha256": "实际哈希"},
  "cases": [{
    "label": "样例名",
    "member": ["来源分片", 0],
    "n": 1,
    "constraints": [[0, 0, 0, 1, 0]],
    "reference": [[0, 0, 2, 3]],
    "draft": [[0.1, 0, 2, 3]],
    "final": [[0, 0, 2, 3]]
  }]
}
```

坐标为 `N×4` 的 XYWH；约束列为 `[F,P,忽略列,G,boundary_bits]`，边界位 L/R/T/B=1/2/4/8。编号从0开始。可用 `layers: [第1层XYWH, …, 第12层XYWH]` 替代 `draft`；只能放实际保存的输出。未保存 reference/final 时留空或省略，页面会明确标注。可选 `metrics` 字段为 `Q/hard_legal/complete_ms`，只显示已存值。

Python 调用为 `visualization.layouts.render(case, path, layer=6)`。已有评估脚本的 `plot_case` 调用仍可用；`layer_capture.py` 保留只读 hook；`layer_comparison.py` 只兼容旧脚本的几何统计导入，不包含第二套界面。

## 文件职责

| 文件 | 职责 |
|---|---|
| `viewer.py` / `viewer.html` | 唯一网页入口和交互；支持任意已存层数 |
| `style.py` | 统一配色、坐标校验、共享坐标范围与边界标记 |
| `layouts.py` | 同款 PNG 导出及旧调用适配 |
| `layer_capture.py` | 由调用方掌管模型前向的只读观察工具 |
| `data/` | 已存真实坐标示例与样式预览，不放另一套 UI |

旧版网页、评分器、图像实验及专用脚本已移到 `history/20261113_workspace_cleanup/visualization_legacy/`，不再作为现行入口。

## E152 v2：普通20例、Loss最差5例、Q最差5例

2026-11-18 按用户请求采集，两组共用固定普通20例，各自追加自身最佳点保存的逐题Loss最差5例及原后处理Q最差5例；极端类别重合时仍保留两个展示条目。A保留reference、best10；B无reference、best8。数据分别位于：

- `experiments/E152_s2_local_repair/runs/e152_s2_ref_seed0/visualization_normal20_loss5_q5/data.json`
- `experiments/E152_s2_local_repair/runs/e152_s2_no_ref_seed0/visualization_normal20_loss5_q5/data.json`

服务启动后本机端口为8812（A）和8813（B）。统一入口仍为 `python -B -m visualization --data <对应data.json> --port <端口>`。两组完整统计与采集记录在 `experiments/E152_s2_local_repair/runs/review_dev500_v1/`。

本模型的功能单元S0/S1/S2与历史展示的逐层S编号不同：L0为初始布局，S0对应L0→L4，S1对应L4→L8，S2对应L8→L12。数据可用可选 `layer_labels`（与真实 `layer_ids` 一一对应）标清这些含义；旧数据不设置时仍沿用原编号。两组都保存真实L0–L12，不把“仅S2可训练”误作“只显示S2”。逐题记录含完整换侧与同组近接触边变化诊断；这些几何指标不能单独证明拓扑不变。后处理与Q直接复用，不重跑求解器。

## E156 暂停最新权重的训练10例

`experiments/E156_two_s0_stage/runs/visualization_train10_latest_v1/data.json` 使用暂停时latest（step26942），每个N区间随机一个已见训练例，保存实际L0/L2/L4。端口8824。Reference为形状修正监督，不保证合法；未推理未训练后层、未后处理，训练继续暂停。同目录png含10张Reference/L0/L2/L4四栏图及zip。

可选字段 `hide_missing_final:true` 让网页与render隐藏未保存的后处理栏；`reference_label` 标记参考含义；`loss_columns:[[key,label],...]`、`loss_label`、`loss_caption` 可明确当前实验的Loss口径，未设置时保持原列和文案。不新增网页入口。
