# 数据集使用与恢复

训练数据与官方公开验证数据已导入；来源版本、大小与 SHA-256 见 [数据清单](../datasets/datasets_manifest.json)。不要重复下载或解压已有内容。

## 当前位置

| 内容 | 当前路径 | 用途 |
|---|---|---|
| 原训练数据 | `datasets/floorset_lite/worker_*/layouts_*.th` | 9,000 分片，按冻结索引读取 |
| 官方公开验证 | `datasets/LiteTensorDataTest/config_21..config_120/` | 100 例，兼容与公开对照 |
| 原始压缩包 | `datasets/archives/` | 保留原件供恢复，不是临时下载缓存 |
| 数据根配置 | [protocol/paths.json](../protocol/paths.json) | 当前物理位置，供数据模块解析 |
| 冻结研究划分 | [split_v1.json](../protocol/split_v1.json) | 唯一 80/10/10 成员与面板索引 |

训练 7,200、开发 900、本地测试 900 个完整分片；不复制三套数据。`split_v1.json` 的旧根字段仅保存历史位置，读取器使用 paths.json；冻结文件字节与分片分配保持不变。详见 [执行协议](protocol.md)。

## 可复用代码

`datasets.paths.dataset_paths()` 返回实际训练/公开验证根，缺失时直接报错；`train_index()` 返回冻结训练白名单并检查交叉。`datasets.cases.load_frozen_cases()` 只开放固定 dev 或公开兼容面板，limit=1..5，禁止 local_test。

样本读取、标签隔离与 fixed/preplaced 输入遮罩属于本模块；图特征、训练采样、增强与监督目标归各实验的 data.py。自由块标签不能进入求解器；fixed/preplaced 公开规格属于合法输入。

```powershell
# 只检查路径和冻结成员，不读取数据张量
python -B scripts/make_splits.py
```

## 数据来源与必要恢复

官方数据仓库 `IntelLabs/FloorSet` 固定版本 `7dc7c1bec07bcae583a274f5ed95e5be399cb8db`。训练使用 `LiteTensorData_v2.tar.gz`（6,627,425,876 bytes），公开验证使用 `LiteTensorDataTest.tar.gz`（69,379,249 bytes）；不混入旧版或 Prime 数据。

仅在数据确实缺失、需要恢复时执行：

```powershell
.\scripts\download_datasets.ps1 -Set All -Extract -WhatIf
# 核实确实需要恢复后，由使用者去掉 -WhatIf
```

脚本保存压缩包到 datasets/archives，验证官方大小/哈希后才解压到 datasets；中断下载用 .part 续传，解压后保留压缩包。它可能重新解压已有内容，日常检查不要运行该恢复命令。

原始 [训练安装证据](../datasets/training_installation.json)、[公开验证安装证据](../datasets/validation_installation.json)保持原样，其中记录的是迁移前路径。当前位置由数据模块和 [迁移清单](../history/20261016_workspace_migration/manifest.json)解释；历史哈希不作为新一次联网核验的证明。用户 Downloads 中的原件不受工作区搬迁影响。

## 官方接口注意事项

- 官方训练加载器默认遍历全量池，直接调用会混入 dev/local_test；训练必须使用冻结 train 白名单。
- 官方加载器的 data_path 应指向 datasets 根；第三方代码与数据现在分离，不再从官方示例的 cwd 推断数据位置。入口先检查本地文件，避免触发官方自动下载。
- 训练标签 fp_sol 为 `(width,height,x,y)`，输出与公开规格为 `(x,y,width,height)`；公开验证标签是多边形，不能套用训练字段顺序。
- 原数据仅在可信来源下使用兼容的 torch.load 设置；不修改上游归档解决兼容问题。
- worker 编号不是块数。五例开发面板只支持排错和机制初筛，不能据此宣称泛化；正式本地测试必须在方案冻结后由用户明确发起。
