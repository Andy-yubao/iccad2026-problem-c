# E58：v6训练数据扩展至64k与缓存准备

2026-10-26用户要求准备新一版数据、处理MIB并做好cache。建议先扩到64,000例，保留16k／32k／64k嵌套视图，开发2000、封存测试2000及两组固定monitor500成员和顺序不变。这里是数据准备，不启动新模型训练，也不改变正在运行的E57。

## 为什么先到64k

现有16k附近仍有数据收益，扩大4倍能检验趋势是否延续，并保留32k中间档。它不是“64k足够”的承诺。64k较16k需要更多训练计算，后续比较必须记录样本暴露与更新步数；固定开发／测试数量不会随训练量扩大而重切分，原始split_v1的分片角色始终不变。

新索引：[protocol/reference_workset_v6.json](../../protocol/reference_workset_v6.json)。原v5和所有冻结协议原件保持字节不变。原16k的记录及顺序为新train的前16000例，追加16000形成32k，再追加32000形成64k；每档实际模块数分布严格为原训练分布的1／2／4倍。不依照Q或自由参考位置筛选，不从开发或测试池扩充训练。

候选只读复用原E16已完成的公开兼容性扫描元数据，训练池有395,061个兼容候选，64k所需各档容量均充足。选定原分片只核对既有大小／mtime，不重新读取测试张量、不全量重哈希原数据，不重做9000片扫描。

## MIB与reference边界

沿用E14.mean_area.v1：正MIB组内使用公开面积算术平均，组总面积在FP32精度内保留；固定／预放置的尺寸和坐标规格原样保留。锁定尺寸之间不同或平均面积与锁定规格不兼容的候选已由原公开扫描拒收，实际构建时再次调用同一转换并核验组面积一致、总面积误差和锁定规格。

reference继续使用原布局中心与log宽高比，不重写为新几何。平均面积变换后的reference不一定是派生任务的合法布局，不能把原标签称为修正后合法解。新48k的MIB核验统计与复用旧16k分别披露。

## 缓存和使用

新缓存：`runs/20261026_reference_v6_64k/cache/`。E42兼容features／batches独立复制，语义哈希和旧缓存一致；新48k构建CPU张量分片。原train16k与dev2k缓存仅通过清单引用，不复制、不修改，且保留其v5来源身份。不会建立测试缓存。

复用数据读取器：[datasets.expanded_reference](../../datasets/expanded_reference/__init__.py)；现有v5读取器、默认registry、共享根模块和docs/protocol都没有为本任务修改，避免使正在运行的E57源码冻结校验失效。后续训练必须显式选择v6与64k数据规模，不可直接把旧16k入口指向新索引；64k也不能假设仍适合整库GPU常驻，需按实际显存设计加载。

```python
from datasets.expanded_reference import ExpandedReferenceLayouts
data = ExpandedReferenceLayouts('train', train_size=64000)
# 本实验提供兼容张量读取；后续模型实验应核对输入特征语义并显式接入。
from experiments.E58_training_data_v6.cache import load_rows
rows = load_rows('train', train_size=32000)
```

上面的cache读取是数据接口，不包含模型、Loss或优化算法；未来实验也可复制该薄读取入口，并直接引用唯一缓存。

## 执行、停止和完成状态

索引已独立准备。缓存流程会等E57整体完成再启动，保护其冻结GPU训练和flash时间证据；随后用4个spawn CPU进程、每进程1库线程构建并验证。预计构建约10–30分钟，排队时间另计，实际以progress为准，不在等待期间偷偷训练模型。

```powershell
& 'E:\project\ICCAD\2026\problemC\.conda\envs\iccad2026\python.exe' -u -B -m experiments.E58_training_data_v6.pipeline --resume
```

在项目根目录执行。总日志`runs/20261026_reference_v6_64k/console.log`；排队／运行状态`preparation_status.json`；缓存分片进度`cache_progress.json`。完成必须同时存在cache/results.json、verification.json和本轮results.json，不能把索引已冻结或排队状态称为缓存已完成。断点只复用同身份、已原子提交并核验的分片。

安全停止：创建本轮`runs/20261026_reference_v6_64k/STOP`，当前分片结束后退出；移除自己创建的STOP再运行原命令。并发锁拒绝重复构建，不覆盖原始数据、旧权重或旧缓存。

[成员准备结果](runs/20261026_reference_v6_64k/membership_results.json) · [准备期检查](runs/20261026_reference_v6_64k/preflight.json) · [来源与冻结文件](provenance.json)
