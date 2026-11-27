# E43：联合正则化矩形96×12，新增重叠面积训练损失

2026-10-23，按用户确认准备**唯一seed0实验，尚未启动**。以已完成E42为控制，只增加重叠面积梯度；E42模型、Dropout .05、AdamW decay .0003、v5成员、原监督/MIB面积变换与后处理保持一致。旧队列和定时任务不恢复。

## 假设与唯一变化

E42完整dev代理0.453814、重叠面积0.063020，仍存在广泛冲突；现有occupancy参与选模却不参与训练。尝试让训练直接看到两两交叠面积，补充原平方穿透项的几何信息：

`L_train = L_geometry_original + ramp(epoch) × 0.25 × L_overlap_area`

`ramp = min(1, max(0, epoch/2))`，与原物理项使用同一实际连续轮数。`L_overlap_area`为无序有效矩形对的交集面积之和除以有效模块输入总面积；与原occupancy里的overlap_area数值和梯度一致。包含自由块—预放置碰撞，排除不可变—不可变对及padding；精确边接触损失为零，不加入正间隙。多重覆盖按两两交叠累加，并非几何并集超额面积。

模型仍为rectangle96×12、1,471,539参数，**从头初始化**，不从E42续训。除新增项外，原Loss权重原样保留：center1、shape .25、relative .25、penetration .5、contact .5、boundary .25、MIB .1、bbox .1。crowding、void、compactness仍只参与评价/选模，不新增它们的训练梯度。

固定dev_monitor500的**原geometry＋occupancy代理**继续负责选best、LR和早停；完整dev2000也报告同一代理。新日志`total`含新增项，不能与E42旧total直接比较；`base_objective`、`weighted_area_objective`单独保存。

系数在正式实验前固定为0.25，没有搜索其他系数。[准备检查](runs/20261023_preparation/results.json)只用原两例train做梯度量级检查：初始化时新增/原参数梯度范数约0.22%，E42冻结best时约14.72%；相对原加权penetration梯度约41.17%/37.95%。仅说明此小样本上没有压倒原目标，不宣称最优或总体量级已知。另用一例dev确认原代理精确一致，不用dev调系数。准备阶段无优化器更新、CUDA训练、求解器调用或测试读取。

## 反例与判读

首先比较相同面板的原代理、geometry、occupancy及重叠；同时检查contact、boundary、MIB、bbox、compactness、void，防止通过布局摊大或破坏分组接触降低碰撞。训练新增项下降本身不构成成功；代理下降也不等同于终末Q或硬合法改善。

预先复查E42保存的救援164/246/276/366：主流程shape候选分别36/36/26/20个，均进入early_projection、project和full_packed_refine。164的raw重叠约.0383，修复平均位移2.462个局部尺度、Q1.927279；总重叠较低仍可能救援。239没有救援，但Q1.339216包含2项边界和1项分组违反。因此本实验检验普遍重叠机制，不能声称会独自解决预放置拓扑或尾部；保存轨迹不够精确定位每次失败的障碍责任，不能据此断言已证明因果。

报告固定全体500、输入预放置最高四分位、E42重叠最高四分位、旧投影22、旧救援4、高group12以及预选尾部164/239/443/329/324。旧模型条件化难例可能均值回归，只作诊断。终末报告硬合法、加权/算术Q、P95、最差5%、最大Q、HPWL/bbox/软违反、救援、位移和耗时，以及病例/worker聚类配对区间；单种子与限时修复运行间不确定性继续保留。没有固定Q/耗时通过线。

## 用户启动

PowerShell从任意目录执行：

```powershell
& 'E:\project\ICCAD\2026\problemC\experiments\E43_overlap_area_loss\launch.ps1' -Stage all -Resume
```

只运行一组：训练→冻结唯一best→完整dev2000原始评价→固定dev500一次后处理→与保存的E42配对报告→退出。不重训/重评E42、E39或E25，不触碰封存测试，不创建定时任务。

只训练：

```powershell
& 'E:\project\ICCAD\2026\problemC\experiments\E43_overlap_area_loss\launch.ps1' -Stage train -Resume
```

之后单独评价：

```powershell
& 'E:\project\ICCAD\2026\problemC\experiments\E43_overlap_area_loss\launch.ps1' -Stage final -Resume
```

加`-DryRun`只显示命令，不创建正式模型/协调目录、不训练/评价。

## 预算、停止与恢复

- 单GPU CUDA、batch32；CPU训练四线程，终末2进程各2线程，与GPU训练分时。数据/缓存身份严格核对，只读复用冻结v5 train16000/dev2000，train/dev_monitor各500。
- 累计训练墙钟最多4小时或200轮，包括缓存载入及固定监测；终末另计。建议预留2～4小时，E42为约113分钟，本新损失尚无正式速度/停止轮数实测。
- LR `1e-3 → 2.5e-4 → 1e-4 → 5e-5`；第15轮起连续3轮无明显改善降档，3次下降且最低档至少完成2轮才允许早停，继承固定改善阈值和clip1。
- `latest`完整保存模型/优化器/调度器、样本顺序与游标、CPU/CUDA RNG、已见成员和累计预算；`best`只由原固定代理选择。首次`-Resume`无latest则从头，已有则严格核对源码/数据/设备/预算恢复，完成则复用，不增加预算。
- Ctrl+C安全停止，或另一个PowerShell写STOP：

```powershell
New-Item -ItemType File -Force -Path 'E:\project\ICCAD\2026\problemC\experiments\E43_overlap_area_loss\runs\STOP' | Out-Null
```

在当前batch/病例边界保存退出。确认退出并明确恢复后移除STOP，再执行原命令：

```powershell
Remove-Item -LiteralPath 'E:\project\ICCAD\2026\problemC\experiments\E43_overlap_area_loss\runs\STOP' -ErrorAction SilentlyContinue
& 'E:\project\ICCAD\2026\problemC\experiments\E43_overlap_area_loss\launch.ps1' -Stage all -Resume
```

如曾手工创建模型目录或协调目录STOP，也须明确恢复时移除。异常保留证据，不自动减batch、切设备或关闭确定性。已完成或预算结束不能重启追加训练；终末只恢复同一冻结best的未完成成员，不按Q另选权重。

## 输出与实现证据

- 正式模型：`runs/20261023_rectangle_w96d12_area025_seed0/`。
- 协调日志：`runs/20261023_rectangle_w96d12_area025_pipeline_seed0/console.log`。
- 模型目录：config、progress、best/latest、full_dev、terminal及comparison.json/md。训练results中的postprocessing_calls仅描述训练阶段；完整评价以独立full_dev和terminal文件为准。
- [area_loss.py](area_loss.py)和[loss.py](loss.py)实现唯一训练变化；[train.py](train.py)记录新增项系数及梯度，模型/原代理/后处理独立复制，禁止跨实验调用私有算法。
- [provenance.json](provenance.json)记录E42来源及修改后哈希；正式启动保存完整源快照与身份。E42源码、checkpoint及结果不改。
- [准备检查代码](runs/20261023_preparation/check.py)、[结果](runs/20261023_preparation/results.json)、[入口检查](runs/20261023_preparation/launch_check.json)：两train一dev，初始化与strict加载、梯度加法/两轮渐入、原选模代理一致、padding/不可变/零间隙接触、实际命令配置。未声称本次已验证CUDA数值恢复或正式训练收益；恢复机制沿用E42。

测试继续封存，E25与baseline冻结。
