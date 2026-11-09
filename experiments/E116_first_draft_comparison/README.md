# E116：原结构前四层 vs 延后读出的独立首稿生成器

2026-11-08 用户明确授权准备对照，并明确暂不启动训练。`authorization.json` 的 `training_enabled=false`、`actual_updates_authorized=false`，本实验 STOP 保留。没有自动队列，不创建训练 run 或 checkpoint，不改旧实验授权或 STOP。

用户所说的“4层 vs 一层”，具体是原结构四个布局更新层，与新版一个首稿生成阶段比较；新版阶段内部同样有四个 Attention 推理块。不是比较新版内部四块与一块。

| arm | 结构 | 参数 | 唯一监督 |
|---|---|---:|---|
| original4 | E114 64维前四个 RectangleBlock，保留初始读出、每层坐标/预测矩形反馈、共享64→64→3头；后面八层彻底移除 | 207,987 | 第四块末端完整 Loss |
| new_first | 四个独立64维 Attention＋公开连接/G分组聚合＋SwiGLU128块，无中间坐标、解码或反馈；末端拼接全局均值，128→64→3读出 | 209,891 | 唯一首稿完整 Loss |

新版比原四层多0.915%参数，是相近预算的完整设计对照，不是只改变几何反馈的单因素消融，也不称等 FLOPs。没有 E65 式独立模块对向量消息、持续边状态或后续修稿器。两组都只对最终输出监督，没有中间辅助 Loss、0.2缩放或Q评价。

## 数据与共同 Loss

使用冻结 `random_reference16k.v1`，train16000、子集内固定train_monitor500，原固定dev_monitor500。仅引用父random40k既有NPZ；不复制数据、不重新抽样或按质量筛选。保持原始面积，无MIB，全模块位置和形状预测；F/P公开规格仅作条件，草稿不覆盖预测，端口网络输入保持总权重/加权中心摘要。

两组共用本实验独立维护的 `base_per_case` 完整目标：位置+.25形状+.08975162861857286重叠三项+.3336832108335229公开任务项；边界/G分组软项归一化排除MIB。反传、固定train/dev监测、Loss-best、LR和早停使用同一个最终 Loss 的逐例等权平均。只比较Loss及分项、拟合差距和实际用时，不做后处理或Q，不读取test载荷。

原E114/E115历史36k联合训练总Loss不作为本对照的同条件结果。独立首稿训练两组均从头初始化，不读取旧权重。

## 初始化与训练规则

- seed0，batch32，FP32确定性路径、TF32关闭，四线程。
- 独立模块入口在导入torch前默认设置CUBLAS_WORKSPACE_CONFIG=:4096:8；若已有不同配置则明确拒绝，避免静默改变数值路径。首次准备检查发现缺失该环境设置，失败记录保留在preflight_attempt1_cublas.json，修复后仅复查原五例。
- 两组新鲜输入编码、各块norm1/norm2、QKV、Attention输出、局部聚合初始张量相同。新版不兼容的FFN/读出在独立固定CPU RNG中初始化；构造后全局CPU RNG与原四层一致。不同FFN的Dropout掩码不宣称相同。
- 轴交换P=.5按seed/epoch/子集内成员确定；FFN Dropout .1、残差Dropout .05；AdamW decay .0003、全局clip1、原每可训练模块联合相对步长保护 .005。
- 预登记每组最多4小时/200轮，原四档LR(1e-3/2.5e-4/1e-4/5e-5)、14轮保护、耐心3、最低档至少两完整轮；这里只是未来运行上限，不构成启动授权。
- 非有限、单例/均值Loss≥1000、裁剪前梯度范数≥10000均在Adam前停止；不跳批、不自动重试、不推进另一组。
- latest保留模型、Adam、CPU/CUDA/NumPy/Python RNG、数据排列/游标、seen、调度、步长保护和累计预算；best独立保存。准备阶段不测试断点恢复，不清零历史预算。

## 准备检查与启动拦截

仅复用冻结16k入口已经核验的五个训练成员，不读dev/test载荷。`inspect.py` 在同五例上检查两组真实前向、共同Loss、反向、FP32 CUDA图捕获/重放、全模块预测及公开输入边界；**不调用optimizer.step**，模型参数保持不变。CUDA图捕获只准备前后向及原步长保护，不能把它算作训练更新。

检查还验证：原四层输出与本地原模型副本完全一致；新版预测头只调用一次；初始公共张量相同；F/P预测不被规格覆盖；训练命令在读取样本、占用CUDA、创建run或优化器更新之前被拒绝。结果见 `preflight.json`。五例检查不是泛化证据，也没有测量真实训练更新吞吐。

准备入口：

```powershell
.\.conda\envs\iccad2026\python.exe -B -m experiments.E116_first_draft_comparison.inspect
```

未来单组训练入口已经准备，**当前执行会拒绝启动**：

```powershell
.\.conda\envs\iccad2026\python.exe -u -B -m experiments.E116_first_draft_comparison.train --arm original4
.\.conda\envs\iccad2026\python.exe -u -B -m experiments.E116_first_draft_comparison.train --arm new_first
```

仅移除STOP仍无法启动，授权开关与允许arm必须在用户将来的明确启动授权下更新；实施检查源码身份也必须匹配。无自动队列，无自动终末dev2000或Q。主比较使用训练中已登记固定dev500的唯一Loss-best，保留全部分项、train500和耗时曲线；不追加未登记候选。

## 独立源码来源

`original_model.py` 为E114当下模型源码的原字节副本，original4只限制块数为4并返回末端状态。数据准备、共同Loss、CUDA/优化器保护和调度均从E114复制到本实验独立维护；不导入其他实验私有算法。共享仅使用datasets，未来绘图归visualization。来源文件及哈希见 `provenance.json`，冻结运行身份见 `plan.json` 和准备检查中的源码哈希。

E114主要研究基线、E101竞赛交付和所有旧STOP保持原身份。本实验准备不是恢复任何旧运行。
