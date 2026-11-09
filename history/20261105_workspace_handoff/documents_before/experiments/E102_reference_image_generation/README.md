# E102：题目输入到 reference 风格图片

这是2026-11-03用户明确新增的兴趣探索，独立于E101最终交付。当前问题只有：**给定题目的公开输入，网络能否生成一张与其原始reference相似、边界清楚的布局图片？**

“有后续研究价值”在本实验中直接指图片与reference的相似程度，以及用户对外观和边界的主观判断。当前没有矩形／矩阵提取、几何修复或后处理，不计算Q、不要求硬合法、也不强制MIB同形。完整配置见[plan.json](plan.json)。

## 当前已完成

- 原始数据读取、公开输入遮罩和固定配色RGB目标；不调用MIB平均面积变换。
- 同五个训练成员的128／256／512／1024整图及最细模块附近局部对照，见[runs/20261103_resolution5](runs/20261103_resolution5)。它检验小模块的像素失真，不做图像往返。
- 条件图像网络、准备缓存／训练／生成对照图的入口；正式训练尚未启动。实施短检记录见[runs/20261103_cpu_update5/result.json](runs/20261103_cpu_update5/result.json)。

## 数据与图片

复用v5冻结成员16k／2k／2k和固定train/dev各500监测；原始split_v1及索引不变。本轮读取原始area和fp_sol，原件不改。注意v5成员历史上经过公开MIB面积相容性筛选，本轮取消的是**面积变换**，并没有把成员变成全量无筛选样本。E102测试仍封存，E101的测试授权不适用于这里。

reference自由位置只用于画训练目标，输入仅使用面积、模块连接、分组/MIB关系、边界要求、固定尺寸、预放置位置以及逐模块端口总权重和加权中心。没有旧模型或教师来源，不将reference外框、标签、指标用作输入。

图像采用固定模块ID配色、白底实色区域，不放文字、纹理或人为加粗的边框。这样直接观察真实小模块在像素下的形状；本轮不从颜色反推几何或模块实例。栅格化统一调用[公共绘图模块](../../visualization/raster_layouts.py)，相邻矩形的亚像素覆盖相加。

画布固定物理原点(0,0)，边长仅由公开输入决定：固定／预放置按公开w×h、其他模块按原始目标面积求有效总面积，S为其平方根；P是预放置公开最远边界，L=max(1.5S,P+0.05S)。训练图片不使用reference外框缩放。缓存准备遇到超界直接停止，不能悄悄裁切，也不能根据dev外框自动扩大；若训练数据证明需要调整，先登记新画布版本。

## 第一轮网络

从头训练约束条件Transformer与卷积图像生成器：模块token编码公开属性，注意力偏置编码连接／MIB／分组；32×32图像查询读取模块token，再经卷积上采样生成512×512 RGB。它输出图片，不输出坐标。

先采用这个单次条件生成基线，便于直接检验“公开输入与reference图片能否建立对应”。暂不叠加扩散／Flow Matching。RGB MSE是唯一Loss，反传、train/dev500监测、best、降LR及早停完全一致；人工图片观察决定后续研究方向，不回头改选本轮checkpoint。

首轮登记batch8／FP32／seed0、AdamW 3e-4、最多100轮或累计8小时；第一个到达的停止条件生效。正式训练需要后续明确启动，不自动接续其他候选，也不占用E101当前运行。

MSE可能产生平均化的模糊图，白底及大模块也可能主导误差；公开输入未必唯一确定reference。这些是要观察的限制，不能据此预先宣称生成路线有效或无效。首轮之后再依据图片决定是否研究随机生成与去噪。

## 命令

在项目根目录用独立环境执行。先准备CPU缓存（不包含test），再按明确授权进行正式训练：

```powershell
& '.\.conda\envs\iccad2026\python.exe' -m experiments.E102_reference_image_generation.prepare --out experiments/E102_reference_image_generation/runs/prepared_rgb512_v1

& '.\.conda\envs\iccad2026\python.exe' -m experiments.E102_reference_image_generation.train --cache experiments/E102_reference_image_generation/runs/prepared_rgb512_v1 --run experiments/E102_reference_image_generation/runs/20261103_rgb512_seed0
```

该配置是新运行身份，与E101的batch32和预算独立。启动时保存实际源码快照、来源哈希与数据身份；best与latest分开。latest保存Adam、RNG、成员排列／游标、调度和累计时间；`--resume`只恢复同身份运行，不清零预算，不预先做断点恢复测试。

安全停止时在run目录创建名为`STOP`的文件，训练／监测均在下一批前保存并退出；未完成的监测不用于best或调度。恢复前移除STOP文件，再对同一命令添加`--resume`。不要强杀进程后把不完整更新保存成latest；意外中断只恢复之前的完整原子latest。检查state.json并确认进程退出。

训练结束后，唯一Loss best生成固定dev100的reference／网络图片对照，不作修复、不生成评分控件、不自动打开界面：

```powershell
& '.\.conda\envs\iccad2026\python.exe' -m experiments.E102_reference_image_generation.render --checkpoint experiments/E102_reference_image_generation/runs/20261103_rgb512_seed0/best.pt --cache experiments/E102_reference_image_generation/runs/prepared_rgb512_v1 --out experiments/E102_reference_image_generation/runs/20261103_rgb512_seed0/gallery
```

首轮采用512是五例分辨率观察支持的起点，不是全数据保证，也不表示512网络已能生成清楚图片。未实施的扩散模型和未来几何研究均不是本轮交付结果。
