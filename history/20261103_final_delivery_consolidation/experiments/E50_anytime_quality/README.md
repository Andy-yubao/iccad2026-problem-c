2026-10-26：当前保留基线为 **pro 质量优先后处理**。共用 E42 best51 权重原位保存；旧实验及共享缓存的迁移见[整理记录](../../history/20261026_flash_pro_consolidation/README.md)。历史命令与源码身份保持原样，不绕过冻结检查续跑。默认不启动评分页面。

# E50：并行保护轨迹与约束反馈驱动的随时可用求解器

2026-10-25完整500已运行完成。[完整报告与35例图集](runs/20261025_full500_review/analysis.md)：500/500硬合法、加权Q1.01204277（常规1.02585963）、395胜93平12负、3例Q>1.2，完整均时4.511742秒/最大4.891637秒。除逐例不退化外的门槛通过，严格联合验收未通过。12个回退中10个已有不差于历史基线的候选；185和393的候选库也略差。全部候选Q/D/核验时间前沿与500例修复负担表保留；当前选解仍偏质量，不能称最小改动器。生成报告入口为`python -m experiments.E50_anytime_quality.analyze_full500 --render`及`python -m experiments.E50_anytime_quality.report_full500`，仅分析/绘制保存数据。

用户查看35例后澄清：“尊重草稿”优先指相对方向和意图邻接保持，而非每个模块小位移；认为当前E50广泛保留了这一拓扑意图，并认为最差10例草稿普遍较差。此反馈已写入[研究协议8.1](../../docs/protocol.md#81-草稿模型的人类反馈研究原则2026-10-24)。D和改形继续作为辅助量，不将大位移直接视为不尊重草稿；现有近邻方向变化代理不等于经校准的邻接意图。

同日按用户要求进行最差10例十秒预算对照：[专用入口](rerun_worst10.py)，固定450、307、467、123、366、202、164、368、246、493；相同算法/模型与4数值线程，只增加预算，核对重新推理草稿一致。原5秒结果保持原样，新结果独立保存在`runs/20261025_worst10_budget10s/`，完成后生成原草稿／5秒E50／10秒E50三相图。命令：`python -m experiments.E50_anytime_quality.rerun_worst10 --resume`；只读重建报告用`--report-only`。该事后难例面板不是独立验证或完整500新成绩，延时无收益也不证明草稿不可修复。

十秒对照已完成：[报告与10例三相图](runs/20261025_worst10_budget10s/analysis.md)、[机制限制解释](runs/20261025_worst10_budget10s/interpretation.md)。10/10硬合法、全10秒内、4胜5平1极小回退；该子集加权Q1.207153→1.152287，Q>1.2从3例减至1例。450和467明显改善，366和164小幅改善；仍无Q<1.03。307、368和493提前达到512次调度上限，不能将其视作完整用满10秒后无收益。全部新draft与原draft逐元素完全一致。

用户2026-10-24授权持续优化、任意必要测试，直到达标或主动叫停；随后将主目标改为加权Q<1.015，并允许彻底重写控制架构。仍固定E42 best51和原固定开发100例，逐例Q不差于原常规steady；最多1例Q>1.2，展示全部超出者的draft。单例5秒、最多4个数值线程；初始化/预热与事后全候选Q诊断单列，最终解独立评分计入5秒。未暴露测试保持封存。

E49已验证：纯串行基线保护占用大量预算，历史时限也会改变基线落点；同时硬化全部几何条件经常不可行。保存E49所有失败，不视为达标。E50独立复制必要内核，重写调度与搜索，不跨实验导入私有算法。

初始架构：2线程保护基线与2线程探索并行，共享同一绝对截止；探索直接从draft出发，使用不可行证书定位冲突分离关系，保留独立状态与合法候选。父进程接收候选、独立检查、保存并回退；真实Q不传给工作进程。基线完成后仍保留它的全部合法轨迹。距离沿用E48的整体软偏好，公开Q不退化证书与事后历史基线验收分开。

来源见[provenance.json](provenance.json)。**当前v3已通过原固定100例验收**，冻结后确认及重复检查已完成；这是开发结果，不宣称任意未知输入均不退化。所有调用、源码快照与逐例证据保存到runs，完整坐标仅本机保存。

用户追加的稳定高质量解与Q–时间–draft距离帕累托前沿研究，持续记录于[frontier_research.md](frontier_research.md)。

## 当前交付结果

[完整实测报告](runs/20261025_final_analysis/analysis.md)：100/100硬合法，加权Q **1.007596**（常规1.024238），逐例76胜/24平/0负，P95 1.076928、最差1.155240，0例Q>1.2。平均完整求解4.528869秒、最长4.898213秒，0例超5秒。超额Q下降68.66%，新目标Q<1.015通过。D中位数0.3508，常规同draft重算0.4253，67例更近；无硬移动上限。

完整100运行见[summary.json](runs/20261025_frozen100_parallel_v3/summary.json)，完整候选留本机cases。写汇总时曾遇Windows瞬时共享锁，已保存83例后以相同冻结源码恢复，仅计算剩余17例；恢复证据见recovery.json。后续仅给评价入口增加文件锁重试和“已完成直接退出”，并通过真实Windows共享锁测试；全部搜索、选解、模型内核保持主100版本，未因此重写旧结果。

[交付核验](runs/20261025_delivery_audit/audit.json)确认68个源码文件可编译，215次正式/确认/重复输出均硬合法且在5秒内；E42核心、唯一checkpoint及五份冻结协议索引保持不变。当前源码快照另存于同目录本机zip。

冻结后从其余400个开发成员中预留100例，见[确认面板](runs/20261025_confirm100_manifest.json)。[确认与稳定性实测](runs/20261025_final_analysis/confirmation_stability.md)：另100全合法、加权Q1.005283（常规1.017490）、79胜/16平/5负，最大回退0.003102，最差Q1.152272，最长4.920428秒，无Q>1.2及超时。五个难例各重复三次，15/15合法、不退化、不超时；259例Q在1.0160–1.0257之间，其余四例三次Q一致。算法/规则在确认前冻结，未读取未暴露测试；这些小回退限制了普遍逐例支配结论，不因平均改善而隐藏。

确认中5个回退案例的候选库都已经含有不差于历史基线的点，证据见[反例诊断](runs/20261025_distance_tradeoff_diagnostic/confirmation_counterexamples.json)。后续重点是公开代理的可靠选解，不应直接将这些回退解释为draft潜力差或搜索未找到好解。

## 复用后处理入口

[service.py](service.py)的`Service`可直接处理任意模型给出的N×4 XYWH draft，不依赖E42权重。若提供E42兼容checkpoint，也可把冻结CPU推理计入单例预算。Windows在主函数保护中调用；先初始化常驻工作进程，再逐例求解：

```python
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'
os.environ['NUMBA_NUM_THREADS'] = '1'
from experiments.E50_anytime_quality.service import Service

def main():
    service = Service()
    try:
        service.ready()
        positions, diagnostics = service.solve(public_inputs, draft=draft_xywh, seconds=5.0)
    finally:
        service.close()

if __name__ == '__main__':
    main()
```

公开输入沿用七项官方接口：模块数、面积、块间连接、引脚连接、引脚坐标、约束、遮罩后的规格。`case=`可在父进程独立评分，不能进入工作进程；不提供时不编造Q。候选都通过独立硬检查，返回全部候选索引、公开代价、距离分量、找到/核验时间、不可行修复轨迹和中断记录。

2线程保护轨迹与2线程探索并行，共享绝对截止；发现少量软冲突时才启用额外的强约束修复，原软修复路径始终保留。父进程核验候选并在截止时终止未完成的工作进程，返回已核验结果。真实Q不传回工作进程。选解是公开装填密度分档的截断代理；不再强制任何参考分母下的保守证书，但仍保留证书值供诊断。

5秒包含转换、推理、全部候选生成/硬检查、调度、回退/终止及最终解官方评分。常驻服务冷启动、载入、预热/必要重启和写盘单列。非选中候选的Q只用于求解结束后的帕累托诊断，单列耗时；主100这部分共340.13秒，没有参与搜索或选解。一般桌面系统不提供严格实时保证，所有实测超支必须报告；本100实测均在5秒内。

## 执行、停止与恢复

### 完整500例开发检查

2026-10-25按用户要求新增`full`入口：完整冻结dev_monitor500，索引0–499，**新跑500例，不复用此前200例结果**，原结果保持原样。单例仍5秒、2+2数值线程；全部候选事后Q诊断另计。入口交付时估计约60–90分钟；现已由用户运行完成，结果见上方报告。输出为`runs/20261025_full500_parallel_v3/`，逐例保存在cases/，汇总为summary.json。

```powershell
& 'E:\project\ICCAD\2026\problemC\experiments\E50_anytime_quality\launch.ps1' -Stage full -Resume
```

相同命令恢复时校验源码、模型与面板，只计算尚未保存的案例。500例汇总要求全硬合法、加权Q<1.015、逐例不退化及全5秒内；Q>1.2容许数按原1%比例扩展为最多5例，所有超出者/回退/超时逐例列出。该检查仍属开发验证，不打开封存测试。搜索/模型/选解算法不变，仅扩面板与相应汇总口径。

安全停止（当前案例保存后退出）：

```powershell
New-Item -ItemType File -Force -Path 'E:\project\ICCAD\2026\problemC\experiments\E50_anytime_quality\runs\20261025_full500_parallel_v3\STOP' | Out-Null
```

恢复前删除自己创建的STOP，再运行原命令。`-DryRun`仅打印计划，不启动推理/求解。

一次只启动一个评价阶段。[launch.ps1](launch.ps1)提供以下入口；已完成阶段只显示结果，不重复推理或求解：

启动脚本使用带BOM的UTF-8，JSON读取显式指定UTF-8，兼容Windows PowerShell 5.1；编辑脚本时保留此编码。已在5.1中实测完整500入口的`-DryRun`及已完成结果退出行为，不启动实际计算。

```powershell
& 'E:\project\ICCAD\2026\problemC\experiments\E50_anytime_quality\launch.ps1' -Stage screen -Resume
& 'E:\project\ICCAD\2026\problemC\experiments\E50_anytime_quality\launch.ps1' -Stage confirm -Resume
& 'E:\project\ICCAD\2026\problemC\experiments\E50_anytime_quality\launch.ps1' -Stage stress -Resume
```

`-DryRun`只打印计划。screen为原固定100，confirm为预留开发100，stress为五难例的三次相同配置重复。新运行均保存源码快照、成员/模型身份及逐例结果。未完成运行`-Resume`只跳过已存案例，源码/权重/面板不同会拒绝恢复。运行目录或E50的runs下创建STOP，在当前案例结束后停止；移除自行创建的STOP后再恢复。禁止并发重复启动同一目录。

不求解、只从已存结果重建主报告：

```powershell
& 'E:\project\ICCAD\2026\problemC\.conda\envs\iccad2026\python.exe' 'E:\project\ICCAD\2026\problemC\experiments\E50_anytime_quality\runs\20261025_final_analysis\analyze.py'
```

## 当前运行

[parallel_gate5_v1_r2](runs/20261024_parallel_gate5_v1_r2/summary.json)：5/5硬合法、5例均不差于历史基线、无Q>1.2、无5秒超支，最大4.778555秒。164/443/246分别从1.927/1.764/1.307改善到1.155/1.021/1.141；21例仅轻微改善。这个难例门不用于声称加权指标达标。

已核验全部源码与五例门一致并完成固定100例[runs/20261024_frozen100_parallel_v1/](runs/20261024_frozen100_parallel_v1/summary.json)：100/100硬合法、0例Q>1.2、最长4.931秒，但加权Q1.020422、9例退化，未通过。该运行冻结期间未修改算法。所有非选中候选Q在求解结束后诊断，明确单列开销，不参与在线搜索或选择。

[公开代理v1候选库复选](runs/20261025_selection_policy_v1/results.json)改善到加权Q1.009719，但确认段仍有4例退化，未通过。后续[密度分档与修复v2](runs/20261025_selection_policy_v2/README.md)在全100上自适应开发，将另做实际运行，不能称独立未见结论。
