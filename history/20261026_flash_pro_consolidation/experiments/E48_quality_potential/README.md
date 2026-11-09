# E48：5秒质量优先的draft潜力评估求解器

2026-10-24用户授权实现。最新边界：**单例完整求解5秒、最多4线程**，不沿用最初10秒建议；逐例目标Q<1.03，允许如实未达标。围绕完整draft开展多候选、回退及质量保持，不设统一移动硬上限。模型/Loss和E25/E42原件不改。此工具衡量在明确算法/预算/距离偏好下已找到的潜力，不是全局最优证明。

方法与验收见[design.md](design.md)，实现入口[solver.py](solver.py)，固定开发评价[run.py](run.py)。来源见[provenance.json](provenance.json)：复制既有内核独立维护，不跨实验导入私有算法。只接收公开输入与draft；原参考位置/形状/线长分母均不能进入搜索。真实Q由scoring在候选接受后独立计算，回调返回值不参与在线搜索或停止条件。

## 当前交付与结果

**已实现，但未达到逐例Q<1.03，也尚不能作为已校准的权威draft评分器；未替换E42常规后处理。** 固定100例全部硬合法，75例达标（同例常规63例），最差Q1.703045；加权Q1.027597，略差于常规1.024238。逐例胜/平/负66/19/15。候选中事后最低Q也仅75例达标，后续需要改善搜索本身，不能只调整选解规则。该面板明确包含已知难例，属于开发诊断。

用相同原draft与公开规格重算距离D，77/100例比常规解更近，D中位数0.4253→0.3378、P95 1.1225→0.7726；距离偏好仍是工程定义，没有用人工评分校准。

| 版本/证据 | 用途与限制 |
|---|---|
| [v3五例](runs/20261024_gate5_v3/analysis.md) | 保留调度导致严重退化的失败证据 |
| [v4五例](runs/20261024_gate5_v4/summary.json) | 修正独立分支状态与保护轨迹后，五例全合法、均在5秒内 |
| [v4固定100报告](runs/20261024_screen100_frozen_v4/analysis.md) | 完整均时4.719884秒，最大5.745934秒，2例超时；这两个失败保留 |
| [外部截止五例](runs/20261024_external_deadline_gate5/summary.json) | 含上述两个超时及三例难例；5/5合法、最大4.960943秒，2次主动终止并返回已核验候选；另通过强制停滞检查 |

## 推荐复用入口

[resident.py](resident.py)提供常驻工作进程及外部截止保护。候选每次先送父进程独立核验、保存，再继续搜索；工作进程卡住或超过剩余时间时，终止该进程并返回已核验的最好候选。父进程仅回复核验完成，不返回Q。下次调用重新启动工作进程，启动/加载/预热和重启时间单独披露，**不属于5秒稳态单例预算**。写盘亦单列。一般桌面操作系统不提供严格实时保证，仍须统计实测超支；当前保护版仅通过上述五例，尚未完成新的100例评价。

使用自己的draft时不需要checkpoint或参考布局。调用方必须在加载数值库之前设置线程环境（运行脚本已设置）；Windows调用放入主函数保护内：

```python
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'
os.environ['NUMBA_NUM_THREADS'] = '1'
from experiments.E48_quality_potential.resident import ResidentPotential

def main():
    service = ResidentPotential()
    try:
        service.ensure_ready()
        positions, diagnostics = service.solve(public_inputs, draft=draft_xywh, seconds=5.0)
    finally:
        service.close()

if __name__ == '__main__':
    main()
```

`public_inputs`沿用官方七项接口：模块数、面积、块间连接、引脚连接、引脚坐标、约束、遮罩后的公开规格。`draft_xywh`为N×4的原草稿坐标。返回全部已核验候选及各项距离、公开代价、首解时间、终止/错误信息；没有参考分母时不编造Q。`case=`仅用于父进程事后评分，不传入工作进程。最终选择使用公开代价＋0.03D，距离没有硬上限。

## 运行、停止与恢复

推荐新入口[launch_resident.ps1](launch_resident.ps1)调用[resident_run.py](resident_run.py)。下面是**尚未执行**的保护版同100例命令，预计约9–12分钟（含可能的重启），输出到`runs/20261024_resident_screen_v1/`，不会覆盖已有v4结果。当前不自动重复100或扩大500。

```powershell
& 'E:\project\ICCAD\2026\problemC\experiments\E48_quality_potential\launch_resident.ps1' -Stage screen -Resume
```

添加`-DryRun`只核验代码/模型/面板与五例身份，不执行推理或求解。`-Resume`校验源码、权重、面板后跳过已保存成员；已完成则直接退出。运行目录或E48的`runs/`下创建`STOP`文件，在当前案例结束后停止，删除STOP再恢复。单次运行总保护900秒，未完成可以恢复，不隐藏超时或货架兜底。

旧[launch.ps1](launch.ps1)/[run.py](run.py)只用于历史软截止版本；它们看到已完成结果会直接返回。不能用原100成绩声称外部截止版100例已通过。

先运行等价开发五例门（含既有难例），检查合法性、截止、回退和来源；若无硬错误及纯货架回退主导，再冻结100例。大面板预计超过20分钟则仅交用户命令。所有已找到的合法候选及质量—距离前沿保存在所属runs；不只保留最终赢家。落在远处的合法兜底单列，不能冒充近草稿好解。未到Q目标和预算超支的案例都计入分母。

本地Q沿用MIB派生任务中性时间口径，不是正式比赛成绩。搜索复杂度同时受算法、数值条件、模块数和约束影响；它是预算条件下的修复负担证据，不能直接当作draft的内在质量。报告首次合法/达到事后Q阈值时间、候选/回退/子问题次数、最佳质量与达到目标的最小已找到距离，保留未知。
