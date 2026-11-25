# E165 后处理软约束修复

冻结 E165 backend best53。在原始面积、原 reference、no-MIB dev2000 上用相同草稿配对重跑。验证权重 SHA-256、冻结成员顺序、数据校验清单和每份样本载荷。网络在 CPU 上推理，每份草稿生成一次。reference 只在独立评分时使用。

## 处理流程

1. 用原版 E98.depth_budget.v1 后处理生成硬合法解。
2. 有剩余软违规，或主合法化没有产出合法解且外框面积异常时，从原合法解出发尝试四步 SOCP。加强贴边和分组接触权重，允许自由模块调形。若仍有违规且该解未被接纳，写入公开预置坐标后重试原草稿的拓扑。
3. 对剩余违规尝试平移已连通分量、沿目标边寻找空槽和小范围推开阻挡模块。对锁定坐标的贴边模块，移动突出其目标边界的模块。
4. 候选先通过原硬合法检查。分组、贴边违规分别不得增加。令 (r_A) 和 (r_W) 分别为新旧外框面积比和线长比，(Delta v) 为软违规数变化，(N_mathrm{soft}) 为公开软约束分母。用 (max(1,r_A,r_W)exp(2Delta v/N_mathrm{soft})) 作为中性耗时 Q 比值上界。软违规数相同时，以面积和线长的 Pareto 改善接纳候选。这个判断只依赖公开输入和两个输出。

## 2000 例配对结果

| 指标 | 同次重跑原版 | 候选 |
| --- | ---: | ---: |
| 硬合法 | 2000/2000 | 2000/2000 |
| 平均 Q | 1.010098 | 1.008974 |
| 加权 Q | 1.007212 | 1.006229 |
| 混合 Q | 1.008655 | 1.007601 |
| P95 Q | 1.061951 | 1.055597 |
| P99 Q | 1.109081 | 1.105172 |
| 最大 Q | 1.382093 | 1.234941 |
| 原版最差 100 例平均 Q | 1.096730 | 1.080795 |
| 分组违规总数 | 201 | 172 |
| 贴边违规总数 | 114 | 104 |
| 全部软约束满足 | 1725 | 1753 |

逐例 Q 改善 74 例，持平 1926 例，退化 0 例，比较容差为 (10^{-9})。两个软违规类别也均无逐例退化。监测 500 例和扩展 1500 例分别改善 17、57 例，四个块数区间的混合 Q 均下降。原主路径无合法候选的 6 例中，2 例改善。原最差的 575 号样本 Q 从 1.382093 降至 1.185355，分组违规从 4 降至 2。候选最差的是 473 号样本，Q 为 1.234941，仍有 1 次分组和 2 次贴边违规。

原版为本次 Linux CPU 重跑。项目此前存档的 E165 混合 Q 为 1.008717，此次重跑为 1.008655。表中使用同次生成的草稿和同一评分环境。质量按项目的中性耗时 Q 计算。

独占单进程串行抽取 80 例，原版与候选交替先运行。后处理均时为 176.0 → 195.8 ms，P95 为 517.3 → 540.1 ms。另复跑 7 个尾部及约束样本。两版在 87 次逐例检查中分别与配对缓存的输出逐元素一致。计时包含完整后处理，排除推理、磁盘和评分。

## 复跑

在仓库根目录使用已有的 E165 评价依赖和 dev 数据。将第一行的权重路径替换为经冻结记录核验的 best53。Windows 使用 `--workers 1`。

```bash
python -m research.postprocess_tail.benchmark --checkpoint path/to/best.pt --out out/postprocess_original
python -m research.postprocess_tail.benchmark --original out/postprocess_original --out out/postprocess_candidate
python -m research.postprocess_tail.summarize --original out/postprocess_original --candidate out/postprocess_candidate --out out/postprocess_comparison
```

每版使用各自的输出目录。汇总核验 2000 对草稿逐元素一致，并输出逐例 CSV 和分组 JSON。实际求解接口是 `research.postprocess_tail.optimizer.optimize(draft, inputs)`，`inputs` 沿用 E165 的 `repair_inputs` 输出。

## 保持坐标逐位一致的加速版

`optimizer.optimize` 使用加速路径。`optimize_reference` 复现首次验收的软约束修复版。加速版用 Numba 完成保持顺序的传递约简、硬合法检查和分组计数，并在同一实例内复用公开输入预处理。no-MIB 形状求解把自由模块的面积校正合并成数组运算。Clarabel 矩阵、容差、求解预算和软违规接纳规则沿用首次验收版。

冻结的 E165 best53 和原 no-MIB dev2000 上，2000/2000 个结果的每个坐标和尺寸逐位一致。复核也匹配了主流程工作计数及软修复触发和接纳记录。混合 Q 为 1.007601，P95 Q 为 1.055597，最大 Q 为 1.234941，硬合法 2000/2000，分组与贴边违规分别为 172 和 104。

| 全量配对 CPU 耗时 | 首次验收版 | 加速版 | 降幅 |
| --- | ---: | ---: | ---: |
| 均值 | 319.4 ms | 279.9 ms | 12.4% |
| 中位数 | 203.5 ms | 179.5 ms | 11.8% |
| P95 | 900.0 ms | 797.8 ms | 11.4% |
| P99 | 1477.5 ms | 1310.3 ms | 11.3% |
| 首次验收版最慢 100 例的均值 | 1277.6 ms | 1101.8 ms | 13.8% |

全量测试用 8 个进程，每例交替先运行两个版本，采集进程 CPU 时间。按模块数分为 21–59、60–89、90–109、110–120 四档，分档均时分别下降 13.5%、12.1%、12.4%、11.9%。启动预热、输入读取和评分排除在计时外。另核对 80 个包含 DAG、循环和跨 64 位字边界的图，传递约简输出边集与顺序均一致。

用固定种子 20261125 均匀抽取 120 例串行复测。三个版本按样本轮换运行顺序，计时为端到端后处理实耗时。

| 串行 120 例 | E165 原版 | 首次验收版 | 加速版 |
| --- | ---: | ---: | ---: |
| 均值 | 171.4 ms | 204.9 ms | 180.3 ms |
| 中位数 | 132.1 ms | 154.7 ms | 130.8 ms |
| P95 | 438.3 ms | 551.9 ms | 494.0 ms |
| P99 | 526.6 ms | 614.3 ms | 532.8 ms |

相对首次验收版，串行均值下降 12.0%，P95 下降 10.5%。加速版均时较 E165 原版多 8.9 ms，保留全部质量收益。运行环境为 Python 3.12.14、NumPy 2.2.6、Clarabel 0.11.1 和 Numba 0.61.2。

```bash
python -m research.postprocess_tail.benchmark_speed --original out/postprocess_original --current out/postprocess_candidate --out out/postprocess_speed
python -m research.postprocess_tail.benchmark_speed --original out/postprocess_original --current out/postprocess_candidate --out out/postprocess_speed --serial-sample
```
