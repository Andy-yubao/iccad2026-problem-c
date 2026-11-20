# 每三层基线的逐层诊断

用户于2026-11-14授权继续分析E134每三层基线，不修改原网络、训练、权重或后处理。

- [分析结论](findings.md)：单独看监督层、展开所有层、量化拓扑何时形成、隐状态探针与资源分配。
- [完整统计表](report.md)：13个输出层、Loss分项、配对同时区间、形成时间、注意力/隐状态/梯度及Q关联。
- [逐题逐层CSV](per_case_layers.csv)：500题×13层，保留成员身份和完整精度。
- [量化方案](plan.json)、[验证](verification.json)、[汇总JSON](summary.json)。

入口权重为E134 `e119_stride3_seed0` best63，SHA256 `1f69a681713255a7adfca09357d70065ad7dff4d62f1a6294a7c233e30ad1df1`。仅读取既定dev500和兼容缓存，未读test。既有S0用共享头hook观察，不增加头、不另造中间布局；全部S12与原评价、既有15例的12层与原展示均逐位一致。

Loss使用原训练域共同完整公式，各层均不乘监督深度权重。几何使用实际保存坐标。拓扑主指标用固定reference K6邻居对，方向近平局按预登记阈值处理，已排除P-P抬分，并用K4/K8、原始头中心和非P子集检查。

`capture.py`采集实际前向/隐状态/注意力并计算Loss；`topology.py`并行计算几何和拓扑；`probe.py`做按源分片五折的固定正则线性诊断拟合；`finish_gradients.py`只补齐eval局部梯度；`analyze.py`计算配对源分片bootstrap并出表/图。线性拟合不更新原布局网络，梯度诊断没有优化器或参数更新。

前向和Loss完成后，初次梯度诊断遇到推理缓存不能用于反向的限制。失败证据保存在 `capture_failure_before_gradient_buffer_fix.json`，改用独立普通缓存后仅补算梯度，未重训或重选checkpoint。`raw_center_followup_plan.json`单独标记了检查非负clamp影响的后续只读诊断，没有把它冒称预注册主检验。

Reference不是唯一好布局；稳定接近S12不保证好。探针留出只针对本dev面板的拟合，原网络已用dev500选模。单seed、重复使用dev、局部eval梯度和未逐层重跑后处理均限制因果解释。
