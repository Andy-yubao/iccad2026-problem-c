# 候选6全局几何容量启动记录

先完成C5及预登记完整dev500分量诊断，见candidate5_result_report.md：只给teacher关系、保留C1几何时Q1.201789/7回退；teacher几何+关系Q1.014894/0回退。均是不可部署oracle反事实，repair-only时钟不作服务成绩，Q1.014894未达目标且不是数学下限。这支持检验更好的几何输入而非继续只拟合关系。

C6冻结完整C1 E74和原小F，新增4块width192/heads4的RectangleBlock全局几何适配器，可训练参数1,972,435。适配器隐藏/input/state_embed来自E74最后4块及input/head，新增draft/shape输入5列零初始化，共享最终读出零初始化。每块.25线性local-scale残差更新公开可变center/log-aspect；不存在新增CPU求解步数或独立port token。

初始输出严格等于C1，预检已验证。模型参数/Dropout/数据顺序的新路径已独立登记，训练仅适配器，旧L/F逐位冻结。Loss逐字等于C1 per_case函数（提取段SHA256 f921d831b499d9a22a423988e9739d7a71c14e6e30a70dcabc147cbb26d4f72f）：teacher center1+shape.25+near-relative.25，反传/固定train/dev500/best/LR/早停同一函数。没有加入C2-C5关系Loss、C3 clearance或dev系数扫描；本候选用于测试全局容量，不保证最终Q。

原五train预检37.17秒通过：真实Dropout/Adam eager/graph更新参数与动量差0；新图重建恢复下一步模型/Adam/RNG逐位一致；捕获无参数或Adam更新；显式非有限teacher在Adam前拒绝；L及C1 F冻结；峰值allocated1234.52 MiB。

新增网络完整公开输入到draft（含特征构建、传输、完整L/F/适配器、返回；统一120模块GPU图，batch1，原五train每例一次）为27.56/11.90/12.04/14.10/10.71毫秒，均值15.26毫秒/P95 24.87毫秒；图捕获.676秒。第一个较慢请求保留，未删异常或重测选时。这仅新结构五例短检，不含最终后处理，不可当作完整dev500达标证据。

后台命令/PID见candidate6_launch.json，日志candidate6.stdout.log/stderr.log，正式训练实际登记6/8。fresh AdamW seed0、batch32 FP32/noAMP/TF32、LR .001/.00025/.0001/.00005，最多1小时/100轮，原UTC2026-11-02T02:58:12全局截止不变。best/latest和初始untrained保存完整Adam/RNG/order/cursor/seen/调度/累计预算。

pipeline已安排训练后自身Loss唯一best一次原动态原生关系slim3逐例完整dev500、新rawdev2000几何诊断、同100四阶段图。已有C1相同预算控制只读复用；若best仍初始化零读出，则剪去恒等适配器并复用相同C1终末/raw记录，只创建缺失展示，不重复相同求解。否则新几何必须做新raw诊断，不能错误沿用C4/C5冻结几何的raw复用规则。推理完全不读teacher/reference自由几何。测试未读。

C1-C5及C6绑定源码哈希已重新核验。活动C6/deep_geometry/模型与数据依赖/计划/流水线/评价器冻结，变更数学/数据/数值路径另登记，不覆盖旧证据。C5、oracle面板及旧结果不重跑。
