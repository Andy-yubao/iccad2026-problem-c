# E114两个已完成模型的验证

2026-11-07用户要求暂停训练，先验证两个已完成模型。训练根STOP和QUEUE_STOP保持；第三组staged在数据准备阶段0更新，不恢复。

terminal best59、dense best52的checkpoint SHA和原random40k dev_monitor500成员在plan.json中先冻结。共同最终原始Loss从各自best当轮已保存监测中读取，不按Q重选。评价使用原始面积、全模块真实草稿、不计MIB的本地研究Q；同一复制的E98.depth_budget.v1后处理，送入后处理的公开输入副本仅MIB列置0，所有面积和固定/预放置规格保持。最终按原始完整题目做硬检查。官方源码、旧后处理、历史权重不修改。

入口：独立环境 `python -u -B -m experiments.E114_supervision_placement.evaluation.run --arm terminal` / `--arm dense`。每组只运行唯一冻结best的一次固定500评价；已有完成结果只读复用，部分结果/错误不自动覆盖或重试。评价停止文件在本evaluation目录或具体run的evaluation_no_mib_dev500目录，不取消训练STOP。两组都完成后，`python -B -m experiments.E114_supervision_placement.evaluation.compare`仅汇总保存结果。

复制来源和哈希见source_manifest.json。只读适配检查使用登记的五个训练成员：公开输入逐位匹配训练特征，原面积/规格保持，一例后处理硬合法；不是额外dev调参。正式每例完整时间含公开输入复制、特征、传输、网络、全部修复/回退及独立硬检查；加载、预热、评分、诊断几何、磁盘/画图另计。CPU内核/BLAS/求解线程固定1，GPU FP32 eval/CUDA graph，关闭TF32。草稿不覆盖固定/预放置预测，硬约束只由后处理满足。

layers入口只在原五个E114训练成员上以CPU FP32 eval/no_grad读取真实13次共享head输出，hooks与plain末层逐位一致。图在公共visualization/E114_supervision/20261107_two_models，原始草稿、跨模型/层统一尺度，不生成100图，不自动打开界面；没有Q控件。五例不承担泛化结论。

本次原始面积/无MIB/开发面板与历史A、E74、E105不同，不能直接排名。仅单seed开发证据；第三组尚未训练，不能回答第4/12层监督是否更优。
