# Reference topology similarity — candidate v1

设计说明：[参考关系图相似度](../../docs/proposals/reference_topology_similarity_v1.md)。这是指标候选与独立原型，没有接入历史实验或启动训练。

```powershell
& .\.conda\envs\iccad2026\python.exe -B research/topology_similarity_v1/check_metric.py
```

只运行合成例、Loss及梯度定义检查；不加载数据集、权重、优化器或后处理。
