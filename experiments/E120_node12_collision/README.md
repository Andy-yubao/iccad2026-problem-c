# E120 Node12-Collision

E119结构对照：保留两层初始全局交互、12层几何主干/共享头，以每层Node12动态几何投影注入Q/K，V保持。tau=.25、每头缩放初值.1可学习；公式见 `formula_spec.md`。725619参数，仍O(N²)。

单组scratch seed0/36k/.2L4+.8L12、FP32/batch32，原面积、无MIB、全部模块预测。共享完整CPU静态缓存，动态几何逐层计算。具体冻结配置、源码和门禁见 `plan.json` 与run登记。

训练和原固定dev500已完成，队列complete；混合Q1.039004，最终500/500硬合法、草稿0/500，未改善原E119。结果在 `runs/node12_seed0/evaluation_no_mib_dev500/`，历史三倍工作预算另存 `evaluation_budget500ms/`。旧启动命令不构成重训/恢复/测试授权。

布局展示统一调用 `python -B -m visualization --e120 --port 8781`，不新增专用网页。2026-11-13用户请求的10普通例＋5最坏例和真实12层缓存位于 `visualization/20261113_dev15/`，末层核对原评价草稿，最终坐标复用原结果；无训练、后处理/Q重跑或test读取。调用与样式见根目录 `visualization/README.md`。
