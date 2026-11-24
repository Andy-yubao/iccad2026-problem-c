# 冻结推理方案

每个阶段T(s)=s+Delta(s)，执行u=s+0.5Delta(s)，v=u+Delta(u)，再把v送入下一阶段。各阶段参数独立，单阶段两遍共享参数。每遍内部双层attention只在入口编码当前布局，两个块间仍不解码、不增加几何反馈；第二遍重新入口编码半步布局。L0不变。七个出口均保存真实坐标，不复制或插值末层。

原监测dev500按各阶段原Loss评价：reference1/.75/.5/.5，overlap .25/.25/1/1，其余原项及MST阻挡不变。同阶段可比；不同阶段总Loss不横比。原单次调用为同一个最终checkpoint的前缀输出。本次仅一个用户指定推理方案，未扫alpha、未按Q重选checkpoint，test封存。全2000沿用原reference/no-MIB与E98.depth_budget.v1预算及原独立硬检查。
