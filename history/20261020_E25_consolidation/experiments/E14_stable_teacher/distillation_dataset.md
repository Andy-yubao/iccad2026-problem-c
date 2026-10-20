# 冻结蒸馏数据：Q最好20000＋尾部1000

当前后续蒸馏唯一入口为[protocol/distillation_workset_v4.json](E:/project/ICCAD/2026/problemC/protocol/distillation_workset_v4.json)，版本`E14.distillation_top20000.v1`。不要使用旧archive库中的teacher.json（那是原16000例教师筛选清单），也不要启动旧E13入口来绕过新版划分。

从完整25000例中按未舍入Q升序选最好20000，Q相同按固定成员哈希排序；最高Q为1.070240635。按用户新要求，此版本不受旧Q≤1.06门槛约束。再按原始分片分组，精确划分：

|学生角色|样本数|分片数|worker数|教师平均Q|
|---|---:|---:|---:|---:|
|训练train|16000|250|100|1.020355|
|验证dev|2000|79|65|1.020424|
|主测试test|2000|127|86|1.020555|
|额外尾部tail_test|1000|261|100|1.259066|

训练、验证、主测试之间样本及分片均不交叉，总计恰好20000例；额外尾部与这20000例样本不交叉。全库还剩4000例，本轮不进入训练、验证、主测试或尾部测试，继续保留在原库。

主划分尝试平衡六个模块数区间、五个教师Q区间和worker分布；不是把Q最好16000全部给训练、然后把稍差的4000给验证/测试。大于100模块的比例分别为21.74%/21.90%/21.65%，总体21.75%。分配器45秒限时得到满足全部硬条件的可行解，均衡目标未证明全局最优。分组结果已冻结，不因求解器再次运行或学生成绩改变。

## 额外尾部测试边界

全库Q最差1000例范围为1.159894831至2.863506809，成员已预先冻结。它们与train/dev/test各自共享192/28/38个原始分片，总计258个；若要求连尾部源分片也与最好20000完全隔离，最好20000只剩3706例可用，无法保持本次目标。因此它的用途是“学生未训练过的具体样本上的尾部压力测试”，不可称完全独立泛化。主test2000提供按分片隔离的留出评价。

全部原始来源仍属于split_v1的train。新角色是本轮学生蒸馏留出，不是把原local_test打开或迁回训练。原split_v1/workset_v2/derived_workset_v3原件及其SHA256保持不变。教师模型可能训练过这些原train成员；这与新学生的样本/分片留出是两回事。选优后的主测试也只代表这一筛选分布，不代表完整未筛选分布。

尾部是按当前教师的Q定义的困难面板。将来应评价学生硬合法率、同口径Q、耗时及相对教师的变化，不把拟合这些较差教师布局的MSE作为唯一成功标准。

## 数据位置和使用

`runs/distillation_top20000_v1/`下包含train.json、dev.json、test.json、tail_test.json，以及固定train_monitor500.json/dev_monitor500.json、group_assignment.json、summary.json和validation.json。清单只引用原完整布局库文件及其校验值，没有复制或覆盖25000份布局。必须保留`runs/teacher_archive25000_v1/layouts/`及冻结原数据。

```python
from experiments.E14_stable_teacher.distillation_data import DistillationData

train = DistillationData('train')
dev = DistillationData('dev')
train_monitor = DistillationData('train_monitor500')
dev_monitor = DistillationData('dev_monitor500')
sample = train[0]
public_inputs = sample['solver_inputs']
teacher_targets = sample['teacher']
```

读取器检查清单身份、原划分文件哈希、单样本布局哈希、标签schema、输入面积版本和结构标签有效性。模型公开输入和教师监督分开返回，不把Q、评分分母或原始自由块参考位置混入网络输入。缓存仅在各进程内保存数据分片，dataset对象可用于Windows spawn的数据加载进程。

`DistillationData('test')`和`DistillationData('tail_test')`默认抛出PermissionError；当前准备工作没有读取这两个集合的布局。未来明确执行冻结学生评价时，评价程序显式设置`evaluation_authorized=True`。训练/调参入口不得设置该参数加载测试。不得用主测试或尾部测试选checkpoint、学习率、早停、后处理或继续训练。

固定监测均500例，分别严格嵌套于train和dev。后续训练应显式绑定此registry的SHA256，并沿用学习率下降、早停和best/latest规范；本轮未启动训练，也未创建新checkpoint。

已检查21000个成员唯一、四集合按成员不交叉、主三集合按分片不交叉、监测面板嵌套正确、原冻结文件未变。仅从train/dev各取样，共读取5份布局验证数据接口；测试布局访问保持关闭。既有25000文件的全量校验结果沿用，单样本读取时仍再次核验SHA256。
