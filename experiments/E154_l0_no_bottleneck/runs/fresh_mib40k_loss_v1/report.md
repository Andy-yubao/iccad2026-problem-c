# E154 best50：新形状修正版完整训练/验证 Loss

Frozen model trained on old MIB32k, not fresh train36k; this checks dataset distribution, not a newly trained-model generalization gap.

| 加权项 | 完整train36k | 完整dev2k | train监测500 | dev监测500 |
|---|---:|---:|---:|---:|
| reference_weighted | 0.29418902 | 0.28860228 | 0.29460846 | 0.29067775 |
| wire_compact_weighted | 0.05081433 | 0.05097242 | 0.05049643 | 0.05110652 |
| grouping_weighted | 0.01188516 | 0.01202730 | 0.01267564 | 0.01259168 |
| boundary_weighted | 0.00475637 | 0.00455533 | 0.00452887 | 0.00448508 |
| public_shape_weighted | 0.00082884 | 0.00080134 | 0.00087379 | 0.00083420 |
| public_preplaced_position_weighted | 0.00232737 | 0.00244863 | 0.00233127 | 0.00273351 |
| mib_weighted | 0.00026221 | 0.00022956 | 0.00025144 | 0.00021544 |
| total | 0.36506331 | 0.35963686 | 0.36576589 | 0.36264419 |

完整dev相对train：-1.486%。
共同N分布标准化：train=0.36552852，dev=0.35722836。
平均N：train=69.957，dev=69.995。
旧E154训练成员重合：新train=2356，新dev=149。

没有更新参数，没有后处理，未读取测试集。所有38000例逐题完整Loss分项保存为NPZ；少数N仅1例，其标准差单独按0记录，不影响任何Loss均值。

非reference的登记Loss项合计：train=0.07087429，dev=0.07103458，dev相对train变化+0.226%。

模块数覆盖补充：新dev缺少N=29, 46, 68, 71, 78, 80, 115, 117, 118，共同N范围覆盖train的90.25%、dev的100%。随机分片划分后N直方图不保证逐档相同；监测500仅在所属父集合内分层随机抽取，不能补充父集合缺失档位。
