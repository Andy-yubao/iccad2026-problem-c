# E55：只拟合 reference 位置与形状

从头训练 seed0；固定 Loss = center_local + 0.25 × shape。全部额外几何惩罚关闭。

训练状态：early_stop；43轮，best40。
best同口径 dev500 Loss：0.245273838；train_monitor500：0.174318209。
原始draft硬合法：0/500。
flash500最终硬合法：500/500，加权Q 1.051851553，P95 1.216195203，最大 1.917892186。
完整均时 0.291117秒；同E42 flash胜/平/负：155/35/310。

草稿主判断应结合[同100例reference／本轮draft／E42 draft](draft_comparison100/index.html)和固定Loss。Q仅作辅助；原始几何及修复负担见analysis.json与terminal/results.json。

本次为已暴露开发集的单种子机制实验，不是独立盲测；不得与E42历史总Loss横向比较。既有E42和pro均未修改或重新选择。
