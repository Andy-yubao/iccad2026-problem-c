# E10 整合与工作区归档

2026-10-18。清理前已提交并推送 `88e7d7c02bef5b3e6ab25e8c884bf8a26153d0e6`。当前入口转为 [E10](../../experiments/E10_integrated_baseline/README.md)，baseline保留。

E10继承 E7 attention_geometry 冻结best与 E8 conic_robust，源码按依赖独立复制，不继承E9退化候选。模型best/latest及同组训练账本移入E10，只保留一份。下列旧实验整体移到本目录，历史命令、哈希、路径、日志和结果均不改写；必要证据与唯一权重保留，后续不默认恢复。

- [conflict_local_repair](experiments/conflict_local_repair/README.md)
- [E8_fast_robust_postprocess](experiments/E8_fast_robust_postprocess/README.md)
- [E9_model_postprocess_alignment](experiments/E9_model_postprocess_alignment/README.md)
- [graph_proposal](experiments/graph_proposal/README.md)
- [intent_aware_repair](experiments/intent_aware_repair/README.md)
- [masked_construction](experiments/masked_construction/README.md)
- [repair_friendly_draft](experiments/repair_friendly_draft/README.md)
- [topology_geometry](experiments/topology_geometry/README.md)

原实验索引及短期计划已由清理前Git提交保存。冻结划分/工作集保持字节不变，未新增训练、未扩读测试；五例开发迁移验证归E10的runs。

精确旧新路径与逐文件字节/哈希在 [manifest.json](manifest.json)。原路径是运行发生时的事实；浏览历史文件内的跨实验旧链接或恢复脚本时，使用本映射定位，不将旧命令直接作为当前入口。原E7 attention_geometry训练子目录已移至E10，其父实验目录其余内容在本历史目录。所有搬迁文件逐一核验SHA-256；复制的算法源文件保持字节一致。

恢复查看源码可用 `git show <snapshot_commit>:<old_path>`；完整旧工作区可从该提交另建checkout，忽略的权重按manifest从本机迁移位置恢复。不要覆盖当前E10，不自动恢复退役研究。Git中没有数据、环境或checkpoint，远程push不等于这些本地文件的备份。
