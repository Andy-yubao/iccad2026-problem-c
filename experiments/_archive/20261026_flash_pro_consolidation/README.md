# 2026-10-26 flash／pro 工作区整理

按用户“规则照旧，只保留 flash、pro 后处理及共用 draft”整理。日常实验区保留 E42 与 E50，25个其他实验（baseline、E25、E30–E41、E43–E49、E51–E54）同盘归档到 experiments/，不恢复旧研究计划。更早 history 和 output 交付件保持原位。

共用 draft 为 E42 best51；E50 直接引用该权重。E25 的共享 train/dev 缓存迁入 E42 runs/prepared_v5_e10_cache_v1，无重复副本。数值内核和模型结构未改变；当前辅助入口只修正资源路径。历史源码身份校验保持严格，不能以本次路径调整为由绕过校验或覆盖、续跑旧运行。

[manifest.json](manifest.json)记录逐文件旧新路径、大小、mtime及源码／权重SHA256，6054个迁移文件完成核对。大缓存不重复全量哈希。历史 JSON、日志、checkpoint、冻结划分均不改写。目录名中“归档”表示退役证据，不表示待运行实验。

整理前实际源码保存于本机 pre_cleanup_sources.zip，已跟踪未提交差异保存于[patch](pre_cleanup_tracked_changes.patch)，[Git状态](pre_cleanup_git_status.txt)记录当时未跟踪范围；不是一次 Git 提交或远程同步。全部原实验运行证据随目录迁移，唯一权重和用户评价保留。仅删除可重建 Python 字节缓存；中途占用导致首轮删除清单未落盘，不声称有逐文件删除计数。

运行状态通过系统进程列表核实；关闭了 E51:8768 和 E53:8769 两个遗留 http.server 看图服务以释放日志占用，未停止其他项目进程。[路径调整清单](navigation_changes.json)说明当前改动。[迁移脚本](archive_workspace.py)、[中断恢复脚本](finish_migration.py)、[导航修订脚本](update_navigation.py)只作审计，不重复执行。

恢复历史时先查 manifest 的 moves。确认目标空闲后，将需要的目录移回旧位置；共享缓存依赖按其单独映射定位。旧 README 相对链接及记录中的绝对路径保持发生时事实，不自动代表可直接恢复运行。

整理没有训练、推理、求解、重评分或读取封存测试。后续用户新授权实验单独记录。
