# Git与源码管理

当前本地main、远程origin：[Andy-yubao/iccad2026-problem-c](https://github.com/Andy-yubao/iccad2026-problem-c)，公开仓库。

- 源码、当前文档、冻结索引、来源及必要小型证据入库；数据、环境、缓存、checkpoint、完整坐标和凭据不入库。
- 先检查status/diff，再显式选择文件、核对暂存内容后commit/push。不reset、强推或一次暂存全部工作区，不覆盖已有无关改动。
- .gitignore不影响已经跟踪的文件。检查新文件与迁移删除；.gitattributes关闭自动换行转换，保护原件与历史字节。
- 归档也要保留可恢复源码和迁移清单；Git不替代被忽略的权重/数据备份。运行记录真实提交与未提交快照，不能只写旧HEAD。
- 拉取先保存本地改动，优先fetch检查及pull --ff-only；分叉时先审阅，不自动强行覆盖。
- 用户未要求发布时不擅自push。清理历史/远程或恢复已明确废弃路线须有对应授权。

```powershell
git status --short --branch
git diff --stat
git add <本次明确选择的文件>
git diff --cached --stat
git commit -m "说明具体变化"
```
