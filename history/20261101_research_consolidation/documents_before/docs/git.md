# Git 与 GitHub 使用约定

本文件说明本工作区的版本管理。实验和文件归属见 [结构规范](workspace_structure.md)，研究边界见 [执行协议](protocol.md)。

## 1. 当前连接

| 项目 | 配置 |
|---|---|
| GitHub 仓库 | [Andy-yubao/iccad2026-problem-c](https://github.com/Andy-yubao/iccad2026-problem-c) |
| 可见性 | Public，公开仓库 |
| 本地主分支 | `main` |
| 远程名 | `origin` |
| 远程地址 | `https://github.com/Andy-yubao/iccad2026-problem-c.git` |

工作区文件是正在编辑的内容；暂存区决定下一次提交包含什么；本地提交保存可追溯版本；push 将提交同步到 GitHub。新修改需要提交并推送后才会出现在远程。

以下命令均在工作区根目录运行。当前已有 Git 仓库，不重复执行 init 或创建远程仓库。

## 2. 日常保存与同步

先查看改动和分支状态：

```powershell
git status --short --branch
git diff --stat
git diff -- README.md docs/git.md
git log -5 --oneline
```

完成与本次改动相关的轻量检查后，明确选择提交文件，复核暂存内容，再提交和推送。例如修改本文和入口：

```powershell
git add docs/git.md README.md
git diff --cached --stat
git diff --cached -- docs/git.md README.md
git commit -m "Update workspace Git documentation"
git push origin main
git status --short --branch
```

删除也需要提交。对已跟踪目录可以使用 `git add -u -- <目录>` 暂存其中的修改和删除；新文件仍需显式 add。避免在未检查内容时一次暂存全部工作区。

提交信息描述具体变化和目的。一次提交围绕一件有意义的事，不将无关实验、数据调整和文档重写混为一谈。实验运行记录中保存实际源码提交；尚未提交的改动必须如实注明并保留必要差异，不能只记录旧 HEAD。

## 3. 从 GitHub 获取更新

先确认本地改动已妥善保存，再获取更新：

```powershell
git status --short --branch
git fetch origin
git log --oneline main..origin/main
git pull --ff-only origin main
```

`fetch` 更新远程分支信息；`pull --ff-only` 只接受可直接前进的更新。若提示分支分叉或存在冲突，先检查本地和远程各自的提交，再决定合并方式；保留双方有用改动，不用强推覆盖远程。

## 4. 入库范围

提交代码、核心文档、官方来源清单、冻结划分与必要的小型基线证据。实验内容按所属目录维护，不能因为使用 Git 就增加第二套源码或结果目录。

以下内容由 `.gitignore` 排除：数据压缩包和解压数据、`.conda/` 等环境、`.cache/`、Python 缓存、checkpoint、本地环境变量与凭据文件。仓库为公开仓库，提交前检查文件列表与内容，避免把本地秘密加入暂存区。

`.gitignore` 对已经跟踪的文件不追溯生效。文件首次入库前应核对范围；若发现误入库，先明确是清理当前版本还是清理全部历史，再处理。

`.gitattributes` 使用 `* -text`，本仓库关闭自动换行转换，以保留官方原件、清单和运行证据的字节及哈希。不要为了格式统一批量改写上游文件或历史结果。

## 5. 删除、恢复与实验边界

普通删除并提交后，文件从工作区和仓库当前版本中移除；旧提交仍保存历史。若用户要求连 Git 历史也清除，应另行明确范围和远程影响，再执行历史清理。

恢复文件前先查看历史内容，确认不会覆盖当前有用改动。已经明确废弃的方向不因历史中仍有文件就自动恢复；架构、权重和实验计划的重新引入必须来自新的明确请求。

Git 不保存未跟踪且被忽略的数据、环境和权重。克隆仓库后，按 [数据说明](datasets.md) 与 `environment.yml` 恢复运行条件；保持冻结索引不变，禁止为配合新机器重新划分数据。第三方源码是固定版本快照，版本记录见 [来源索引](../references/README.md)。
