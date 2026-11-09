"""Authorized same-volume archival, gated on a real final-run optimizer update."""
from pathlib import Path
import ctypes,json,os,shutil,subprocess,time
from .common import ROOT,HERE,RUN,ARCHIVE,read,save,sha,utc

def alive(pid):
    h=ctypes.windll.kernel32.OpenProcess(0x1000,False,int(pid))
    if not h:return False
    try:
        code=ctypes.c_ulong();ctypes.windll.kernel32.GetExitCodeProcess(h,ctypes.byref(code));return code.value==259
    finally:ctypes.windll.kernel32.CloseHandle(h)

def main():
    beat=read(RUN/'L/heartbeat.json');progress=read(RUN/'L/progress.json')
    assert beat['actual_optimizer_updates'] and beat['steps']>=1 and progress['step' if 'step' in progress else 'steps']>=1
    assert alive(beat['pid']),'Final training process must be alive before archival'
    assert HERE.resolve().is_relative_to(ROOT.resolve()) and ARCHIVE.resolve().is_relative_to(ROOT.resolve())
    ARCHIVE.mkdir(parents=True,exist_ok=True)
    if (ARCHIVE/'manifest.json').exists():raise RuntimeError('Archive already registered; do not repeat')
    before=ARCHIVE/'documents_before';before.mkdir(exist_ok=True)
    for name in ('README.md','AGENTS.md','docs/protocol.md','docs/workspace_structure.md','docs/datasets.md','experiments/README.md'):
        p=ROOT/name;t=before/name;t.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,t)
    for args,name in [(['git','status','--short'],'git_status_before.txt'),(['git','diff','--stat'],'git_diff_stat_before.txt')]:
        (ARCHIVE/name).write_text(subprocess.check_output(args,cwd=ROOT,text=True,encoding='utf-8',errors='replace'),encoding='utf-8')
    targets=[p for p in (ROOT/'experiments').iterdir() if p.is_dir() and p!=HERE and p.name!='__pycache__']
    targets += [p for p in (ROOT/'output').iterdir()] if (ROOT/'output').exists() else []
    if (ROOT/'scripts').exists():targets.append(ROOT/'scripts')
    identities=[];moves=[]
    for source in targets:
        source=source.resolve();relative=source.relative_to(ROOT);target=(ARCHIVE/relative).resolve()
        assert source.is_relative_to(ROOT.resolve()) and target.is_relative_to(ARCHIVE.resolve()) and not target.exists()
        assert source!=HERE.resolve() and not HERE.resolve().is_relative_to(source)
        # Every file is inventoried; small source/evidence and all weights get SHA.
        files=[source] if source.is_file() else [p for p in source.rglob('*') if p.is_file()]
        for p in files:
            if p.is_symlink():raise RuntimeError('Review symlink before archival: '+str(p))
            st=p.stat();hash_needed=p.suffix.lower() in ('.py','.md','.ps1','.toml','.yml','.yaml') or (p.suffix.lower() in ('.pt','.pth') and ('checkpoints' in p.parts or 'best' in p.name or 'latest' in p.name)) or st.st_size<=65536
            identities.append(dict(old=p.relative_to(ROOT).as_posix(),new=(target/p.relative_to(source)).relative_to(ROOT).as_posix() if p!=source else target.relative_to(ROOT).as_posix(),size=st.st_size,mtime_ns=st.st_mtime_ns,sha256=sha(p) if hash_needed else None))
        moves.append(dict(old=relative.as_posix(),new=target.relative_to(ROOT).as_posix(),file_count=len(files)))
    save(ARCHIVE/'manifest.json',dict(status='registered_before_moves',created_utc=utc(),authorization='User: after confirmed final training start retain final solver only in active workspace; archive old experiments preserving evidence and effective caches',final_training_pid=beat['pid'],actual_optimizer_steps=beat['steps'],moves=moves,files=identities,large_cache_check='same-volume rename; every size/mtime preserved; existing producer hashes remain unchanged; not a destructive deletion'))
    for move in moves:
        source=(ROOT/move['old']).resolve();target=(ROOT/move['new']).resolve();assert source.is_relative_to(ROOT.resolve()) and target.is_relative_to(ARCHIVE.resolve())
        target.parent.mkdir(parents=True,exist_ok=True);source.rename(target)
        print('Archived',move['old'],'files',move['file_count'],flush=True)
    for item in identities:
        p=ROOT/item['new'];st=p.stat();assert (st.st_size,st.st_mtime_ns)==(item['size'],item['mtime_ns']),item['new']
        if item['sha256']:assert sha(p)==item['sha256'],item['new']
    manifest=read(ARCHIVE/'manifest.json');manifest.update(status='verified_complete',verified_utc=utc(),file_count=len(identities),sha256_checked_files=sum(x['sha256'] is not None for x in identities),preserved_bytes=sum(x['size'] for x in identities));save(ARCHIVE/'manifest.json',manifest)
    (ARCHIVE/'README.md').write_text('# 2026-11-03 最终交付整理\n\n用户结束方法研究并选定C5＋原完整后处理。确认E101最终64k运行有真实Adam更新且训练进程存活后，将全部旧实验、旧输出和历史辅助脚本同盘归档。无删除、无重训旧实验、无修改旧JSON或唯一权重。\n\n所有文件旧新位置、大小、mtime及源码／权重哈希见manifest.json。大缓存保留原生产者清单，移动后全部文件size/mtime逐项一致；不声称重哈希全部大型数值载荷。旧记录中的原路径保持发生时事实，由本清单解释映射。当前最终目录experiments/E101_final_c5_solver独立持有源码和C5起点，运行缓存已独立准备；仍需复用的E42有效缓存由新路径映射解析。\n\nGit已有改动保留，未重置或批量提交；源码和权重保护，测试只在最终冻结后按授权打开。\n',encoding='utf-8')
    print('ARCHIVAL VERIFIED',len(identities),'files',manifest['preserved_bytes'],'bytes',flush=True)

if __name__=='__main__':main()
