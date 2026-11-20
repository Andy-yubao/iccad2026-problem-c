"""Resume registered archival file by file when a directory has an open handle."""
from pathlib import Path
from .common import ROOT,ARCHIVE,read,save,sha,utc

def main():
    manifest=read(ARCHIVE/'manifest.json');assert manifest['status']=='registered_before_moves';files=manifest['files'];left=[]
    for i,item in enumerate(files):
        a=(ROOT/item['old']).resolve();b=(ROOT/item['new']).resolve()
        assert a.is_relative_to(ROOT.resolve()) and b.is_relative_to(ARCHIVE.resolve())
        if a.exists():
            assert not b.exists(),item['new'];b.parent.mkdir(parents=True,exist_ok=True)
            try:a.rename(b)
            except PermissionError:
                import shutil
                shutil.copy2(a,b);left.append(item['old'])
        st=b.stat();assert (st.st_size,st.st_mtime_ns)==(item['size'],item['mtime_ns'])
        if item['sha256']:assert sha(b)==item['sha256'],item['new']
        if (i+1)%1000==0:
            save(ARCHIVE/'progress.json',dict(verified_files=i+1,total_files=len(files),source_files_retained_due_open_handle=left,updated_utc=utc()));print('Archive verified',i+1,'/',len(files),flush=True)
    empty_left=[]
    for move in manifest['moves']:
        source=ROOT/move['old']
        if not source.exists() or source.is_file():continue
        dirs=[p for p in source.rglob('*') if p.is_dir()]+[source]
        for p in sorted(dirs,key=lambda p:len(p.parts),reverse=True):
            assert p.resolve().is_relative_to(ROOT.resolve()) and p.resolve()!=ROOT.resolve()
            if not any(p.iterdir()):
                try:p.rmdir()
                except PermissionError:empty_left.append(p.relative_to(ROOT).as_posix())
    manifest.update(status='verified_complete' if not left else 'verified_archived_copies_with_locked_originals',verified_utc=utc(),file_count=len(files),sha256_checked_files=sum(x['sha256'] is not None for x in files),preserved_bytes=sum(x['size'] for x in files),locked_original_files=left,empty_directory_handles=empty_left,initial_directory_rename_error='E100 directory WinError5; resumed per-file without overwriting originals')
    save(ARCHIVE/'manifest.json',manifest)
    (ARCHIVE/'README.md').write_text('# 2026-11-03 最终交付归档\n\n用户选定C5＋原完整后处理并授权后台64k重训。确认真实Adam更新后，同盘归档所有旧实验、旧输出及辅助脚本。首次E100目录整体改名遇Windows目录占用；按已登记清单逐文件迁移，逐项核对size/mtime，源码、小证据及唯一权重另核对SHA256。被占用原件若存在则保护保留，详见manifest.json；没有删除唯一证据或强行终止其他应用。\n\nmanifest.json解释旧记录到新位置的映射。旧JSON、历史命令和权重不改写。大缓存沿用生产者哈希，全部size/mtime核验，不声称重新哈希所有大型数值载荷。Git已有改动保留，未批量提交或push。\n\n日常唯一求解器：experiments/E101_final_c5_solver。运行有效缓存独立准备，E42旧缓存仍由归档映射供必要复用。\n',encoding='utf-8')
    print('Archival complete',len(files),'files; locked originals',len(left),'empty handles',len(empty_left),flush=True)
if __name__=='__main__':main()
