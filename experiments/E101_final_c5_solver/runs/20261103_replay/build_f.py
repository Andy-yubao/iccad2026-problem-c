"""Prepare an independent historical C5 F runtime; no model inference."""
from pathlib import Path
import json,shutil,os,ast
from prepare import ROOT,BUNDLE,ARCHIVE,link,junction,sha,save,read
OWNER='E101_final_c5_solver'
def main():
    if (BUNDLE/'state.json').exists():raise RuntimeError('Prepared runtime is protected once execution begins')
    root=BUNDLE/'sf';here=root/'experiments'/OWNER;here.mkdir(parents=True,exist_ok=True)
    original=ROOT/'experiments'/OWNER
    names=['__init__.py','common.py','data.py','model.py','loss_f.py','train.py','solver.py','evaluate.py']
    for name in names:shutil.copy2(original/name,here/name)
    for directory in ['nn','repair','placer']:
        for source in (original/directory).rglob('*.py'):
            target=here/source.relative_to(original);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    stage74=BUNDLE/'s74'
    for directory in ['datasets','scoring','third_party','visualization']:
        base=stage74/directory
        if base.exists():
            for source in base.rglob('*.py'):
                # The only reparse points in datasets are tensor/public-data roots.
                if 'floorset_lite' in source.parts or 'LiteTensorDataTest' in source.parts:continue
                target=root/source.relative_to(stage74);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    (root/'visualization').mkdir(exist_ok=True);shutil.copy2(ROOT/'visualization/layouts.py',root/'visualization/layouts.py')
    # Official evaluator imports these loader/plot modules at import time,
    # even though final scoring does not instantiate their dataset classes.
    for name in ('litetestLoader.py','liteLoader.py','lite_dataset.py','lite_dataset_test.py','visualize.py','prime_dataset.py'):
        shutil.copy2(ROOT/'third_party/FloorSet'/name,root/'third_party/FloorSet'/name)
    (root/'protocol').mkdir(exist_ok=True)
    for source in (ROOT/'protocol').glob('*.json'):shutil.copy2(source,root/'protocol'/source.name)
    for _,name in read(ROOT/'protocol/paths.json').items():junction(ROOT/name,root/name)
    p=here/'common.py';text=p.read_text();text=text.replace("RUN=HERE/'runs/20261103_final64k_seed0'","RUN=HERE/'runs/replay_F_seed0'");text=text.replace("'runs' not in p.parts", "'runs' not in p.relative_to(HERE).parts");text=text.replace("    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(files))}","    files += [ROOT/'third_party/FloorSet/iccad2026contest/iccad2026_evaluate.py',ROOT/'third_party/FloorSet/cost.py',ROOT/'third_party/FloorSet/utils.py']\n    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(files))}");p.write_text(text,encoding='utf-8')
    p=here/'data.py';text=p.read_text().replace("rows=meta[role]['records'];self.members", "rows=meta[role]['records'];rows=rows[:16000] if role=='train' else rows;self.members");p.write_text(text,encoding='utf-8')
    p=here/'train.py';text=p.read_text();old="if elapsed()>=cfg['max_seconds'] or state['epoch']>=cfg['max_epochs']:";assert old in text;text=text.replace(old,"if False:  # Authorized normal-early-stop only; no wall/epoch cap.");text=text.replace('weights-only C5 migration; fresh optimizer/schedule','historical C3 refiner weights + new E74 placer; fresh F optimizer/schedule');p.write_text(text,encoding='utf-8')
    p=here/'evaluate.py';text=p.read_text();text=text.replace('local final held-out evaluation of the public-MIB-mean-area task; not official contest score; original v2 test exposure recorded in protocol','Repeated evaluation on the registered2000 previously exposed by E101; not a fresh unseen holdout or official contest score; no test feedback');text=text.replace('scope=\'Repeated','scope=\'Repeated');p.write_text(text,encoding='utf-8')
    src=ARCHIVE/'E97_gpu_hybrid_solver/runs/20261102_candidate3_clearance_seed0/checkpoints/best.pt';assert sha(src)=='d784e96f46decf7a487c25bde2ab16db8cf6ebae1bc0632434cc5e2f78173fce';link(src,here/'assets/c5_baseline.pt')
    original_run=original/'runs/20261103_final64k_seed0';run=here/'runs/replay_F_seed0';run.mkdir(parents=True,exist_ok=True)
    for role in ['train','dev']:
        for source in (original_run/'cache'/role).iterdir():
            if source.is_file():link(source,run/'cache'/role/source.name)
    plan=dict(revision='E101.historical_chain.F.v1',authorization='User2026-11-03: full E42->E74->historical C5 F training through dev500 and registered repeatedtest2000; normal early-stop only; no test feedback',source_sha256=sha(src),source_checkpoint=str(src),data=dict(train_size=16000,train='original v5 first16k, read through nestedv6 prefix',dev='same devmonitor500',test='same registered2000, already exposed E101'),stages=[dict(name='F',lr_levels=[.00025,.0001,.00005,.00002],max_seconds=None,max_epochs=None)],repair=read(original/'plan.json')['repair'],teacher='Original E85 target tensors unchanged; new E74 best supplies base only',selection='original C5 identical count-weighted total for backward/monitor/best/LR/earlystop',scope='No new architecture, loss weights, coefficients, postprocessor or Q-based checkpoint selection')
    save(here/'plan.json',plan);save(here/'provenance.json',dict(network_loss_repair='Copied independent E101 implementation matching original C5 algorithms; F start is original C3 as in historical C5',baseline_sha256=sha(src),shared_cache='read-only hard links to exact E101 cache; training is first16k only',test='repeated existing2000; no independent unseen claim'))
    for source in here.rglob('*.py'):ast.parse(source.read_text(encoding='utf-8'))
    save(BUNDLE/'F_runtime.json',dict(root=str(root),here=str(here),run=str(run),train_size=16000,source_sha256=sha(src),prepared=True))
    print('Independent F runtime prepared; no inference, optimizer update or test read',flush=True)
if __name__=='__main__':main()
