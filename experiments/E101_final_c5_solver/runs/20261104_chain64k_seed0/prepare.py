"""Create a separate 64k/64k/64k runtime; never alter the previous run."""
from pathlib import Path
import ast,hashlib,json,shutil,zipfile,datetime
from support import ROOT,BUNDLE,ARCHIVE,E58,read,save,sha,link,junction,mirror,original,STAGES

PREVIOUS=ROOT/'experiments/E101_final_c5_solver/runs/20261103_replay'
OWNER='E101_final_c5_solver'
OLDTEACHER=ARCHIVE/'E97_gpu_hybrid_solver/runs/20261101_public_teacher'

def write(p,text):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text,encoding='utf-8')

def shared(root):
    for p in (ROOT/'protocol').glob('*.json'):
        target=root/'protocol'/p.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    for name in read(ROOT/'protocol/paths.json').values():junction(ROOT/name,root/name)
    target=root/'history/20261029_model_research_consolidation/experiments/E58_training_data_v6';target.mkdir(parents=True,exist_ok=True)
    for p in E58.glob('*.py'):shutil.copy2(p,target/p.name)
    mirror(E58/'runs/20261026_reference_v6_64k',target/'runs/20261026_reference_v6_64k')
    mirror(ARCHIVE/'E42_joint_regularization/runs/prepared_v5_e10_cache_v1',root/'experiments/E42_joint_regularization/runs/prepared_v5_e10_cache_v1')

def numeric_cache(folder):
    source=ROOT/'experiments/E101_final_c5_solver/runs/20261103_final64k_seed0/cache'
    identity={'registry_sha256':sha(ROOT/'protocol/reference_workset_v6.json'),'arrays':{}}
    for role in ('train','dev'):
        identity['arrays'][role]={}
        for p in (source/role).iterdir():
            if not p.is_file():continue
            link(p,folder/role/p.name)
            if p.suffix=='.npy':
                s=p.stat();identity['arrays'][role][p.name]=dict(size=s.st_size,mtime_ns=s.st_mtime_ns,source=str(p))
    save(folder/'manifest.json',identity)

def main():
    assert not (BUNDLE/'state.json').exists(),'Runtime preparation cannot overwrite an executing run'
    audits=[]
    for label,(owner,run,sub) in STAGES.items():
        old=ARCHIVE/owner/'runs'/run;root=BUNDLE/sub;cfg=read(old/'config.json')
        with zipfile.ZipFile(old/'source_snapshot.zip') as z:
            for name in z.namelist():
                payload=z.read(name);digest=hashlib.sha256(payload).hexdigest();assert cfg['identity']['sources'][name]==digest
                p=root/name;assert p.resolve().is_relative_to(root.resolve());p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(payload)
                audits.append(dict(stage=label,source=name,sha256=digest))
        # Both stages use exactly the same frozen v6 64k view and shared tensors.
        s74=PREVIOUS/'s74/experiments/E74_collision_32k_posttrain';here=root/'experiments'/owner
        (root/'datasets/expanded_reference').mkdir(parents=True,exist_ok=True)
        shutil.copy2(PREVIOUS/'s74/datasets/expanded_reference/__init__.py',root/'datasets/expanded_reference/__init__.py')
        shared(root)
        cache=(s74/'cache.py').read_text(encoding='utf-8').replace('32000','64000').replace('nested32k','nested64k')
        write(here/'cache.py',cache)
        write(here/'dataset_view.py',(s74/'dataset_view.py').read_text(encoding='utf-8').replace('32000','64000').replace('nested32k','nested64k'))
        shutil.copy2(BUNDLE/'stream_data.py',here/'data.py');numeric_cache(here/'runs/prepared64k_memmap')
        text=(here/'train.py').read_text(encoding='utf-8')
        text=text.replace("'max_seconds': args.max_hours*3600, 'max_epochs': args.max_epochs","'max_seconds': None, 'max_epochs': None")
        cap=4 if label=='E42' else 2
        text=text.replace(f'if args.max_hours > {cap} or args.max_epochs > 200 or args.batch_size != 32:', 'if args.batch_size != 32:')
        text=text.replace("def exhausted(): return elapsed()>=identity['max_seconds']",'def exhausted(): return False')
        text=text.replace("while progress['epoch']<args.max_epochs or progress['pending_monitor']:",'while True:')
        if label=='E42':
            text=text.replace('from datasets.reference_layouts import ReferenceLayouts, REGISTRY, registry','from .dataset_view import ReferenceLayouts, REGISTRY, registry')
            text=text.replace("'revision': REVISION,", "'revision': REVISION+'.v6_64k_memmap',")
            text=text.replace('Read-only frozen E25/E17 train/dev tensors','Exact FP32 disk memmap; original E58 nested64k; dynamic original E42 padding')
            io=(here/'io.py').read_text(encoding='utf-8')
            marker="    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(set(files))}"
            assert marker in io
            write(here/'io.py',io.replace(marker,"    files += [ROOT/'datasets/expanded_reference/__init__.py', HERE/'plan.json']\n"+marker))
        else:
            shutil.copy2(BUNDLE/'stream_resident.py',here/'resident.py')
            text=text.replace("'dataset_size':32000","'dataset_size':64000").replace('nested32k','nested64k')
            text=text.replace("'gpu_resident_inputs': args.engine != 'reference'","'gpu_resident_inputs': False, 'input_engine': 'exact_FP32_disk_memmap_pinned_batch'")
            text=text.replace("resident_train.batch(order_cuda[progress['cursor']:progress['cursor']+len(ids)],out=captured.inputs)","resident_train.batch(ids,out=captured.inputs)")
            text=text.replace('GPU resident inputs:', 'Disk memmap staging; dataset VRAM allocation:')
            provenance=read(here/'provenance.json')
            for name,digest in provenance['protected_checkpoints_sha256'].items():
                source=original(name);assert sha(source)==digest;link(source,root/name)
        write(here/'train.py',text)
        stageplan=read(here/'plan.json') if (here/'plan.json').exists() else {'historical_source':str(old/'config.json')};stageplan['new_run_override']=dict(run='20261104_chain64k_seed0',train_size=64000,registry_sha256=sha(root/'protocol/reference_workset_v6.json'),initialization='random96x12' if label=='E42' else 'new E42 unique best, exact widening',time_limit=None,epoch_limit=None,input_engine='exact FP32 disk memmap; original mathematical batch and Loss',original_plan='Historical fields retained as provenance; this override is the current data/budget registration')
        save(here/'plan.json',stageplan)
        link(old/'checkpoints/best.pt',BUNDLE/'historical'/label/'best.pt')
        print(label,'64k runtime restored and linked',flush=True)
    # F runtime is copied independently from the working, repaired C5 runtime.
    root=BUNDLE/'sf';here=root/'experiments'/OWNER;source=PREVIOUS/'sf/experiments'/OWNER
    for p in source.rglob('*.py'):
        if 'runs' in p.relative_to(source).parts:continue
        target=here/p.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    for folder in ('datasets','scoring','visualization','third_party'):
        for p in (PREVIOUS/'sf'/folder).rglob('*.py'):
            if '__pycache__' in p.parts:continue
            target=root/p.relative_to(PREVIOUS/'sf');target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    shared(root);run=here/'runs/replay_F_seed0';numeric_cache(run/'cache')
    text=(here/'data.py').read_text(encoding='utf-8');text=text.replace("rows=rows[:16000] if role=='train' else rows;",'');write(here/'data.py',text)
    # Include the full official import closure in the frozen dependency list.
    text=(here/'common.py').read_text(encoding='utf-8');needle='    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(files))}'
    assert needle in text
    text=text.replace(needle,"    files += list((ROOT/'third_party/FloorSet').glob('*.py'))\n"+needle);write(here/'common.py',text)
    fplan=read(source/'plan.json');fplan.update(revision='E101.all_stages64k.v1',authorization='User2026-11-04 explicitly authorizes preparing then background-running E42(64k)->E74(64k)->C5F(64k)->freeze->dev500->registered repeatedtest2000; normal early stop, no time/epoch cap')
    fplan['data']['train_size']=64000;fplan['data']['train']='original frozen v6 nested64k for all stages'
    fplan['teacher']='Original E85 targets for prefix16k/dev500; extend train indices16000..63999 using same frozen E85 model and FULL teacher repair; new E74 supplies only F base'
    save(here/'plan.json',fplan);save(here/'provenance.json',dict(source_runtime=str(source),F_start='historical C3, same as historical C5 and preceding replay',new_run='E42 random64k -> E74 64k -> F64k',data_engine='exact existing FP32 memmaps; numerical path registered separately from original resident dataset',test='registered repeated2000; no selection or training feedback'))
    link(source/'assets/c5_baseline.pt',here/'assets/c5_baseline.pt')
    tc=read(OLDTEACHER/'config.json');tpath=original(tc['candidate_plan']['source_checkpoint']);assert sha(tpath)==tc['checkpoint_sha256'];link(tpath,here/'assets/teacher_E85.pt')
    with zipfile.ZipFile(OLDTEACHER/'source_snapshot.zip') as z:
        for name in z.namelist():
            if name.startswith(('nn85/','repair85/')) and name.endswith('.py'):
                payload=z.read(name);assert hashlib.sha256(payload).hexdigest()==tc['source_sha256'][name]
                folder,tail=name.split('/',1);target=here/('teacher_nn' if folder=='nn85' else 'teacher_repair')/tail;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(payload)
    save(here/'teacher_source.json',dict(checkpoint_sha256=tc['checkpoint_sha256'],FULL=tc['FULL'],source_config_sha256=sha(OLDTEACHER/'config.json'),workers=4,worker_torch_threads=1,worker_solver_threads=2,seed='per original index; np.random.seed(index), torch.manual_seed(index)',prefix='existing16k/dev500 copied unchanged',expansion='only original train indices16000..63999',batch=32,precision='FP32',free_reference=False))
    save(BUNDLE/'F_runtime.json',dict(root=str(root),here=str(here),run=str(run),train_size=64000))
    save(BUNDLE/'frozen_source_manifest.json',audits)
    for folder in ('s42','s74','sf'):
        for p in (BUNDLE/folder/'experiments').rglob('*.py'):
            if 'runs' not in p.relative_to(BUNDLE/folder/'experiments').parts:ast.parse(p.read_text(encoding='utf-8'))
    print('Prepared all three64k runtimes; no model update, teacher generation or test read',flush=True)

if __name__=='__main__':main()
