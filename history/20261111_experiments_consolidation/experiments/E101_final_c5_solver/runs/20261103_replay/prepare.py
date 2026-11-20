"""Restore the original E42/E74 runtime snapshots and immutable shared inputs."""
from pathlib import Path
import hashlib,json,os,shutil,zipfile,datetime
ROOT=Path(__file__).resolve().parents[4]
BUNDLE=Path(__file__).resolve().parent
ARCHIVE=ROOT/'history/20261103_final_delivery_consolidation/experiments'
E58=ROOT/'history/20261029_model_research_consolidation/experiments/E58_training_data_v6'
STAGES={'E42':('E42_joint_regularization','20261023_rectangle_w96d12_ab_seed0','s42'),'E74':('E74_collision_32k_posttrain','20261029_e42_topology_seed0','s74')}
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def link(a,b):
    a=a.resolve();b=b.absolute();assert a.is_relative_to(ROOT) and b.is_relative_to(BUNDLE)
    b.parent.mkdir(parents=True,exist_ok=True)
    if b.exists():assert os.path.samefile(a,b)
    else:os.link(a,b)
def mirror(a,b):
    for p in a.rglob('*'):
        if p.is_file() and p.suffix in ('.pt','.pth','.json'):link(p,b/p.relative_to(a))
def original(name):
    for base in [ROOT,ROOT/'history/20261103_final_delivery_consolidation',ROOT/'history/20261101_research_consolidation',ROOT/'history/20261029_model_research_consolidation',ROOT/'history/20261026_flash_pro_consolidation']:
        p=base/name
        if p.exists():return p
    raise FileNotFoundError(name)
def junction(a,b):
    # Historical dataset_paths requires resolved files to remain under its
    # isolated ROOT. Directory junctions escape that guard; file hardlinks do not.
    import stat
    a=a.resolve();b=b.absolute();assert a.is_relative_to(ROOT) and b.is_relative_to(BUNDLE)
    if b.exists() and b.lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
        assert b.resolve()==a,'Refuse to replace an unrelated directory mapping'
        os.rmdir(b)  # Remove only the verified junction itself, never its target.
    b.mkdir(parents=True,exist_ok=True);assert b.resolve()==b
    for source in a.rglob('*'):
        if source.is_file():link(source,b/source.relative_to(a))

def main():
    if (BUNDLE/'state.json').exists():raise RuntimeError('Prepared runtime is protected once execution begins')
    assert (ROOT/'protocol/paths.json').exists(),ROOT
    facts={};source_audit=[];adaptations=[]
    for label,(owner,run,sub) in STAGES.items():
        old=ARCHIVE/owner/'runs'/run;cfg=read(old/'config.json');root=BUNDLE/sub;root.mkdir(exist_ok=True)
        with zipfile.ZipFile(old/'source_snapshot.zip') as z:
            for name in z.namelist():
                target=(root/name).resolve();assert target.is_relative_to(root.resolve())
                payload=z.read(name);digest=hashlib.sha256(payload).hexdigest();assert cfg['identity']['sources'][name]==digest,name
                target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(payload);source_audit.append(dict(stage=label,path=name,sha256=digest))
        evidence=BUNDLE/'historical'/label;evidence.mkdir(parents=True,exist_ok=True)
        for name in ['config.json','results.json','progress.json','invocations.json','panels.json','budget.json','source_snapshot.zip']:
            shutil.copy2(old/name,evidence/name)
        for tag in ('best','latest'):
            link(old/'checkpoints'/f'{tag}.pt',evidence/f'{tag}.pt')
        for p in (ROOT/'protocol').glob('*.json'):
            target=root/'protocol'/p.name;target.parent.mkdir(exist_ok=True);shutil.copy2(p,target)
        assert sha(root/'protocol/reference_workset_v5.json')=='1409ccf825b9c7cae1097a22e8dc4068e1283d79541c85ce54c5627d0c2b4208'
        paths=read(root/'protocol/paths.json')
        for key in ('training_root','public_validation_root'):junction(ROOT/paths[key],root/paths[key])
        prefix=ARCHIVE/'E42_joint_regularization/runs/prepared_v5_e10_cache_v1'
        cache_path='experiments/E25_integrated_baseline/runs/prepared_v5_e10_cache_v1' if label=='E42' else 'experiments/E42_joint_regularization/runs/prepared_v5_e10_cache_v1'
        mirror(prefix,root/cache_path)
        if label=='E74':
            new_owner=root/'history/20261029_model_research_consolidation/experiments/E58_training_data_v6'
            new_owner.mkdir(parents=True,exist_ok=True)
            for p in E58.glob('*.py'):shutil.copy2(p,new_owner/p.name)
            mirror(E58/'runs/20261026_reference_v6_64k',new_owner/'runs/20261026_reference_v6_64k')
            provenance=read(root/f'experiments/{owner}/provenance.json')
            for name,h in provenance['protected_checkpoints_sha256'].items():
                a=original(name);assert sha(a)==h;link(a,root/name)
        train=root/f'experiments/{owner}/train.py';source=train.read_text(encoding='utf-8');before=sha(train)
        # User explicitly replaces historical time/epoch caps with normal early stopping.
        source=source.replace("'max_seconds': args.max_hours*3600, 'max_epochs': args.max_epochs", "'max_seconds': None, 'max_epochs': None")
        cap=4 if label=='E42' else 2
        old_guard=f"if args.max_hours > {cap} or args.max_epochs > 200 or args.batch_size != 32:"
        assert old_guard in source
        source=source.replace(old_guard,"if args.batch_size != 32:")
        assert "def exhausted(): return elapsed()>=identity['max_seconds']" in source
        source=source.replace("def exhausted(): return elapsed()>=identity['max_seconds']","def exhausted(): return False  # Authorized: no wall-time cap; STOP/finite gates remain.")
        assert "while progress['epoch']<args.max_epochs or progress['pending_monitor']:" in source
        source=source.replace("while progress['epoch']<args.max_epochs or progress['pending_monitor']:","while True:  # Authorized: terminate only on normal early stop, user STOP or error.")
        train.write_text(source,encoding='utf-8');adaptations.append(dict(stage=label,file=str(train.relative_to(BUNDLE)),old_sha256=before,new_sha256=sha(train),change='Remove wall-time and epoch caps only; historical Loss, schedule, LR, optimizer, order, monitor, best and early-stop code unchanged'))
        result=read(old/'results.json');p=read(old/'progress.json')
        facts[label]=dict(owner=owner,original_run=run,original_config=cfg['identity'],original_result=result,original_best_checkpoint_sha256=sha(old/'checkpoints/best.pt'),original_latest_checkpoint_sha256=sha(old/'checkpoints/latest.pt'),best_epoch=p['best_epoch'],latest_epoch=p['epoch'],latest_steps=p['step'],latest_cursor=p['cursor'],source_snapshot_sha256=sha(old/'source_snapshot.zip'))
        print(label,'restored source and shared cache paths',flush=True)
    plan=dict(revision='E101.E42_to_E74.normal_earlystop.replay.v1',registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),authorization='User explicitly requests both E42 and E74 train until normal early stop, ignoring historical budgets',training_members=dict(E42='v5 first16k',E74='v6 nested32k',monitor='unchanged original train/dev500',split='unchanged split_v1'),stages=[dict(name='E42',initialization='random 96x12',backward='Original eight-term G, geometric weights ramp over first2 fractional epochs',selection='Historical G+occupancy, not backward total; explicit historical reproduction exception',optimizer='AdamW fresh, lr1e-3, decay3e-4, grad clip1, FP32 batch32 seed0, CPU4',stop='Original plateau controller, protection15, three bad checks per drop; tiers1e-3/2.5e-4/1e-4/5e-5; lowest at least2epochs; normal early_stop only'),dict(name='E74',initialization='Stage E42 unique best; exact function-preserving 96to192 widening; graph-local mappings frozen; fresh optimizer',backward='Original fixed G+0.5 collision-gated C',selection='Same fixed total mean on original dev500',optimizer='Original guarded CUDA graph with0.5percent joint module step bound; fresh AdamW same four LR tiers; FP32 batch32 seed0 CPU4',stop='Original plateau controller to normal early_stop; no time cap or epoch cap')],limits=dict(time=None,epochs=None,numeric_gates='preserved; error saves and exits',user_stop='preserved; blocks next stage'),historical_difference='Original E74 stopped on7200s budget at42338steps, best42; new run intentionally continues to normal early stop, so it is a convergence extension, not a byte-identical historical replay',scope='Only E42/E74 placer stages; no F candidate, postprocessor search, devQ checkpoint selection, or new test evaluation',historical_facts=facts,training_not_started=True)
    plan['scope']='Full new E42 -> E74 -> historical C5 F training -> dev500 -> frozen repeated local_test2000; no test-based parameter or checkpoint selection'
    plan['F_stage']=dict(initialization='Historical E97 C3 F, same warm start as original round2 C5; new E74 L frozen',teacher='Reuse original E85 public solver teacher target tensors; recompute only L bases for new L',train_size=16000,loss='Exact original C5 count-weighted group-aware teacher/public boundary/dynamicMST',stop='Normal original early stop, no wall-time/epoch cap')
    plan['evaluation']=dict(dev_monitor=500,test=2000,order='All training finished -> unique own-Loss best composition frozen -> complete dev500 -> repeated registered2000 test',test_scope='Explicitly authorized by latest user request; previously exposed by E101, so not a fresh sealed holdout',no_test_feedback=True)
    save(BUNDLE/'plan.json',plan);save(BUNDLE/'frozen_source_manifest.json',source_audit);save(BUNDLE/'adaptations.json',adaptations)
    print('Prepared; no model forward, optimizer update or test payload read',flush=True)

if __name__=='__main__':main()
