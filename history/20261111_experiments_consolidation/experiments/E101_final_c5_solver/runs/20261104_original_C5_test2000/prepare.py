"""Freeze the original selected C5 for its first complete registered test2000."""
from pathlib import Path
import hashlib,json,shutil,sys,zipfile,datetime
ROOT=Path(__file__).resolve().parents[4]
BUNDLE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from experiments.E101_final_c5_solver.common import sources,sha,read,save

def main():
    assert not (BUNDLE/'state.json').exists(),'Do not overwrite a queued or running evaluation'
    owner=ROOT/'experiments/E101_final_c5_solver';old=ROOT/'history/20261103_final_delivery_consolidation/experiments/E97_gpu_hybrid_solver'
    historic=old/'runs/20261102_round2_candidate5_countweighted_seed0/evaluation';frozen=read(historic/'frozen.json');result=read(historic/'results.json')
    checkpoint=owner/'assets/c5_baseline.pt';assert sha(checkpoint)==frozen['checkpoint_sha256']=='028ea3aa124c8cd9e7c3cc7bf8587c699ab9b4932f256899e3780d4d8c8816e0'
    assert result['terminal']['count']==500 and result['raw_count']==2000 and result['test_read'] is False
    repairs=read(owner/'plan.json')['repair'];assert repairs==read(old/'round2_control_plan.json')['full_repair']
    proof=[]
    for name,digest in frozen['source_sha256'].items():
        prefix='experiments/E97_gpu_hybrid_solver/'
        if not name.startswith(prefix):continue
        rel=name[len(prefix):]
        if rel.startswith('repair85_one_thread/'):target=owner/'repair'/rel.split('/',1)[1]
        elif rel.startswith('nn85/'):target=owner/'nn'/rel.split('/',1)[1]
        else:continue
        assert sha(target)==digest;proof.append(dict(historical=name,current=target.relative_to(ROOT).as_posix(),sha256=digest))
    assert (old/'student.py').read_text(encoding='utf-8').replace('.nn85.','.nn.')==(owner/'model.py').read_text(encoding='utf-8')
    registry=read(ROOT/'protocol/reference_workset_v6.json');members=[r['member'] for r in registry['sets']['local_test']['records']];assert len(members)==2000
    previous=[]
    for path in [owner/'runs/20261103_final64k_seed0/evaluation/local_test/frozen.json',owner/'runs/20261103_replay/sf/experiments/E101_final_c5_solver/runs/replay_F_seed0/evaluation/local_test/frozen.json']:
        value=read(path);assert value['members']==members and value['checkpoint_sha256']!=sha(checkpoint)
        previous.append(dict(path=str(path),checkpoint_sha256=value['checkpoint_sha256'],same_2000_members=True))
    available=sources()
    names=('__init__.py','common.py','model.py','solver.py','evaluate.py','plan.json','provenance.json')
    base='experiments/E101_final_c5_solver/'
    identity={n:h for n,h in available.items() if (n.startswith(base) and (n[len(base):] in names or n.startswith(base+'nn/') or n.startswith(base+'repair/'))) or n.startswith('scoring/') or n in ['datasets/'+v for v in ('__init__.py','paths.py','cases.py','reference_layouts.py','worksets.py','expanded_reference/__init__.py')] or n.startswith('third_party/')}
    for p in (ROOT/'third_party/FloorSet').glob('*.py'):identity[p.relative_to(ROOT).as_posix()]=sha(p)
    for name in ('paths.json','reference_workset_v6.json','reference_workset_v5.json','split_v1.json'):identity['protocol/'+name]=sha(ROOT/'protocol'/name)
    save(BUNDLE/'sources.json',identity)
    with zipfile.ZipFile(BUNDLE/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in identity:z.write(ROOT/name,name)
    save(BUNDLE/'identity_proof.json',dict(model_package_rename_only=True,unchanged_NN_and_repair_files=proof,repair_settings_identical=True,historical_dev500=result,existing_completed_test_models=previous,original_C5_full_test_found=False,test_payloads_read=False))
    save(BUNDLE/'plan.json',dict(revision='E101.original_R2C5.test2000.v1',authorization='User2026-11-04 asks whether the previously best trained configuration ran these2000; if not prepare to run it',checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),configuration='Original E97 round2 C5 + original complete E85 single-thread repair',training=False,dev500='Reuse historical completed500, no repeated dev solve',repair=repairs,role='local_test',members=members,registry_sha256=sha(ROOT/'protocol/reference_workset_v6.json'),frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),test_scope='First full test2000 for this original C5 checkpoint, on the registered2000 already exposed by E101 retrains; not an unseen holdout',selection='Immutable original own-Loss best; no test selection, parameter scan or retraining',timing='Same E101 resident batch1 complete-request protocol as the two completed retrains, including public clone/masking/meanarea/features/transfers/L+F/decode/full repair/hard checks; cold/disk/scoring separate',timing_difference_from_historical_dev='E101 additionally returns the L draft for consistent deployment measurement; model and repair math unchanged',serial_after=str(owner/'runs/20261104_chain64k_seed0/state.json'),on_error='Record and exit; no automatic retry',stop='BUNDLE/STOP, checked while queued and between cases; no effect on other runs'))
    # Static load/import check, no GPU context or train/test payload.
    import torch
    from experiments.E101_final_c5_solver.model import DeploymentModel
    from experiments.E101_final_c5_solver.evaluate import summary
    from scoring.official import official_evaluator
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False);model=DeploymentModel(**saved['model_config']);model.load_state_dict(saved['model'],strict=True)
    evaluator=official_evaluator();assert evaluator.SHAPELY_AVAILABLE
    for name in ('iccad2026_evaluate','litetestLoader','liteLoader','lite_dataset','lite_dataset_test','prime_dataset','visualize','cost','utils'):
        assert Path(sys.modules[name].__file__).resolve().is_relative_to(ROOT/'third_party/FloorSet')
    save(BUNDLE/'preflight.json',dict(status='passed',checkpoint_cpu_strict_load=True,official_import_closure=True,source_identity_files=len(proof),same_ordered_test_members=True,test_payloads_read=False,GPU_used=False))
    names=[p for p in BUNDLE.glob('*.py')]+[BUNDLE/'plan.json',BUNDLE/'sources.json']
    save(BUNDLE/'controller_sources.json',{p.name:sha(p) for p in names})
    print('Original C5 frozen; model, repair, members and complete scorer imports verified; no GPU or test read',flush=True)

if __name__=='__main__':main()
