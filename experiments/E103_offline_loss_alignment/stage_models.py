"""Stage five completed draft models; no dataset loading, inference or training."""
from pathlib import Path
import hashlib
import importlib
import json
import shutil
import sys
import torch

ROOT=Path(__file__).resolve().parents[2]
OWNER=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

SPECS=[
 dict(id='e42',experiment='E42',owner='history/20261103_final_delivery_consolidation/experiments/E42_joint_regularization',run='20261023_rectangle_w96d12_ab_seed0',reason='Historical mature reference/geometric baseline; compressed public port summaries; supplies a classic draft family.',chinese_reason='历史成熟基线，提供原始reference/几何目标下的草稿风格。'),
 dict(id='e68',experiment='E68',owner='history/20261029_model_research_consolidation/experiments/E68_port_tokens',run='20261028_full_ports_nosummary_v2_seed0',reason='Full public-port representation trained from scratch with original G; separates representation diversity from later task losses.',chinese_reason='完整端口表示、原G目标；提供不同输入表示下的成熟草稿。'),
 dict(id='e70',experiment='E70',owner='history/20261029_model_research_consolidation/experiments/E70_reference_task_loss',run='20261028_full_ports_taskloss_v1_seed0',reason='Independent scratch training with explicit public task/overlap/wirelength objective; adds a different geometry trade-off.',chinese_reason='独立从头训练的任务Loss模型，提供避碰、线长与紧凑度的不同取舍。'),
 dict(id='e71',experiment='E71',owner='history/20261101_research_consolidation/experiments/E71_density_reference_curriculum',run='20261028_full_ports_density_curriculum_v1_seed0',reason='Completed density plus reference-curriculum training; adds compactness/interior-space organization behavior.',chinese_reason='密度项＋reference课程训练，补充紧凑布局与空间组织取舍。'),
 dict(id='e98_generalization',experiment='E98 generalization baseline',owner='experiments/E98_loss_weight_learning',run='20261104_generalization96_seed0',reason='Completed scratch-trained enhanced baseline with axis-swap and stronger dropout; current unmodified mature baseline, not any discarded search branch.',chinese_reason='轴交换与增强Dropout的独立成熟基线；只取泛化基线，不取已放弃调权分支。'),
]

def main():
    sys.path.insert(0,str(ROOT))
    torch.set_num_threads(1)
    records=[]
    for spec in SPECS:
        origin=ROOT/spec['owner'];run=origin/'runs'/spec['run']
        result=read(run/'results.json');assert result['status']=='early_stop'
        source_checkpoint=run/'checkpoints/best.pt'
        destination=OWNER/'assets'/spec['id']/'best.pt'
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():assert sha(destination)==sha(source_checkpoint)
        else:shutil.copy2(source_checkpoint,destination)
        assert sha(destination)==sha(source_checkpoint)
        checkpoint=torch.load(destination,map_location='cpu',weights_only=False)
        assert checkpoint['progress']['epoch']==result['best_epoch']
        files=['model.py','features.py','batches.py']
        if spec['id'] in ('e68','e70','e71'):files+=['base_model.py','ports.py']
        if spec['id']=='e98_generalization':files+=['ports.py']
        source_dir=OWNER/'sources'/spec['id'];source_dir.mkdir(parents=True,exist_ok=True)
        (source_dir/'__init__.py').write_text('"""Frozen historical inference implementation; see model_manifest.json."""\n',encoding='utf-8')
        copied=[]
        registered_sources=checkpoint['identity']['sources']
        for name in files:
            original=origin/('generalization_baseline/model.py' if spec['id']=='e98_generalization' and name=='model.py' else name)
            frozen=source_dir/name
            logical=original.relative_to(ROOT).as_posix()
            if logical.startswith('history/'):
                logical=logical[logical.index('/experiments/')+1:]
            assert logical in registered_sources,(spec['id'],logical)
            assert sha(original)==registered_sources[logical],(spec['id'],logical,'source changed since training')
            shutil.copy2(original,frozen)
            assert sha(original)==sha(frozen)
            copied.append(dict(source=original.relative_to(ROOT).as_posix(),copy=frozen.relative_to(ROOT).as_posix(),sha256=sha(frozen)))
        evidence=OWNER/'evidence'/spec['id'];evidence.mkdir(parents=True,exist_ok=True)
        for name in ('results.json','config.json'):
            shutil.copy2(run/name,evidence/name)
        cls=importlib.import_module(f'experiments.E103_offline_loss_alignment.sources.{spec["id"]}.model').GeometryModel
        model=cls(**checkpoint['model_config'])
        model.load_state_dict(checkpoint['model'],strict=True)
        assert all(torch.isfinite(v).all() for v in checkpoint['model'].values())
        tensor_hash=hashlib.sha256()
        for name,value in sorted(checkpoint['model'].items()):
            a=value.detach().cpu().contiguous().numpy()
            tensor_hash.update(name.encode());tensor_hash.update(str(a.dtype).encode());tensor_hash.update(str(a.shape).encode());tensor_hash.update(a.tobytes())
        rec=dict(**spec,origin_run=run.relative_to(ROOT).as_posix(),source_checkpoint=source_checkpoint.relative_to(ROOT).as_posix(),checkpoint=destination.relative_to(ROOT).as_posix(),checkpoint_sha256=sha(destination),model_tensor_sha256=tensor_hash.hexdigest(),model_config=checkpoint['model_config'],parameters=sum(p.numel() for p in model.parameters()),stop_reason=result['status'],completed_epochs=result['epochs'],best_epoch=result['best_epoch'],steps=result.get('steps'),copied_sources=copied,strict_state_load=True,finite_weights=True,source_matches_training_checkpoint_identity=True,
            evidence=dict(results_sha256=sha(evidence/'results.json'),config_sha256=sha(evidence/'config.json')),
            input_contract='Original public-input representation retained; no labels/reference positions in network inputs',
            convergence_scope='Completed original validation-based early stopping; not proof of global optimization or flat train Loss')
        records.append(rec)
        print(spec['experiment'],result['status'],'completed',result['epochs'],'best',result['best_epoch'],'strict CPU load verified',flush=True)
        del model,checkpoint
    assert len({r['model_tensor_sha256'] for r in records})==5
    save(OWNER/'model_manifest.json',dict(status='models_staged_for_user_review',models=records,count=5,distinct_model_tensor_hashes=5,dataset_payloads_read=0,generated_drafts=0,postprocessing_calls=0,test_read=False))
    save(OWNER/'selection_verification.json',dict(status='passed',checked_models=5,all_completed_normal_early_stop=True,all_hash_equal_to_original=True,all_sources_match_training_identity=True,all_strict_CPU_load=True,distinct_model_tensors=5,dataset_payloads_read=0,inference_calls=0,training_updates=0,postprocessing_calls=0,test_read=False))

if __name__=='__main__':main()
