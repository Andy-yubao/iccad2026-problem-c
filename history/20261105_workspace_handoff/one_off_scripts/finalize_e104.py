from pathlib import Path
import json,hashlib
p=Path('experiments/E104_scaling32k')
plan=json.loads((p/'plan.json').read_text());base=Path(plan['comparison']['baseline'])
plan['baseline_evaluation_sha256']=hashlib.sha256((base/'evaluation_depth_dev500/results.json').read_bytes()).hexdigest()
plan['baseline_progress_sha256']=hashlib.sha256((base/'progress.json').read_bytes()).hexdigest()
plan['maximum_seconds']=None;plan['maximum_epochs']=None
(p/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
s=(p/'train.py').read_text();s=s.replace("RUN.mkdir(parents=True, exist_ok=False)","RUN.mkdir(parents=True, exist_ok=False)\n        save(RUN/'pipeline_state.json',dict(status='training',controller_pid=os.getppid(),training_pid=os.getpid(),test_read=False))")
(p/'train.py').write_text(s,encoding='utf-8')
s=(p/'preflight.py').read_text();s=s.replace("pack=ResidentRows(rows,'cuda',max_nodes=120,max_pins=146)","pack=ResidentRows(rows,'cuda',max_nodes=120,max_pins=max(146,max(r['graph']['pin_coordinates'].shape[1] for r in rows)))")
# Same training fixture provides an actual end-to-end public inference / full official scoring check.
s=s.replace("result.update(status='passed',sources=sources(),actual_updates=1,",'''from .io import checkpoint
        from .predictor import Predictor
        from .evaluate import request
        from .depth_evaluation.repair.canonical_native import warmup
        from threadpoolctl import threadpool_limits
        import numba
        numba.set_num_threads(1)
        fixture=OWNER/'runs/preflight_checkpoint.pt'
        checkpoint(fixture,dict(model=model.state_dict(),model_config=MODEL))
        with threadpool_limits(limits=1),torch.inference_mode():
            warmup()
            predictor=Predictor(fixture,rows[0]['case'].solver_inputs)
            record=request(predictor,rows[0])
        fixture.unlink()
        result.update(status='passed',sources=sources(),actual_updates=1,
            full_evaluation_training_case=dict(member=record['member'],hard_legal=record['metrics']['is_feasible'],independent_hard_legal=record['independent_hard_checks']['is_feasible'],seconds=record['complete_seconds']),''')
s=s.replace('postprocessing_calls=0','postprocessing_calls=1')
(p/'preflight.py').write_text(s,encoding='utf-8')
s=(p/'evaluate.py').read_text();s=s.replace("baseline=json.loads(baseline_path.read_text(encoding='utf-8'))", "if sha(BASE/'progress.json')!=plan['baseline_progress_sha256']:raise ValueError('Baseline Loss history drift')\n    baseline=json.loads(baseline_path.read_text(encoding='utf-8'))")
(p/'evaluate.py').write_text(s,encoding='utf-8')
# Generated tensors, weights, layouts and source snapshots stay local.
with Path('.gitignore').open('a',encoding='utf-8') as f:f.write('\n# E104 authorized scaling experiment generated evidence.\nexperiments/E104_scaling32k/runs/**/*.pt\nexperiments/E104_scaling32k/runs/**/*.zip\nexperiments/E104_scaling32k/runs/**/cases/\nexperiments/E104_scaling32k/runs/**/gallery/\nexperiments/E104_scaling32k/*.log\n')
# Register the new authorization without changing any prior run or protocol file.
entry='\n\n用户2026-11-04另行授权[E104扩容实验](experiments/E104_scaling32k/README.md)：E98泛化增强设计从随机初始化扩大为冻结train32k、128宽×16层、3,463,747参数，原固定Loss/增强/Dropout与正常LR早停保持，无硬时间/轮数上限。准备完成后后台训练，正常结束冻结唯一Loss-best，自动同固定dev500与E98.depth_budget.v1评价并报告，用户已确认不是封存test；不按Q重选或追加候选，不恢复旧E98搜索，不改变E101/E102/E103。状态入口experiments/E104_scaling32k/runs/20261104_scaling128x16_seed0。\n'
with Path('README.md').open('a',encoding='utf-8') as f:f.write(entry)
for name in ['docs/protocol.md','docs/workspace_structure.md','experiments/README.md']:
    with Path(name).open('a',encoding='utf-8') as f:f.write(entry.replace('(experiments/E104','(../experiments/E104'))
