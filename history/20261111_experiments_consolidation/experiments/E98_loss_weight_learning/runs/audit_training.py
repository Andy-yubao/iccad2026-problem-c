"""Read-only audit of the completed fixed-loss run; no optimizer or solver steps."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from experiments.E98_loss_weight_learning.io import ROOT, HERE, save, sha, sources, exclusive
from experiments.E98_loss_weight_learning.train import seed_all, MODEL
from experiments.E98_loss_weight_learning.model import GeometryModel
from experiments.E98_loss_weight_learning.schedule import PlateauController
from experiments.E98_loss_weight_learning.loss_config import COEFFICIENTS
from experiments.E98_loss_weight_learning.loss import per_case
from experiments.E98_loss_weight_learning.batches import inputs_only
from experiments.E98_loss_weight_learning.data import load_rows
from experiments.E98_loss_weight_learning.resident import ResidentRows

RUN = HERE/'runs/20261102_fixed_rich96_seed0'
OUT = RUN/'audit_20261102'
OUT.mkdir(exist_ok=True)
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
result = read(RUN/'results.json')
progress = read(RUN/'progress.json')
identity = read(RUN/'config.json')['identity']
history = progress['history']
assert [h['epoch'] for h in history] == list(range(62))
assert progress['step'] == 30500 and progress['seen_count'] == 16000
assert result['status'] == 'early_stop' and result['error'] is None
assert not (RUN/'failure.pt').exists()
assert identity['sources'] == sources()
plan = read(HERE/'plan.json')
assert plan == identity['plan']
assert all(h['loss'] == identity['loss'] for h in history)
for h in history:
    assert h['step'] == h['epoch']*500
    for role in ('train_monitor','dev_monitor'):
        values = h[role]
        assert values['count'] == 500
        assert len(values['per_case_total']) == 500
        assert np.isfinite(values['per_case_total']).all()
        weighted = sum(COEFFICIENTS[k]*values[k] for k in COEFFICIENTS)
        assert abs(weighted-values['total']) < 2e-6

epochs = [h['epoch'] for h in history]
best_epoch = min(epochs, key=lambda e: history[e]['dev_monitor']['total'])
assert best_epoch == result['best_epoch'] == 58
assert abs(history[58]['dev_monitor']['total']-result['best_fixed_loss']) < 1e-12
seed_all(0)
scalar = torch.nn.Parameter(torch.zeros(()))
dummy_opt = torch.optim.AdamW([scalar], lr=1e-3)
scheduler = PlateauController(dummy_opt)
drops = []
for h in history:
    before = scheduler.drops
    scheduler.step(h['dev_monitor']['total'], h['epoch'])
    assert scheduler.state_dict() == h['scheduler']
    assert dummy_opt.param_groups[0]['lr'] == h['lr_next']
    if scheduler.drops != before:
        drops.append(dict(epoch=h['epoch'], lr=h['lr_next']))
assert scheduler.should_stop()
with exclusive(HERE/'runs/training.lock'):
    lock_free = True

paths = {name: RUN/f'checkpoints/{name}.pt' for name in ('best','latest')}
before_hashes = {k: sha(p) for k,p in paths.items()}
checkpoints = {}
for name,path in paths.items():
    state = torch.load(path, map_location='cpu', weights_only=False)
    assert state['identity'] == identity and state['model_config'] == MODEL
    expected_epoch = 58 if name == 'best' else 61
    assert state['progress']['epoch'] == expected_epoch
    assert state['progress']['step'] == expected_epoch*500
    assert state['projection']['steps'] == expected_epoch*500
    assert state['progress']['cursor'] == 0 and state['progress']['order'] is None
    assert int(state['progress']['seen'].sum()) == 16000
    for value in state['model'].values():
        assert torch.isfinite(value).all()
    adam_steps = []
    for value in state['optimizer']['state'].values():
        for v in value.values():
            if isinstance(v,torch.Tensor):
                assert torch.isfinite(v).all()
        adam_steps.append(int(value['step']))
    assert set(adam_steps) == {expected_epoch*500}
    assert all(k in state['rng'] for k in ('python','numpy','torch','cuda'))
    checkpoints[name] = dict(epoch=expected_epoch, step=expected_epoch*500,
         sha256=before_hashes[name], tensors=len(state['model']), adam_parameter_states=len(adam_steps),
         finished=state['finished'], status=state['status'], rng_saved=True,
         max_projection_clip_fraction=float(max(state['projection']['clipped_steps_per_group']))/(expected_epoch*500))
    del state

# Reproduce only five already-frozen train-monitor members, no dev inference.
rows = load_rows('train_monitor500', range(5))
pack = ResidentRows(rows, 'cuda', max_nodes=120, max_pins=146)
batch = pack.batch([i%5 for i in range(32)])
model = GeometryModel(**MODEL).cuda().eval()
assert sum(p.numel() for p in model.parameters()) == 1471539
for name,path in paths.items():
    state = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'])
    with torch.no_grad():
        predicted = model(inputs_only(batch))
        values = per_case(predicted,batch)
    actual = values['total'][:5].cpu().numpy()
    expected = np.asarray(history[checkpoints[name]['epoch']]['train_monitor']['per_case_total'][:5])
    assert np.isfinite(predicted.cpu().numpy()).all()
    np.testing.assert_allclose(actual,expected,rtol=3e-5,atol=3e-6)
    checkpoints[name]['five_train_reproduction_max_abs'] = float(np.max(np.abs(actual-expected)))
    del state
assert {k:sha(p) for k,p in paths.items()} == before_hashes

step_rows = [json.loads(line) for line in (RUN/'steps.jsonl').read_text().splitlines()]
assert len(step_rows) == 305 and [r['step'] for r in step_rows] == list(range(100,30501,100))
assert np.isfinite([r['gradient_norm'] for r in step_rows]).all()
timing = [r['seconds_per_step'] for r in step_rows]
tail = history[-4:]
last_train = [h['train_monitor']['total'] for h in tail]
audit = dict(status='passed', termination=result, best=history[58], latest=history[61],
             initial=history[0], scheduler_drop_epochs=drops, checkpoints=checkpoints,
             source_identity_unchanged=True, immutable_weights_unchanged=True,
             training_lock_free=lock_free, seen_count=16000, full_epochs=61,
             all_62_train_dev_monitors_finite=True, constant_coefficients=True,
             maximum_logged_gradient_norm=float(max(r['gradient_norm'] for r in step_rows)),
             median_step_seconds=float(np.median(timing)), p95_step_seconds=float(np.quantile(timing,.95)),
             last_3_epoch_train_relative_change=(last_train[-1]/last_train[0]-1),
             last_4_train_range_relative=(max(last_train)-min(last_train))/abs(last_train[0]),
             operational_train_plateau_threshold=.001, train_plateau_confirmed=False,
             additional_inference_members=[list(r['member']) for r in rows],
             additional_inference_role='five existing train_monitor500 members only',
             optimizer_updates_during_audit=0, solver_calls_during_audit=0,
             dev_inference_during_audit=0, test_read=False)
save(OUT/'audit.json',audit)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(11,4),dpi=170)
for role,label,color in [('train_monitor','Train500','#2274a5'),('dev_monitor','Dev500','#e07a28')]:
    axes[0].plot(epochs[1:],[h[role]['total'] for h in history[1:]],label=label,color=color)
    axes[1].plot(epochs[-15:],[h[role]['total'] for h in history[-15:]],marker='.',label=label,color=color)
for ax in axes:
    ax.axvline(58,ls='--',color='#777777',lw=1,label='Selected epoch 58')
    ax.set_xlabel('Epoch');ax.set_ylabel('Fixed total loss');ax.grid(alpha=.2);ax.legend(fontsize=8)
axes[0].set_yscale('log');axes[0].set_title('Fixed rich-loss training')
axes[1].set_title('Final 15 epochs: train loss still decreasing')
fig.tight_layout();fig.savefig(OUT/'learning_curve.png');plt.close(fig)

best,latest=history[58],history[61]
lines=['# E98固定丰富Loss基底：训练完成核查','',
       '训练正常早停，无数值错误；训练集充分收敛尚未确认，不能仅凭开发集平台进入调权归因实验。','',
       f'- 61轮／30,500更新，16,000训练成员全部见过；完整累计{result["elapsed_seconds"]/60:.2f}分钟。',
       '- best为第58轮，latest为第61轮，模型／Adam／RNG／排列游标／调度／更新保护状态完整，哈希保持不变。',
       '- 固定10项系数和源码身份未变化；全部62次train/dev500监测有限；调度重放与保存记录一致。',
       '- 训练锁可获取，进程已退出；本次只复现五个原train_monitor成员，无参数更新，无后处理，无新dev推理，测试封存。','',
       '| 指标 | best58 | latest61 |','|---|---:|---:|']
for role,label in [('train_monitor','train500 Loss'),('dev_monitor','dev500 Loss')]:
    lines.append(f'| {label} | {best[role]["total"]:.9f} | {latest[role]["total"]:.9f} |')
for key,label in [('position','dev位置'),('shape','dev形状'),('overlap_total','dev重叠组合代理'),('task','dev任务组合代理'),('pair_intersection_fraction','dev成对交面积／总模块面积')]:
    lines.append(f'| {label} | {best["dev_monitor"][key]:.9f} | {latest["dev_monitor"][key]:.9f} |')
lines+=['',f'最后四轮train500 Loss为{last_train}；第58→61轮下降{(1-last_train[-1]/last_train[0])*100:.3f}%，四轮相对范围{audit["last_4_train_range_relative"]*100:.3f}%，高于预设0.1%平台口径。',
        '这说明训练仍有近期下降信号，不能断言充分收敛；也不能据此保证继续训练一定有收益。',
        '',f'学习率下降记录：{drops}。最终已在5e-5完成5轮。',
        f'每100更新窗口的步时中位{audit["median_step_seconds"]*1000:.1f}ms、P95 {audit["p95_step_seconds"]*1000:.1f}ms；这是batch32训练吞吐指标，不是逐例求解延迟。',
        '', '草稿成对交面积不是模块覆盖并集，当前记录不能提供精确模块≤10%达标率或证明硬合法。尚未运行后处理，不能判断最终Q或最终求解质量。',
        '', '后续先保留本基线。如要求“充分训练后再调权”，应先独立登记固定权重的平台确认／匹配继续训练对照，不能把后训练的全部收益归为系数改变。当前没有启动任何后续训练。',
        '', '[训练曲线](learning_curve.png) · [完整只读审计](audit.json)']
(OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({k:audit[k] for k in ('status','scheduler_drop_epochs','checkpoints','maximum_logged_gradient_norm','median_step_seconds','last_3_epoch_train_relative_change','last_4_train_range_relative')},ensure_ascii=False,indent=2))
print('Best and latest scalar metrics:')
for h in (best,latest):
    print(h['epoch'],{role:{k:h[role][k] for k in ('total','position','shape','overlap_total','task','pair_intersection_fraction','module_load_max')} for role in ('train_monitor','dev_monitor')})
print('Report:',OUT/'report.md')
