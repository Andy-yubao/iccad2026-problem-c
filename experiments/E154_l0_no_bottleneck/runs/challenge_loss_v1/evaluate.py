"""Evaluate frozen E154 best50 on the newly frozen challenge500; no updates."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import json
import pathlib
import time
import traceback
import torch
from experiments.E154_l0_no_bottleneck import train
from experiments.E154_l0_no_bottleneck.io import ROOT, HERE as EXP, sha, sources, save
from experiments.E154_l0_no_bottleneck.model import GeometryModel
from experiments.E154_l0_no_bottleneck.data import load_rows, register_resident
from experiments.E154_l0_no_bottleneck.resident import ResidentRows

HERE = pathlib.Path(__file__).resolve().parent
DATA = ROOT/'datasets/mib_dev500_challenge_v1'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def challenge_rows():
    manifest = read(DATA/'manifest.json')
    cache = read(DATA/'cache/manifest.json')
    assert cache['identity']==manifest['identity'] and len(manifest['records'])==500
    found = {}
    for part in cache['parts']:
        path = DATA/part['path']
        assert sha(path)==part['sha256']
        content = torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        assert content['identity']==cache['identity'] and content['indices']==part['indices']
        for i,row in zip(content['indices'],content['rows']):
            assert i not in found and list(row['member'])==cache['members'][i]
            found[i]=row
    assert sorted(found)==list(range(500))
    return [found[i] for i in range(500)], manifest


def main():
    started=time.perf_counter()
    assert not (HERE/'results.json').exists(), 'Preserve completed evaluation'
    save(HERE/'state.json',dict(status='running',activity='loading frozen best',pid=os.getpid()))
    train.seed_all(0)
    train.ARM='e154_initial'
    frozen=read(EXP/'frozen_initial.json')
    checkpoint=ROOT/frozen['checkpoint']
    assert sha(checkpoint)==frozen['checkpoint_sha256']
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert state['identity']['sources']==sources() and state['identity']['arm']=='e154_initial'
    assert state['progress']['best_epoch']==frozen['best_epoch']==50
    model=GeometryModel(**state['model_config']).cuda().eval()
    model.load_state_dict(state['model'])
    parameters={k:v.detach().clone() for k,v in model.state_dict().items()}
    history=read(EXP/'runs/e154_initial_seed0/progress.json')['history']
    best=next(h for h in history if h['epoch']==50)
    old=load_rows('dev_monitor500')
    old_pack=ResidentRows(old,'cuda',max_nodes=120,max_pins=146)
    register_resident(old_pack)
    def beat(**kw):
        save(HERE/'state.json',dict(status='running',pid=os.getpid(),elapsed_seconds=time.perf_counter()-started,**kw))
    reproduced=train.monitor(model,old,beat=beat)
    max_old_error=max(abs(a-b) for a,b in zip(reproduced['per_case_total'],best['dev_monitor']['per_case_total']))
    assert max_old_error<2e-5, ('Original monitor mismatch',max_old_error)
    rows,manifest=challenge_rows()
    new_pack=ResidentRows(rows,'cuda',max_nodes=120,max_pins=max(146,max(r['graph']['pin_coordinates'].shape[1] for r in rows)))
    register_resident(new_pack)
    new=train.monitor(model,rows,beat=beat)
    for k,v in model.state_dict().items():
        assert torch.equal(v,parameters[k]), 'Inference changed weights'
    assert new['group_bottleneck_weighted']==0 and new['overlap_weighted']==0
    result=dict(status='complete',checkpoint=frozen,arm='e154_initial',dataset_identity=manifest['identity'],
        loss_scope='Unmodified registered E154 L0 training Loss; arithmetic per-case mean, eval mode, FP32 batch64; no overlap or grouping bottleneck in objective.',
        best_train_monitor=best['train_monitor'],best_old_dev_monitor=best['dev_monitor'],
        reproduced_old_dev=reproduced,old_monitor_max_per_case_difference=max_old_error,
        challenge_dev500=new,relative_to_old_dev=new['total']/reproduced['total']-1,
        relative_to_train_monitor=new['total']/best['train_monitor']['total']-1,
        per_case=[dict(index=i,member=manifest['records'][i]['member'],n=manifest['records'][i]['n'],loss=v)
                  for i,v in enumerate(new['per_case_total'])],
        elapsed_seconds=time.perf_counter()-started,training_updates=0,postprocessing_calls=0,test_read=False)
    save(HERE/'results.json',result)
    terms=('reference_weighted','wire_compact_weighted','grouping_weighted','boundary_weighted',
           'public_shape_weighted','public_preplaced_position_weighted','mib_weighted','total')
    lines=['# E154 best50：新挑战 dev500 Loss','',
        '使用冻结best50及原登记L0损失，不训练、不后处理。训练监测采用第50轮原始记录，旧dev500重新推理核对，新500按已冻结成员计算。','',
        '| 加权项 | 训练监测500 | 旧dev500复算 | 新挑战dev500 |','|---|---:|---:|---:|']
    for k in terms:
        lines.append(f'| {k} | {best["train_monitor"][k]:.8f} | {reproduced[k]:.8f} | {new[k]:.8f} |')
    lines+=['',f'旧dev逐题Loss最大复现误差：{max_old_error:.3g}。',
        f'新集合相对旧dev均值变化：{result["relative_to_old_dev"]:+.2%}；相对训练监测均值：{result["relative_to_train_monitor"]:+.2%}。',
        f'新集合P95={new["p95"]:.8f}，最大={new["max"]:.8f}。',
        '', '挑战集按公开结构复杂度富集，不是总体无偏估计。没有改变E154历史最佳权重、学习率、早停或监测成员。']
    (HERE/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(HERE/'state.json',dict(status='complete',count=500,elapsed_seconds=time.perf_counter()-started))
    print(json.dumps(dict(old_dev=reproduced['total'],new_dev=new['total'],train_monitor=best['train_monitor']['total'],
        relative_to_old_dev=result['relative_to_old_dev'],relative_to_train=result['relative_to_train_monitor'],
        old_monitor_max_per_case_difference=max_old_error),ensure_ascii=False),flush=True)


if __name__=='__main__':
    try:
        main()
    except BaseException as error:
        save(HERE/'state.json',dict(status='error',error=repr(error),traceback=traceback.format_exc()))
        raise
