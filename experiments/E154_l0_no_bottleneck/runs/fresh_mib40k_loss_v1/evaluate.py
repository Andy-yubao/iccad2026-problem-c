"""Frozen best50 Loss on all fresh train36k/dev2k, with streaming CPU caches."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import json
import pathlib
import time
import traceback
import numpy as np
import torch
from experiments.E154_l0_no_bottleneck import train
from experiments.E154_l0_no_bottleneck.io import ROOT,HERE as EXP,sha,save,sources
from experiments.E154_l0_no_bottleneck.model import GeometryModel
from experiments.E154_l0_no_bottleneck.resident import ResidentRows
from experiments.E154_l0_no_bottleneck.public import inputs_only
from experiments.E154_l0_no_bottleneck.loss import per_case

HERE=pathlib.Path(__file__).resolve().parent
DATA=ROOT/'datasets/random_mib40k_v1'
TERMS=('reference_weighted','wire_compact_weighted','grouping_weighted','boundary_weighted',
       'public_shape_weighted','public_preplaced_position_weighted','mib_weighted','total')


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def summarize(array,keys):
    out={k:float(array[:,j].mean()) for j,k in enumerate(keys)}
    total=array[:,keys.index('total')]
    out.update(count=len(total),std=float(total.std(ddof=1)) if len(total)>1 else 0.,p95=float(np.quantile(total,.95)),max=float(total.max()))
    return out


@torch.inference_mode()
def main():
    started=time.perf_counter()
    assert not (HERE/'results.json').exists(),'Preserve completed evaluation'
    train.seed_all(0)
    state=read(DATA/'state.json');assert state['status']=='complete'
    registry=read(DATA/'manifest.json');identity=registry['manifest_sha256']
    assert state['identity']==identity
    for name,value in registry['producer_hashes'].items():assert sha(DATA/name)==value
    frozen=read(EXP/'frozen_initial.json');path=ROOT/frozen['checkpoint']
    assert sha(path)==frozen['checkpoint_sha256'] and frozen['best_epoch']==50
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    assert checkpoint['identity']['sources']==sources()
    model=GeometryModel(**checkpoint['model_config']).cuda().eval()
    model.load_state_dict(checkpoint['model'])
    before={k:v.detach().clone() for k,v in model.state_dict().items()}
    profile=read(DATA/'cache/profile.json')
    padding=dict(max_nodes=120,max_pins=max(146,profile['max_pins']),max_ports=profile['max_ports'])
    old=read(EXP/'runs/e154_initial_seed0/panels.json')
    old_train={tuple(v) for v in old['train_members']}
    results={};arrays={};all_keys=None
    for role in ('train','dev'):
        records=registry['sets'][role]['records'];count=len(records)
        manifest=read(DATA/f'cache/{role}/manifest.json')
        assert manifest['identity']==identity and manifest['members']==[r['member'] for r in records]
        covered=np.zeros(count,bool);array=None;completed=0
        save(HERE/'state.json',dict(status='running',pid=os.getpid(),role=role,completed=0,total=count))
        for part in manifest['parts']:
            p=DATA/part['path'];assert sha(p)==part['sha256']
            content=torch.load(p,map_location='cpu',weights_only=False,mmap=True)
            assert content['identity']==identity and content['indices']==part['indices']
            rows=content['rows'];ids=content['indices']
            assert all(list(r['member'])==records[i]['member'] for i,r in zip(ids,rows))
            pack=ResidentRows(rows,'cuda',**padding)
            for start in range(0,len(rows),64):
                batch=pack.batch(range(start,min(start+64,len(rows))))
                output=model(inputs_only(batch));values=per_case(output,batch,'e154_initial')
                train.numerical_check(values,output)
                keys=list(values)
                if all_keys is None:all_keys=keys
                assert all_keys==keys
                if array is None:array=np.empty((count,len(keys)),np.float64)
                target=ids[start:start+64]
                assert not covered[target].any()
                array[target]=torch.stack([values[k] for k in keys],-1).cpu().double().numpy()
                covered[target]=True;completed+=len(target)
            del pack,batch,output,values,content,rows
            save(HERE/'state.json',dict(status='running',pid=os.getpid(),role=role,completed=completed,total=count,
                elapsed_seconds=time.perf_counter()-started,training_updates=0,postprocessing_calls=0,test_read=False))
            if completed//1000!=(completed-part['count'])//1000 or completed==count:
                print(f'[LOSS] {role} {completed}/{count}',flush=True)
        assert covered.all() and completed==count
        assert not array[:,keys.index('group_bottleneck_weighted')].any()
        assert not array[:,keys.index('overlap_weighted')].any()
        arrays[role]=array
        totals=array[:,keys.index('total')]
        seen=np.array([tuple(r['member']) in old_train for r in records])
        ns=np.array([r['n'] for r in records])
        results[role]=dict(full=summarize(array,keys),
            monitor500=summarize(array[registry['panels'][role+'_monitor500']],keys),
            prior_training_overlap=int(seen.sum()),
            prior_seen_loss=float(totals[seen].mean()) if seen.any() else None,
            prior_unseen_loss=float(totals[~seen].mean()),
            mean_n=float(ns.mean()),
            by_n={str(n):summarize(array[ns==n],keys) for n in np.unique(ns)})
        np.savez_compressed(HERE/f'{role}_per_case.npz',values=array,keys=np.asarray(keys),n=ns,seen_in_E154_training=seen)
    for k,v in model.state_dict().items():assert torch.equal(v,before[k])
    assert sha(path)==frozen['checkpoint_sha256']
    a=results['train']['full'];b=results['dev']['full']
    common=set(results['train']['by_n'])&set(results['dev']['by_n'])
    weight={n:results['train']['by_n'][n]['count']+results['dev']['by_n'][n]['count'] for n in common}
    standardized={role:sum(weight[n]*results[role]['by_n'][n]['total'] for n in common)/sum(weight.values()) for role in results}
    result=dict(status='complete',frozen_checkpoint=frozen,dataset_identity=identity,results=results,
        difference_dev_minus_train=b['total']-a['total'],relative_dev_minus_train=b['total']/a['total']-1,
        common_n_standardized_loss=standardized,
        note='Frozen model was trained on old MIB32k, not the fresh train36k. This is a cross-dataset distribution check, not a newly trained-model generalization gap.',
        loss='Unmodified E154 L0 Loss; reference1; MIB retained; no direct overlap or grouping bottleneck; FP32 eval mode; common padding, batch<=64; no updates or postprocessing.',
        elapsed_seconds=time.perf_counter()-started,training_updates=0,postprocessing_calls=0,test_read=False)
    save(HERE/'results.json',result)
    lines=['# E154 best50 在新形状修正版上的 Loss','',result['note'],'',
        '| 加权项 | 完整train36k | 完整dev2k | train监测500 | dev监测500 |',
        '|---|---:|---:|---:|---:|']
    for k in TERMS:
        lines.append(f'| {k} | {a[k]:.8f} | {b[k]:.8f} | {results["train"]["monitor500"][k]:.8f} | {results["dev"]["monitor500"][k]:.8f} |')
    lines+=['',f'完整dev相对train：{result["relative_dev_minus_train"]:+.3%}。',
        f'共同N分布标准化：train={standardized["train"]:.8f}，dev={standardized["dev"]:.8f}。',
        '',f'旧E154训练成员重合数：新train={results["train"]["prior_training_overlap"]}，新dev={results["dev"]["prior_training_overlap"]}。',
        '', '所有Loss均使用E154原登记公式，无更新、无后处理、未读取test。完整逐题分项保存于train_per_case.npz/dev_per_case.npz。历史权重和原选模结果保持不变。']
    (HERE/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(HERE/'state.json',dict(status='complete',elapsed_seconds=time.perf_counter()-started,count=38000))
    print(json.dumps(dict(train=a['total'],dev=b['total'],train500=results['train']['monitor500']['total'],
        dev500=results['dev']['monitor500']['total'],relative=result['relative_dev_minus_train'],standardized=standardized),ensure_ascii=False),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        save(HERE/'state.json',dict(status='error',error=repr(error),traceback=traceback.format_exc()))
        raise
