"""New held-out MIB panel: match36k N quotas, rank public structural complexity."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
import collections
import hashlib
import json
import math
import pathlib
import time
import traceback
import numpy as np
import torch
from datasets.paths import ROOT
from experiments.E153_mib_role_topology.prepare_data import resolve, SOURCE
from experiments.E153_mib_role_topology.batches import sample
from experiments.E153_mib_role_topology.ports import public_ports
from experiments.E153_mib_role_topology.role_features import build

HERE = pathlib.Path(__file__).resolve().parent
WEIGHTS = dict(group_edges=.20, largest_group=.10, mib_edges=.20,
               boundary_sides=.10, locked_modules=.10, coupled_modules=.20, net_density=.10)


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(pathlib.Path(p).read_text(encoding='utf-8'))


def save(p, v):
    p = pathlib.Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(v, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    for attempt in range(20):
        try:
            os.replace(tmp, p)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.05)


def metadata(row, index):
    inputs = row['case'].solver_inputs
    n, areas, b2b, p2b, pins, flags, specs = inputs
    c = np.asarray(flags).astype(int)
    counts = [collections.Counter(c[:, k][c[:, k] > 0]) for k in (2, 3)]
    locked = c[:, :2].any(1)
    active = np.column_stack((locked, c[:, 2]>0, c[:, 3]>0, c[:, 4]>0))
    # Public tensors only; no reference coordinates, teacher values, Loss, or Q.
    group_edges = sum(v-1 for v in counts[1].values())
    mib_edges = sum(v-1 for v in counts[0].values())
    boundary_sides = sum(int(v).bit_count() for v in c[:, 4])
    return dict(index=index, member=list(row['member']), n=int(n),
        group_edges=group_edges, largest_group=max(counts[1].values(), default=0),
        mib_edges=mib_edges, boundary_sides=boundary_sides,
        locked_modules=int(locked.sum()), preplaced_modules=int((c[:, 1]>0).sum()),
        coupled_modules=int((active.sum(1)>=2).sum()),
        net_density=float((len(b2b)+len(p2b))/max(int(n), 1)))


def main():
    started = time.perf_counter()
    torch.set_num_threads(1)
    state = dict(status='running', phase='public_metadata', pid=os.getpid(), test_payloads_read=0,
                 model_inference_calls=0, training_started=False)
    assert not (HERE/'manifest.json').exists(), 'Preserve frozen panel; no implicit reselection'
    save(HERE/'state.json', state)
    reg = read(ROOT/'protocol/reference_workset_v6.json')
    manifest = read(SOURCE/'dev/manifest.json')
    records = reg['sets']['dev']['records']
    assert manifest['members'] == [r['member'] for r in records]
    old = {tuple(r['member']) for r in reg['sets']['dev_monitor500']['records']}
    train = {tuple(r['member']) for r in reg['sets']['train']['records']}
    train_shards = {s for s, i in train}
    infos = {}
    used_parts = []
    for part in manifest['parts']:
        path = resolve(part['path'])
        st = path.stat()
        assert (st.st_size, st.st_mtime_ns) == (part['size'], part['mtime_ns'])
        content = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        assert content['indices']==part['indices'] and content['identity']==manifest['part_identities'][part['identity_key']]
        for i, row in zip(content['indices'], content['rows']):
            assert list(row['member'])==records[i]['member']
            infos[i]=metadata(row, i)
        used_parts.append(dict(path=str(path.relative_to(ROOT)), size=st.st_size, mtime_ns=st.st_mtime_ns))
        print(f'[SELECT] public metadata {len(infos)}/2000', flush=True)
        save(HERE/'state.json', dict(state, scanned=len(infos), elapsed_seconds=time.perf_counter()-started))
    assert len(infos)==2000
    for n in sorted({r['n'] for r in infos.values()}):
        group=[r for r in infos.values() if r['n']==n]
        for r in group:
            r['complexity_score']=sum(w*(sum(o[k]<r[k] for o in group)+.5*sum(o[k]==r[k] for o in group))/len(group)
                                      for k,w in WEIGHTS.items())
    candidates=[r for r in infos.values() if tuple(r['member']) not in old]
    assert len(candidates)==1500
    available=collections.Counter(r['n'] for r in candidates)
    report36=ROOT/'datasets/random_reference40k_v1/report.json'
    h={int(k):v for k,v in read(report36)['module_count_histograms']['train'].items()}
    assert sum(h.values())==36000
    desired={n:500*v/36000 for n,v in h.items()}
    quota={n:min(math.floor(v),available[n]) for n,v in desired.items()}
    while sum(quota.values())<500:
        n=max((n for n in desired if quota[n]<available[n]), key=lambda n:(desired[n]-quota[n],n))
        quota[n]+=1
    selected=[]
    for n,q in sorted(quota.items()):
        pool=sorted((r for r in candidates if r['n']==n),
                    key=lambda r:(-r['complexity_score'], hashlib.sha256(json.dumps(r['member']).encode()).hexdigest()))
        selected.extend(pool[:q])
    assert len(selected)==500 and len({tuple(r['member']) for r in selected})==500
    members={tuple(r['member']) for r in selected}
    assert not members & (old | train) and not {s for s,i in members}&train_shards
    source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'protocol/reference_workset_v6.json', SOURCE/'dev/manifest.json', report36, pathlib.Path(__file__)]}
    for name in ('prepare_data.py','batches.py','features.py','role_features.py','ports.py'):
        p=ROOT/'experiments/E153_mib_role_topology'/name
        source_hashes[str(p.relative_to(ROOT))]=sha(p)
    panel=dict(version='MIB.dev500.challenge.v1',task='E14.mean_area.v1',role='dev',count=500,
        parent_pool='reference_workset_v6 dev2000',old_panel_overlap=0,train64k_overlap=0,
        train64k_shard_overlap=0,selection_uses_model_loss=False,selection_uses_Q=False,
        selection_uses_free_labels=False,test_payloads_read=0,rank_weights=WEIGHTS,
        n_distribution_reference='random_reference40k_v1 train36000',desired_n_counts=desired,
        selected_n_counts=quota,records=selected,sources=source_hashes,
        current_run_monitor_changed=False,training_started=False,
        caveat='Public structural challenge panel; intentionally enriched, not an unbiased population estimate. Higher neural Loss is not guaranteed. Historical dev exposure is possible.')
    identity=hashlib.sha256(json.dumps(panel,sort_keys=True).encode()).hexdigest()
    panel['identity']=identity
    save(HERE/'manifest.json',panel)
    save(ROOT/'protocol/mib_dev500_challenge_v1.json',panel)
    save(HERE/'public_metadata.json',dict(rows=list(infos.values()),source_parts=used_parts))
    # Only after membership is frozen, retrieve supervision and build the same predict-all features as E153.
    wanted={r['index']:j for j,r in enumerate(selected)}
    parts=[]
    for part in manifest['parts']:
        if not set(part['indices']) & wanted.keys():
            continue
        content=torch.load(resolve(part['path']),map_location='cpu',weights_only=False,mmap=True)
        output=[];indices=[]
        for i,row in zip(content['indices'],content['rows']):
            if i not in wanted:
                continue
            inputs=tuple(v.detach().cpu().clone() if isinstance(v,torch.Tensor) else v for v in row['case'].solver_inputs)
            old_graph=row['graph'];reference=old_graph['reference']*old_graph['scale']
            graph=sample(inputs,reference);graph.update(public_ports(inputs,graph['scale']))
            graph['target']=old_graph['target'].clone();graph['target'][:,:2]*=old_graph['scale']/graph['scale']
            assert bool((graph['mask']==1).all()) and graph['features'].shape[-1]==17
            flags=inputs[5]
            for label in torch.unique(flags[:,2]):
                if label>0:
                    mask=flags[:,2]==label
                    assert bool((inputs[1][mask]==inputs[1][mask][0]).all())
            output.append(dict(member=tuple(row['member']),graph=graph,role=build(graph),inputs=inputs,
                reference_xywh=reference,baseline_metrics=row['case'].baseline_metrics))
            indices.append(wanted[i])
        path=HERE/f'cache/{len(parts):04d}.pt';path.parent.mkdir(exist_ok=True)
        torch.save(dict(identity=identity,indices=indices,rows=output),path)
        parts.append(dict(path=str(path.relative_to(HERE)),sha256=sha(path),indices=indices))
        done=sum(len(p['indices']) for p in parts)
        save(HERE/'state.json',dict(state,phase='selected_cache',completed=done,total=500))
        print(f'[CACHE] {done}/500',flush=True)
    assert sorted(i for p in parts for i in p['indices'])==list(range(500))
    save(HERE/'cache/manifest.json',dict(identity=identity,members=[r['member'] for r in selected],parts=parts))
    old_rows=[r for r in infos.values() if tuple(r['member']) in old]
    metrics=(*WEIGHTS,'complexity_score','preplaced_modules','n')
    means=lambda rows:{k:sum(r[k] for r in rows)/len(rows) for k in metrics}
    comparison=dict(old500=means(old_rows),new500=means(selected),remaining1500=means(candidates))
    save(HERE/'comparison.json',comparison)
    lines=['# 新 MIB dev500 挑战集','',
        '从原dev2000中排除旧500例，按36k训练集模块数直方图分配500个名额，再在各N内部按公开约束复杂度择优。模型Loss、Q和自由reference几何均未参与选择。',
        '', '与旧500、MIB train64k及其分片均无重合。当前E154已结束，历史选模结果不变。',
        '', '| 每题均值 | 旧500 | 新500 |','|---|---:|---:|']
    for k in metrics:
        lines.append(f'| {k} | {comparison["old500"][k]:.4f} | {comparison["new500"][k]:.4f} |')
    lines+=['','模块数21–120均覆盖；每种N名额3–7例。N80仅余4例可选，其余名额按与36k目标的缺额分配。',
        '', '这里的“更强”指公开结构约束更复杂，不保证任意模型Loss更高。本集合有意富集难约束，不能当作总体无偏估计；是否提高Loss尚未做模型推理验证。旧dev历史曝光可能存在，不能称全新未见测试。',
        '', '已生成独立清单和500例predict-all/MIB/增强角色CPU缓存，不修改任何旧注册表、训练源码、checkpoint或数据缓存。']
    (HERE/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(HERE/'state.json',dict(state,status='complete',phase='complete',completed=500,elapsed_seconds=time.perf_counter()-started,identity=identity))
    print('[COMPLETE] '+json.dumps(comparison,ensure_ascii=False),flush=True)


if __name__=='__main__':
    try:
        main()
    except BaseException as error:
        save(HERE/'state.json',dict(status='error',error=repr(error),traceback=traceback.format_exc()))
        raise
