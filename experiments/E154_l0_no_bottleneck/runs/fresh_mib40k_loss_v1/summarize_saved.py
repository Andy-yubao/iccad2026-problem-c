"""Summarize the saved complete inference arrays, including singleton N bins."""
import json
import pathlib
import numpy as np
from .evaluate import summarize,TERMS,HERE,DATA,EXP,read,sha,save,ROOT


def main():
    assert not (HERE/'results.json').exists()
    registry=read(DATA/'manifest.json');frozen=read(EXP/'frozen_initial.json')
    assert sha(ROOT/frozen['checkpoint'])==frozen['checkpoint_sha256']
    state=read(HERE/'state.json')
    save(HERE/'prior_summary_state.json',state)
    results={}
    for role,count in [('train',36000),('dev',2000)]:
        with np.load(HERE/f'{role}_per_case.npz',allow_pickle=False) as saved:
            a=saved['values'];keys=saved['keys'].tolist();ns=saved['n'];seen=saved['seen_in_E154_training']
        assert len(a)==count and np.isfinite(a).all()
        assert np.array_equal(ns,[r['n'] for r in registry['sets'][role]['records']])
        total=a[:,keys.index('total')]
        results[role]=dict(full=summarize(a,keys),monitor500=summarize(a[registry['panels'][role+'_monitor500']],keys),
            prior_training_overlap=int(seen.sum()),prior_seen_loss=float(total[seen].mean()) if seen.any() else None,
            prior_unseen_loss=float(total[~seen].mean()),mean_n=float(ns.mean()),
            by_n={str(n):summarize(a[ns==n],keys) for n in np.unique(ns)})
    common=set(results['train']['by_n'])&set(results['dev']['by_n'])
    w={n:results['train']['by_n'][n]['count']+results['dev']['by_n'][n]['count'] for n in common}
    standardized={role:sum(w[n]*results[role]['by_n'][n]['total'] for n in common)/sum(w.values()) for role in results}
    a=results['train']['full'];b=results['dev']['full']
    result=dict(status='complete',frozen_checkpoint=frozen,dataset_identity=registry['manifest_sha256'],results=results,
        difference_dev_minus_train=b['total']-a['total'],relative_dev_minus_train=b['total']/a['total']-1,
        common_n_standardized_loss=standardized,
        common_n_coverage={role:sum(results[role]['by_n'][n]['count'] for n in common)/results[role]['full']['count'] for role in results},
        note='Frozen model trained on old MIB32k, not fresh train36k; this checks dataset distribution, not a newly trained-model generalization gap.',
        loss='Unmodified E154 L0 Loss; reference1; MIB retained; overlap and grouping bottleneck absent; FP32 eval common padding; no parameter updates.',
        inference_arrays_sha256={role:sha(HERE/f'{role}_per_case.npz') for role in results},
        summary_note='Statistics assembled from complete finite saved inference arrays. Singleton N-bin sample standard deviation reported as0.',
        training_updates=0,postprocessing_calls=0,test_read=False)
    save(HERE/'results.json',result)
    lines=['# E154 best50：新形状修正版完整训练/验证 Loss','',result['note'],'',
        '| 加权项 | 完整train36k | 完整dev2k | train监测500 | dev监测500 |','|---|---:|---:|---:|---:|']
    for k in TERMS:
        lines.append(f'| {k} | {a[k]:.8f} | {b[k]:.8f} | {results["train"]["monitor500"][k]:.8f} | {results["dev"]["monitor500"][k]:.8f} |')
    lines+=['',f'完整dev相对train：{result["relative_dev_minus_train"]:+.3%}。',
        f'共同N分布标准化：train={standardized["train"]:.8f}，dev={standardized["dev"]:.8f}。',
        f'平均N：train={results["train"]["mean_n"]:.3f}，dev={results["dev"]["mean_n"]:.3f}。',
        f'旧E154训练成员重合：新train={results["train"]["prior_training_overlap"]}，新dev={results["dev"]["prior_training_overlap"]}。',
        '', '没有更新参数，没有后处理，未读取测试集。所有38000例逐题完整Loss分项保存为NPZ；少数N仅1例，其标准差单独按0记录，不影响任何Loss均值。']
    (HERE/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(HERE/'state.json',dict(status='complete',count=38000,summary_from_saved_arrays=True))
    print(json.dumps(dict(train=a['total'],dev=b['total'],relative=result['relative_dev_minus_train'],standardized=standardized),ensure_ascii=False))


if __name__=='__main__':main()
