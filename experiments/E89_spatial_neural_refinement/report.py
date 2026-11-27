"""Same-case saved development comparison after automatic evaluation."""
import argparse,json
import numpy as np
from .io import ROOT,HERE,sha,save


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=('spatial','pooled'),required=True);args=parser.parse_args()
    folder=HERE/f'runs/20261031_{args.arm}_seed0'
    old_folder=ROOT/'experiments/E90_auxiliary_weight_placer/runs/20261031_aux3_seed0'
    raw=read(folder/'full_dev.json');old_raw=read(old_folder/'full_dev.json')
    current=read(folder/'terminal/results.json')
    if 'reused_from' in current:
        path=ROOT/current['reused_from'];assert sha(path)==current['source_sha256'];current=read(path)
    old=read(old_folder/'terminal/results.json')
    records=sorted(current['records'],key=lambda r:r['index']);baseline=sorted(old['records'],key=lambda r:r['index'])
    assert len(records)==len(baseline)==500
    assert all(a['member']==b['member'] for a,b in zip(records,baseline))
    n=np.array([r['n'] for r in records]);w=np.exp((n-n.max())/12)
    delta=np.array([a['metrics']['cost']-b['metrics']['cost'] for a,b in zip(records,baseline)])
    groups=np.array([r['member'][0].split('/')[0] for r in records]);labels=sorted(set(groups))
    numerator=np.array([(delta*w)[groups==g].sum() for g in labels]);denominator=np.array([w[groups==g].sum() for g in labels])
    draw=np.random.default_rng(20261031).integers(0,len(labels),size=(5000,len(labels)))
    interval=np.percentile(numerator[draw].sum(1)/denominator[draw].sum(1),[2.5,97.5]).tolist()
    a,b=raw['records'],old_raw['records'];assert len(a)==len(b)==2000
    assert [r['member'] for r in a]==[r['member'] for r in b]
    features=read(ROOT/'experiments/E69_case_loss_statistics/runs/20261028_dev2000/features.json')['records']
    assert [r['member'] for r in features]==[r['member'] for r in a]
    med={nn:np.median([f['connection_density'] for f in features if f['n']==nn]) for nn in {f['n'] for f in features}}
    ids=[i for i,f in enumerate(features) if f['pin_connected_fraction']<=.3 and f['connection_density']<=med[f['n']]]
    assert len(ids)==157
    def metrics(rows):
        count=sum(len(r['covered_fractions']) for r in rows)
        return dict(count=len(rows),pair_overlap=float(np.mean([r['overlap_area_fraction'] for r in rows])),
            module_pass10=1-sum(r['N10'] for r in rows)/count,
            worst_module_coverage_mean=float(np.mean([r['max_covered_fraction'] for r in rows])),
            all_modules_pass10=sum(r['all_modules_pass10'] for r in rows))
    result=dict(status='complete',terminal_summary=current['summary'],E90_terminal_summary=old['summary'],
        weighted_Q_difference=float(np.average(delta,weights=w)),worker_bootstrap95=interval,
        raw2000=metrics(a),E90_raw2000=metrics(b),sparse157=metrics([a[i] for i in ids]),E90_sparse157=metrics([b[i] for i in ids]),
        limitations='Single-seed adaptive development. Frozen E90 source, fresh corrector. Spatial/pooled matched control must be compared before attributing gain to explicit spatial distribution.',
        test_read=False)
    save(folder/'comparison.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
