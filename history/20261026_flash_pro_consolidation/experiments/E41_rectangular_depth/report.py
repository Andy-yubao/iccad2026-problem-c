"""Read saved depth16 results and frozen E39 control; never infer or solve."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .io import ROOT,save

CONTROL=ROOT/'experiments/E39_rectangular_attention/runs/20261023_rectangle_w96d12_r2_rectangle_seed0'
FEATURES=ROOT/'experiments/E39_rectangular_attention/runs/20261023_failure_mode_audit/case_features_enriched.csv'

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def saved_summary(folder):
    result=read(folder/'results.json');progress=read(folder/'progress.json')
    best=next((r for r in progress['history'] if r['epoch']==progress['best_epoch']),None)
    out={k:result[k] for k in ['status','epochs','steps','best_epoch','best_proxy','elapsed_seconds']}
    out['best_monitor']=best
    if (folder/'full_dev.json').exists():out['full_dev']=read(folder/'full_dev.json')['losses']
    if (folder/'terminal/results.json').exists():out['terminal']=read(folder/'terminal/results.json')['summary']
    return out

def summarize(rows):
    n=np.array([r['n'] for r in rows]);q=np.array([r['metrics']['cost'] for r in rows]);w=np.exp((n-n.max())/12)
    return {'count':len(rows),'hard_legal':sum(r['metrics']['is_feasible'] for r in rows),
        'weighted_Q':float(np.average(q,weights=w)),'mean_Q':float(q.mean()),
        'p95_Q':float(np.percentile(q,95)),'max_Q':float(q.max()),
        'worst5pct_mean_Q':float(np.sort(q)[-max(1,int(np.ceil(.05*len(q)))):].mean()),
        'mean_seconds':float(np.mean([r['seconds'] for r in rows])),
        'postprocess_seconds':float(np.mean([r['postprocess_seconds'] for r in rows])),
        'mean_center_move':float(np.mean([r['mean_center_move_local'] for r in rows])),
        'overlap_fraction':float(np.mean([r['draft_diagnostics']['overlap_area_fraction'] for r in rows])),
        'raw_grouping':float(np.mean([r['raw_metrics']['grouping_violations'] for r in rows])),
        'raw_mib':float(np.mean([r['raw_metrics']['mib_violations'] for r in rows])),
        'raw_boundary':float(np.mean([r['raw_metrics']['boundary_violations'] for r in rows])),
        'rescue_cases':sum(r['rescue_used'] for r in rows),
        'early_projection_cases':sum(any(t['phase']=='early_projection' for t in r['trace']) for r in rows),
        'nonlegal_shape_candidates':sum(not t.get('legal',False) for r in rows for t in r['trace'] if t['phase']=='shape')}

def run(output):
    current=saved_summary(output);control=saved_summary(CONTROL)
    result={'candidate':current,'E39_depth12_control':control,'test_read':False,'solver_calls':0,
        'scope':'Adaptive development, one seed, once-only fixed-best final; subgroup results do not select checkpoint'}
    if (output/'terminal/results.json').exists():
        candidate=read(output/'terminal/results.json')['records'];old=read(CONTROL/'terminal/results.json')['records']
        candidate.sort(key=lambda r:r['index']);old.sort(key=lambda r:r['index'])
        assert len(candidate)==len(old)==500
        assert [r['member'] for r in candidate]==[r['member'] for r in old]
        with FEATURES.open(encoding='utf-8-sig') as f:features={int(r['index']):r for r in csv.DictReader(f)}
        assert all(features[r['index']]['member']==r['member'][0]+':'+str(r['member'][1]) for r in old)
        pre_cut=float(np.quantile([float(r['preplaced_fraction']) for r in features.values()],.75))
        overlap_cut=float(np.quantile([float(r['raw_overlap']) for r in features.values()],.75))
        definitions={'all':lambda i:True,
            'preplaced_top_quartile_input':lambda i:float(features[i]['preplaced_fraction'])>=pre_cut,
            'E39_overlap_top_quartile':lambda i:float(features[i]['raw_overlap'])>=overlap_cut,
            'E39_early_projection25':lambda i:any(t['phase']=='early_projection' for t in old[i]['trace']),
            'E39_rescue6':lambda i:old[i]['rescue_used'],
            'group_fraction_ge_half12':lambda i:float(features[i]['group_fraction'])>=.5}
        result['subgroups']={}
        for label,condition in definitions.items():
            ids=[i for i in range(500) if condition(i)]
            pair={'depth16':summarize([candidate[i] for i in ids]),'depth12':summarize([old[i] for i in ids])}
            pair['delta']={k:pair['depth16'][k]-pair['depth12'][k] for k in pair['depth16'] if k!='count'}
            result['subgroups'][label]=pair
        result['strata_definition']={'preplaced_top_quartile_cut':pre_cut,'E39_overlap_top_quartile_cut':overlap_cut,
            'fixed_before_depth16_training':True,'overlap_and_tail_strata_are_E39_control_conditioned':True,
            'warning':'Control-conditioned difficult subgroups can regress to the mean; do not infer causal subgroup improvement'}
    save(output/'comparison.json',result)
    lines=['# E41 16层与冻结E39 12层对照','','单种子开发结果；不按终末Q反选checkpoint。MIB均值面积与Loss均保持原样。','',
        '| 模型 | 状态 | 轮数 | best轮 | monitor代理 |','|---|---|---:|---:|---:|']
    for name,row in [('E39 12层',control),('E41 16层',current)]:lines.append(f'| {name} | {row["status"]} | {row["epochs"]} | {row["best_epoch"]} | {row["best_proxy"]:.9f} |')
    if 'subgroups' in result:
        lines+=['','固定分层及总体配对结果见comparison.json；预放置/重叠/旧难例分层不改变全体评价分母。',
            '旧模型条件化难例存在均值回归，6/12/25例子组只作诊断；病例配对不覆盖运行间波动和种子不确定性。']
    else:lines+=['','尚未执行完整开发/终末评价；这里没有候选Q。']
    with (output/'comparison.md').open('w',encoding='utf-8',newline='\n') as f:f.write('\n'.join(lines)+'\n')

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();run(args.output)
if __name__=='__main__':main()
