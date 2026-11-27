"""Read E52 module-overlap result and E42 control; never infer or solve."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .io import ROOT,save

CONTROL=ROOT/'experiments/E42_joint_regularization/runs/20261023_rectangle_w96d12_ab_seed0'
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
    complete=[r for r in rows if 'error' not in r]
    def mean(fn):return float(np.mean([fn(r) for r in complete])) if complete else None
    return {'count':len(rows),'hard_legal':sum(r['metrics']['is_feasible'] for r in rows),
        'weighted_Q':float(np.average(q,weights=w)),'mean_Q':float(q.mean()),
        'p95_Q':float(np.percentile(q,95)),'max_Q':float(q.max()),
        'worst5pct_mean_Q':float(np.sort(q)[-max(1,int(np.ceil(.05*len(q)))):].mean()),
        'mean_seconds':float(np.mean([r['seconds'] for r in rows])),
        'errors':len(rows)-len(complete),'available_diagnostic_count':len(complete),
        'postprocess_seconds':mean(lambda r:r['postprocess_seconds']),
        'mean_center_move':mean(lambda r:r['mean_center_move_local']),
        'overlap_fraction':mean(lambda r:r['draft_diagnostics']['overlap_area_fraction']),
        'raw_grouping':mean(lambda r:r['raw_metrics']['grouping_violations']),
        'raw_mib':mean(lambda r:r['raw_metrics']['mib_violations']),
        'raw_boundary':mean(lambda r:r['raw_metrics']['boundary_violations']),
        'hpwl_gap':mean(lambda r:r['metrics']['hpwl_gap']),
        'area_gap':mean(lambda r:r['metrics']['area_gap']),
        'rescue_cases':sum(r.get('rescue_used',False) for r in rows),
        'early_projection_cases':sum(any(t['phase']=='early_projection' for t in r.get('trace',[])) for r in rows),
        'nonlegal_shape_candidates':sum(not t.get('legal',False) for r in rows for t in r.get('trace',[]) if t['phase']=='shape')}

def run(output):
    current=saved_summary(output);control=saved_summary(CONTROL)
    result={'candidate':current,'E42_AB_control':control,'test_read':False,'solver_calls':0,
        'scope':'Adaptive development, one seed, once-only fixed-best final; subgroup results do not select checkpoint'}
    if (output/'terminal/results.json').exists():
        candidate=read(output/'terminal/results.json')['records'];old=read(CONTROL/'terminal/results.json')['records']
        candidate.sort(key=lambda r:r['index']);old.sort(key=lambda r:r['index'])
        assert len(candidate)==len(old)==500
        assert [r['index'] for r in candidate]==list(range(500))==[r['index'] for r in old]
        assert [r['member'] for r in candidate]==[r['member'] for r in old]
        with FEATURES.open(encoding='utf-8-sig') as f:features={int(r['index']):r for r in csv.DictReader(f)}
        assert all(features[r['index']]['member']==r['member'][0]+':'+str(r['member'][1]) for r in old)
        pre_cut=float(np.quantile([float(r['preplaced_fraction']) for r in features.values()],.75))
        overlap_cut=float(np.quantile([r['draft_diagnostics']['overlap_area_fraction'] for r in old],.75))
        definitions={'all':lambda i:True,
            'preplaced_top_quartile_input':lambda i:float(features[i]['preplaced_fraction'])>=pre_cut,
            'E42_overlap_top_quartile':lambda i:old[i]['draft_diagnostics']['overlap_area_fraction']>=overlap_cut,
            'E42_early_projection22':lambda i:any(t['phase']=='early_projection' for t in old[i]['trace']),
            'E42_rescue4':lambda i:old[i]['rescue_used'],
            'E42_preselected_tail5':lambda i:i in (164,239,443,329,324),
            'group_fraction_ge_half12':lambda i:float(features[i]['group_fraction'])>=.5}
        from .paired import paired
        result['paired']=paired(candidate,old)
        result['subgroups']={}
        for label,condition in definitions.items():
            ids=[i for i in range(500) if condition(i)]
            pair={'module_loss12':summarize([candidate[i] for i in ids]),'control12':summarize([old[i] for i in ids])}
            pair['delta']={k:pair['module_loss12'][k]-pair['control12'][k] for k in pair['module_loss12']
                if k!='count' and pair['module_loss12'][k] is not None and pair['control12'][k] is not None}
            result['subgroups'][label]=pair
        result['strata_definition']={'preplaced_top_quartile_cut':pre_cut,'E42_overlap_top_quartile_cut':overlap_cut,
            'fixed_before_E52_training':True,'overlap_and_tail_strata_are_E42_control_conditioned':True,
            'warning':'Control-conditioned difficult subgroups can regress to the mean; do not infer causal subgroup improvement'}
    save(output/'comparison.json',result)
    lines=['# E52 逐模块相对重叠与冻结E42联合AB对照','','保留E42模型及原几何目标，新增0.25×逐模块相对重叠；面积平方根加权，使用连续饱和个数因子，第一批起固定完整权重。训练、开发监测和选模统一使用同一个总Loss。历史E42代理未按新Loss重算，数值不能与新total比较。单种子开发，不按终末Q反选checkpoint。','',
        '| 模型 | 状态 | 轮数 | best轮 | 统一Loss |','|---|---|---:|---:|---:|']
    for name,row in [('E42 AB',control),('E52 unified-v2',current)]:
        value = f"{row['best_proxy']:.9f}" if row is current else '未按新定义重算'
        lines.append(f'| {name} | {row["status"]} | {row["epochs"]} | {row["best_epoch"]} | {value} |')
    if 'subgroups' in result:
        lines+=['','| 模型 | full-dev统一Loss | 硬合法 | 加权Q | 算术Q | P95 | 最差5% | 最大Q | 秒/例 | 救援 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for name,key,summary in [('E42 AB','control12',control),('E52 module-overlap','module_loss12',current)]:
            r=result['subgroups']['all'][key]
            proxy=summary.get('full_dev',{}).get('total') if summary is current else '未按新定义重算'
            lines.append(f'| {name} | {proxy} | {r["hard_legal"]}/500 | {r["weighted_Q"]:.6f} | {r["mean_Q"]:.6f} | {r["p95_Q"]:.6f} | {r["worst5pct_mean_Q"]:.6f} | {r["max_Q"]:.6f} | {r["mean_seconds"]:.6f} | {r["rescue_cases"]} |')
        lines+=['',f'加权Q配对差/区间：{result["paired"]}', '同时核对full_dev的bbox、boundary、contact、MIB及三相图留白，防止靠摊大布局换重叠下降。','','固定分层及总体配对结果见comparison.json；预放置/重叠/旧难例分层不改变全体评价分母。',
            '旧模型条件化难例存在均值回归，4/12/22例子组只作诊断；病例配对不覆盖运行间波动和种子不确定性。']
    else:lines+=['','尚未执行完整开发/终末评价；这里没有候选Q。']
    if 'full_dev' in current:
        lines += ['', '逐模块累计覆盖负荷及小/中/大面积三分组见full_dev.json的draft_diagnostics；负荷保留重复覆盖，不等于覆盖并集比例。旧E42记录缺少这些新字段，不能凭缺失值宣称逐模块负荷已经改善。',
                  '', '人工评分仍是首要参考；运行review阶段生成固定随机100例评分入口，图片准备完成不代表用户评分完成。']
    with (output/'comparison.md').open('w',encoding='utf-8',newline='\n') as f:f.write('\n'.join(lines)+'\n')
    plot_curves(output)


def plot_curves(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    history = read(output/'progress.json')['history']
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, key, title in zip(axes, ('total', 'module_overlap_counted', 'module_load_gt05_fraction'),
                              ('Unified fixed Loss', 'Common counted module objective', 'Nodes with pair-load > 5%')):
        for role, label in (('train_monitor', 'train500'), ('dev_monitor', 'dev500')):
            ax.plot([r['epoch'] for r in history], [r[role][key] for r in history], label=label)
        ax.set(title=title, xlabel='Completed epochs'); ax.grid(alpha=.2); ax.legend()
    fig.tight_layout(); fig.savefig(output/'training_curves.png', dpi=160); plt.close(fig)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();run(args.output)
if __name__=='__main__':main()
