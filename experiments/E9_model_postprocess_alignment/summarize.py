"""Rebuild the matched E9 research table from saved complete dev evaluations."""
import argparse
import json
from pathlib import Path
import numpy as np
from scoring.official import official_evaluator
from .records import save,read_rows,run_path
from .probe import BASELINE


def read_metrics(path):
    rows=read_rows(path);out={}
    for row in rows:
        key='model' if 'model' in row['variants'] else 'conic_robust'
        record=row['variants'][key]
        out[row['case_id']]={'metrics':record['final']['metrics'],'n':row['block_count'],
            'end_to_end_seconds':row.get('inference_seconds',0.)+record['solve_seconds']}
    return out


def write_report(run):
    run=run_path(run);baseline=read_metrics(BASELINE)
    names=['retained_E7'];evaluations={'retained_E7':baseline};selections={}
    for name in ('geometry_control','quality_aligned','center_refiner'):
        path=run/name/'selection.json'
        if not path.exists():continue
        selection=json.loads(path.read_text());metrics=read_metrics(Path(selection['evaluation'])/'cases.jsonl')
        if set(metrics)!=set(baseline):raise ValueError('Report requires the same complete dev100 members')
        evaluations[name]=metrics;selections[name]=selection;names.append(name)
    summary={}
    for name in names:
        values=evaluations[name];keys=list(baseline)
        q=np.array([values[k]['metrics']['cost'] for k in keys])
        delta=q-np.array([baseline[k]['metrics']['cost'] for k in keys])
        summary[name]={'n':len(q),'Q_mean':float(q.mean()),'Q_weighted':float(official_evaluator().compute_total_score(q.tolist(),[values[k]['n'] for k in keys])),
                       'Q_p95':float(np.quantile(q,.95)),'Q_max':float(q.max()),
                       'feasible':sum(v['metrics']['is_feasible'] for v in values.values()),
                       'mean_difference_vs_retained':float(delta.mean()),'wins':int((delta < -1e-8).sum()),
                       'ties':int((np.abs(delta)<=1e-8).sum()),'losses':int((delta>1e-8).sum()),
                       'end_to_end_seconds_mean':float(np.mean([v['end_to_end_seconds'] for v in values.values()])),
                       'target_met_on_dev100':bool(q.mean()<1.20)}
        if name in selections:summary[name]['selection']=selections[name]
    contrast=None
    if 'geometry_control' in evaluations and 'quality_aligned' in evaluations:
        differences=[evaluations['quality_aligned'][k]['metrics']['cost']-evaluations['geometry_control'][k]['metrics']['cost'] for k in baseline]
        contrast={'quality_minus_control_mean':float(np.mean(differences)),
                  'note':'Selected complete development results; selection/seed reuse must be inspected before attributing any benefit to the new loss.'}
    candidates=[]
    for name in ('geometry_control','quality_aligned','center_refiner'):
        path=run/name/'results.json'
        if not path.exists():continue
        record=json.loads(path.read_text())
        for label,node in record.get('quality_nodes',{}).items():
            result_path=Path(node['evaluation'])/'results.json'
            stats=json.loads(result_path.read_text()).get('summary',{}) if result_path.exists() else {}
            candidates.append({'variant':name,'node':label,'Q_mean':node['Q_mean'],
                'same_as_retained':node['model_digest']==selections.get(name,{}).get('model_digest') and selections.get(name,{}).get('source') is not None,
                'feasible':stats.get('feasible'),'Q_max':stats.get('Q_max'),'mean_difference_vs_E8':stats.get('mean_difference_vs_E8'),
                'evaluation':node['evaluation']})
    result={'variants':summary,'candidates':candidates,'paired_objective_contrast':contrast,'target':1.20,'test_payload_reads':0}
    save(run/'comparison.json',result)
    lines=['# E9：同起点模型训练对照','',
           '固定 dev100，官方中性时间 Q，越低越好。所有失败均计入。模型在开发面板上被选择，不是封存测试结果。', '',
           '| 方法 | 平均 Q | 块数加权 Q | P95 Q | 最大 Q | 硬合法 | 对原模型均值差 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for name,record in summary.items():
        lines.append(f'| {name} | {record["Q_mean"]:.6f} | {record["Q_weighted"]:.6f} | {record["Q_p95"]:.6f} | {record["Q_max"]:.6f} | {record["feasible"]}/{record["n"]} | {record["mean_difference_vs_retained"]:+.6f} |')
    lines+=['','每组 selection.json 指向真实被选权重与完整评价。若仍指向原 E7 权重，表示保留旧模型，不能写成新模型改善。候选质量节点和终末 dev2500 是否完成以各组 results.json 为准；本表不掩盖资源停止或未完成评价。','']
    if candidates:
        lines+=['## 实际候选，而非保留旧模型后的结果','','| 训练组 | 评价节点 | 平均 Q | 硬合法 | 与旧模型相同 |','|---|---|---:|---:|---|']
        for c in candidates:
            lines.append(f'| {c["variant"]} | {c["node"]} | {c["Q_mean"]:.6f} | {c["feasible"]}/100 | {c["same_as_retained"]} |')
        lines.append('')
    if contrast:lines.append(f'新目标组减控制组的平均 Q 差为 {contrast["quality_minus_control_mean"]:+.6f}；相同后处理下，该差值用于区分新目标与追加训练的贡献。只有开发证据，不作统计显著性或隐藏集排名声明。')
    if (run/'results.json').exists():
        run_result=json.loads((run/'results.json').read_text())
        lines+=['','## 运行状态与解释','',f'总体状态：{run_result.get("status")}；累计 {run_result.get("cumulative_seconds",0.)/60:.2f} 分钟。']
        for name in names[1:]:
            path=run/name/'results.json'
            if not path.exists():continue
            arm=json.loads(path.read_text())
            lines.append(f'{name}：{arm["stop_reason"]}，{arm["epoch"]:.2f}轮，step {arm["step"]}；终末dev2500：{arm["terminal_dev2500"]}。')
        if run_result.get('status')=='completed' and all(not value['target_met_on_dev100'] for value in summary.values()):
            lines+=['','本轮未达到平均Q＜1.20。若最终选择仍为旧模型，表示新候选没有胜出，不能把三行同分解释为新训练有效。',
                    '不据此延长当前训练或反复调同一组权重；原目标、代理改善和最终布局质量分别判断。']
        path=run/'quality_aligned/run.log'
        if path.exists():
            monitors=[r for r in read_rows(path) if r.get('phase')=='monitor']
            if monitors:
                first,last=monitors[0],monitors[-1]
                lines+=['','新目标组固定dev500的草稿代理变化：','',
                        '| 代理项 | 初始 | 最后 |','|---|---:|---:|']
                for key in ('aligned','geometry','center_local','boundary','wire_excess','area_excess'):
                    lines.append(f'| {key} | {first["dev"][key]:.6f} | {last["dev"][key]:.6f} |')
                lines+=['','代理项下降不保证最终改善。该轮真实Q评价的边界/分组违反与MIB分别见各quality节点results.json；不能将草稿边界距离下降等同于最终边界违反次数下降。']
    (run/'analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(write_report(args.run)),flush=True)


if __name__=='__main__':main()
