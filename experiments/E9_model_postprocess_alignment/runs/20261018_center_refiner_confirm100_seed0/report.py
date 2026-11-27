"""Rebuild this frozen candidate's paired report from saved outputs only."""
import json
from pathlib import Path
import sys
import hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from experiments.E9_model_postprocess_alignment.records import save,read_rows
from scoring.official import official_evaluator

HERE=Path(__file__).resolve().parent
TRAIN=HERE.parent/'20261018_center_refiner_seed0'
PANELS={'dev100':TRAIN/'center_refiner/quality/proxy_best','confirm100':HERE}


def main():
    panels={}
    for name,path in PANELS.items():
        result=json.loads((path/'results.json').read_text(encoding='utf-8'))
        if result['status']!='completed' or result['completed_cases']!=100:
            raise ValueError('Only complete frozen panels enter the comparison')
        rows=read_rows(path/'cases.jsonl')
        assert len(rows)==100 and len({r['case_id'] for r in rows})==100
        old=np.array([r['baseline_metrics']['cost'] for r in rows])
        new=np.array([r['variants']['model']['final']['metrics']['cost'] for r in rows])
        delta=new-old;rng=np.random.default_rng(20261018)
        interval=np.quantile(rng.choice(delta,(10000,len(delta)),replace=True).mean(1),[.025,.975])
        def stats(q,metrics):
            return {'Q_mean':float(q.mean()),'Q_weighted':float(official_evaluator().compute_total_score(q.tolist(),[r['block_count'] for r in rows])),
                    'Q_p95':float(np.quantile(q,.95)),'Q_max':float(q.max()),
                    'worst10percent_mean':float(np.sort(q)[-10:].mean()),
                    **{k:float(np.mean([m[k] for m in metrics])) for k in ['boundary_violations','grouping_violations','mib_violations','hpwl_gap','area_gap']}}
        panels[name]={'old':stats(old,[r['baseline_metrics'] for r in rows]),
                      'new':stats(new,[r['variants']['model']['final']['metrics'] for r in rows]),
                      'mean_difference':float(delta.mean()),'relative_difference':float(new.mean()/old.mean()-1),
                      'wins':int((delta < -1e-8).sum()),'ties':int((np.abs(delta)<=1e-8).sum()),'losses':int((delta > 1e-8).sum()),
                      'paired_bootstrap95_descriptive':interval.tolist(),
                      'feasible':sum(r['variants']['model']['final']['metrics']['is_feasible'] for r in rows),
                      'inference_and_postprocess_mean_seconds':result['summary']['end_to_end_seconds_mean'],
                      'wall_seconds':result['cumulative_seconds'],
                      'cases_sha256':hashlib.sha256((path/'cases.jsonl').read_bytes()).hexdigest()}
    selection=json.loads((TRAIN/'center_refiner/selection.json').read_text(encoding='utf-8'))
    config=json.loads((HERE/'config.json').read_text(encoding='utf-8'))
    assert selection['checkpoint']==config['identity']['checkpoint']
    checkpoint=Path(selection['checkpoint']);digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert digest==config['identity']['checkpoint_sha256']
    decision={'status':'frozen_confirmation_completed','candidate':selection['checkpoint'],'checkpoint_sha256':digest,
              'panels':panels,'target_Q_mean':1.20,'target_met':False,'test_payload_reads':0,
              'decision':'Retain original E7 plus E8 as main reference. Keep center-refiner checkpoint and results; do not promote or extend this training route.',
              'scope':'Both panels are adaptive development data; bootstrap is descriptive, not a generalization guarantee or official contest score.'}
    save(HERE/'paired_analysis.json',decision)
    lines=['# E9中心修正：冻结候选的两组开发复核','',
           '**结论：固定dev100有小幅收益，但额外开发100例退化，没有达到平均Q＜1.20。保留旧模型＋E8作为主线，本轮候选保留用于诊断，不继续增加训练预算。**','',
           '| 面板 | 原模型＋E8 | 中心修正＋E8 | 均值差（新−旧） | 胜/平/负 | 新模型硬合法 |','|---|---:|---:|---:|---:|---:|']
    for name,p in panels.items():
        lines.append(f'| {name} | {p["old"]["Q_mean"]:.6f} | {p["new"]["Q_mean"]:.6f} | {p["mean_difference"]:+.6f} | {p["wins"]}/{p["ties"]}/{p["losses"]} | {p["feasible"]}/100 |')
    lines+=['','## 为什么不提升为默认模型','',
            '训练共10轮，按原规则验证早停；best为第9轮step5625。总墙钟17.91分钟，最终dev2500原始几何评价完成。dev100与confirm100均使用这一个冻结best，权重哈希在配置及paired_analysis.json中记录，没有看确认分数后再换checkpoint。',
            'dev100原始草稿宽高逐值未变，骨干与形状冻结机制已核验；平均Q下降0.001438（约0.12%），但32胜/35负，收益不是普通例逐例改善。两组MIB违反均值与原参照相同，位置修正没有处理主要MIB罚项。',
            '确认组平均Q上升0.007060（约0.57%），最大Q由2.047516增至2.221291，最差10%也变差。它反对“中心微调收益足以稳定迁移”的解释。不能据此证明所有中心学习都无效，但当前证据不支持再续训或只放宽早停。','',
            '| 面板 | 原/新P95 Q | 原/新最大Q | 原/新最差10%均值 | 原/新边界违反均值 | 原/新分组违反均值 |','|---|---:|---:|---:|---:|---:|']
    for name,p in panels.items():
        a,b=p['old'],p['new']
        lines.append(f'| {name} | {a["Q_p95"]:.6f}/{b["Q_p95"]:.6f} | {a["Q_max"]:.6f}/{b["Q_max"]:.6f} | {a["worst10percent_mean"]:.6f}/{b["worst10percent_mean"]:.6f} | {a["boundary_violations"]:.2f}/{b["boundary_violations"]:.2f} | {a["grouping_violations"]:.2f}/{b["grouping_violations"]:.2f} |')
    lines+=['','## 证据边界与复现','',
            'Q为官方中性时间质量分，不是比赛成绩。两组均已参与历史开发；dev100还用于本轮checkpoint选择。配对bootstrap只是描述性的开发面板区间，不能消除选择偏差、生成相关性或单seed限制。主面板和额外面板分别报告，不用合并平均掩盖退化。']
    for name,p in panels.items():
        lines.append(f'{name}：描述性均值差95%区间{p["paired_bootstrap95_descriptive"]}；模型推理＋后处理均值{p["inference_and_postprocess_mean_seconds"]:.3f}秒；本次评价墙钟{p["wall_seconds"]:.1f}秒。原结果与时间复用各自E8保存参照，未冒充同步在线重测。')
    lines+=['','确认运行仅读取固定workset_v2内E8已固定的confirm100，100/100完成，零执行异常；本轮由助手运行，整次保护900秒，实际约4.8分钟。没有训练、读取封存测试、改数据、改评分或修改原训练selection。',
            '根训练selection仍记录“dev100选中中心修正模型”的真实事实；本报告额外记录“确认后不提升主线”的决定，两者用途不同。保护原权重、候选best/latest及完整运行证据。',
            '下一步需要新的可证伪机制，优先核查剩余可避免MIB违反与真实后处理反馈如何进入模型；不再盲目追加局部位置微调训练。兼容测试面板仍未筛选，后续另立协议。','',
            '复算本报告只读取已保存JSON：从工作区根目录运行本目录report.py；源码快照与实际评价命令见config.json/source_snapshot.zip。更换输出目录可复现同一冻结权重的评价，已完成目录不可覆盖。','']
    (HERE/'analysis.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({name:{k:v for k,v in p.items() if k in ['mean_difference','wins','ties','losses','paired_bootstrap95_descriptive']} for name,p in panels.items()}),flush=True)


if __name__=='__main__':main()
