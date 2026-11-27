"""Aggregate original 500 traces as recorded; no solver and no rescoring."""
from .resources import configure
configure()
from collections import Counter
import json
from .io import HERE,ROOT,save,sha


def main():
    path=ROOT/'experiments/E25_integrated_baseline/runs/20261020_controls_steady_monitor500/results.json';result=json.loads(path.read_text());rows=result['records'];stats={}
    for row in rows:
        phases=set()
        for event in row['trace']:
            name=event['phase'];s=stats.setdefault(name,{'cases':0,'calls':0,'legal_candidates':0,'legal_field_present':0,'accepted_moves':0,'accepted_cases':0,'tested_moves':0,'seconds_inclusive':0.,'statuses':Counter(),'keys':set()})
            s['calls']+=1;s['keys'].update(event);s['seconds_inclusive']+=event.get('seconds',0.);s['statuses'][str(event.get('status','not_recorded'))]+=1
            if 'legal' in event:s['legal_field_present']+=1;s['legal_candidates']+=bool(event['legal'])
            if event.get('accepted',0):s['accepted_cases']+=name not in phases;s['accepted_moves']+=event['accepted']
            s['tested_moves']+=event.get('tested',0)
            if name not in phases:s['cases']+=1;phases.add(name)
    for s in stats.values():s['keys']=sorted(s['keys']);s['case_rate']=s['cases']/500
    out={'source_sha256':sha(path),'source':str(path.relative_to(ROOT)),'summary':result['summary'],'phases':stats,
        'original_resources':'two worker processes times two library threads; new E35 uses one times two',
        'missing':['incumbent update source','intermediate layouts/Q','gate skip reasons','recursive internal trace','packing seed construction time','canonicalization movement and effect'],'test_read':False}
    folder=HERE/'runs/20261022_historical500';folder.mkdir(parents=True,exist_ok=True);save(folder/'summary.json',out)
    lines=['# E25原始500例轨迹盘点','','这是原记录聚合，不重跑、不重评分。原资源2进程×2线程；新诊断总计两线程，两者耗时不能作纯实现比较。','',
        '| 阶段 | 触发例数 | 调用数 | 硬合法候选 | 接受移动数 | 累计记录秒 |','|---|---:|---:|---:|---:|---:|']
    for name,s in stats.items():lines.append(f"| {name} | {s['cases']}/500 | {s['calls']} | {s['legal_candidates'] if s['legal_field_present'] else '未记录'} | {s['accepted_moves'] if name=='certified_rigid_moves' else '未记录'} | {s['seconds_inclusive']:.3f} |")
    lines+=['','legal表示候选硬合法，不代表更新最佳解。认证调用包含零修改，须看accepted_cases。未记录字段不是零；递归救援内部未展开，seconds不完整且不能与子阶段重复相加。','',f"认证实际接受移动的实例{stats['certified_rigid_moves']['accepted_cases']}/500，接受移动总数{stats['certified_rigid_moves']['accepted_moves']}。",'']
    (folder/'analysis.md').write_text('\n'.join(lines),encoding='utf-8');print(json.dumps(out,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
