"""Offline function-level geometry/contact diagnostics and stored counterexamples."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
from experiments.E35_optimizer_attribution.resources import configure
configure()
import json
import sqlite3
import zlib
from collections import Counter
import numpy as np
from experiments.E35_optimizer_attribution.cases import prepare_cache
from experiments.E35_optimizer_attribution.io import save,sha
from scoring.evaluation import evaluate_case
from visualization.layouts import plot_case,geometry_details

HERE=ROOT/'experiments/E35_optimizer_attribution';OUT=Path(__file__).resolve().parent


def main():
    path=HERE/'runs/20261022_attribution100/outcomes.sqlite3'
    c=sqlite3.connect('file:'+path.resolve().as_posix()+'?mode=ro',uri=True)
    records=[json.loads(zlib.decompress(row[0])) for row in c.execute('SELECT payload FROM records')];c.close()
    byarm={a:{r['index']:r for r in records if r['arm']==a} for a in {r['arm'] for r in records}}
    rows=prepare_cache()['rows'];summary=json.loads((OUT/'summary.json').read_text());geometry={};guards=Counter();details=[];layout_cache=json.loads((OUT/'layout_scores_local.json').read_text())
    interesting={'shape_solve','fixed_shape_projection','canonicalize','negative_roundoff_guard','certified_polish','packed_seed'}
    for index,r in byarm['full'].items():
        case=rows[index]['case'];area=np.asarray(case.solver_inputs[1]);unit=np.sqrt(area);layouts=r['layouts'];scored=layout_cache.setdefault(str(index),{})
        def score(k):
            if k not in scored:
                try:scored[k]=evaluate_case(case,layouts[k],0.)['metrics']
                except Exception as error:scored[k]={'cost':10.,'is_feasible':False,'error':repr(error)}
            return scored[k]
        for e in r['events']:
            if e['kind']!='call':continue
            name=e['module']
            if name=='hard_guard':guards['accepted' if e.get('returned_bool') else 'rejected']+=1
            if name not in interesting or not e.get('input') or not e.get('output'):continue
            a=np.asarray(layouts[e['input']]);b=np.asarray(layouts[e['output']]);delta=b-a;shift=(b[:,:2]+b[:,2:]/2)-(a[:,:2]+a[:,2:]/2)
            distance=np.linalg.norm(shift,axis=1)/np.maximum(unit,1e-12)
            s=geometry.setdefault(name,{'calls_with_geometry':0,'changed_calls':0,'changed_blocks_above_1e_6_block':0,'changed_blocks_above_1e_12_block':0,'sum_center_move_block_units':0.,'max_center_move_block_units':0.,'shape_changed_blocks':0,
                'raw_illegal_to_legal':0,'soft_violation_reduction_sum':0,'Q_gain_sum':0.,'Q_worse_calls':0})
            s['calls_with_geometry']+=1;s['changed_calls']+=e['input']!=e['output'];s['changed_blocks_above_1e_6_block']+=int((distance>1e-6).sum());s['changed_blocks_above_1e_12_block']+=int((distance>1e-12).sum())
            s['sum_center_move_block_units']+=float(distance.sum());s['max_center_move_block_units']=max(s['max_center_move_block_units'],float(distance.max()));s['shape_changed_blocks']+=int((np.abs(delta[:,2:]).max(1)>1e-6*unit).sum())
            if name in ('canonicalize','negative_roundoff_guard','certified_polish') and e['input']!=e['output']:
                before,after=score(e['input']),score(e['output']);s['raw_illegal_to_legal']+=not before['is_feasible'] and after['is_feasible']
                if before['is_feasible'] and after['is_feasible']:
                    s['Q_gain_sum']+=before['cost']-after['cost'];s['Q_worse_calls']+=after['cost']>before['cost']+1e-8
                    s['soft_violation_reduction_sum']+=before.get('total_soft_violations',0)-after.get('total_soft_violations',0)
            if name=='certified_polish' and e.get('stats',{}).get('accepted',0):
                before,after=score(e['input']),score(e['output'])
                details.append({'index':index,'member':r['member'],'n':r['n'],'max_center_move_block_units':float(distance.max()),'accepted':e['stats']['accepted'],'before':before,'after':after})
    figures=[]
    for arm,tag in [('no_robust_seeds','rescue_tail_counterexample'),('no_projection','projection_local_counterexample'),('no_boundary_penalty','boundary_counterexample')]:
        index=max(byarm[arm],key=lambda i:byarm[arm][i]['metrics']['cost']-byarm['reference'][i]['metrics']['cost']);case=rows[index]['case'];constraints=np.asarray(case.solver_inputs[5]).tolist();variants={}
        for name,record in [('baseline',byarm['reference'][index]),('removed',byarm[arm][index])]:
            variants[name]={'positions':record['positions'],'metrics':record['metrics'],'soft_details':geometry_details(record['positions'],constraints)}
        plot_case({'panel':'E35 stored matched removal','case_id':str(byarm[arm][index]['member']),'constraints':constraints,'variants':variants},OUT/(tag+'.png'),stages=(('baseline','E25 original logic'),('removed',arm)))
        figures.append({'file':tag+'.png','arm':arm,'index':index,'member':rows[index]['member'],'n':case.block_count,'baseline':byarm['reference'][index]['metrics'],'removed':byarm[arm][index]['metrics']})
    old=json.loads((HERE/'runs/20261022_historical500/summary.json').read_text());old_config=json.loads((ROOT/'experiments/E25_integrated_baseline/runs/20261020_controls_steady_monitor500/config.json').read_text())
    parity={}
    for name in ('solver.py','fast_geometry.py','projection.py','certify_polish.py','model.py'):
        old_name='experiments/E24_reference_fast/'+name
        parity[name]={'original_recorded_sha256':old_config['sources'].get(old_name),'current_E25_sha256':sha(ROOT/'experiments/E25_integrated_baseline'/name)}
        parity[name]['same']=parity[name]['original_recorded_sha256']==parity[name]['current_E25_sha256']
    out={'function_geometry_all_recursive_calls':geometry,'hard_guard_calls':dict(guards),'certified_effective_cases':details,'figures':figures,
        'original500_critical_source_bytes_match':parity,'scope':'offline saved states; local canonical/cleanup before-after is not a full-system removal','source_script_sha256':sha(Path(__file__)),'test_read':False}
    save(OUT/'mechanisms.json',out)
    text=['# 机制补充：微小移动、硬守卫和反例','','仅分析已保存布局，没有新增求解。模块函数的输入/输出几何统计包含递归调用，不能和主流程候选表直接相加。块单位为中心位移除以该块公开面积的平方根。','',
        '| 函数 | 有布局输入/输出的调用 | 几何有变调用 | 移动块次 >1e-6块单位 | 移动块次 >1e-12块单位 | 最大中心位移/块单位 |','|---|---:|---:|---:|---:|---:|']
    for name,s in geometry.items():text.append(f"| {name} | {s['calls_with_geometry']} | {s['changed_calls']} | {s['changed_blocks_above_1e_6_block']} | {s['changed_blocks_above_1e_12_block']} | {s['max_center_move_block_units']:.4g} |")
    text+=['','硬守卫调用计数：'+str(dict(guards))+'。其中包括每个递归返回的最终断言，不把所有通过次数解释为修复次数。','',
        '认证微调的有效案例：','']
    for d in details:text.append(f"- index={d['index']}，n={d['n']}，Q {d['before']['cost']:.6f}→{d['after']['cost']:.6f}，软违反{d['before'].get('total_soft_violations')}→{d['after'].get('total_soft_violations')}，最大中心位移{d['max_center_move_block_units']:.4g}块单位。")
    text+=['','极小几何变化也可能跨越精确接触边界，显著改变分组连通性；用大位移阈值统计会把有效微调误计为零。canonical化的局部前后差只解释实际函数作用，不替代关闭该模块后的全流程消融。','']
    for f in figures:text.extend([f"## {f['arm']}：index={f['index']}，n={f['n']}",'',f"Q {f['baseline']['cost']:.6f}→{f['removed']['cost']:.6f}；两者硬合法={f['baseline']['is_feasible']}/{f['removed']['is_feasible']}。",'',f"![保存布局对照]({f['file']})",''])
    text+=['[精确机制记录与来源字节校验](mechanisms.json)',''];(OUT/'mechanisms.md').write_text('\n'.join(text),encoding='utf-8')
    print(json.dumps({'guards':dict(guards),'geometry':geometry,'certified':details,'source_parity':parity,'figures':[{k:f[k] for k in ('arm','index','n')} for f in figures]},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
