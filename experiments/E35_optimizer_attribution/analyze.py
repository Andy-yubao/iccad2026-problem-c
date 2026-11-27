"""Saved-state attribution and paired removals; never invoke an optimizer."""
from .resources import configure
configure()
from collections import Counter,defaultdict
import json
import os
import sqlite3
import zlib
import numpy as np
from .cases import prepare_cache
from .io import HERE,ROOT,save,sha
from scoring.evaluation import evaluate_case

LABELS={'reference':'E25原逻辑','full':'完整观测','no_shape':'关闭形状迭代','no_projection':'关闭末端/周期投影','no_quick_project':'关闭早期投影','no_robust_seeds':'关闭打包救援','no_certified':'关闭认证微调','no_plateau':'关闭平台早停','no_late_extension':'关闭预算延长','no_incumbent_reset':'关闭最佳解重置','linear_projection':'投影改线性目标','no_contact_penalty':'关闭接触惩罚','no_boundary_penalty':'关闭边界惩罚','no_reweight':'关闭惩罚重加权'}


def paired(base,other):
    keys=sorted(base);a=[base[k] for k in keys];b=[other[k] for k in keys]
    n=np.asarray([r['n'] for r in a]);w=np.exp((n-n.max())/12)
    d=np.asarray([y['metrics']['cost']-x['metrics']['cost'] for x,y in zip(a,b)]);td=np.asarray([y['seconds']-x['seconds'] for x,y in zip(a,b)])
    shards=sorted({r['member'][0] for r in a});lookup={s:i for i,s in enumerate(shards)};g=np.asarray([lookup[r['member'][0]] for r in a]);rng=np.random.default_rng(20261022)
    samples=rng.integers(0,len(shards),(2000,len(shards)));dgroup=np.bincount(g,weights=w*d);wg=np.bincount(g,weights=w);tg=np.bincount(g,weights=td);ng=np.bincount(g)
    return {'weighted_Q_delta':float(np.average(d,weights=w)),'weighted_Q_delta_95CI':np.percentile(dgroup[samples].sum(1)/wg[samples].sum(1),[2.5,97.5]).tolist(),
        'seconds_delta':float(td.mean()),'seconds_delta_95CI':np.percentile(tg[samples].sum(1)/ng[samples].sum(1),[2.5,97.5]).tolist(),
        'improved_cases':int((d<-1e-8).sum()),'worse_cases':int((d>1e-8).sum()),'max_regression':float(d.max()),'shards':len(shards),
        'worst_cases':[{'index':keys[i],'member':a[i]['member'],'n':int(n[i]),'delta_Q':float(d[i]),'baseline_Q':a[i]['metrics']['cost'],'removed_Q':b[i]['metrics']['cost']} for i in np.argsort(-d)[:5]]}


def movement(a,b,areas):
    a=np.asarray(a);b=np.asarray(b);shift=(b[:,:2]+b[:,2:]/2)-(a[:,:2]+a[:,2:]/2)
    norm=np.sqrt(np.asarray(areas));dist=np.linalg.norm(shift,axis=1)/np.maximum(norm,1e-12)
    return {'moved_blocks':int((dist>1e-6).sum()),'mean_center_move_block_units':float(dist.mean()),'sum_center_move_block_units':float(dist.sum()),
        'shape_changed_blocks':int((np.max(np.abs(a[:,2:]-b[:,2:]),axis=1)>1e-6*norm).sum())}


def main():
    folder=HERE/'runs/20261022_attribution100';raw=json.loads((folder/'summary.json').read_text())
    if raw['status']!='complete':raise ValueError('Complete fixed panel required')
    database=folder/'outcomes.sqlite3';con=sqlite3.connect('file:'+database.resolve().as_posix()+'?mode=ro',uri=True)
    records=[json.loads(zlib.decompress(r[0])) for r in con.execute('SELECT payload FROM records ORDER BY idx,arm')];con.close()
    byarm={arm:{r['index']:r for r in records if r['arm']==arm} for arm in LABELS}
    effects={arm:paired(byarm['reference'],values) for arm,values in byarm.items() if arm!='reference'}
    rows=prepare_cache()['rows'];modules={};gates={};functions={};final_sources=Counter();case_diagnostics=[];layout_scores={}
    def module(name):
        return modules.setdefault(name,{'calls':0,'cases':set(),'legal':0,'updates':0,'update_cases':set(),'first_exit_row':0,'subsequent_updates':0,'subsequent_Q_gain_sum':0.,'subsequent_proxy_gain_sum':0.,'proxy_better_Q_worse':0,'moved_blocks_sum':0,'center_move_sum_block_units':0.,'shape_changed_blocks_sum':0})
    for index,r in byarm['full'].items():
        if 'error' in r:continue
        case=rows[index]['case'];layouts=r['layouts'];areas=case.solver_inputs[1];scorecache={}
        def score(key):
            if key not in scorecache:
                x=np.asarray(layouts[key],dtype=float)
                try:scorecache[key]=evaluate_case(case,x,0.)['metrics']
                except Exception as error:scorecache[key]={'cost':10.,'is_feasible':False,'error':repr(error)}
            return scorecache[key]
        row_id=next(e['row'] for e in r['events'] if e['kind']=='initial_state' and e['depth']==1);producer='row_fallback';local_updates=Counter()
        calls=[e for e in r['events'] if e['kind']=='call']
        for e in calls:
            name=e['module'];s=functions.setdefault(name,{'calls':0,'cases':set(),'seconds_inclusive':0.,'seconds_exclusive':0.,'output_geometry_changes':0})
            duration=e.get('seconds_excluding_observer',0.);children=sum(c.get('seconds_excluding_observer',0.) for c in calls if c['parent']==e['id'])
            s['calls']+=1;s['cases'].add(index);s['seconds_inclusive']+=duration;s['seconds_exclusive']+=max(0.,duration-children)
            if e.get('input') and e.get('output') and e['input']!=e['output']:s['output_geometry_changes']+=1
        for e in r['events']:
            if e['kind']=='gate':
                g=gates.setdefault(e['module'],{'checks':0,'triggered':0,'cases':set(),'triggered_cases':set(),'reasons':Counter()});g['checks']+=1;g['cases'].add(index);g['reasons'][e['reason']]+=1
                if e['triggered']:g['triggered']+=1;g['triggered_cases'].add(index)
            if e['kind']=='acceptance' and e['depth']==1:
                s=module(e['module']);s['calls']+=1;s['cases'].add(index);s['legal']+=bool(e['hard_legal'])
                if e['updated']:
                    s['updates']+=1;s['update_cases'].add(index);producer=e['module'];local_updates[producer]+=1
                    move=movement(layouts[e['before']],layouts[e['after']],areas);s['moved_blocks_sum']+=move['moved_blocks'];s['center_move_sum_block_units']+=move['sum_center_move_block_units'];s['shape_changed_blocks_sum']+=move['shape_changed_blocks']
                    before,after=score(e['before']),score(e['after'])
                    if e['before']==row_id:s['first_exit_row']+=1
                    else:
                        s['subsequent_updates']+=1;s['subsequent_Q_gain_sum']+=before['cost']-after['cost'];s['subsequent_proxy_gain_sum']+=e['before_proxy']-e['after_proxy'];s['proxy_better_Q_worse']+=after['cost']>before['cost']+1e-8
            if e['kind']=='rigid_accepted' and e['depth']==1:
                s=module('certified_rigid_move');s['calls']+=1;s['cases'].add(index);s['legal']+=1;s['updates']+=1;s['update_cases'].add(index);s['subsequent_updates']+=1
                s['subsequent_Q_gain_sum']+=score(e['before'])['cost']-score(e['after'])['cost'];move=movement(layouts[e['before']],layouts[e['after']],areas)
                s['moved_blocks_sum']+=move['moved_blocks'];s['center_move_sum_block_units']+=move['sum_center_move_block_units'];s['shape_changed_blocks_sum']+=move['shape_changed_blocks'];producer='certified_rigid_move'
        final_sources[producer]+=1;case_diagnostics.append({'index':index,'member':r['member'],'n':r['n'],'final_producer':producer,'root_updates':dict(local_updates),'observer_seconds':r['observer_seconds'],'seconds':r['seconds']})
        layout_scores[str(index)]=scorecache
    for s in modules.values():
        s['cases']=len(s['cases']);s['update_cases']=len(s['update_cases'])
    for s in functions.values():s['cases']=len(s['cases'])
    for s in gates.values():s['cases']=len(s['cases']);s['triggered_cases']=len(s['triggered_cases'])
    output={'scope':'fixed adaptive dev100, input-only solver, neutral runtime Q, no training/test','summaries':raw['summaries'],'paired_vs_original_logic':effects,
        'root_acceptance_modules':modules,'gates_all_recursive_calls':gates,'functions_all_recursive_calls':functions,'final_update_sources':dict(final_sources),'cases':case_diagnostics,
        'observer_seconds_mean':float(np.mean([r.get('observer_seconds',0.) for r in byarm['full'].values()])),
        'sources':{'run_summary':sha(folder/'summary.json'),'database':sha(database)},'test_read':False}
    report=HERE/'runs/20261022_attribution_report';report.mkdir(parents=True,exist_ok=True);save(report/'summary.json',output);save(report/'layout_scores_local.json',layout_scores)
    write_report(report,output)
    print(json.dumps({'modules':modules,'final_sources':dict(final_sources),'effects':effects},ensure_ascii=False),flush=True)


def write_report(folder,r):
    lines=['# E25优化器：功能触发与贡献诊断','','固定100例来自原monitor500按模块数分层抽取，不按成绩或救援选例。CPU一进程、两个执行槽；所有14种设置各执行同一100例。Q为MIB派生自适应开发中性时间指标，非比赛成绩。','',
        '原始500轨迹见[历史盘点](../20261022_historical500/analysis.md)。触发率、更新率、几何改动、删除效应是四种证据，不互相替代。','',
        '| 主流程候选来源 | 调用 | 涉及实例 | 硬合法 | 更新最佳解 | 首次替换行布局 | 后续Q改善之和 |','|---|---:|---:|---:|---:|---:|---:|']
    for name,s in r['root_acceptance_modules'].items():lines.append(f"| {name} | {s['calls']} | {s['cases']} | {s['legal']} | {s['updates']} | {s['first_exit_row']} | {s['subsequent_Q_gain_sum']:.6f} |")
    lines+=['','Q改善之和仅是轨迹中的直接、等权收益，排除首次替换行布局，不是模块的因果贡献或加权总Q。内部目标改善也可能使参考归一化Q恶化，精确计数另见summary。','',
        '| 设置 | 加权Q | 平均秒 | CVaR95 | 最大Q | 硬合法 | ΔQ 对原逻辑 |','|---|---:|---:|---:|---:|---:|---:|']
    for arm,s in r['summaries'].items():
        delta=r['paired_vs_original_logic'].get(arm,{}).get('weighted_Q_delta',0.)
        lines.append(f"| {LABELS[arm]} | {s['weighted_Q']:.6f} | {s['seconds_mean']:.6f} | {s['cvar95_Q']:.6f} | {s['max_Q']:.6f} | {s['hard_legal']}/{s['count']} | {delta:+.6f} |")
    lines+=['','## 解释范围','',
        '单模块关闭对整个递归调用树生效。释放的墙钟预算可由其他分支使用，结果包含后续轨迹改变与预算重分配；多个模块具有互补/替代关系，不能把删除效应相加。硬守卫与行布局保底保留，因此最终合法率不能独自证明求解质量。',
        '原逻辑臂只调整独立副本的相对导入，算法保持E25。完整观测臂扣除了直接记录开销对控制器/认证时钟的影响，但缓存、分配及系统计时扰动仍存在；其相对原逻辑的配对差是扰动对照。消融主要对照原逻辑，并非对历史四线程数据作速度结论。',
        '95%区间按完整来源分片做配对bootstrap，仅描述本次开发面板的不确定性，不是多重比较后的显著性认证。100例未触发或收益为零的机制不能据此全局删除，尤其应结合原500的低频反例。',
        '最后更新来源不代表之前步骤可删除；形状初始化产生的拓扑、救援的可行起点、后续投影等都有路径依赖。所有候选和布局留在本机数据库，评分在求解之后完成，参考信息没有参与决策。','',
        '[精确汇总、门控原因、几何改动与最差反例](summary.json)','']
    (folder/'analysis.md').write_text('\n'.join(lines),encoding='utf-8')
    plot(folder,r)


def plot(folder,r):
    os.environ['MPLCONFIGDIR']=str(ROOT/'.cache/matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    keys=[k for k in r['paired_vs_original_logic'] if k!='full'];keys.sort(key=lambda k:r['paired_vs_original_logic'][k]['weighted_Q_delta'])
    fig,axes=plt.subplots(1,2,figsize=(14,7));y=np.arange(len(keys))
    for ax,key,ci,title in [(axes[0],'weighted_Q_delta','weighted_Q_delta_95CI','Removal effect on weighted Q'),(axes[1],'seconds_delta','seconds_delta_95CI','Removal effect on complete runtime')]:
        v=np.array([r['paired_vs_original_logic'][k][key] for k in keys]);bounds=np.array([r['paired_vs_original_logic'][k][ci] for k in keys]);ax.errorbar(v,y,xerr=np.maximum(0,np.stack((v-bounds[:,0],bounds[:,1]-v))),fmt='o',capsize=3)
        ax.axvline(0,color='black',lw=.8);ax.set_yticks(y,keys);ax.grid(alpha=.2);ax.set_title(title);ax.set_xlabel('Delta; positive is worse');ax.set_xscale('symlog',linthresh=.001)
    fig.tight_layout();fig.savefig(folder/'removal_effects.png',dpi=170);plt.close(fig)


if __name__=='__main__':main()
