"""Predeclared case/node subgroups from each unique G-selected best; no new inference."""
import json
import numpy as np
from .io import ROOT,HERE,save
from .pipeline import OUTPUTS
from .data import load_rows


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def member(row):return tuple(row['member'])


def public_strata(rows):
    data={}
    for row in rows:
        n,area,b2b,p2b,pins,c,s=row['case'].solver_inputs
        adj=np.zeros((n,n),bool);pin=np.zeros(n,bool)
        for i,j,w in np.asarray(b2b):
            if w>0:adj[int(i),int(j)]=adj[int(j),int(i)]=True
        for pi,bi,w in np.asarray(p2b):
            if w>0:pin[int(bi)]=True
        coverage=float(pin.mean());data[member(row)]={'n':n,'pin_coverage':coverage,
            'pin_group':'low' if coverage<=.3 else ('high' if coverage>=.8 else 'middle'),
            'density':float(adj.sum()/max(n*(n-1),1))}
    med={n:np.median([r['density'] for r in data.values() if r['n']==n]) for n in {v['n'] for v in data.values()}}
    for r in data.values():r['sparse_within_N']=r['density']<=med[r['n']]
    return data


def summarize_records(records,features):
    result={}
    for label in ('all','low','middle','high','low_sparse'):
        selected=[r for r in records if label=='all' or features[member(r)]['pin_group']==label or
                  (label=='low_sparse' and features[member(r)]['pin_group']=='low' and features[member(r)]['sparse_within_N'])]
        result[label]={'common_G':float(np.mean([r.get('loss_old_G',r['loss_total']) for r in selected])),'count':len(selected),**{k:float(np.mean([r[k] for r in selected])) for k in
                     ('loss_total','overlap_area_fraction')},
                     'members':[list(member(r)) for r in selected]}
    return result


def module_errors(terminal,rows):
    mapping={member(r):r for r in rows};values={}
    for rec in terminal['records']:
        if 'draft_positions' not in rec:continue
        row=mapping[member(rec)];g=row['graph'];p=np.asarray(rec['draft_positions']);ref=(g['reference']*g['scale']).numpy()
        c=np.asarray(row['case'].solver_inputs[5]);pin=np.asarray(g['pin_weights']).sum(1)>0;mutable=c[:,1]==0
        size=np.sqrt(np.asarray(g['area'])*float(g['scale'])**2)
        err=np.linalg.norm((p[:,:2]+p[:,2:]/2)-(ref[:,:2]+ref[:,2:]/2),axis=1)/size
        record={}
        for label,mask in (('pin',pin&mutable),('no_pin',~pin&mutable)):
            record[label]=float(err[mask].mean()) if mask.any() else None
        i,j=np.triu_indices(len(p),1);pairsize=(size[i]+size[j])/2
        rc=ref[:,:2]+ref[:,2:]/2;pc=p[:,:2]+p[:,2:]/2;rd=rc[i]-rc[j];pd=pc[i]-pc[j]
        margin=np.abs(rd)-(ref[i,2:]+ref[j,2:])/2
        near=(np.linalg.norm(np.maximum(margin,0),axis=1)<=.25*pairsize)&(mutable[i]|mutable[j])
        clear=(np.abs(rd)>.25*pairsize[:,None])&near[:,None]
        record['near_relative_error']=float((np.linalg.norm(pd-rd,axis=1)/pairsize)[near].mean()) if near.any() else None
        record['near_axis_flip_fraction']=float(((pd*rd<0)&clear).sum()/max(clear.sum(),1))
        values[member(rec)]=record
    return values


def main():
    rows=load_rows('dev');features=public_strata(rows);all_values={};histories={}
    references={'E64_baseline':ROOT/'experiments/E64_e42_unified_retrain/runs/20261027_e42_fixedG_scratch_seed0',
                'E65_geometry':ROOT/'experiments/E65_geometry_messages/runs/20261027_geometry_message_seed0',
                'E68_full_ports':ROOT/'experiments/E68_port_tokens/runs/20261028_full_ports_nosummary_v2_seed0',
                'E70_taskloss':ROOT/'experiments/E70_reference_task_loss/runs/20261028_full_ports_taskloss_v1_seed0'}
    for arm,output in {**references,**OUTPUTS}.items():
        raw=read(output/'full_dev.json');train=read(output/'results.json');progress=read(output/'progress.json')
        terminal=read(output/'terminal/results.json');node=module_errors(terminal,rows)
        groups=summarize_records(raw['records'],features)
        for label,group in groups.items():
            eligible={tuple(v) for v in group['members']};ns=[v for k,v in node.items() if k in eligible]
            group['saved500_node_diagnostics']={k:float(np.mean(vs)) if vs else None for k in ('pin','no_pin','near_relative_error','near_axis_flip_fraction')
                for vs in [[v[k] for v in ns if v[k] is not None]]}
            group['saved500_count']=len(ns)
        all_values[arm]={'training':train,'strata':groups,'terminal':read(output/'terminal/summary.json'),
                        'checkpoint_sha256':raw['checkpoint_sha256']}
        histories[arm]=progress['history']
    paired={}
    for baseline in references:
        paired[baseline]={}
        for label in all_values['full_ports']['strata']:
            f=all_values['full_ports']['strata'][label];c=all_values[baseline]['strata'][label]
            assert f['members']==c['members']
            paired[baseline][label]={'count':f['count'],'common_G_delta':f['common_G']-c['common_G'],
                           'overlap_delta_percentage_points':100*(f['overlap_area_fraction']-c['overlap_area_fraction'])}
    from .saved_geometry import compare
    audit=read(ROOT/'experiments/E70_reference_task_loss/runs/20261028_bbox_void_audit/saved_geometry.json')
    geometries={arm:compare(read(output/'terminal/results.json'),audit['records']['E70']) for arm,output in OUTPUTS.items()}
    save(HERE/'runs/geometry_vs_E70.json',geometries)
    gallery=read(ROOT/'experiments/E70_reference_task_loss/runs/20261028_bbox_void_audit/gallery100_E42.json')
    versus42={arm:compare(None,gallery['records']['E42'],current_records=g['current_records']) for arm,g in geometries.items()}
    save(HERE/'runs/geometry_vs_E42_gallery100.json',versus42)
    save(HERE/'runs/review.json',{'arms':all_values,'combined_minus_historical_baselines':paired,
        'geometry_vs_E70':{k:{a:b for a,b in v.items() if a not in ('current_records','paired')} for k,v in geometries.items()},
        'geometry_vs_E42_gallery100':{k:{a:b for a,b in v.items() if a not in ('current_records','paired')} for k,v in versus42.items()},
        'attribution':'Combined proposal versus saved single-seed baselines; no component-wise causal attribution or matched-capacity control',
        'strata':'Public thresholds coverage<=.3 / >=.8; sparse<=exact-N median on unchanged dev2000',
        'all_seeds':[0],'multi_seed_confirmation':False,'baseline_promoted':False,'test_read':False})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for arm,history in histories.items():
        for role,style in (('train_monitor','--'),('dev_monitor','-')):
            for ax,key in zip(axes,('old_G','raw_overlap_area_fraction')):
                ax.plot([r['step'] for r in history],[r[role].get('old_G',r[role]['total']) if key=='old_G' else r[role]['draft_diagnostics']['overlap_area_fraction']['mean'] for r in history],style,label=arm+' '+role)
                ax.set(xlabel='Optimizer steps',title=key);ax.grid(alpha=.2)
    axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(HERE/'runs/learning_curves.png',dpi=160);plt.close(fig)
    text='# E71：温和reference＋多尺度密度＋紧凑约束\n\n一个seed0组合候选，保持96×12和16k。位置0.375降至0.25，紧凑0.5，新增密度0.5；reference仅监督位置/形状。唯一best只在第16轮起按稳定最终Loss选出；旧G和Q仅诊断，不跨阶段解释当期总Loss。复用历史E70及其他既有证据，不作单项因果归因。\n\n'
    text+='| 组 | 分层 | 数量 | dev2000共同旧G | 交面积均值 |\n|---|---|---:|---:|---:|\n'
    for arm,value in all_values.items():
        for label,g in value['strata'].items():text+=f"| {arm} | {label} | {g['count']} | {g['common_G']:.6f} | {g['overlap_area_fraction']:.3%} |\n"
    text+='\n精确外框和空白比较（开发500，同例E70）：\n\n| 模型 | B/A | 空白/B | 重复覆盖/A | 相对E70外框变化 | 外框增大例数 |\n|---|---:|---:|---:|---:|---:|\n'
    for arm,g in geometries.items():
        r=g['current'];text+=f"| {arm} | {r['B_over_A']:.6f} | {r['empty_over_B']:.3%} | {r['redundant_over_A']:.3%} | {g['bbox_relative_change_mean']:+.3%} | {g['bbox_larger']}/{r['count']} |\n"
    text+='\n[精确并集同例证据](geometry_vs_E70.json)。这张表与dev2000交面积使用不同面板，不混算。\n'
    text+='\n同100例相对已接受E42的外框比较：\n\n'
    for arm,g in versus42.items():
        text+=f"- {arm}：同{g['current']['count']}例外框平均变化{g['bbox_relative_change_mean']:+.3%}，{g['bbox_larger']}例增大，当前空白/B为{g['current']['empty_over_B']:.3%}。\n"
    text+='\n[同100例精确几何证据](geometry_vs_E42_gallery100.json)。\n'
    text+='\n组合方案相对历史参照的同例差及500例无端口模块误差、近邻相对误差/次序翻转见[review.json](review.json)。只作单seed自适应开发证据，不以flash Q重选checkpoint，不自动升级；草稿观感待用户评价。\n\n![训练曲线](learning_curves.png)\n\n'
    for arm,output in OUTPUTS.items():text+=f"- [{arm}同100例reference/current/E42无评分图]({output.name}/draft_comparison100/index.html)\n"
    text+='\n本候选完成即结束；不自动恢复旧长期研究或启动其他模型，测试保持封存。\n'
    (HERE/'runs/review.md').write_text(text,encoding='utf-8');print('E71 new-task-loss candidate and sparse/low-pin subgroups summarized.',flush=True)


if __name__=='__main__':main()
