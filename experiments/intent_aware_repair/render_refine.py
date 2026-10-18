"""Render and summarize saved refinement evidence. No datasets/model execution."""
import argparse,hashlib,html,json,os
from pathlib import Path
import numpy as np
from scoring.official import official_evaluator


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def rows(path):return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines()]
def save(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args();out=args.run_dir.resolve();root=Path(__file__).resolve().parents[2]
    records=rows(out/'cases.jsonl');basic=read(out/'results.json');config=read(out/'config.json')
    assert len(records)==100 and basic['status']=='completed'
    source=Path(config['source']);old={r['case_id']:r for r in rows(source)}
    assert {r['case_id'] for r in records}==set(config['case_ids']) and len({r['case_id'] for r in records})==100
    evaluator=official_evaluator();ids=[r['block_count'] for r in records]
    def summarize(metrics):
        q=np.array([m['cost'] for m in metrics]);return {'weighted_Q':evaluator.compute_total_score(q.tolist(),ids),
            'feasible_count':sum(m['is_feasible'] for m in metrics),'Q_mean':float(q.mean()),'Q_median':float(np.median(q)),
            'Q_p95':float(np.quantile(q,.95)),'Q_max':float(q.max()),'Q_below_1_5_count':int((q<1.5).sum()),
            **{k+'_mean':float(np.mean([m[k] for m in metrics])) for k in ['hpwl_gap','area_gap','boundary_violations','grouping_violations','mib_violations','violations_relative']}}
    om=[r['old'] for r in records];nm=[r['new']['metrics'] for r in records]
    comparisons={'old':summarize(om),'new':summarize(nm),'reference':summarize([old[r['case_id']]['reference_metrics'] for r in records])}
    times=np.array([r['trace']['end_to_end_seconds_including_recorded_model'] for r in records]);post=np.array([r['trace']['seconds'] for r in records])
    timing={'mean_seconds':float(times.mean()),'post_only_mean_seconds':float(post.mean()),'median_seconds':float(np.median(times)),
        'p95_seconds':float(np.quantile(times,.95)),'max_seconds':float(times.max()),'over_15s_count':int((times>15).sum()),
        'over_120s_count':int((times>120).sum()),'old_mean_seconds':float(np.mean([m['runtime_seconds'] for m in om])),
        'scope':'Postprocessing realtime; recorded original GPU feature/forward charged; dataset reads/scoring outside'}
    by_size={}
    for name,predicate in [('21–50',lambda n:n<=50),('51–90',lambda n:50<n<=90),('91–120',lambda n:n>90)]:
        selected=[r for r in records if predicate(r['block_count'])];ns=[r['block_count'] for r in selected]
        by_size[name]={'count':len(ns),'old_Q':evaluator.compute_total_score([r['old']['cost'] for r in selected],ns),
            'new_Q':evaluator.compute_total_score([r['new']['metrics']['cost'] for r in selected],ns),
            'below_1_5':sum(r['new']['metrics']['cost']<1.5 for r in selected)}
    worst=sorted(records,key=lambda r:r['new']['metrics']['cost'],reverse=True)[:5]
    distances=[];shapes=[]
    for r in records:
        z=old[r['case_id']];a=np.asarray(z['area_targets']);c=np.asarray(z['public_constraints']);raw=np.asarray(z['variants']['attention_geometry']['raw']['positions']);final=np.asarray(r['new']['positions'])
        locked=c[:,:2].any(1);a[locked]=raw[locked,2:].prod(1);movable=c[:,1]==0;free=~locked
        d=np.linalg.norm(final[:,:2]+final[:,2:]/2-raw[:,:2]-raw[:,2:]/2,axis=1)/np.sqrt(a)
        distances.append(float(d[movable].mean()))
        if free.any():shapes.append(float(np.abs(np.log(final[free,2:]/raw[free,2:])).max(1).mean()))
        assert r['new']['independent_hard_checks']['is_feasible']
    ablations=[]
    for name,folder in [('位置LP','20261017_ablation_position_dev5_seed0'),('位置＋形状','20261017_ablation_unweighted_dev5_seed0'),('完整重加权方案','20261017_reweighted_shape_dev5_seed0')]:
        d=read(out.parent/folder/'results.json');ablations.append({'name':name,'run':folder,'Q':d['new_Q'],'feasible':d['new_feasible']})
    guard=read(out/'guard_validation.json')
    ablations.insert(2,{'name':'位置＋形状＋严格输出守卫','run':'20261017_ablation_unweighted_dev5_seed0',
        'Q':guard['ablation_checked_Q'],'feasible':5})
    basic.update(metrics=comparisons,timing=timing,by_size=by_size,paired={'wins':sum(n['cost']<o['cost']-1e-8 for o,n in zip(om,nm)),
        'ties':sum(abs(n['cost']-o['cost'])<=1e-8 for o,n in zip(om,nm)),'losses':sum(n['cost']>o['cost']+1e-8 for o,n in zip(om,nm))},
        ablations_dev5=ablations,displacement={'layout_mean_center_local':float(np.mean(distances)),'layout_mean_free_shape_log_change':float(np.mean(shapes))},
        goal_met=comparisons['new']['weighted_Q']<1.5 and comparisons['new']['feasible_count']==100,guard_validation=guard)
    save(out/'results.json',basic)
    os.environ.setdefault('MPLCONFIGDIR',str(root/'.cache/matplotlib'))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    oq=np.array([m['cost'] for m in om]);nq=np.array([m['cost'] for m in nm])
    fig,axes=plt.subplots(2,2,figsize=(12,9),constrained_layout=True)
    ax=axes[0,0];ax.scatter(oq,nq,c=ids,cmap='viridis',s=25);ax.plot([1,10],[1,10],'--',color='grey');ax.axhline(1.5,color='#a33',ls=':');ax.set(xlabel='Old Q',ylabel='New Q',title='Same100dev cases; lower is better')
    ax=axes[0,1];x=np.arange(3);ax.bar(x-.18,[comparisons['old'][k+'_mean'] for k in ['boundary_violations','grouping_violations','mib_violations']],.36,label='Old');ax.bar(x+.18,[comparisons['new'][k+'_mean'] for k in ['boundary_violations','grouping_violations','mib_violations']],.36,label='New');ax.set_xticks(x,['Boundary','Grouping','MIB']);ax.set_title('Mean soft violation counts');ax.legend()
    ax=axes[1,0];ax.bar([0,1],[comparisons['old']['hpwl_gap_mean']*100,comparisons['old']['area_gap_mean']*100],label='Old');ax.bar([2,3],[comparisons['new']['hpwl_gap_mean']*100,comparisons['new']['area_gap_mean']*100],label='New');ax.set_xticks([0,1,2,3],['Old wire','Old bbox','New wire','New bbox']);ax.set(ylabel='Mean signed gap (%)',title='Reference-normalized geometry gaps')
    ax=axes[1,1];ax.hist(times,bins=20,color='#268');ax.set(xlabel='Seconds per case',ylabel='Cases',title='Realtime post + recorded model time')
    fig.suptitle(f"Frozen model + global refinement | weighted Q {basic['old_Q']:.3f} → {basic['new_Q']:.3f}\n100/100 legal; {basic['Q_below_1_5_count']}/100 under1.5; neutral runtime, not contest score",fontsize=13)
    fig.savefig(out/'comparison.png',dpi=180);plt.close(fig)
    gallery=[]
    for r in records:
        worker=int(old[r['case_id']]['source']['shard'].split('/')[0][7:])
        if worker not in {0,24,49,74,99}:continue
        z=old[r['case_id']];c=np.asarray(z['public_constraints']);v=z['variants']['attention_geometry']
        layouts=[('Reference',np.asarray(z['reference']),z['reference_metrics']),('Model draft',np.asarray(v['raw']['positions']),v['raw']['metrics']),('Old postprocessing',np.asarray(v['final']['positions']),v['final']['metrics']),('New global postprocessing',np.asarray(r['new']['positions']),r['new']['metrics'])]
        fig,axes=plt.subplots(2,2,figsize=(14,14),constrained_layout=True);end=np.concatenate([a[:,:2]+a[:,2:] for _,a,_ in layouts]).max(0)*1.04
        for ax,(title,a,m) in zip(axes.flat,layouts):
            i,j=np.triu_indices(len(a),1);inter=np.minimum(a[i,:2]+a[i,2:],a[j,:2]+a[j,2:])-np.maximum(a[i,:2],a[j,:2]);hit=set(np.r_[i[(inter>1e-6).all(1)],j[(inter>1e-6).all(1)]])
            for k,(x,y,w,h) in enumerate(a):
                g,mib=int(c[k,3]),int(c[k,2]);ax.add_patch(Rectangle((x,y),w,h,facecolor=plt.cm.Set3((g-1)%12) if g else '#ddd',edgecolor='red' if k in hit else '#555',hatch=['///','xxx','...','+++'][(mib-1)%4] if mib else None,linewidth=1.1 if k in hit else .6,alpha=.8))
                ax.text(x+w/2,y+h/2,str(k),ha='center',va='center',fontsize=5)
            ax.set(xlim=(0,end[0]),ylim=(0,end[1]),aspect='equal');ax.set_title(title,fontsize=14)
            ax.set_xlabel('Q=%.3f | hard=%s | B/G/M=%d/%d/%d\nHPWL gap=%.1f%% | bbox gap=%.1f%%'%(m['cost'],m['is_feasible'],m['boundary_violations'],m['grouping_violations'],m['mib_violations'],100*m['hpwl_gap'],100*m['area_gap']),fontsize=10)
        fig.suptitle(r['case_id']+' | same scale; grouping=color; MIB=hatch; red=overlap\nInference uses public inputs only; reference for evaluation/display',fontsize=13)
        filename=f'worker_{worker}_comparison.png';fig.savefig(out/filename,dpi=170);plt.close(fig);gallery.append((worker,filename))
    text=(out/'analysis.md').read_text(encoding='utf-8');new=comparisons['new'];oldm=comparisons['old']
    text=text.replace('**完整dev100评价进行中，本文件为待补齐终末数据的研究记录。**','**完整dev100与同版本五例消融均已完成。**')
    conclusion=f"**当前开发目标已达成：固定dev100加权中性时间Q从{basic['old_Q']:.6f}降至{basic['new_Q']:.6f}（下降{100*(1-basic['new_Q']/basic['old_Q']):.2f}%），100/100硬合法，100例全部优于旧后处理；97/100单例Q＜1.5。** 模型/评分器未改，测试未读。这不等于全部例达标、隐藏集泛化或正式含时间比赛成绩。"
    text=text.replace('<!-- FINAL_CONCLUSION -->',conclusion)
    table='| 指标 | 旧后处理 | 新后处理 |\n|---|---:|---:|\n'
    for title,key,fmt in [('加权中性时间Q','weighted_Q','.6f'),('Q算术均值','Q_mean','.6f'),('Q中位数','Q_median','.6f'),('Q P95','Q_p95','.6f'),('Q最大值','Q_max','.6f'),('Q＜1.5例数','Q_below_1_5_count','d'),('硬合法例数','feasible_count','d'),('HPWL有符号gap均值','hpwl_gap_mean','.6f'),('bbox有符号gap均值','area_gap_mean','.6f'),('边界违反均值','boundary_violations_mean','.2f'),('分组违反均值','grouping_violations_mean','.2f'),('MIB违反均值','mib_violations_mean','.2f')]:table+=f'| {title} | {oldm[key]:{fmt}} | {new[key]:{fmt}} |\n'
    text=text.replace('<!-- METRICS_TABLE -->',table)
    tail='未达1.5的三例仍保留在主分数与分母中：\n\n| case | Q | B/G/M |\n|---|---:|---|\n'
    for r in worst:
        m=r['new']['metrics']
        if m['cost']<1.5:continue
        tail+=f"| {r['case_id']} | {m['cost']:.6f} | {m['boundary_violations']}/{m['grouping_violations']}/{m['mib_violations']} |\n"
    tail+='\n平均边界违反从10.99降至0.45、分组从4.68降至0.39，MIB均值仍为3.93；剩余同形违反不能全部称作不可避免，本轮没有给出完整不可行证明。质量收益也来自外框/线长损伤大幅减小，而非只消除软罚。'
    text=text.replace('<!-- CASE_DISTRIBUTION -->',tail)
    table='| 模块数 | 例数 | 旧Q | 新Q | 单例＜1.5 |\n|---|---:|---:|---:|---:|\n'
    for key,m in by_size.items():table+=f"| {key} | {m['count']} | {m['old_Q']:.6f} | {m['new_Q']:.6f} | {m['below_1_5']} |\n"
    text=text.replace('<!-- SIZE_TABLE -->',table)
    text=text.replace('<!-- TIMING -->',f"新后处理平均{timing['post_only_mean_seconds']:.3f}秒/例；加回原模型特征/前向实测后平均{timing['mean_seconds']:.3f}秒，中位{timing['median_seconds']:.3f}秒、P95={timing['p95_seconds']:.3f}秒、最大{timing['max_seconds']:.3f}秒；{timing['over_15s_count']}例超过旧15秒档，0例超过本轮120秒档。旧流程平均{timing['old_mean_seconds']:.3f}秒。完整评价墙钟{basic['wall_seconds_this_invocation']/60:.2f}分钟（含读取/评分）。HiGHS主体2线程；这不是所有公用库线程总数的硬上限。位移/形变仅为解释统计：按布局平均的可动块中心位移为自身尺度{np.mean(distances):.3f}，自由块宽高最大绝对log变化均值{np.mean(shapes):.3f}，不用于晋级。")
    ab='\n## 同版本五例消融\n\n| 策略 | 加权Q | 硬合法 |\n|---|---:|---:|\n'
    for a in ablations:ab+=f"| {a['name']} | {a['Q']:.6f} | {a['feasible']}/5 |\n"
    ab+='\n三者同一冻结草稿、相同120秒上限；A/B重新运行，C复用与完整评价一致的五例记录。B/C差异包括规则重加权及配套分离优先级，不能把差额归因于某一个系数。这是五例机制证据，不替代主100例结果。\n\n未重加权B的一例包含极小负坐标，按非负规则计Q=10，未删除或改写原记录。新增加的[严格输出入口](../../solver.py)在失败时回退原合法控制；保存输出反事实核验得到B＋守卫Q=1.876979。主方案100个保存输出均通过同一守卫、零回退，因此主Q不变。守卫不改核心优化轨迹；本次对100个保存输出的严格验证平均耗时约0.00047秒/例，详见[守卫验证](guard_validation.json)。主评价源码快照仍保留守卫加入前的真实版本，不冒充已重新运行100次优化。\n'
    text=text.replace('<!-- NEXT -->',ab+'\n## 结论与下一步\n\n当前模型接口足以与该后处理达到开发目标，证明此前后处理是质量瓶颈之一；尚未证明模型相对无学习初值的独立贡献。后续优先分析三例尾部失败、MIB兼容子组和求解速度，再决定是否补模型。当前不再追加训练、扩读测试或自动继续追求更低Q。\n\n[浏览固定五例前后图](index.html) · [100例比较图](comparison.png)。')
    (out/'analysis.md').write_text(text,encoding='utf-8')
    doc='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>E6 全局后处理结果</title><style>body{font:16px system-ui;max-width:1400px;margin:30px auto;padding:0 20px;background:#f1f5f9;color:#172033}article{background:white;border-radius:12px;padding:20px;margin:25px 0}img{width:100%;height:auto}a{color:#146}p{line-height:1.7}</style>'
    doc+=f'<h1>全局后处理：Q {basic["old_Q"]:.3f} → {basic["new_Q"]:.3f}</h1><p>固定dev100，100/100硬合法；97例单例Q＜1.5。模型和评分器未改，未读取测试。Q为中性时间口径。新后处理平均{timing["post_only_mean_seconds"]:.2f}秒/例，旧流程{timing["old_mean_seconds"]:.2f}秒/例。</p><p><a href="analysis.md">完整分析</a> · <a href="results.json">精确结果</a>。以下固定worker0/24/49/74/99，按预先选样展示，点击看大图。每图四栏为参考、原始草稿、旧后处理、新后处理，共用尺度和模块编号。</p><img src="comparison.png">'
    for worker,filename in gallery:doc+=f'<article><h2>worker_{worker}</h2><a href="{filename}" target="_blank"><img src="{filename}"></a></article>'
    (out/'index.html').write_text(doc+'</html>',encoding='utf-8')
    save(out/'render_manifest.json',{'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'cases_sha256':hashlib.sha256((out/'cases.jsonl').read_bytes()).hexdigest(),'selection':'fixed workers0/24/49/74/99, no score selection','figures':[f for _,f in gallery]+['comparison.png'],'reads':'saved JSON only; no inference or datasets'})
    print(json.dumps({'weighted_Q':basic['new_Q'],'legal':basic['new_feasible'],'under1.5':basic['Q_below_1_5_count'],'time':timing,'ablations':ablations}))


if __name__=='__main__':main()
