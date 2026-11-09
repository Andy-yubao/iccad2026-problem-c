"""Pooled cross-problem correlations and one nonnegative global linear fit."""
import csv
import numpy as np
from scipy.optimize import nnls
from scipy.stats import pearsonr,spearmanr
from .common import ROOT,OWNER,RUN,KEYS,read,save,sha
from ..loss_config import COEFFICIENTS

def corr(a,b):
    if np.ptp(a)<1e-14 or np.ptp(b)<1e-14:return dict(pearson=None,spearman=None)
    return dict(pearson=float(pearsonr(a,b).statistic),spearman=float(spearmanr(a,b).statistic))

def nonnegative_fit(x,q):
    mean=x.mean(0);scale=x.std(0);scale=np.where(scale>1e-12,scale,1.)
    target_mean=q.mean();target_scale=q.std();assert target_scale>1e-12
    z=(x-mean)/scale;y=(q-target_mean)/target_scale
    beta,residual=nnls(z,y,maxiter=3000)
    physical=beta/scale;assert physical.sum()>1e-12,'No positive-correlation nonnegative combination found'
    multiplier=sum(COEFFICIENTS.values())/physical.sum()
    co=physical*multiplier
    return dict(coefficients=co.tolist(),standardized_beta=beta.tolist(),feature_mean=mean.tolist(),
        feature_std=scale.tolist(),Q_mean=float(target_mean),Q_std=float(target_scale),
        prediction_slope=float(target_scale/multiplier),
        prediction_intercept=float(target_mean-target_scale*np.dot(mean,physical)),
        normalized_residual_norm=float(residual))

def grouped_indices(groups,rng):
    unique=np.unique(groups);sample=rng.choice(unique,len(unique),replace=True)
    return np.concatenate([np.flatnonzero(groups==g) for g in sample])

def main():
    panel=read(OWNER/'panel.json');plan=read(OWNER/'plan.json')
    observations=[read(RUN/f'observations/{i:04d}.json') for i in range(2000)]
    assert len({tuple(r['member']) for r in observations})==2000
    assert all(r['member']==panel['samples'][i]['member'] for i,r in enumerate(observations))
    x=np.array([[r['loss_terms'][k] for k in KEYS] for r in observations]);q=np.array([r['Q'] for r in observations])
    n=np.array([r['n'] for r in observations]);models=np.array([r['model_id'] for r in observations])
    groups=np.array([r['member'][0].split('/')[0] for r in observations])
    split=np.array([r['split'] for r in observations]);fit=split=='fit';hold=split=='check'
    assert fit.sum()==1400 and hold.sum()==600 and np.isfinite(x).all() and np.isfinite(q).all()
    regression=nonnegative_fit(x[fit],q[fit]);new_co=np.array(regression['coefficients'])
    proposal=dict(method='Global centered nonnegative least squares on1400 independent cross-problem observations; free regression intercept, no pair construction, no within-problem centering, no ranking objective, no hyperparameter scan',
        coefficients=dict(zip(KEYS,new_co.tolist())),regression=regression,
        coefficient_total=float(new_co.sum()),fit_samples=1400,check_samples=600,
        source_model_ids=['enhanced_best71','current_best29'],test_read=False,
        note='Free intercept calibrates Q prediction only; exported training Loss is sum of nonnegative coefficient*raw term. Global positive rescaling does not affect correlation.')
    save(RUN/'proposal.json',proposal)  # Freeze before inspecting the holdout correlations.
    current_co=plan['current_training_coefficients']
    coefficient_sets={'E98_original':COEFFICIENTS,'current_v1':current_co,'global_fitted':proposal['coefficients']}
    scores={name:x@np.array([co[k] for k in KEYS]) for name,co in coefficient_sets.items()}
    comparisons={}
    for label,loss in scores.items():
        comparisons[label]=dict(fit=corr(loss[fit],q[fit]),check=corr(loss[hold],q[hold]),all_descriptive=corr(loss,q),
            model_check={mid:corr(loss[hold&(models==mid)],q[hold&(models==mid)]) for mid in np.unique(models)},
            plot_mean=float(loss[fit].mean()),plot_std=float(loss[fit].std()))
    # A fixed secondary diagnostic controls linear module-count effects only.
    control=np.c_[np.ones(len(q)),n]
    rq=q-control@np.linalg.lstsq(control,q,rcond=None)[0]
    term_stats=[];fit_mean=x[fit].mean(0);fit_std=np.maximum(x[fit].std(0),1e-12)
    z=(x[fit]-fit_mean)/fit_std;y=(q[fit]-q[fit].mean())/q[fit].std()
    signed=np.linalg.lstsq(z,y,rcond=None)[0]
    for j,k in enumerate(KEYS):
        rx=x[:,j]-control@np.linalg.lstsq(control,x[:,j],rcond=None)[0]
        others=np.c_[np.ones(fit.sum()),np.delete(z,j,axis=1)]
        partial_x=z[:,j]-others@np.linalg.lstsq(others,z[:,j],rcond=None)[0]
        partial_y=y-others@np.linalg.lstsq(others,y,rcond=None)[0]
        term_stats.append(dict(term=k,all_descriptive=corr(x[:,j],q),fit=corr(x[fit,j],q[fit]),check=corr(x[hold,j],q[hold]),
            model_all={mid:corr(x[models==mid,j],q[models==mid]) for mid in np.unique(models)},
            partial_linear_n_descriptive=corr(rx,rq)['pearson'],
            partial_other9_fit=corr(partial_x,partial_y)['pearson'],
            signed_standardized_beta_diagnostic=float(signed[j]),nonnegative_standardized_beta=float(regression['standardized_beta'][j]),
            weight=float(new_co[j])))
    # Refit bootstrap samples for stability only; never select or average a winner.
    rng=np.random.default_rng(2026110464);bootstrap_weights=[];check_deltas=[]
    for _ in range(128):
        sample=grouped_indices(groups[fit],rng)
        boot=nonnegative_fit(x[fit][sample],q[fit][sample]);bootstrap_weights.append(boot['coefficients'])
        sample_check=grouped_indices(groups[hold],rng)
        check_deltas.append([corr(scores['global_fitted'][hold][sample_check],q[hold][sample_check])['pearson']-
            corr(scores[label][hold][sample_check],q[hold][sample_check])['pearson'] for label in ('E98_original','current_v1')])
    bootstrap_weights=np.array(bootstrap_weights);check_deltas=np.array(check_deltas)
    stability={k:dict(p025=float(np.quantile(bootstrap_weights[:,j],.025)),median=float(np.median(bootstrap_weights[:,j])),
        p975=float(np.quantile(bootstrap_weights[:,j],.975)),nonzero_fraction=float(np.mean(bootstrap_weights[:,j]>1e-8))) for j,k in enumerate(KEYS)}
    for j,label in enumerate(('E98_original','current_v1')):
        comparisons['global_fitted'][f'check_Pearson_gain_vs_{label}_cluster_bootstrap95']=np.quantile(check_deltas[:,j],[.025,.5,.975]).tolist()
    summary=dict(status='complete',observations=2000,distinct_problems=2000,fit_samples=1400,check_samples=600,
        hard_legal=sum(r['metrics']['is_feasible'] for r in observations),
        Q_distribution=dict(min=float(q.min()),median=float(np.median(q)),p95=float(np.quantile(q,.95)),max=float(q.max()),equal1=int((q==1).sum())),
        comparisons=comparisons,term_statistics=term_stats,coefficients=proposal['coefficients'],bootstrap_weight_stability=stability,
        feature_correlation=np.corrcoef(x[fit],rowvar=False).tolist(),source_model_counts={mid:int((models==mid).sum()) for mid in np.unique(models)},
        old_observations_used=0,within_problem_ranking_used=False,neural_training_started=False,test_read=False,
        evidence_boundary='New independent problems exclude previous1000; all are source-network training members, not unseen to networks. Holdout600 are unseen only to this statistical fit. Pooled association intentionally retains between-problem differences; n-control and within-model correlations are diagnostics, not causal proof. Highly correlated features can exchange coefficients.')
    save(RUN/'results.json',summary)
    np.savez_compressed(RUN/'analysis_arrays.npz',x=x,q=q,n=n,models=models,split=split,
        old_loss=scores['E98_original'],current_loss=scores['current_v1'],new_loss=scores['global_fitted'])
    with (RUN/'observations.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f);writer.writerow(['index','shard','row','model','split','n','Q',*KEYS])
        for i,r in enumerate(observations):writer.writerow([i,*r['member'],r['model_id'],r['split'],r['n'],r['Q'],*x[i]])
    from .plots import draw
    draw(summary)
    lines=['# E103：跨题目全局Loss–Q相关性','',
        '全新2000道不同训练题，每题一份草稿；两个模型各1000。排除旧1000题，未复用旧草稿/Q或旧排序结果。没有同题差分或排序目标。',
        '1400样本拟合一个非负全局线性组合，600样本仅检查；按样本等权。所有指标使用原始Q，包括Q=1平台与全部高尾。',
        '', '|Loss|拟合Pearson|检查Pearson|检查Spearman|','|---|---:|---:|---:|']
    for label,s in comparisons.items():lines.append(f"|{label}|{s['fit']['pearson']:.4f}|{s['check']['pearson']:.4f}|{s['check']['spearman']:.4f}|")
    lines+=['','## 全局拟合的实际系数','', '|项|E98原系数|当前v1|全局拟合|','|---|---:|---:|---:|']
    for k,w in proposal['coefficients'].items():lines.append(f"|{k}|{COEFFICIENTS[k]:.8g}|{current_co[k]:.8g}|{w:.8g}|")
    lines+=['','## 单项全局关联','', '|项|全2000 Pearson|全2000 Spearman|控制线性N后Pearson|联合非负标准系数|','|---|---:|---:|---:|---:|']
    for t in sorted(term_stats,key=lambda t:abs(t['all_descriptive']['pearson']),reverse=True):
        lines.append(f"|{t['term']}|{t['all_descriptive']['pearson']:.4f}|{t['all_descriptive']['spearman']:.4f}|{t['partial_linear_n_descriptive']:.4f}|{t['nonnegative_standardized_beta']:.4f}|")
    lines+=['',summary['evidence_boundary'],'', '系数为0不等于该物理指标不重要，可能因共线性、取值范围或当前模型分布而被替代。新系数尚未用于训练。',
        '', '[检查集Loss–Q对比图](loss_q_comparison.png)','[单项相关性和联合权重图](term_correlations.png)','']
    (RUN/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    return summary
