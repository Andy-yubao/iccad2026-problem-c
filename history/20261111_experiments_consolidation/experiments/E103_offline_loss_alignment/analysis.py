"""Case-separated convex ranking fit; final check never chooses coefficients."""
import itertools
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import spearmanr
from .common import OWNER,RUN,KEYS,read,save,case_weights,q_summary,setup
from .loss_config import COEFFICIENTS

def arrays():
    panel=read(OWNER/'panel.json')
    rows=[read(RUN/f'bank/{i:04d}.json') for i in range(1000)]
    ids=[r['id'] for r in read(OWNER/'model_manifest.json')['models']]
    assert all([d['model_id'] for d in r['drafts']]==ids for r in rows)
    x=np.array([[[d['loss_terms'][k] for k in KEYS] for d in r['drafts']] for r in rows])
    q=np.array([[d['Q'] for d in r['drafts']] for r in rows])
    n=np.array([r['n'] for r in rows])
    assert x.shape==(1000,5,10) and np.isfinite(x).all() and np.isfinite(q).all()
    return panel,rows,ids,x,q,n

def pairs(x,q,n,source_ids):
    differences=[];weights=[]
    cw=case_weights(n)
    for i in range(len(x)):
        local=[]
        for a,b in itertools.combinations(source_ids,2):
            if abs(q[i,a]-q[i,b])<=1e-4:continue
            good,bad=(a,b) if q[i,a]<q[i,b] else (b,a)
            local.append(x[i,bad]-x[i,good])
        if local:
            differences.extend(local);weights.extend([cw[i]/len(local)]*len(local))
    d=np.asarray(differences);a=np.asarray(weights)
    assert len(d)>0
    return d,a/a.sum()

def fit(x,q,n,regularization,source_ids):
    d,a=pairs(x,q,n,source_ids)
    # Fit-only robust within-problem differences determine all ten fixed scales.
    scale=np.maximum(np.quantile(np.abs(d),.75,axis=0),1e-6)
    prior=np.array([COEFFICIENTS[k] for k in KEYS])*scale
    prior/=prior.sum()
    z=d/scale
    def objective(w):
        margin=z@w
        value=np.sum(a*np.logaddexp(0.,-margin))+regularization*np.sum((w-prior)**2)
        grad=-(z.T@(a*expit(-margin)))+2*regularization*(w-prior)
        return value,grad
    result=minimize(objective,prior,jac=True,method='SLSQP',bounds=[(0.,1.)]*10,
        constraints=[dict(type='eq',fun=lambda w:w.sum()-1.,jac=lambda w:np.ones_like(w))],
        options=dict(ftol=1e-10,maxiter=1000))
    if not result.success:raise RuntimeError('Convex ranking fit failed: '+result.message)
    assert np.isfinite(result.x).all() and abs(result.x.sum()-1)<1e-7
    physical=np.maximum(result.x,0)/scale
    physical*=sum(COEFFICIENTS.values())/physical.sum()
    return dict(regularization=regularization,scale=scale.tolist(),simplex_weights=result.x.tolist(),
        physical_coefficients=dict(zip(KEYS,physical.tolist())),pair_count=len(d),
        objective=float(result.fun),optimizer='SLSQP convex logistic ranking; nonnegative simplex; fixed robust scales',
        optimizer_success=True,iterations=int(result.nit))

def evaluate(x,q,n,co,source_pair_filter=None):
    values=x@np.array([co[k] for k in KEYS])
    # Prespecified tie order: prefer current generalization source, then source order.
    best=np.min(values,axis=1);selected=np.array([
        4 if abs(v[4]-m)<=1e-12 else int(np.flatnonzero(np.abs(v-m)<=1e-12)[0])
        for v,m in zip(values,best)])
    chosen=q[np.arange(len(q)),selected];oracle=q.min(axis=1)
    accuracy=[];active_weights=[];pair_count=0
    for i in range(len(q)):
        scores=[]
        for a,b in itertools.combinations(range(5),2):
            if source_pair_filter is not None and source_pair_filter not in (a,b):continue
            dq=q[i,a]-q[i,b]
            if abs(dq)<=1e-4:continue
            dl=values[i,a]-values[i,b]
            scores.append(.5 if abs(dl)<=1e-12 else float(dl*dq>0));pair_count+=1
        if scores:accuracy.append(float(np.mean(scores)));active_weights.append(i)
    w=case_weights(n)
    aw=w[active_weights];aw=aw/aw.sum()
    return dict(**q_summary(chosen,n),mixed_regret=float((chosen-oracle)@w),
        pair_accuracy=float(np.array(accuracy)@aw),pair_count=pair_count,
        selected_model_counts=np.bincount(selected,minlength=5).tolist(),
        oracle=q_summary(oracle,n),source_pair_filter=source_pair_filter)

def fit_job(args):
    return fit(*args)

def analyze():
    panel,rows,ids,x,q,n=arrays()
    split=panel['analysis_splits'];tr=np.array(split['fit']);tu=np.array(split['tune']);te=np.array(split['check'])
    candidates=[]
    jobs=[(x[tr],q[tr],n[tr],lam,list(range(5))) for lam in (1e-4,.01,1.,10.)]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=setup) as pool:
        fitted=list(pool.map(fit_job,jobs))
    for result in fitted:
        result['tune']=evaluate(x[tu],q[tu],n[tu],result['physical_coefficients'])
        candidates.append(result)
    # Only tune cases choose lambda; final check is read once after this freeze.
    choice=min(candidates,key=lambda c:(c['tune']['mixed_regret'],-c['tune']['pair_accuracy'],-c['regularization']))
    proposal={k:v for k,v in choice.items() if k!='tune'}
    proposal.update(selection='minimum mixed regret on internal tune200 only; regularization tie favors stronger penalty',
        fit_cases=600,tune_cases=200,check_cases=200,loss_keys=list(KEYS),
        feature_semantics='Ten common unweighted terms evaluated at original raw network state; decoded geometry determines postprocessed Q. Clamping may make different states share geometry; direction check tests its effect.',
        fixed_total_physical_coefficient_sum=sum(COEFFICIENTS.values()),test_read=False)
    save(RUN/'analysis/proposal.json',proposal)
    baselines={name:evaluate(x[te],q[te],n[te],co) for name,co in (
        ('original',COEFFICIENTS),('learned',proposal['physical_coefficients']))}
    leave_source=[]
    jobs=[(x[tr],q[tr],n[tr],proposal['regularization'],[j for j in range(5) if j!=held]) for held in range(5)]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=setup) as pool:
        fitted=list(pool.map(fit_job,jobs))
    for held,trained in enumerate(fitted):
        leave_source.append(dict(held_source=ids[held],
            check=evaluate(x[te],q[te],n[te],trained['physical_coefficients'],held),
            coefficients=trained['physical_coefficients']))
    # Within-problem ranks remove between-problem difficulty from diagnostics.
    correlations=[]
    for j,k in enumerate(KEYS):
        per=[]
        for i in tr:
            if np.ptp(x[i,:,j])>1e-12 and np.ptp(q[i])>1e-12:
                r=float(spearmanr(x[i,:,j],q[i]).statistic)
                if np.isfinite(r):per.append(r)
        correlations.append(dict(term=k,fit_mean_within_case_spearman=float(np.mean(per)) if per else None,cases=len(per)))
    counts=np.array([[d['metrics']['is_feasible'] for d in r['drafts']] for r in rows])
    full={mid:dict(**q_summary(q[:,j],n),hard_legal=int(counts[:,j].sum())) for j,mid in enumerate(ids)}
    result=dict(status='complete',source_summary=full,candidates=candidates,proposal=proposal,
        final_internal_check=baselines,leave_one_source_out=leave_source,correlations=correlations,
        boundary='All1000 cases are original training members and in source-model training histories. Internal check is unseen only to coefficient fitting, not unseen to source networks. No dev/test Q used.',test_read=False)
    save(RUN/'analysis/results.json',result)
    return result

def direction_summary():
    indices=read(OWNER/'panel.json')['direction_indices']
    records=[read(RUN/f'direction/{i:04d}.json') for i in indices]
    n=[r['n'] for r in records]
    summary={}
    for name in ('baseline','learned'):
        for step in (.002,.01):
            chosen=[next(v for v in r['results'] if v['objective']==name and v['step']==step) for r in records]
            delta=np.array([v['Q_delta'] for v in chosen]);w=case_weights(n)
            summary[f'{name}_{step}']=dict(mixed_Q_delta=float(delta@w),
                improved=int((delta< -1e-4).sum()),worsened=int((delta>1e-4).sum()),
                unchanged=int((np.abs(delta)<=1e-4).sum()),hard_legal=sum(v['metrics']['is_feasible'] for v in chosen))
    improves_both=all(summary[f'learned_{s}']['mixed_Q_delta']< -1e-4 and
        summary[f'learned_{s}']['mixed_Q_delta']<summary[f'baseline_{s}']['mixed_Q_delta'] for s in (.002,.01))
    result=dict(status='complete',cases=40,source='e98_generalization',source_split='internal tune only',
        evaluations=160,summary=summary,descent_gate_passed=improves_both,
        training_recommendation='prepare fixed-objective matched-start training' if improves_both else 'do not launch long training; diagnose missing features or misleading descent directions',
        interpretation='Direct layout-space perturbations preserve fixed specs and area; they do not prove a realizable neural parameter update or causal training improvement.',test_read=False)
    save(RUN/'direction/results.json',result)
    return result
