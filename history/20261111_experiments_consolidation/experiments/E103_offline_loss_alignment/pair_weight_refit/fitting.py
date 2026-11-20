"""One convex fit, fixed lambda1 and E98 prior; exactly two model sources."""
import numpy as np
from ..analysis import fit
from ..loss_config import COEFFICIENTS
from .common import OWNER,RUN,BANK,KEYS,read,save,case_weights,q_summary

def evaluate(x,q,n,co):
    loss=x@np.array([co[k] for k in KEYS])
    # Equal predicted Loss retains the current version (source1).
    pick=np.where(loss[:,0]<loss[:,1]-1e-12,0,1)
    chosen=q[np.arange(len(q)),pick]
    gap=q[:,1]-q[:,0];difference=loss[:,1]-loss[:,0];active=abs(gap)>1e-4
    correct=np.where(abs(difference)<=1e-12,.5,(gap*difference>0).astype(float))
    w=case_weights(n)
    accuracy=float(correct[active]@w[active]/w[active].sum()) if active.any() else None
    return dict(**q_summary(chosen,n),mixed_regret=float((chosen-q.min(1))@w),
        pair_accuracy=accuracy,informative_pairs=int(active.sum()),Q_ties=int((~active).sum()),
        selected_enhanced=int((pick==0).sum()),selected_current=int((pick==1).sum()))

def main():
    panel=read(OWNER.parent/'panel.json');xs=[];qs=[];ns=[];legal=[0,0]
    for i in range(1000):
        old=read(BANK/f'bank/{i:04d}.json')
        enhanced=next(d for d in old['drafts'] if d['model_id']=='e98_generalization')
        current=read(RUN/f'current/{i:04d}.json')
        assert current['member']==old['member']==panel['members'][i]
        pair=[enhanced,current]
        xs.append([[d['loss_terms'][k] for k in KEYS] for d in pair]);qs.append([d['Q'] for d in pair]);ns.append(old['n'])
        for j,d in enumerate(pair):legal[j]+=int(d['metrics']['is_feasible'])
    x=np.array(xs);q=np.array(qs);n=np.array(ns)
    assert x.shape==(1000,2,10) and np.isfinite(x).all() and np.isfinite(q).all()
    split=panel['analysis_splits'];tr=np.array(split['fit']);tu=np.array(split['tune']);te=np.array(split['check'])
    # No lambda grid and no prior changes: only the source population differs from v1.
    proposal=fit(x[tr],q[tr],n[tr],1.,[0,1])
    proposal.update(revision='E103.enhanced_current_pair.v1',sources=['e98_enhanced_best71','learned_v1_best29'],
        fit_cases=600,regularization_fixed=True,hyperparameter_trials=1,
        prior='Original E98 coefficients, same prior and algorithm as first fit; source restriction is the change',
        selection='One fixed-lambda fit on train600; frozen before tune200/check200 reports; no coefficient selection using either panel',
        feature_semantics='Same ten E98 unweighted terms, evaluated on original raw network states for both sources',test_read=False)
    save(RUN/'proposal.json',proposal)
    current_co=read(OWNER.parent/'runs/20261104_bank1000x5_seed0/analysis/proposal.json')['physical_coefficients']
    reports={}
    for name,indices in [('fit600',tr),('tune200',tu),('previously_exposed_check200',te)]:
        reports[name]={label:evaluate(x[indices],q[indices],n[indices],co) for label,co in (
            ('E98_original',COEFFICIENTS),('currently_used_v1',current_co),('two_source_refit',proposal['physical_coefficients']))}
    total=sum(COEFFICIENTS.values())
    comparison={k:dict(E98=COEFFICIENTS[k],current_v1=current_co[k],refitted=proposal['physical_coefficients'][k],
        normalized_E98=COEFFICIENTS[k]/total,normalized_current=current_co[k]/total,
        normalized_refitted=proposal['physical_coefficients'][k]/total) for k in KEYS}
    result=dict(status='complete',cases=1000,models=2,paired_drafts=2000,
        enhanced_results_reused=1000,new_current_results=1000,
        source_summary={label:dict(**q_summary(q[:,j],n),hard_legal=legal[j]) for j,label in enumerate(('enhanced_best71','current_best29'))},
        proposal=proposal,reports=reports,coefficient_comparison=comparison,neural_training_started=False,test_read=False,
        evidence_boundary='Same original1000 training members; check200 was already exposed in earlier research and is descriptive, not fresh independent validation. Only two related model versions give at most one informative pair per problem; a fitted ranking Loss is not proof of better training.')
    save(RUN/'results.json',result)
    lines=['# E103：仅增强E98与当前best29的一次调权','',
        '同1000训练题，仅两个已完成模型。增强E98的1000结果只读复用；仅补齐当前best29同题结果。四进程CPU1，与原数据库相同数值路径和E98固定深度后处理。',
        '','只在原fit600上进行一次lambda=1的非负排序拟合，使用与第一轮相同的E98先验，不扫描超参。先冻结系数，随后才报告原200调参题与已曝光200检查题；不读取dev/test或自动训练。',
        '', '|Loss项|E98原系数|当前v1|本次两源拟合|','|---|---:|---:|---:|']
    for k,v in comparison.items():lines.append(f"|{k}|{v['E98']:.9g}|{v['current_v1']:.9g}|{v['refitted']:.9g}|")
    for panel_name,values in reports.items():
        lines+=['',f'## {panel_name}','', '|Loss|配对排序准确率|选草稿混合Q|混合遗憾|有效配对|','|---|---:|---:|---:|---:|']
        for label,v in values.items():
            accuracy='N/A' if v['pair_accuracy'] is None else f"{v['pair_accuracy']:.2%}"
            lines.append(f"|{label}|{accuracy}|{v['mixed_Q']:.9f}|{v['mixed_regret']:.9f}|{v['informative_pairs']}|")
    lines+=['',result['evidence_boundary'],'', '不能把本次两模型的准确率与此前五模型准确率直接当作同难度指标比较。',
        '新系数仅保存为候选，不根据这些报告自动开始训练或扩方向。','']
    (RUN/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    return result
