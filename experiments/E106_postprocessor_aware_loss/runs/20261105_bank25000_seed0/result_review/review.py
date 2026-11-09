"""Read-only frozen-best diagnostics on existing internal panels, no solver or updates."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[5]))
import math
import numpy as np
import torch
from scipy.stats import pearsonr,spearmanr
from experiments.E106_postprocessor_aware_loss.common import OWNER,RUN,read,save,sha,sources,setup
from experiments.E106_postprocessor_aware_loss.data import load_case,batch_cases
from experiments.E106_postprocessor_aware_loss.model import PostprocessorCritic
from experiments.E106_postprocessor_aware_loss.objective import fitting_loss,role_exp_mean


OUT=Path(__file__).resolve().parent


def weights(ns):
    ns=np.asarray(ns,dtype=float)
    v=np.exp((ns-120)/12)
    return .5/len(ns)+.5*v/v.sum()


def corr(x,y):
    x,y=np.asarray(x).flatten(),np.asarray(y).flatten()
    if np.std(x)<1e-14 or np.std(y)<1e-14:
        return dict(pearson=None,spearman=None)
    return dict(pearson=float(pearsonr(x,y).statistic),spearman=float(spearmanr(x,y).statistic))


def main():
    setup()
    result=read(RUN/'critic/results.json')
    assert result['status']=='early_stop'
    assert sha(RUN/'critic/best.pt')==result['checkpoint_sha256']
    config=read(RUN/'critic/config.json')
    assert config['identity']['sources']==sources()
    best=torch.load(RUN/'critic/best.pt',map_location='cpu',weights_only=False)
    latest=torch.load(RUN/'critic/latest.pt',map_location='cpu',weights_only=False)
    assert all(torch.isfinite(v).all() for v in best['model'].values())
    history=latest['progress']['history']
    assert result['best_epoch']==7 and result['epochs']==26 and result['updates']==13000
    assert abs(min(h['validation']['total'] for h in history)-result['best_validation_Loss'])<1e-10
    normalization=read(RUN/'normalization.json')
    panel=read(OWNER/'panel.json')['cases']
    model=PostprocessorCritic(**config['plan']['model']).cuda().eval().requires_grad_(False)
    model.load_state_dict(best['model'],strict=True)
    arrays={}
    reproduced={}
    for role in ('fit','validation','check'):
        specs=[r for r in panel if r['split']==role]
        data={k:[] for k in ('Q','n','radius','index','identical_pair_geometry')}
        if role!='fit':
            data['prediction']=[]
            accum=dict(total=0.,value=0.,directions=0.,auxiliary=0.)
        with torch.inference_mode():
            for start in range(0,len(specs),6):
                subset=specs[start:start+6]
                rows=[load_case(r['index']) for r in subset]
                data['Q'].extend([r['Q'].numpy() for r in rows])
                data['n'].extend([r['n'] for r in rows])
                data['index'].extend([r['index'] for r in rows])
                data['radius'].extend([r['pairs'][0]['radius'] for r in rows])
                data['identical_pair_geometry'].extend([torch.equal(r['finals'][3],r['finals'][4]) for r in rows])
                if role!='fit':
                    batch=batch_cases(rows,'cuda')
                    prediction=model(batch['graph'],batch['states'],group_size=5)
                    data['prediction'].extend(prediction['Q'].view(-1,5).cpu().numpy())
                    loss=fitting_loss(prediction,batch,normalization,role_exp_mean(role))
                    for key in accum:
                        accum[key]+=float(loss[key])*len(rows)
        arrays[role]={k:np.asarray(v) for k,v in data.items()}
        if role!='fit':
            reproduced[role]={key:value/len(specs) for key,value in accum.items()}
        print('Reviewed stored-label panel',role,len(specs),flush=True)
    fit=arrays['fit']
    fit_w=weights(fit['n'])
    constant=float(fit['Q'].mean(1)@fit_w)
    fit_delta=fit['Q'][:,3]-fit['Q'][:,4]
    nonzero=abs(fit_delta)>1e-6
    majority=1 if float(np.sum(fit_w[nonzero]*np.sign(fit_delta[nonzero])))>=0 else -1
    evaluations={}
    labels={}
    for role,data in arrays.items():
        w=weights(data['n'])
        true=data['Q']
        delta=true[:,3]-true[:,4]
        slope=delta/(2*data['radius'])
        labels[role]=dict(cases=len(w),drafts=true.size,Q_mean=float(true.mean()),Q_max=float(true.max()),
            exact_Q1=int(np.sum(true==1)),pair_delta_abs_max=float(np.max(abs(delta))),
            nonzero_pair_count=int(np.sum(abs(delta)>1e-6)),
            identical_pair_final_geometry_count=int(np.sum(data['identical_pair_geometry'])),
            zero_slope_baseline_direction_Loss=float((w*slope**2).sum()/normalization['slope_scale']**2))
        if role=='fit':
            continue
        pred=data['prediction']
        error=pred-true
        mse=float(w@(error**2).mean(1))
        baseline_mse=float(w@((constant-true)**2).mean(1))
        mean=float(w@true.mean(1))
        variance=float(w@((true-mean)**2).mean(1))
        pred_delta=pred[:,3]-pred[:,4]
        pred_slope=pred_delta/(2*data['radius'])
        changed=abs(delta)>1e-6
        resolved=abs(pred_delta)>1e-6
        direction_error=float(w@((pred_slope-slope)**2))/normalization['slope_scale']**2
        baseline_direction=labels[role]['zero_slope_baseline_direction_Loss']
        direction_rms=math.sqrt(float(w@(slope**2)))
        pred_direction_rms=math.sqrt(float(w@(pred_slope**2)))
        target_case_std=np.std(true,axis=1)
        pred_case_std=np.std(pred,axis=1)
        evaluations[role]=dict(composite_Loss=reproduced[role],
            Q_prediction=dict(MAE=float(w@np.abs(error).mean(1)),RMSE=math.sqrt(mse),
                fit_constant_baseline=constant,fit_constant_RMSE=math.sqrt(baseline_mse),
                MSE_improvement_over_fit_constant=1-mse/baseline_mse,
                weighted_R2=1-mse/variance,**corr(true,pred)),
            direction=dict(fitted_direction_Loss=direction_error,zero_slope_baseline_Loss=baseline_direction,
                MSE_improvement_over_zero_slope=1-direction_error/baseline_direction,
                actual_slope_RMS=direction_rms,predicted_slope_RMS=pred_direction_rms,
                predicted_to_actual_slope_RMS=pred_direction_rms/max(direction_rms,1e-12),
                changed_pair_threshold=1e-6,changed_pairs=int(changed.sum()),
                resolved_changed_pairs=int(np.sum(changed&resolved)),
                sign_accuracy_including_unresolved_as_wrong=float(np.mean((np.sign(pred_delta[changed])==np.sign(delta[changed]))&resolved[changed])) if changed.any() else None,
                fit_majority_sign=majority,
                fit_majority_sign_accuracy=float(np.mean(np.sign(delta[changed])==majority)) if changed.any() else None,
                **corr(delta,pred_delta)),
            within_case_variation=dict(actual_Q_std_median=float(np.median(target_case_std)),
                predicted_Q_std_median=float(np.median(pred_case_std))))
    assert abs(reproduced['check']['total']-result['check']['total'])<2e-6
    audit=dict(status='complete',training_result=result,source_identity_unchanged=True,
        best_checkpoint_sha256=sha(RUN/'critic/best.pt'),best_epoch=7,epochs=26,updates=13000,
        fitted_constant_from_fit_only=constant,labels=labels,evaluations=evaluations,
        reproduction_absolute_difference={k:abs(reproduced['check'][k]-result['check'][k]) for k in reproduced['check']},
        formal_model_or_optimizer_updates=0,new_postprocessor_calls=0,dev_test_read=False,
        evidence='Existing train-partition internal validation/check, not unseen to source placers; no checkpoint reselection, no layout continuation')
    save(OUT/'results.json',audit)
    np.savez_compressed(OUT/'predictions.npz',**{role+'_'+k:v for role,data in arrays.items() for k,v in data.items()})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    epochs=[h['epoch'] for h in history]
    axes[0,0].plot(epochs,[h['train']['total'] for h in history],label='Train monitor500')
    axes[0,0].plot(epochs,[h['validation']['total'] for h in history],label='Internal validation1000')
    axes[0,0].axvline(7,color='gray',linestyle='--',label='Frozen best: epoch7')
    axes[0,0].set(xlabel='Epoch',ylabel='Composite fitting Loss',title='Training completed normally at epoch26')
    axes[0,0].legend(fontsize=8)
    for key,mult,label in (('value',1,'Q value'),('directions',1,'Local direction'),('auxiliary',.1,'0.1 x auxiliary')):
        axes[0,1].plot(epochs,[h['validation'][key]*mult for h in history],label=label)
    axes[0,1].set(xlabel='Epoch',ylabel='Contribution to validation Loss',title='Most improvement is auxiliary fitting')
    axes[0,1].legend(fontsize=8)
    data=arrays['check']
    true,pred=data['Q'].flatten(),data['prediction'].flatten()
    low,high=min(1.,float(true.min())),float(max(true.max(),pred.max()))
    axes[1,0].scatter(true,pred,s=5,alpha=.2,rasterized=True)
    axes[1,0].plot([low,high],[low,high],color='gray',linestyle='--',label='Perfect prediction')
    axes[1,0].axhline(constant,color='orange',linestyle=':',label='Fit-only constant baseline')
    axes[1,0].set(xlabel='Actual postprocessed Q',ylabel='Predicted Q',title='Frozen best: internal check5000 drafts')
    axes[1,0].legend(fontsize=8)
    d=data['Q'][:,3]-data['Q'][:,4]
    pd=data['prediction'][:,3]-data['prediction'][:,4]
    extent=float(max(np.max(abs(d)),np.max(abs(pd)),.001))
    axes[1,1].scatter(d,pd,s=8,alpha=.3,rasterized=True)
    axes[1,1].plot([-extent,extent],[-extent,extent],color='gray',linestyle='--')
    axes[1,1].set_xscale('symlog',linthresh=.001)
    axes[1,1].set_yscale('symlog',linthresh=.001)
    axes[1,1].set(xlabel='Actual Q(plus)-Q(minus)',ylabel='Predicted Q(plus)-Q(minus)',
                  title='Local response: full tails, symmetric log axes')
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.savefig(OUT/'results_overview.png',dpi=180)
    fig.savefig(OUT/'results_overview.pdf')
    plt.close(fig)
    print('Frozen-best diagnostics complete; no training or new Q calls.',flush=True)


if __name__=='__main__':
    main()
