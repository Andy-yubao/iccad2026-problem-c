"""Fixed grouped-CV linear diagnostic; never update the frozen layout model."""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_key]='1'
import hashlib
import json
from pathlib import Path
import numpy as np
from .topology import HERE,read,center_setup,center_metrics

METRICS=('local_axis_accuracy','knn6_recall','global_axis_accuracy')

def nullable(array):
    if isinstance(array,np.ndarray):return [nullable(x) for x in array]
    return float(array) if np.isfinite(array) else None

def main():
    manifest=read(HERE/'manifest.json')
    areas=np.load(HERE/'areas.npy');reference=np.load(HERE/'reference.npy');flags=np.load(HERE/'flags.npy')
    positions=np.load(HERE/'positions.npy',mmap_mode='r')
    active=[np.flatnonzero((areas[i]>0)&(flags[i,:,1]==0)) for i in range(500)]
    eligible=[i for i in range(500) if len(active[i])>=2]
    groups={}
    for i in eligible:groups.setdefault(manifest['members'][i][0],[]).append(i)
    names=list(groups);np.random.default_rng(20261114).shuffle(names)
    names.sort(key=lambda key:-len(groups[key]))
    folds=[[] for _ in range(5)]
    for name in names:folds[min(range(5),key=lambda j:len(folds[j]))].extend(groups[name])
    targets={};setups={};scales={};weights={};permutations={}
    for i in eligible:
        ids=active[i];p=reference[i,ids];center=p[:,:2]+p[:,2:]/2
        scales[i]=np.sqrt(areas[i].sum());target=center/scales[i];targets[i]=target-target.mean(0)
        setups[i]=center_setup(center,areas[i,ids])
        weights[i]=np.full(len(ids),1/len(ids))
        permutations[i]=np.random.default_rng(20261114+i).permutation(len(ids))
    stages=['features',*manifest['hidden_names']]
    output={};fold_notes={}
    actual=np.full((500,13,len(METRICS)),np.nan)
    for i in eligible:
        ids=active[i]
        for layer in range(13):
            p=positions[i,layer,ids];metrics,_,_=center_metrics(p[:,:2]+p[:,2:]/2,setups[i])
            actual[i,layer]=[metrics[name] for name in METRICS]
    for stage in stages:
        array=np.load(HERE/('features.npy' if stage=='features' else f'hidden/{stage}.npy'),mmap_mode='r')
        xs={}
        for i in eligible:
            x=np.asarray(array[i,active[i]],float);xs[i]=x-x.mean(0)
        predictions=np.full((500,len(METRICS)),np.nan);control=np.full_like(predictions,np.nan)
        notes=[]
        shuffled=stage in ('features','S0','S3','S6','S9','S12')
        for fold,test in enumerate(folds):
            test=set(test);train=[i for i in eligible if i not in test]
            if {manifest['members'][i][0] for i in train}&{manifest['members'][i][0] for i in test}:raise ValueError('Source-shard CV leakage')
            x=np.concatenate([xs[i] for i in train]);y=np.concatenate([targets[i] for i in train])
            w=np.concatenate([weights[i] for i in train])/len(train)
            sigma=np.sqrt((w[:,None]*x.square()).sum(0)) if hasattr(x,'square') else np.sqrt((w[:,None]*(x*x)).sum(0))
            sigma=np.where(sigma>1e-8,sigma,1.);z=x/sigma
            gram=z.T@(w[:,None]*z);regularizer=.01*np.trace(gram)/gram.shape[0]
            matrix=gram+max(regularizer,1e-12)*np.eye(gram.shape[0])
            fitted=np.linalg.solve(matrix,z.T@(w[:,None]*y))
            null_fit=None
            if shuffled:
                null_y=np.concatenate([targets[i][permutations[i]] for i in train])
                null_fit=np.linalg.solve(matrix,z.T@(w[:,None]*null_y))
            for i in test:
                p=(xs[i]/sigma)@fitted*scales[i]
                metrics,_,_=center_metrics(p,setups[i]);predictions[i]=[metrics[name] for name in METRICS]
                if null_fit is not None:
                    p=(xs[i]/sigma)@null_fit*scales[i]
                    metrics,_,_=center_metrics(p,setups[i]);control[i]=[metrics[name] for name in METRICS]
            notes.append(dict(fold=fold,train_cases=len(train),test_cases=len(test),lambda_value=regularizer,feature_dimension=x.shape[1]))
        output[stage]=dict(metrics=nullable(predictions),permuted_label_control=nullable(control) if shuffled else None)
        fold_notes[stage]=notes
        print('Grouped-CV probe',stage,'axis/knn',np.nanmean(predictions,axis=0)[:2],flush=True)
    result=dict(metric_names=METRICS,stages=output,actual_readout_metrics=nullable(actual),folds=folds,fold_notes=fold_notes,
        eligible_cases=len(eligible),node_subset='non-preplaced only',target='per-case relative reference centers / sqrt(total public area)',
        model_training=False,diagnostic_closed_form_fit=True,hyperparameter_selection=False,Q_selection=False,
        frozen_model_modified=False,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (HERE/'probe.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

if __name__=='__main__':main()
