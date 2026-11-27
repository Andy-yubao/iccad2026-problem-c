"""Freeze an internal-train confidence gate before outer dev is evaluated."""
from .resources import configure
configure()
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from .inference import load_gate
from .prepare import load_matrix,FOLDER,ACTION_COUNT
from .inspect_labels import metrics
from .io import HERE,ROOT,save,sha,sources

THRESHOLDS=(.5,.6,.7,.8,.9,.95)

def select(probability,threshold):
    action=probability.argmax(1);return np.where(probability.max(1)>=threshold,action,0)

def acceptable(candidate,baseline):
    return candidate['weighted_Q']<baseline['weighted_Q']-1e-4 and candidate['seconds_mean']<baseline['seconds_mean'] and candidate['cvar95_Q']<=baseline['cvar95_Q']+1e-8 and candidate['max_Q']<=baseline['max_Q']+.05 and candidate['seconds_p95']<=baseline['seconds_p95']*1.05 and candidate['seconds_max']<=baseline['seconds_max']*1.1

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--checkpoint',type=Path,required=True);args=ap.parse_args()
    folder=args.checkpoint.parents[1];path=folder/'calibration.json'
    if path.exists():raise ValueError('Completed calibration immutable')
    model,state=load_gate(args.checkpoint);rows=load_matrix();ids=state['identity']['inner_split']['calibration']
    epochs=state['progress']['best_epoch'];epochs=epochs if isinstance(epochs,list) else [epochs]
    learned_best=all(epoch is not None and epoch>0 for epoch in epochs)
    if not state.get('assembly_only',False):
        training=json.loads((folder/'results.json').read_text())
        if training['status'] not in ('early_stop','max_epochs','budget') or training['best_epoch']!=epochs[0]:raise ValueError('Gate training is not finished at the registered best')
    x=torch.tensor(np.stack([rows[i]['features'] for i in ids]));q=np.stack([rows[i]['q'] for i in ids]);t=np.stack([rows[i]['seconds'] for i in ids]);standalone=np.stack([rows[i]['standalone_seconds'] for i in ids]);n=np.asarray([rows[i]['n'] for i in ids]);k=np.arange(len(ids))
    with torch.inference_mode():
        model(x[:1]);timings=[]
        for i in range(min(100,len(x))):
            started=time.perf_counter();model(x[i:i+1]);timings.append(time.perf_counter()-started)
        gate_seconds=float(np.mean(timings));probability=torch.cat([model(x[start:start+32])[0].softmax(-1) for start in range(0,len(x),32)]).numpy()
    baseline=metrics(q[:,0],standalone[:,0],n);trials=[]
    for threshold in THRESHOLDS:
        action=select(probability,threshold);candidate=metrics(q[k,action],t[k,action]+gate_seconds,n)
        trials.append({'threshold':threshold,'summary':candidate,'actions':np.bincount(action,minlength=ACTION_COUNT).tolist(),'acceptable':acceptable(candidate,baseline),'max_Q_regression':float((q[k,action]-q[:,0]).max())})
    eligible=[r for r in trials if learned_best and r['acceptable'] and r['max_Q_regression']<=.2]
    chosen=min(eligible,key=lambda r:(r['summary']['weighted_Q'],r['summary']['seconds_mean'])) if eligible else None
    bins=[(21,60),(60,90),(90,110),(110,121)];simple=[]
    for lo,hi in bins:
        use=(n>=lo)&(n<hi);base=metrics(q[use,0],standalone[use,0],n[use]);candidates=[]
        for action in range(ACTION_COUNT):
            value=metrics(q[use,action],standalone[use,action],n[use])
            if value['weighted_Q']<=base['weighted_Q']+1e-4 and value['cvar95_Q']<=base['cvar95_Q']+1e-8:candidates.append((value['seconds_mean'],value['weighted_Q'],action))
        action=min(candidates)[2] if candidates else 0;simple.append({'lo':lo,'hi':hi,'action':action})
    simple_actions=np.zeros(len(n),dtype=int)
    for rule in simple:simple_actions[(n>=rule['lo'])&(n<rule['hi'])]=rule['action']
    simple_summary=metrics(q[k,simple_actions],standalone[k,simple_actions],n);simple_regression=float((q[k,simple_actions]-q[:,0]).max())
    simple_safe=acceptable(simple_summary,baseline) and simple_regression<=.2
    deployment='gate' if chosen else ('n_rule' if simple_safe else None)
    if chosen and simple_safe and (simple_summary['weighted_Q'],simple_summary['seconds_mean'])<(chosen['summary']['weighted_Q'],chosen['summary']['seconds_mean']):deployment='n_rule'
    report={'status':'frozen_candidate' if deployment else 'stopped_no_safe_gate','deployment_method':deployment,'learned_best_has_gradient_updates':learned_best,'checkpoint_sha256':sha(args.checkpoint),'baseline':baseline,'trials':trials,'chosen':chosen,'simple_n_rule':simple,
        'simple_n_rule_summary':simple_summary,'simple_n_rule_max_regression':simple_regression,'simple_n_rule_acceptable':simple_safe,
        'single_case_gate_seconds_mean':gate_seconds,'calibration_members':[rows[i]['member'] for i in ids],'calibration_scope':'held-out original train shards, disjoint from fit and monitor shards','outer_dev_used':False,'test_read':False}
    save(path,report)
    if deployment:
        label_identity=json.loads((FOLDER/'identity.json').read_text())
        save(folder/'frozen.json',{'checkpoint':str(args.checkpoint.resolve().relative_to(ROOT)),'checkpoint_sha256':sha(args.checkpoint),'calibration_sha256':sha(path),
            'threshold':chosen['threshold'] if chosen else 1.01,'simple_n_rule':simple,'deployment_method':deployment,'sources':sources(),'expert_weights':label_identity['weights'],
            'confirmation_reservation_sha256':sha(ROOT/'experiments/E32_learned_relation_feedback/runs/research_contract.json'),'outer_dev_used':False,'test_read':False})
    print(json.dumps({key:report[key] for key in ('status','deployment_method','baseline','chosen','simple_n_rule_summary','single_case_gate_seconds_mean')},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
