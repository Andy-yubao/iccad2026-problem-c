"""Read saved paired evidence only; never train, infer, score or select a checkpoint.

Run from the workspace with its independent conda Python after both small arms
finish. Writes a provisional numerical recommendation; visual review and the
root state transition remain the researcher's responsibility.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
import json
import hashlib
from datetime import datetime,timezone
import numpy as np

HERE=ROOT/'experiments/E59_centerline_overlap'
CONTROL=Path(__file__).resolve().parent
NAMES={'original':'20261027_small4096_original_seed0','centerline_v1':'20261027_small4096_centerline_seed0'}
KEYS=('overlap_area_fraction','overlap_pairs','penetration_squared_per_block',
      'loss_overlap_original','loss_overlap_centerline','loss_center_local',
      'loss_shape','loss_relative_local','loss_contact','loss_boundary','loss_mib','loss_bbox')


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def save(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def stats(values):
    v=np.asarray(values,dtype=float)
    if not len(v) or not np.isfinite(v).all():raise ValueError('Empty or nonfinite paired metric')
    return {'mean':float(v.mean()),'p95':float(np.percentile(v,95)),
            'worst5pct_mean':float(np.sort(v)[-max(1,int(np.ceil(.05*len(v)))):].mean()),'max':float(v.max())}


def relative(a,b):return None if abs(a)<1e-10 else float((b-a)/abs(a))


def no_worse(a,b,limit):
    # Explicit absolute safeguard when a denominator is effectively zero.
    return bool(b<=a*(1+limit)) if abs(a)>=1e-10 else bool(b<=a+1e-8)


def bootstrap(delta,members):
    rng=np.random.default_rng(20261027);case_means=[];cluster_means=[]
    groups={}
    for i,m in enumerate(members):groups.setdefault(m[0],[]).append(i)
    totals=np.array([delta[ids].sum() for ids in groups.values()]);counts=np.array([len(ids) for ids in groups.values()])
    for _ in range(20):
        ix=rng.integers(0,len(delta),(100,len(delta)));case_means.extend(delta[ix].mean(1).tolist())
        ix=rng.integers(0,len(totals),(100,len(totals)))
        cluster_means.extend((totals[ix].sum(1)/counts[ix].sum(1)).tolist())
    return {'resamples':2000,'seed':20261027,'case_paired_95ci':np.percentile(case_means,[2.5,97.5]).tolist(),
            'source_shard_cluster_paired_95ci':np.percentile(cluster_means,[2.5,97.5]).tolist(),
            'source_shards':len(groups),'primary_gate':'source-shard cluster interval upper bound <0',
            'limit':'Adaptive development evidence; neither interval measures across-seed training variation'}


def main():
    registry=read(ROOT/'protocol/reference_workset_v6.json')
    members=[tuple(x['member']) for x in registry['sets']['dev']['records']]
    n=np.array([x['n'] for x in registry['sets']['dev']['records']])
    if len(members)!=2000 or len(set(members))!=2000:raise ValueError('Frozen dev panel changed')
    full={};terminal={};arrays={};identity={}
    for label,name in NAMES.items():
        folder=HERE/'runs'/name;result=read(folder/'results.json');frozen=read(folder/'frozen.json')
        if result['status'] not in ('early_stop','budget','max_epochs'):raise ValueError('Unfinished training')
        full[label]=read(folder/'full_dev.json');terminal[label]=read(folder/'terminal/results.json')
        summary=read(folder/'terminal/summary.json')
        if full[label]['checkpoint_sha256']!=frozen['checkpoint_sha256'] or summary['checkpoint_sha256']!=frozen['checkpoint_sha256']:raise ValueError('Raw/flash checkpoint differs')
        if len(full[label]['records'])!=2000 or len(terminal[label]['records'])!=500:raise ValueError('Incomplete evaluation')
        mapped={tuple(x['member']):x for x in full[label]['records']}
        if set(mapped)!=set(members):raise ValueError('Paired dev members differ')
        arrays[label]={key:np.array([mapped[m][key] for m in members]) for key in KEYS}
        identity[label]={'run':name,'checkpoint_sha256':frozen['checkpoint_sha256'],
                         'training_steps':result['steps'],'own_best_monitor_round':result['best_epoch'],
                         'own_best_monitor_loss':result['best_proxy'],'flash_summary':summary,
                         'own_total_loss_is_not_cross_objective_ranking':True}
    comparisons={}
    for key in KEYS:
        a=stats(arrays['original'][key]);b=stats(arrays['centerline_v1'][key])
        comparisons[key]={'original':a,'centerline_v1':b,'relative_changes':{k:relative(a[k],b[k]) for k in a},
                          'paired_improved_count':int((arrays['centerline_v1'][key]<arrays['original'][key]).sum())}
    delta=arrays['centerline_v1']['overlap_area_fraction']-arrays['original']['overlap_area_fraction']
    uncertainty=bootstrap(delta,members)
    ao=comparisons['overlap_area_fraction']['original'];bo=comparisons['overlap_area_fraction']['centerline_v1']
    gates={'overlap_mean_down5pct':bo['mean']<=.95*ao['mean'],
           'paired_cluster_ci_below_zero':uncertainty['source_shard_cluster_paired_95ci'][1]<0,
           'overlap_p95_within2pct':no_worse(ao['p95'],bo['p95'],.02),
           'overlap_worst5_within2pct':no_worse(ao['worst5pct_mean'],bo['worst5pct_mean'],.02)}
    for key,limit in [('loss_center_local',.02),('loss_shape',.02),('loss_relative_local',.02),('loss_boundary',.05),('loss_mib',.05)]:
        gates[key+'_guard']=no_worse(comparisons[key]['original']['mean'],comparisons[key]['centerline_v1']['mean'],limit)
    qa=identity['original']['flash_summary'];qb=identity['centerline_v1']['flash_summary']
    gates['flash_hard_legal_nonregression']=qb['hard_legal']>=qa['hard_legal']
    gates['flash_weighted_Q_within1pct']=qb['weighted_Q']<=qa['weighted_Q']*1.01
    # Only cached public constraints are used for strata; no additional model run.
    import torch
    torch.set_num_threads(1)
    from experiments.E59_centerline_overlap.cache import load_rows
    rows=load_rows('dev');public={tuple(x['member']):np.asarray(x['case'].solver_inputs[5]) for x in rows}
    strata={'n_le40':n<=40,'n_41_80':(n>40)&(n<=80),'n_gt80':n>80}
    for label,predicate in [('has_mib',lambda c:bool((c[:,2]>0).any())),
                            ('has_preplaced',lambda c:bool((c[:,1]>0).any())),
                            ('has_boundary_request',lambda c:bool((c[:,4]>0).any())),
                            ('has_fixed_shape',lambda c:bool((c[:,0]>0).any()))]:
        mask=np.array([predicate(public[m]) for m in members]);strata[label]=mask;strata['no_'+label]=~mask
    stratified={}
    for label,mask in strata.items():
        if not mask.any():continue
        stratified[label]={'count':int(mask.sum()),'overlap_original':stats(arrays['original']['overlap_area_fraction'][mask]),
                           'overlap_centerline_v1':stats(arrays['centerline_v1']['overlap_area_fraction'][mask]),
                           'paired_mean_difference':float(delta[mask].mean())}
    # Five cases already present in both fixed galleries, covering improvement,
    # regression, typical difference, and worst-overlap drafts of both models.
    gallery={label:read(HERE/'runs'/name/'draft_comparison100/manifest.json') for label,name in NAMES.items()}
    available={x['id'] for x in gallery['original']['cases']}
    if available!={x['id'] for x in gallery['centerline_v1']['cases']}:raise ValueError('Gallery panels differ')
    ta={x['index']:x for x in terminal['original']['records']};tb={x['index']:x for x in terminal['centerline_v1']['records']}
    ids=sorted(available)
    differences={i:tb[i]['draft_diagnostics']['overlap_area_fraction']-ta[i]['draft_diagnostics']['overlap_area_fraction'] for i in ids}
    order=sorted(ids,key=lambda i:differences[i]);chosen=[]
    proposals=[order[0],order[-1],order[len(order)//2],max(ids,key=lambda i:ta[i]['draft_diagnostics']['overlap_area_fraction']),max(ids,key=lambda i:tb[i]['draft_diagnostics']['overlap_area_fraction'])]
    for i in proposals+order:
        if i not in chosen:chosen.append(i)
        if len(chosen)==5:break
    images=[]
    for i in chosen:
        images.append({'index':i,'member':ta[i]['member'],'overlap_delta':differences[i],
                       'original_image':str(HERE/'runs'/NAMES['original']/f'draft_comparison100/images/case_{i:04d}.png'),
                       'centerline_image':str(HERE/'runs'/NAMES['centerline_v1']/f'draft_comparison100/images/case_{i:04d}.png')})
    report={'status':'numeric_comparison_complete_visual_review_pending','created_utc':datetime.now(timezone.utc).isoformat(),
            'runs':identity,'common_metrics':comparisons,'uncertainty':uncertainty,'gates':gates,
            'provisional_numeric_choice':'centerline_v1' if all(gates.values()) else 'original',
            'failed_gates':[k for k,v in gates.items() if not v],'stratified':stratified,'visual_review_cases':images,
            'member_panel_sha256':hashlib.sha256(json.dumps(members,separators=(',',':')).encode()).hexdigest(),
            'test_read':False,'new_inference_or_postprocessing':False,
            'limits':['Single seed and adaptive development evidence','No human ratings collected','Q is execution-sensitive; inspect reused_control_parity.json','Fixed pretraining gradient calibration does not equalize every future gradient']}
    save(CONTROL/'paired_comparison.json',report)
    lines=['# E59 small paired Loss comparison','',f"Numeric recommendation: {report['provisional_numeric_choice']}; visual inspection pending.",'',
           '| Common metric | Original mean | Centerline mean | Relative change |','|---|---:|---:|---:|']
    for key,v in comparisons.items():
        change=v['relative_changes']['mean'];text='near-zero denominator' if change is None else f'{change:+.2%}'
        lines.append(f"| {key} | {v['original']['mean']:.9g} | {v['centerline_v1']['mean']:.9g} | {text} |")
    lines+=['',f"Failed preregistered gates: {', '.join(report['failed_gates']) or 'none'}.",'',
            'Each candidate is its own fixed-Loss best. Cross-objective total Loss is not used to rank definitions.',
            'Development comparisons and a single seed do not establish independent generalization or universal superiority.',
            'See paired_comparison.json for paired shard/case bootstrap intervals, tails, strata and five existing visual pairs.']
    (CONTROL/'paired_comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'],'numeric_choice':report['provisional_numeric_choice'],'failed_gates':report['failed_gates']},ensure_ascii=False))


if __name__=='__main__':main()
