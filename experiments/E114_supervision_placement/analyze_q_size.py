"""Read saved dev results only: size weights, outliers and training cost."""
import json
import math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/supervision_q_analysis_20261108'
SOURCES={
 'E98 enhanced': 'experiments/E98_loss_weight_learning/runs/20261104_generalization96_seed0/evaluation_depth_dev500',
 'E109 A (96x12)': 'experiments/E109_deep_supervision_shared/runs/20261107_shared_seed0/evaluation_depth_dev500',
 'E114 terminal': 'experiments/E114_supervision_placement/runs/terminal_seed0/evaluation_no_mib_dev500',
 'E114 dense': 'experiments/E114_supervision_placement/runs/dense_seed0/evaluation_no_mib_dev500',
 'E114 4+12': 'experiments/E114_supervision_placement/runs/staged_seed0/evaluation_no_mib_dev500',
 'E115 1+9': 'experiments/E115_first_head_supervision/runs/first9_seed0/evaluation_no_mib_dev500',
}

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def summary(rows):
 n=np.array([r['n'] for r in rows]);q=np.array([r['metrics']['cost'] for r in rows]);w=np.exp((n-n.max())/12);w/=w.sum()
 geom=np.array([.5*(max(0,r['metrics']['hpwl_gap'])+max(0,r['metrics']['area_gap'])) for r in rows])
 soft=q-1-geom
 delta=(w-1/len(q))*(q-1)
 worst=np.argsort(-q,kind='stable');keep=np.ones(len(q),bool);keep[worst[:5]]=False
 ww=np.exp((n[keep]-n[keep].max())/12);ww/=ww.sum()
 bins=[]
 for lo,hi in [(21,40),(41,60),(61,80),(81,100),(101,110),(111,120)]:
  idx=(n>=lo)&(n<=hi);s=q[idx]
  bins.append(dict(range=f'{lo}-{hi}',count=int(idx.sum()),sample_share=float(idx.mean()),weight_share=float(w[idx].sum()),mean_Q=float(s.mean()),median_Q=float(np.median(s)),p95_Q=float(np.percentile(s,95)),max_Q=float(s.max()),gap_contribution=float(delta[idx].sum()),geometry_excess=float(geom[idx].mean()),soft_excess=float(soft[idx].mean()),boundary_mean=float(np.mean([r['metrics']['boundary_violations'] for r,j in zip(rows,idx) if j])),group_mean=float(np.mean([r['metrics']['grouping_violations'] for r,j in zip(rows,idx) if j]))))
 top=[]
 for i in worst[:10]:
  r=rows[i];m=r['metrics']
  top.append(dict(index=r['index'],member=r['member'],n=r['n'],Q=m['cost'],weight=float(w[i]),gap_contribution=float(delta[i]),hpwl_gap=m['hpwl_gap'],area_gap=m['area_gap'],soft_relative=m['violations_relative'],boundary=m['boundary_violations'],grouping=m['grouping_violations'],mib=m['mib_violations'],soft_max=m['max_possible_violations'],geometry_excess=float(geom[i]),soft_excess=float(soft[i]),work=r.get('work'),raw_geometry=r.get('raw_geometry')))
 return dict(count=len(q),weighted_Q=float(w@q),mean_Q=float(q.mean()),gap=float(w@q-q.mean()),effective_sample_size=float(1/(w@w)),n_mean=float(n.mean()),n_weighted_mean=float(w@n),n_Q_pearson=float(np.corrcoef(n,q)[0,1]),median_Q=float(np.median(q)),Q_above_1_2=int((q>1.2).sum()),bins=bins,worst=top,worst5_gap_contribution=float(delta[worst[:5]].sum()),worst5_weighted_excess_share=float((w[worst[:5]]*(q[worst[:5]]-1)).sum()/(w@(q-1))),diagnostic_excluding_worst5=dict(weighted_Q=float(ww@q[keep]),mean_Q=float(q[keep].mean()),gap=float(ww@q[keep]-q[keep].mean())),geometry_gap=float(w@geom-geom.mean()),soft_gap=float(w@soft-soft.mean()))

def main():
 OUT.mkdir(parents=True,exist_ok=True);models={};raw={}
 for label,p in SOURCES.items():
  rows=[read(f) for f in sorted((ROOT/p/'cases').glob('*.json'))];assert len(rows)==500
  raw[label]=rows;models[label]=summary(rows)
 # Six-bin standardization holds the empirical size-bin mixture constant, not task identity.
 for label,s in models.items():
  means=np.array([b['mean_Q'] for b in s['bins']]);counts=np.array([b['count'] for b in models['E114 4+12']['bins']]);centers=np.array([30.5,50.5,70.5,90.5,105.5,115.5]);weights=counts*np.exp((centers-120)/12)
  s['diagnostic_common_new_bin_mix']=dict(mean_Q=float(np.average(means,weights=counts)),weighted_Q=float(np.average(means,weights=weights)))
 timings=[]
 for label in list(SOURCES)[2:]:
  run=(ROOT/SOURCES[label]).parent;r=read(run/'results.json');p=read(run/'progress.json');h=next(x for x in p['history'] if x['epoch']==r['best_epoch'])
  tr=h['train_monitor'];dv=h['dev_monitor']
  timings.append(dict(model=label,seconds=r['elapsed_seconds'],epochs=r['epochs'],steps=r['steps'],best_epoch=r['best_epoch'],mean_seconds_per_epoch=r['elapsed_seconds']/r['epochs'],steps_per_second=r['steps']/r['elapsed_seconds'],best_own_train_loss=tr['total'],best_own_dev_loss=dv['total'],relative_own_gap=dv['total']/tr['total']-1,final_train_loss=tr['final_total'],final_dev_loss=dv['final_total'],relative_final_gap=dv['final_total']/tr['final_total']-1))
 result=dict(models=models,timings=timings,test_read=False,new_model_runs=0,new_Q_evaluations=0,method='Saved dev cases; weights exp(N/12). Exact gap contributions (normalized weight - 1/500)*(Q-1); geometry excess .5*(positive HPWL gap+positive area gap), remainder includes multiplicative soft interaction. Worst-five removal is diagnosis only, all published scores retain all cases.',limits='Historical task/panel/MIB/fixed-output semantics differ; width is confounded; common-bin standardization does not make an architecture control.')
 (OUT/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 lines=['# 加权Q与规模、训练耗时诊断（2026-11-08）','',result['method'],'',result['limits'],'','|模型|加权Q|平均Q|差|最差5例对差值的贡献|移去最差5例后的差（仅诊断）|','|---|---:|---:|---:|---:|---:|']
 for label,s in models.items():lines.append(f"|{label}|{s['weighted_Q']:.6f}|{s['mean_Q']:.6f}|{s['gap']:+.6f}|{s['worst5_gap_contribution']:+.6f}|{s['diagnostic_excluding_worst5']['gap']:+.6f}|")
 lines+=['','## 规模分组','', '|模型|N|例数|权重占比|平均Q|中位Q|P95 Q|差值贡献|','|---|---|---:|---:|---:|---:|---:|---:|']
 for label,s in models.items():
  for b in s['bins']:lines.append(f"|{label}|{b['range']}|{b['count']}|{100*b['weight_share']:.2f}%|{b['mean_Q']:.6f}|{b['median_Q']:.6f}|{b['p95_Q']:.6f}|{b['gap_contribution']:+.6f}|")
 lines+=['','## 四模型训练用时','', '|模型|轮次|最佳轮|总秒数|每轮平均秒数|更新步/秒|Train Loss|Dev Loss|相对差|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for t in timings:lines.append(f"|{t['model']}|{t['epochs']}|{t['best_epoch']}|{t['seconds']:.2f}|{t['mean_seconds_per_epoch']:.2f}|{t['steps_per_second']:.2f}|{t['best_own_train_loss']:.6f}|{t['best_own_dev_loss']:.6f}|{100*t['relative_own_gap']:.2f}%|")
 (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print(json.dumps(dict(models={k:{key:v[key] for key in ('gap','worst5_gap_contribution','diagnostic_excluding_worst5','geometry_gap','soft_gap','worst5_weighted_excess_share','diagnostic_common_new_bin_mix')} for k,v in models.items()},timings=timings,staged_worst=models['E114 4+12']['worst'][:5],staged_bins=models['E114 4+12']['bins']),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
