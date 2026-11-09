"""Additional saved-coordinate diagnostics and figures; no model or solver calls."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='4'
from pathlib import Path
import sys,json,csv,time
import numpy as np
import torch
from scipy.stats import rankdata,spearmanr
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];sys.path.insert(0,str(ROOT))
torch.set_num_threads(4);torch.set_num_interop_threads(1)
from experiments.E39_rectangular_attention.data import load_rows
from experiments.E39_rectangular_attention.diagnostics import geometry_checks
from experiments.E39_rectangular_attention.backend.frozen.packing import prepare_problem,_dimensions
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d):
    with p.open('w',encoding='utf-8',newline='\n') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
def avg(r,k):return float(np.mean([x[k] for x in r]))
def fixed_n_rho(rows,x,y):
    a=rankdata([r[x] for r in rows]);b=rankdata([r[y] for r in rows]);n=np.array([r['n'] for r in rows])
    for k in np.unique(n):
        ids=n==k;a[ids]-=a[ids].mean();b[ids]-=b[ids].mean()
    den=np.linalg.norm(a)*np.linalg.norm(b);return float(a@b/den) if den else None
def main():
    start=time.perf_counter();summary=read(HERE/'summary.json')
    with (HERE/'case_features.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    for row in rows:
        for k,v in list(row.items()):
            if k not in ['member','shard','worker']:row[k]=None if v=='' else float(v)
        row['index']=int(row['index']);row['n']=int(row['n'])
    ds=load_rows('dev_monitor500')
    source=ROOT/'experiments/E39_rectangular_attention/runs/20261023_rectangle_w96d12_r2_rectangle_seed0/terminal/results.json'
    terminal={r['index']:r for r in read(source)['records']}
    gallery=read(ROOT/'experiments/E39_rectangular_attention/runs/20261023_rectangle_triptych100/drafts_local.json')
    raw={r['index']:np.asarray(r['raw_positions']) for r in gallery};extra=np.load(HERE/'drafts_additional_local.npz')
    raw.update({int(k):extra[k] for k in extra.files});assert len(raw)==500
    for r,row in zip(rows,ds):
        i=r['index'];case=row['case'];c=np.asarray(case.solver_inputs[5]).astype(int);p=raw[i];n=len(p);areas=np.asarray(case.solver_inputs[1],dtype=float)
        low=np.maximum(p[:,None,:2],p[None,:,:2]);hi=np.minimum((p[:,:2]+p[:,2:])[:,None,:],(p[:,:2]+p[:,2:])[None,:,:])
        inter=np.maximum(hi-low,0).prod(-1);a,b=np.triu_indices(n,1);pre=c[:,1]>0;obstacle=pre[a]|pre[b]
        r['preplaced_overlap_area_fraction']=float(inter[a[obstacle],b[obstacle]].sum()/areas.sum())
        r['preplaced_overlap_pairs']=int((inter[a[obstacle],b[obstacle]]>1e-8).sum())
        bits=((c[:,4,None]&np.array([1,2,4,8]))>0).sum(1)
        r['corner_fraction']=float((bits>1).mean());r['mib_count_blocks']=int((c[:,2]>0).sum())
        r['shape_iterations_per_call']=r['shape_iterations']/max(r['shape_calls'],1)
        problem=prepare_problem(*case.solver_inputs);w,h=_dimensions(problem,p[:,2]/p[:,3],True)
        initial=np.column_stack((p[:,:2],w,h));_,_,diag=geometry_checks(initial,case.solver_inputs)
        r['mib_init_overlap']=diag['overlap_area_fraction'];r['mib_init_overlap_increase']=r['mib_init_overlap']-r['raw_overlap']
        r['mib_init_log_ratio_change']=float(np.abs(np.log(w/h)-np.log(p[:,2]/p[:,3])).mean())
    xs=['group_fraction','largest_group_fraction','group_mst_gap','mib_count_blocks','mib_fraction','boundary_fraction','corner_fraction',
        'locked_fraction','preplaced_fraction','preplaced_overlap_area_fraction','raw_overlap','raw_cluster','raw_contact_loss','mib_init_overlap','mib_init_overlap_increase']
    ys=['raw_overlap','center_move','topology_change_fraction','failed_shape_calls','early_projection','rescue','postprocess_seconds','final_Q','final_grouping','shape_iterations_per_call']
    assoc=[{'x':x,'y':y,'fixed_n_partial_rho':fixed_n_rho(rows,x,y)} for x in xs for y in ys if x!=y]
    rng=np.random.default_rng(8314);workers=sorted(set(r['worker'] for r in rows));by={w:[r for r in rows if r['worker']==w] for w in workers}
    cis=[]
    for x,y in [('group_fraction','center_move'),('group_fraction','final_Q'),('raw_overlap','center_move'),('raw_overlap','topology_change_fraction'),('preplaced_fraction','failed_shape_calls'),('mib_count_blocks','final_Q')]:
        vals=[]
        for _ in range(600):
            rr=[r for w in rng.choice(workers,len(workers),replace=True) for r in by[w]];vals.append(fixed_n_rho(rr,x,y))
        cis.append({'x':x,'y':y,'fixed_n_partial_rho':fixed_n_rho(rows,x,y),'worker_bootstrap95':np.percentile(vals,[2.5,97.5]).tolist()})
    event=[]
    for name in ['early_projection','rescue']:
        for value in [0,1]:
            selected=[r for r in rows if r[name]==value]
            event.append({'event':name,'value':value,'count':len(selected),**{k:avg(selected,k) for k in ['n','group_fraction','locked_fraction','preplaced_fraction','corner_fraction','raw_overlap','preplaced_overlap_area_fraction','mib_init_overlap_increase','failed_shape_calls','shape_iterations_per_call','center_move','final_Q','postprocess_seconds']}})
    q=np.quantile([r['group_fraction'] for r in rows],[0,.25,.5,.75,1]);bins=[]
    for j in range(4):
        selected=[r for r in rows if q[j]<=r['group_fraction'] and (r['group_fraction']<q[j+1] if j<3 else r['group_fraction']<=q[j+1])]
        bins.append({'range':[float(q[j]),float(q[j+1])],'count':len(selected),**{k:avg(selected,k) for k in ['n','center_move','postprocess_seconds','final_Q','raw_overlap','failed_shape_calls','rescue']}})
    group_high=[r for r in rows if r['group_fraction']>=.5];highgroupeasy=min(group_high,key=lambda r:r['final_Q'])
    selected=[('high_group_easy',highgroupeasy),('high_group_hard',max(group_high,key=lambda r:r['final_Q'])),
              ('largest_repair',max(rows,key=lambda r:r['center_move'])),('rescue',max([r for r in rows if r['rescue']],key=lambda r:r['final_Q'])),
              ('large_move_good_Q',max([r for r in rows if r['final_Q']<1.001],key=lambda r:r['center_move']))]
    examples=[]
    from visualization.layouts import plot_case,geometry_details
    from scoring.evaluation import evaluate_case
    (HERE/'examples').mkdir(exist_ok=True)
    for label,r in selected:
        i=r['index'];row=ds[i];case=row['case'];c=np.asarray(case.solver_inputs[5]).astype(int).tolist()
        g=row['graph'];ref=(g['reference']*g['scale']).numpy().astype(float)
        variants={}
        for key,positions,metrics in [('reference',ref,evaluate_case(case,ref,0.)['metrics']),('raw',raw[i],terminal[i]['raw_metrics']),('final',np.asarray(terminal[i]['positions']),terminal[i]['metrics'])]:
            variants[key]={'positions':positions.tolist(),'metrics':metrics,'soft_details':geometry_details(positions.tolist(),c)}
        path=HERE/'examples'/f'{label}_index{i:03d}.png'
        plot_case({'case_id':f'{label} | zero-based index {i} | group {r["group_fraction"]:.1%}',
            'panel':'E39 failure-mode audit | existing dev500','constraints':c,'variants':variants,
            'caption':'Original reference uses old areas; final layout/metrics reused from saved evaluation. No training or solver rerun.'},path,
            stages=(('reference','Original reference / old areas'),('raw','E39 rectangle draft'),('final',f'Saved final | Q={r["final_Q"]:.5f}')))
        examples.append({'label':label,'image':path.relative_to(HERE).as_posix(),**{k:r[k] for k in ['index','member','n','group_fraction','center_move','final_Q','rescue','early_projection']}})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    for a,x,y,xlabel,ylabel in [(ax[0],'group_fraction','center_move','Grouped block fraction','Mean repair displacement / sqrt(area)'),(ax[1],'raw_overlap','topology_change_fraction','Draft overlap pair area / total area','Changed pair separation axes'),(ax[2],'preplaced_fraction','failed_shape_calls','Preplaced block fraction','Nonlegal shape candidates')]:
        dots=a.scatter([r[x] for r in rows],[r[y] for r in rows],c=[r['n'] for r in rows],s=15,alpha=.6,cmap='viridis')
        rr=[r for r in rows if r['rescue']];a.scatter([r[x] for r in rr],[r[y] for r in rr],s=70,facecolors='none',edgecolors='red',label='Packing rescue')
        a.set(xlabel=xlabel,ylabel=ylabel);a.grid(alpha=.15)
    ax[2].legend();fig.colorbar(dots,ax=ax,label='Module count');fig.savefig(HERE/'repair_associations.png',dpi=160);plt.close(fig)
    keys=['group_fraction','raw_overlap','raw_contact_loss','preplaced_fraction','locked_fraction','mib_count_blocks','corner_fraction']
    outcomes=['center_move','topology_change_fraction','failed_shape_calls','postprocess_seconds','final_Q']
    matrix=np.array([[fixed_n_rho(rows,k,y) for y in outcomes] for k in keys]);fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    im=ax.imshow(matrix,vmin=-.65,vmax=.65,cmap='RdBu_r');ax.set_xticks(range(len(outcomes)),['Repair move','Axis changes','Nonlegal candidates','Repair time','Final Q'],rotation=20,ha='right')
    ax.set_yticks(range(len(keys)),['Grouped fraction','Draft overlap','Reference-contact loss','Preplaced fraction','Locked fraction','MIB block count','Corner fraction'])
    for i in range(len(keys)):
        for j in range(len(outcomes)):ax.text(j,i,f'{matrix[i,j]:+.2f}',ha='center',va='center',color='white' if abs(matrix[i,j])>.4 else 'black')
    ax.set_title('Partial rank association within identical module counts (exploratory)');fig.colorbar(im,ax=ax,label='Partial rank correlation');fig.savefig(HERE/'conditional_associations.png',dpi=170);plt.close(fig)
    with (HERE/'case_features_enriched.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save(HERE/'additional_summary.json',{'fixed_n_associations':assoc,'worker_bootstrap':cis,'group_quartiles':bins,'events':event,'examples':examples,
        'mean_cross_group_or_unassigned_overlap_share':avg(rows,'cross_group_overlap_share'),
        'mib_init_overlap_mean':avg(rows,'mib_init_overlap'),'mib_init_overlap_increase_mean':avg(rows,'mib_init_overlap_increase'),
        'seconds':time.perf_counter()-start,'training':False,'solver_calls':0,'test_read':False})
    print('Saved-coordinate enrichment and figures complete',flush=True)
if __name__=='__main__':main()
