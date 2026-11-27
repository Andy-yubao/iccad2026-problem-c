"""Read frozen train/dev caches and saved repairs; no training or solver calls."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='2'
from pathlib import Path
import sys, json, time, hashlib, csv, argparse, multiprocessing as mp
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import torch
from scipy.stats import spearmanr, rankdata
from scipy.sparse.csgraph import minimum_spanning_tree

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT))
SOURCE=ROOT/'experiments/E39_rectangular_attention/runs/20261023_rectangle_w96d12_r2_rectangle_seed0'
CACHE=ROOT/'experiments/E25_integrated_baseline/runs/prepared_v5_e10_cache_v1'
torch.set_num_threads(2)
torch.set_num_interop_threads(1)

def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
    t=p.with_suffix(p.suffix+'.tmp')
    with t.open('w',encoding='utf-8',newline='\n') as f: json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
    os.replace(t,p)
def csvsave(p,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def mean(x): return float(np.mean(x)) if len(x) else 0.
def cos(a,b):
    a=a.reshape(-1).double().numpy();b=b.reshape(-1).double().numpy()
    den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/den) if den>1e-15 else None
def groups(c,col): return [np.flatnonzero(c[:,col]==k) for k in np.unique(c[:,col]) if k>0]
def contact_edges(p,c,tol):
    center=p[:,:2]+p[:,2:]/2
    margin=np.abs(center[:,None]-center[None,:])-(p[:,None,2:]+p[None,:,2:])/2
    same=(c[:,3,None]==c[None,:,3])&(c[:,3,None]>0)
    cx=(np.abs(margin[...,0])<tol)&(margin[...,1]<-tol)&same
    cy=(np.abs(margin[...,1])<tol)&(margin[...,0]<-tol)&same
    return np.triu(cx|cy,1), margin
def components_count(edges,ids):
    parent={int(i):int(i) for i in ids}
    def find(k):
        while parent[k]!=k: k=parent[k]
        return k
    for i,j in zip(*np.where(edges)):
        if i in parent and j in parent: parent[find(int(i))]=find(int(j))
    return len({find(int(i)) for i in ids})
def shape_spread(p,gs):
    ratios=np.log(p[:,2]/p[:,3]);pairs=[]
    for ids in gs:
        i,j=np.triu_indices(len(ids),1)
        pairs.extend(((ratios[ids[i]]-ratios[ids[j]])**2/4).tolist())
    return mean(pairs)

def load_selected(role):
    from datasets.reference_layouts import ReferenceLayouts
    from experiments.E39_rectangular_attention.cache import verify
    parent='train' if role.startswith('train') else 'dev'
    ds=ReferenceLayouts(role); wanted={tuple(r['member']):i for i,r in enumerate(ds.records)}
    rows=[None]*len(ds)
    m=verify(parent)
    for part in m['parts']:
        payload=torch.load(CACHE/parent/part['name'],map_location='cpu',weights_only=False)
        for row in payload['rows']:
            member=tuple(row['member'])
            if member in wanted: rows[wanted[member]]=row
    assert all(r is not None for r in rows)
    return rows

def target_audit(row):
    from experiments.E39_rectangular_attention.batches import collate, decode
    from experiments.E39_rectangular_attention.losses import components
    from experiments.E39_rectangular_attention.diagnostics import geometry_checks
    g=collate([row['graph']],[0],'cpu'); case=row['case']; c=np.asarray(case.solver_inputs[5]).astype(int)
    scale=float(g['scale'][0]);ref=g['reference'][0].numpy()*scale
    oracle=decode(g['target'],g)[0].numpy()*scale
    values=components(g['target'],g)
    n=len(ref);areas=np.asarray(case.solver_inputs[1]);old=ref[:,2:].prod(1);locked=c[:,:2].any(1)
    changed=(np.abs(areas/old-1)>.01)&~locked
    mg=groups(c,2);gg=groups(c,3)
    refedge,_=contact_edges(ref,c,scale*1e-6)
    newedge,newmargin=contact_edges(oracle,c,scale*1e-6)
    _,_,diag=geometry_checks(oracle,case.solver_inputs)
    gmib=0;gmismatch=0;shape_constraints=0
    for ids in mg:
        if len(ids)<2:continue
        gmib+=1
        spread=np.ptp(np.log(ref[ids,2]/ref[ids,3]))
        gmismatch+=int(spread>1e-4)
        shape_constraints+=len(ids)-1
    broken=refedge&~newedge
    contact_norm=[]
    for i,j in zip(*np.where(refedge)):
        # Signed normal gap at the ORIGINAL contact orientation; original relation supervision.
        _,rm=contact_edges(ref[[i,j]],c[[i,j]],scale*1e-6)
        axis=int(np.argmin(np.abs(rm[0,1])))
        contact_norm.append(float(newmargin[i,j,axis]/((np.sqrt(areas[i])+np.sqrt(areas[j]))/2)))
    return dict(n=n,group_fraction=float((c[:,3]>0).mean()),mib_fraction=float((c[:,2]>0).mean()),
        changed_area_blocks=int(changed.sum()),changed_area_fraction=float(changed.mean()),
        mean_abs_area_change=mean(np.abs(areas/old-1)[~locked]),
        reference_mib_groups=gmib,reference_mib_ratio_conflict_groups=gmismatch,
        reference_mib_pair_logwh_loss=shape_spread(ref,mg),
        original_contact_edges=int(refedge.sum()),oracle_broken_contact_edges=int(broken.sum()),
        contact_normal_gap_abs_mean=mean(np.abs(contact_norm)),
        oracle_overlap_fraction=diag['overlap_area_fraction'],
        oracle_group_components_excess=sum(components_count(newedge,ids)-1 for ids in gg),
        original_group_components_excess=sum(components_count(refedge,ids)-1 for ids in gg),
        oracle_center_clip_fraction=float((np.linalg.norm((oracle[:,:2]+oracle[:,2:]/2)-(ref[:,:2]+ref[:,2:]/2),axis=1)>scale*1e-6).mean()),
        **{'oracle_loss_'+k:float(v[0]) for k,v in values.items()})

def initialize():
    global MODEL
    from experiments.E39_rectangular_attention.model import GeometryModel
    torch.use_deterministic_algorithms(True)
    state=torch.load(SOURCE/'checkpoints/best.pt',map_location='cpu',weights_only=False)
    MODEL=GeometryModel(**state['model_config']).eval();MODEL.load_state_dict(state['model'],strict=True)
    for p in MODEL.parameters():p.requires_grad_(False)

def analyze_case(task):
    index,row,record,saved_raw=task
    from experiments.E39_rectangular_attention.inference import graph
    from experiments.E39_rectangular_attention.model import decode
    from experiments.E39_rectangular_attention.batches import collate
    from experiments.E39_rectangular_attention.losses import components
    from experiments.E39_rectangular_attention.diagnostics import geometry_checks
    from experiments.E39_rectangular_attention.fast_geometry import topology
    from scoring.evaluation import evaluate_case
    inputs=row['case'].solver_inputs;c=np.asarray(inputs[5]).astype(int);n=int(inputs[0]);areas=np.asarray(inputs[1],dtype=float);local=np.sqrt(areas)
    ig=graph(inputs)
    with torch.no_grad():
        state=MODEL(ig)
        draft=(decode(state,ig)[0]*ig['scale'][0]).numpy().astype(float)
    specs=np.asarray(inputs[6]);locked=c[:,:2].any(1);pre=c[:,1]>0
    draft[locked,2:]=specs[locked,2:];draft[pre]=specs[pre]
    if saved_raw is not None: assert np.array_equal(draft,np.asarray(saved_raw)), 'Gallery draft mismatch'
    metrics=evaluate_case(row['case'],draft,0.)['metrics']
    for k in ('overlap_violations','boundary_violations','grouping_violations','mib_violations','hpwl_gap','area_gap'):
        assert np.isclose(metrics[k],record['raw_metrics'][k],rtol=1e-9,atol=1e-10),(index,k,metrics[k],record['raw_metrics'][k])
    _,_,diag=geometry_checks(draft,inputs)
    assert all(np.isclose(diag[k],v,rtol=1e-9,atol=1e-10) for k,v in record['draft_diagnostics'].items())
    final=np.asarray(record['positions']);center=draft[:,:2]+draft[:,2:]/2;fc=final[:,:2]+final[:,2:]/2
    moves=np.linalg.norm(fc-center,axis=1)/local
    assert np.isclose(moves.mean(),record['mean_center_move_local'],rtol=1e-9)
    e0=topology(draft,c,array=True);e1=topology(final,c,array=True)
    mg=groups(c,2);gg=groups(c,3);boundary=c[:,4]>0;membership=c[:,3]>0
    pairgroup=(c[:,3,None]==c[None,:,3])&(c[:,3,None]>0)
    # Explicit broadcasting keeps module-pair axes unambiguous.
    dx=np.maximum(np.minimum((draft[:,:2]+draft[:,2:])[:,None,:],(draft[:,:2]+draft[:,2:])[None,:,:])-np.maximum(draft[:,:2][:,None,:],draft[:,:2][None,:,:]),0)
    inter=dx.prod(-1);ii,jj=np.triu_indices(n,1);same=pairgroup[ii,jj]
    total=inter[ii,jj].sum();within=inter[ii[same],jj[same]].sum()
    treegaps=[];largest_group=0
    for ids in gg:
        largest_group=max(largest_group,len(ids))
        p=draft[ids];gap=np.maximum(np.maximum(p[:,None,:2]-p[None,:,:2]-p[None,:,2:],p[None,:,:2]-p[:,None,:2]-p[:,None,2:]),0)
        dist=np.linalg.norm(gap,axis=-1)/((local[ids,None]+local[None,ids])/2)
        dist=dist+1e-12;np.fill_diagonal(dist,0)
        treegaps.extend(minimum_spanning_tree(dist).data.tolist())
    bg=collate([row['graph']],[0],'cpu');s=state.detach().requires_grad_(True);loss=components(s,bg)
    grads={k:torch.autograd.grad(loss[k].sum(),s,retain_graph=True)[0] for k in ('shape','mib','center_local','contact','overlap','boundary')}
    trace=record['trace'];shape=[t for t in trace if t['phase']=='shape']
    out=dict(index=index,member=row['member'][0]+':'+str(row['member'][1]),shard=row['member'][0],worker=row['member'][0].split('/')[0],n=n,
        group_fraction=float(membership.mean()),group_count=len(gg),largest_group_fraction=largest_group/n,
        group_tree_edges=sum(max(0,len(ids)-1) for ids in gg),group_mst_gap=mean(treegaps),
        mib_fraction=float((c[:,2]>0).mean()),boundary_fraction=float(boundary.mean()),locked_fraction=float(locked.mean()),preplaced_fraction=float(pre.mean()),
        group_boundary_fraction=float((membership&boundary).mean()),group_mib_fraction=float((membership&(c[:,2]>0)).mean()),
        area_cv=float(np.std(areas)/np.mean(areas)),
        raw_overlap=diag['overlap_area_fraction'],raw_cluster=diag['largest_conflict_cluster_fraction'],
        within_group_overlap_share=float(within/max(total,1e-12)),cross_group_overlap_share=float((total-within)/max(total,1e-12)),
        raw_grouping=metrics['grouping_violations'],raw_grouping_normalized=metrics['grouping_violations']/max(1,sum(len(ids)-1 for ids in gg)),
        raw_boundary=metrics['boundary_violations'],raw_mib=metrics['mib_violations'],raw_mib_pair_loss=shape_spread(draft,mg),
        raw_geometry=float(loss['geometry'][0]),raw_contact_loss=float(loss['contact'][0]),raw_shape_loss=float(loss['shape'][0]),
        raw_supervision=float(loss['local'][0]),
        shape_mib_gradient_cos=cos(grads['shape'],grads['mib']),
        center_contact_gradient_cos=cos(grads['center_local'],grads['contact']),
        overlap_contact_gradient_cos=cos(grads['overlap'],grads['contact']),
        overlap_boundary_gradient_cos=cos(grads['overlap'],grads['boundary']),
        weighted_mib_shape_gradient_norm_ratio=float(torch.linalg.vector_norm(.1*grads['mib'])/torch.linalg.vector_norm(.25*grads['shape']).clamp_min(1e-12)),
        center_move=record['mean_center_move_local'],max_move=record['max_center_move_local'],
        group_center_move=mean(moves[membership]),nongroup_center_move=mean(moves[~membership]),
        moved_gt_half_fraction=float((moves>.5).mean()),moved_gt_one_fraction=float((moves>1).mean()),
        log_ratio_change=mean(np.abs(np.log(final[:,2]/final[:,3])-np.log(draft[:,2]/draft[:,3]))),
        topology_axis_change_fraction=float((e0[:,2]!=e1[:,2]).mean()),topology_change_fraction=float((e0!=e1).any(1).mean()),
        shape_calls=len(shape),shape_iterations=sum(t.get('iterations',0) for t in shape),
        failed_shape_calls=sum(not t.get('legal',False) for t in shape),
        early_projection=int(any(t['phase']=='early_projection' for t in trace)),rescue=int(record['rescue_used']),
        postprocess_seconds=record['postprocess_seconds'],final_Q=metrics['cost'],
        final_grouping=record['metrics']['grouping_violations'],final_boundary=record['metrics']['boundary_violations'],
        final_mib=record['metrics']['mib_violations'])
    out['final_Q']=record['metrics']['cost']
    return out,None if saved_raw is not None else draft

def partial_rho(rows,x,y,controls):
    valid=[r for r in rows if r.get(x) is not None and r.get(y) is not None]
    a=rankdata([r[x] for r in valid]);b=rankdata([r[y] for r in valid])
    z=np.column_stack([np.ones(len(valid))]+[rankdata([r[k] for r in valid]) for k in controls])
    a=a-z@np.linalg.lstsq(z,a,rcond=None)[0];b=b-z@np.linalg.lstsq(z,b,rcond=None)[0]
    den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/den) if den>1e-9 else None

def summaries(rows):
    keys=['raw_overlap','raw_contact_loss','raw_mib_pair_loss','center_move','max_move','log_ratio_change','topology_change_fraction',
          'shape_calls','failed_shape_calls','early_projection','rescue','postprocess_seconds','final_Q','final_grouping','final_boundary']
    out={'count':len(rows)}
    for k in keys:
        x=[r[k] for r in rows];out[k+'_mean']=mean(x)
    out['weighted_Q']=float(np.average([r['final_Q'] for r in rows],weights=np.exp((np.array([r['n'] for r in rows])-120)/12))) if rows else None
    return out

def statistics(rows,targets):
    for r in rows:
        n=r['n'];r.update(n_squared=n*n,budget60=int(n>=60),budget90=int(n>=90),budget110=int(n>=110))
    controls=['n','n_squared','budget60','budget90','budget110']
    xs=['group_fraction','largest_group_fraction','mib_fraction','boundary_fraction','group_boundary_fraction',
        'locked_fraction','area_cv','raw_overlap','raw_cluster','group_mst_gap','raw_contact_loss',
        'raw_mib_pair_loss','reference_mib_pair_logwh_loss','changed_area_fraction']
    ys=['raw_overlap','center_move','log_ratio_change','topology_change_fraction','failed_shape_calls','early_projection','rescue','postprocess_seconds','final_Q','final_grouping']
    associations=[]
    for x in xs:
        for y in ys:
            if x==y:continue
            associations.append({'x':x,'y':y,'spearman':float(spearmanr([r[x] for r in rows],[r[y] for r in rows]).statistic),
                'partial_size_budget':partial_rho(rows,x,y,controls),
                'partial_extra_constraints':partial_rho(rows,x,y,controls+[k for k in ['mib_fraction','boundary_fraction','locked_fraction'] if k!=x])})
    bins=[]
    for size,a,b in [('all',0,1000),('n20_59',20,60),('n60_89',60,90),('n90_109',90,110),('n110_120',110,121)]:
        for label,lo,hi in [('g0_25',0,.25),('g25_50',.25,.5),('g50_75',.5,.75),('g75_100',.75,1.001)]:
            chosen=[r for r in rows if a<=r['n']<b and lo<=r['group_fraction']<hi]
            if chosen:bins.append({'size':size,'group_bin':label,**summaries(chosen)})
    bootstrap=[];rng=np.random.default_rng(20261023)
    shardids=sorted(set(r['shard'] for r in rows));ids={s:[i for i,r in enumerate(rows) if r['shard']==s] for s in shardids}
    for y in ['center_move','topology_change_fraction','postprocess_seconds','final_Q','failed_shape_calls','raw_overlap']:
        vals=[]
        for _ in range(700):
            sample=[rows[i] for s in rng.choice(shardids,len(shardids),replace=True) for i in ids[s]]
            vals.append(partial_rho(sample,'group_fraction',y,controls))
        bootstrap.append({'x':'group_fraction','y':y,'partial_size_budget':partial_rho(rows,'group_fraction',y,controls),
            'shard_bootstrap95':np.percentile(vals,[2.5,97.5]).tolist()})
    paired=[]
    for arm,path in [('plain96x12','20261022_capacity_deep96x12_seed0'),('plain96x4','20261022_capacity_base96x4_seed0')]:
        other=read(ROOT/'experiments/E38_model_architecture/runs'/path/'terminal/results.json')['records']
        by={tuple(r['member']):r for r in other};pr=[]
        for r in rows:
            key=(r['shard'],int(r['member'].rsplit(':',1)[1]));o=by[key]
            pr.append({**r,'delta_overlap':r['raw_overlap']-o['draft_diagnostics']['overlap_area_fraction'],
                'delta_move':r['center_move']-o['mean_center_move_local'],
                'delta_postprocess':r['postprocess_seconds']-o['postprocess_seconds'],
                'delta_Q':r['final_Q']-o['metrics']['cost']})
        for lo,hi in [(0,.5),(.5,.75),(.75,1.001)]:
            p=[r for r in pr if lo<=r['group_fraction']<hi]
            if p:paired.append({'control':arm,'group_range':[lo,hi],'count':len(p),
                **{k:mean([r[k] for r in p]) for k in ['delta_overlap','delta_move','delta_postprocess','delta_Q']}})
    target_summary={}
    for role in ['train_monitor500','dev_monitor500']:
        rr=[r for r in targets if r['role']==role]
        target_summary[role]={'count':len(rr),'with_mib':sum(r['reference_mib_groups']>0 for r in rr),
            'area_changed_cases':sum(r['changed_area_blocks']>0 for r in rr),
            'mib_ratio_conflict_cases':sum(r['reference_mib_ratio_conflict_groups']>0 for r in rr),
            'mib_ratio_conflict_groups':sum(r['reference_mib_ratio_conflict_groups'] for r in rr),
            'mib_groups':sum(r['reference_mib_groups'] for r in rr),
            'original_contact_edges':sum(r['original_contact_edges'] for r in rr),
            'oracle_broken_contact_edges':sum(r['oracle_broken_contact_edges'] for r in rr),
            'oracle_overlap_positive_cases':sum(r['oracle_overlap_fraction']>1e-6 for r in rr),
            **{k:mean([r[k] for r in rr]) for k in rr[0] if k.startswith('oracle_loss_') or k in ['mean_abs_area_change','changed_area_fraction','oracle_overlap_fraction','reference_mib_pair_logwh_loss','oracle_center_clip_fraction']}}
    grad={}
    for key in ['shape_mib_gradient_cos','center_contact_gradient_cos','overlap_contact_gradient_cos','overlap_boundary_gradient_cos','weighted_mib_shape_gradient_norm_ratio']:
        v=[r[key] for r in rows if r[key] is not None];grad[key]={'count':len(v),'mean':mean(v),'median':float(np.median(v)),'negative_fraction':mean(np.array(v)<0)}
    hard=[]
    for kind in ['rescue','early_projection']:
        for val in [0,1]:hard.append({'event':kind,'value':val,**summaries([r for r in rows if r[kind]==val])})
    tails={k:[{q:r[q] for q in ['index','member','n','group_fraction','mib_fraction','boundary_fraction',k,'center_move','final_Q','rescue','early_projection']} for r in sorted(rows,key=lambda r:r[k],reverse=True)[:10]] for k in ['center_move','final_Q','postprocess_seconds','raw_overlap','group_fraction']}
    return {'overall':summaries(rows),'unique_shards':len(shardids),'unique_workers':len(set(r['worker'] for r in rows)),
        'bins':bins,'associations':associations,'group_partial_bootstrap':bootstrap,'paired':paired,
        'targets':target_summary,'gradients':grad,'events':hard,'tails':tails,
        'phase_counts':dict(Counter(t['phase'] for r in read(SOURCE/'terminal/results.json')['records'] for t in r['trace']))}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pilot',action='store_true');args=ap.parse_args()
    started=time.perf_counter()
    records=read(SOURCE/'terminal/results.json')['records'];records.sort(key=lambda r:r['index'])
    frozen=read(SOURCE/'frozen.json');assert sha(SOURCE/'checkpoints/best.pt')==frozen['checkpoint_sha256']
    sources=[SOURCE/'terminal/results.json',SOURCE/'checkpoints/best.pt',ROOT/'protocol/reference_workset_v5.json',Path(__file__),
        ROOT/'datasets/reference_layouts.py',ROOT/'experiments/E39_rectangular_attention/losses.py',ROOT/'experiments/E39_rectangular_attention/batches.py']
    identity={p.relative_to(ROOT).as_posix():sha(p) for p in sources}
    save(HERE/('pilot_config.json' if args.pilot else 'config.json'),{'source_hashes':identity,'command':sys.argv,'workers':2,'threads_per_worker':2,
        'seed':20261023,'training':False,'solver_calls':0,'test_read':False,'roles':['train_monitor500','dev_monitor500'],
        'wall_limit_seconds':900,'original_area_note':'Initial cache audit uses reference WH product as area proxy; verified separately against raw public areas.'})
    gallery={r['index']:r['raw_positions'] for r in read(ROOT/'experiments/E39_rectangular_attention/runs/20261023_rectangle_triptych100/drafts_local.json')}
    dev=load_selected('dev_monitor500');print('Loaded dev monitor cache',flush=True)
    if args.pilot:
        initialize();a=target_audit(dev[0]);b,_=analyze_case((0,dev[0],records[0],gallery[0]));save(HERE/'pilot.json',{'target':a,'case':b,'seconds':time.perf_counter()-started});print('Pilot passed',flush=True);return
    done=read(HERE/'case_features_partial.json') if (HERE/'case_features_partial.json').exists() else []
    completed={r['index'] for r in done};newdrafts={}
    targetrows=[]
    for i,row in enumerate(dev):targetrows.append({'role':'dev_monitor500','index':i,'member':str(row['member']),**target_audit(row)})
    print('Dev target audit complete',flush=True)
    with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
        futures={pool.submit(analyze_case,(i,row,records[i],gallery.get(i))):i for i,row in enumerate(dev) if i not in completed}
        for f in as_completed(futures):
            row,draft=f.result();done.append(row)
            if draft is not None:newdrafts[str(row['index'])]=draft
            if len(done)%25==0:
                save(HERE/'case_features_partial.json',done);print(f'Draft diagnostics {len(done)}/500, seconds={time.perf_counter()-started:.1f}',flush=True)
            assert time.perf_counter()-started<900,'Audit wall limit reached'
    existing=dict(np.load(HERE/'drafts_additional_local.npz')) if (HERE/'drafts_additional_local.npz').exists() else {}
    np.savez_compressed(HERE/'drafts_additional_local.npz',**existing,**newdrafts)
    train=load_selected('train_monitor500');print('Loaded train monitor cache',flush=True)
    for i,row in enumerate(train):targetrows.append({'role':'train_monitor500','index':i,'member':str(row['member']),**target_audit(row)})
    bytarget={r['index']:r for r in targetrows if r['role']=='dev_monitor500'}
    for r in done:r.update({k:bytarget[r['index']][k] for k in ['reference_mib_pair_logwh_loss','changed_area_fraction','oracle_overlap_fraction','oracle_broken_contact_edges']})
    done.sort(key=lambda r:r['index']);csvsave(HERE/'case_features.csv',done);csvsave(HERE/'target_audit.csv',targetrows)
    summary=statistics(done,targetrows);summary.update(status='complete',seconds=time.perf_counter()-started,source_hashes=identity,
        raw_metric_parity_count=len(done),training=False,solver_calls=0,test_read=False)
    assert all(sha(ROOT/p)==h for p,h in identity.items())
    save(HERE/'summary.json',summary);print('Audit complete: '+str(summary['seconds'])+' seconds',flush=True)

if __name__=='__main__':main()
