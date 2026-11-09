"""Read-only audit of source quality, frozen correction and inherited frames."""
import json,sys,time,csv,hashlib
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scipy.stats import spearmanr,rankdata
from experiments.E81_multisource_refinement.io import HERE,sha,save,sources
from experiments.E81_multisource_refinement.model import GeometryModel,decode
from experiments.E81_multisource_refinement.data import load_rows,collate
from experiments.E81_multisource_refinement.batches import inputs_only
from experiments.E81_multisource_refinement.train import seed_all
from experiments.E81_multisource_refinement.losses import WEIGHTS
OUT=HERE/'runs/20261029_placer_corrector_audit'
PATHS={
 'E42':ROOT/'experiments/E42_joint_regularization/runs/20261023_rectangle_w96d12_ab_seed0',
 'E74':ROOT/'experiments/E74_collision_32k_posttrain/runs/20261029_e42_topology_seed0',
 'E79':ROOT/'experiments/E79_center_refinement/runs/20261029_e78_center_seed0',
 'E80':ROOT/'experiments/E80_modular_transfer/runs/20261029_e42_frozen_e79_transfer',
 'E81':HERE/'runs/20261029_e42_multisource_seed0',
 'E82':ROOT/'experiments/E82_global_pressure_refinement/runs/20261029_e81_pressure_seed0'}
SOURCE={'E79':'E74','E80':'E42','E81':'E42','E82':'E42'}
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def scalar(v):
    v=float(v);return v if np.isfinite(v) else None
def mean(x):return float(np.mean(x))
def original_geometry(r):
    if 'loss_common_original_G' in r:return r['loss_common_original_G']
    return sum(weight*r['loss_'+term] for term,weight in WEIGHTS.items())
def correlation(a,b):return scalar(spearmanr(a,b).statistic) if np.std(a)>1e-12 and np.std(b)>1e-12 else None
def partial_rank(a,b,features):
    control=np.column_stack([np.ones(len(a)),*[rankdata([r[k] for r in features]) for k in ('n','pin_connected_fraction','connection_density')]])
    x,y=rankdata(a),rankdata(b)
    x=x-control@np.linalg.lstsq(control,x,rcond=None)[0];y=y-control@np.linalg.lstsq(control,y,rcond=None)[0]
    return scalar(np.corrcoef(x,y)[0,1]) if x.std()>1e-12 and y.std()>1e-12 else None
def geom(p,area):
    lo=p[:,:2];hi=lo+p[:,2:];b=float(np.prod(hi.max(0)-lo.min(0)))
    inter=np.maximum(np.minimum(hi[:,None],hi[None,:])-np.maximum(lo[:,None],lo[None,:]),0).prod(-1)
    return {'overlap':float(np.triu(inter,1).sum()/area),'bbox_over_required_area':b/area,
            'output_area_over_required':float(p[:,2:].prod(1).sum()/area),'unavoidable_overlap_lower_bound':max(0.,1-b/area)}
def win_counts(d,tol=1e-8):return [int((d< -tol).sum()),int((abs(d)<=tol).sum()),int((d>tol).sum())]
def bins(base,corrected):
    order=np.argsort(base,kind='stable');result=[];assignment=np.zeros(len(base),dtype=int)
    for level,ids in enumerate(np.array_split(order,5)):
        assignment[ids]=level;delta=corrected[ids]-base[ids]
        result.append({'quintile':level+1,'count':len(ids),'base_min':float(base[ids].min()),'base_max':float(base[ids].max()),
          'base_mean':mean(base[ids]),'corrected_mean':mean(corrected[ids]),'absolute_reduction_mean':mean(-delta),
          'relative_reduction_fraction':float(1-corrected[ids].sum()/base[ids].sum()),
          'corrected_overlap_gt5percent':int((corrected[ids]>.05).sum()),'better_equal_worse':win_counts(delta)})
    return result,assignment
def qgroup(rows,base_rows,ids):
    q=np.array([rows[i]['metrics']['cost'] for i in ids]);b=np.array([base_rows[i]['metrics']['cost'] for i in ids]);n=np.array([rows[i]['n'] for i in ids]);w=np.exp((n-n.max())/12)
    return {'count':len(ids),'corrected_pipeline_weighted_Q':float(np.average(q,weights=w)),
      'placer_only_pipeline_weighted_Q':float(np.average(b,weights=w)),'mean_paired_Q_delta':mean(q-b),'better_equal_worse':win_counts(q-b)}

def main():
    start=time.perf_counter();seed_all(0,4);OUT.mkdir(parents=True,exist_ok=True)
    plan=read(OUT/'plan.json');plan['figures']['models']=['E79','E80','E81','E82'];save(OUT/'plan.json',plan)
    state=torch.load(PATHS['E81']/'checkpoints/best.pt',map_location='cpu',weights_only=False,mmap=True)
    assert state['identity']['sources']==sources() and not state['identity']['implementation_check']
    model=GeometryModel(**state['model_config']).cuda().eval();model.load_state_dict(state['model']);model.requires_grad_(False)
    source_states={name:torch.load(PATHS[name]/'checkpoints/best.pt',map_location='cpu',weights_only=False,mmap=True) for name in ('E42','E74')}
    for name,module in (('E42',model.placer),('E74',model.alternative)):
        assert all(torch.equal(v.cpu(),source_states[name]['model'][k]) for k,v in module.state_dict().items())
    raw={name:read(path/'full_dev.json') for name,path in PATHS.items() if name!='E42'}
    old=read(ROOT/'experiments/E69_case_loss_statistics/runs/20261028_dev2000/forwards.json')
    assert old['identities']['E42']['checkpoint_sha256']==sha(PATHS['E42']/'checkpoints/best.pt')
    base_saved={'E42':old['models']['E42'],'E74':raw['E74']['records']}
    terminal={name:read(path/'terminal/results.json') for name,path in PATHS.items()}
    rows=load_rows('dev');members=[list(r['member']) for r in rows]
    for records in [*base_saved.values(),*[raw[n]['records'] for n in SOURCE]]:assert [r['member'] for r in records]==members
    features=read(ROOT/'experiments/E69_case_loss_statistics/runs/20261028_dev2000/features.json')['records'];assert [r['member'] for r in features]==members
    max_nodes=max(r['graph']['n'] for r in rows);coordinates={n:np.zeros((len(rows),max_nodes,4),np.float64) for n in ('E42','E74')};base={n:[] for n in coordinates}
    with torch.no_grad():
        for first in range(0,len(rows),32):
            selected=rows[first:first+32];g,_=collate(selected,'cuda');pub=inputs_only(g)
            for name,module in (('E42',model.placer),('E74',model.alternative)):
                p=(decode(module(pub),g)*g['scale'][:,None,None]).cpu().double().numpy()
                for j,row in enumerate(selected):
                    n=row['graph']['n'];p0=p[j,:n];inp=row['case'].solver_inputs;c=np.asarray(inp[5]);s=np.asarray(inp[6]);locked=c[:,:2].any(1)
                    p0[locked,2:]=s[locked,2:];p0[c[:,1]!=0]=s[c[:,1]!=0]
                    area=np.where(locked,s[:,2:].prod(1),np.asarray(inp[1])).sum();v=geom(p0,area)
                    assert abs(v['output_area_over_required']-1)<5e-6
                    coordinates[name][first+j,:n]=p0;base[name].append({'member':list(row['member']),'n':n,'required_area':float(area),**v})
            if (first//32)%15==0:print('Frozen public placer audit:',min(first+32,len(rows)),'/2000',flush=True)
            if time.perf_counter()-start>plan['max_numerical_seconds']:raise TimeoutError('Read-only audit wall limit')
    diffs={name:max(abs(a['overlap']-b['overlap_area_fraction']) for a,b in zip(base[name],base_saved[name])) for name in base}
    assert max(diffs.values())<5e-5,('Fresh public source differs from authenticated frozen metrics',diffs)
    np.savez_compressed(OUT/'source_drafts.npz',**coordinates,n=np.array([r['graph']['n'] for r in rows]))
    save(OUT/'source_metrics.json',{'members':members,'models':base,'coordinate_file':'source_drafts.npz'})
    lookup={tuple(m):i for i,m in enumerate(members)};panel_members=[r['member'] for r in terminal['E81']['records']]
    for name,doc in terminal.items():assert [r['member'] for r in doc['records']]==panel_members,name
    panel_ids=[lookup[tuple(m)] for m in panel_members]
    cache=ROOT/'history/20261026_flash_pro_consolidation/experiments/E51_module_overlap/runs/20261025_reference_E51_E42_draft100/e42_drafts_local'
    fixed_ids=[r['id'] for r in read(PATHS['E81']/'draft_comparison100/manifest.json')['cases']]
    parity=[]
    for index in fixed_ids[:5]:
        saved=read(cache/f'case_{index:04d}.json');i=panel_ids[index];assert saved['member']==members[i]
        assert saved['checkpoint_sha256']==sha(PATHS['E42']/'checkpoints/best.pt')
        delta=np.max(abs(coordinates['E42'][i,:base['E42'][i]['n']]-np.asarray(saved['positions'])))/np.sqrt(base['E42'][i]['required_area'])
        assert delta<3e-5;parity.append({'index':index,'max_normalized_coordinate_delta':float(delta)})
    results={};area_cert={}
    for name in base:
        ratios=np.array([r['bbox_over_required_area'] for r in base[name]]);ix=np.flatnonzero(ratios<1-1e-5)
        area_cert[name]={'count':len(ix),'minimum_B_over_A':float(ratios.min()),'below_99percent':int((ratios<.99).sum()),
            'members':[members[i] for i in ix],'max_unavoidable_pair_overlap_lower_bound':float(max(r['unavoidable_overlap_lower_bound'] for r in base[name]))}
    for name,source in SOURCE.items():
        x=np.array([r['overlap_area_fraction'] for r in base_saved[source]]);y=np.array([r['overlap_area_fraction'] for r in raw[name]['records']]);table,assignment=bins(x,y)
        original_g=np.array([original_geometry(r) for r in base_saved[source]])
        g_order=np.argsort(original_g,kind='stable');g_bins=[]
        for level,ix in enumerate(np.array_split(g_order,5)):
            g_bins.append({'quintile':level+1,'count':len(ix),'base_G_mean':mean(original_g[ix]),'base_overlap_mean':mean(x[ix]),'corrected_overlap_mean':mean(y[ix]),'absolute_reduction_mean':mean(x[ix]-y[ix])})
        results[name]={'source':source,'base_overlap':mean(x),'corrected_overlap':mean(y),'better_equal_worse':win_counts(y-x),
          'source_overlap_quintiles':table,'source_G_quintiles':g_bins,'source_vs_remaining_spearman':correlation(x,y),
          'partial_rank_n_pin_density':partial_rank(x,y,features),'source_G_vs_remaining_spearman':correlation(original_g,y),
          'Q_by_source_overlap_quintile':[qgroup(terminal[name]['records'],terminal[source]['records'],[j for j,i in enumerate(panel_ids) if assignment[i]==q]) for q in range(5)]}
    p42=np.array([r['overlap_area_fraction'] for r in base_saved['E42']]);p74=np.array([r['overlap_area_fraction'] for r in base_saved['E74']]);better=p74<p42-1e-6
    same_f={}
    for name in ('E79/E80','E81','E82'):
        if name=='E79/E80':a=np.array([r['overlap_area_fraction'] for r in raw['E80']['records']]);b=np.array([r['overlap_area_fraction'] for r in raw['E79']['records']])
        else:
            a=np.array([r['overlap_area_fraction'] for r in raw[name]['records']]);b=np.array([r['loss_P74_true_intersection'] for r in raw[name]['records']])
            assert max(abs(a-np.array([r['loss_true_intersection'] for r in raw[name]['records']])))<5e-5
        same_f[name]={'P74_source_better_raw_overlap_cases':int(better.sum()),'among_those_P74_corrected_better_equal_worse':win_counts((b-a)[better]),
          'P42_corrected_overlap':mean(a),'P74_corrected_overlap':mean(b),
          'source_difference_corrected_difference_spearman':correlation(p74-p42,b-a),
          'limits':'Same cases/shared frozen F; frame changes with source. E79 F trained on P74, later F trained on both. No new P74 postprocessing Q for F81/F82.'}
    result={'status':'numeric_audit_complete','train_updates':0,'solver_calls':0,'new_Q_evaluations':0,'test_read':False,
       'source_checkpoint_sha256':{n:sha(PATHS[n]/'checkpoints/best.pt') for n in ('E42','E74','E81','E82')},
       'source_metrics_match_saved_max_abs_overlap_error':diffs,'cached_E42_coordinate_parity_five':parity,
       'initial_frame_area_necessary_infeasibility':area_cert,'models':results,'same_case_shared_corrector':same_f,
       'source_result_sha256':{n:sha(PATHS[n]/'terminal/results.json') for n in PATHS},
       'limitations':['Quantile/correlation analyses are observational, not proof of causal irreparability or a universal failure threshold.',
          'B<A certifies no nonoverlapping placement can fit this inherited frame at mandatory areas; B>=A does not prove packability.',
          'Reference shapes precede public MIB area transform; reference positions used only for posthoc G and visualization, never input.',
          'ExistingQ is after postprocessing, not a raw neural-network score; raw/legalized legality remain distinct.'],
       'seconds':time.perf_counter()-start,'sources':sources()}
    save(OUT/'audit.json',result)
    with (OUT/'case_table.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['member','n','P42_overlap','P74_overlap','P42_B_over_A','P74_B_over_A','E79_overlap','E80_overlap','E81_overlap','E82_overlap']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for i,m in enumerate(members):w.writerow({'member':m[0]+':'+str(m[1]),'n':base['E42'][i]['n'],'P42_overlap':p42[i],'P74_overlap':p74[i],
            'P42_B_over_A':base['E42'][i]['bbox_over_required_area'],'P74_B_over_A':base['E74'][i]['bbox_over_required_area'],
            **{n+'_overlap':raw[n]['records'][i]['overlap_area_fraction'] for n in SOURCE}})
    print(json.dumps({'frames':{n:{k:v for k,v in z.items() if k!='members'} for n,z in area_cert.items()},
       'E81_bins':results['E81']['source_overlap_quintiles'],'same_f':same_f},ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
