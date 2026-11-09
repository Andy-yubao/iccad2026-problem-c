"""Read saved drafts; optional five-case frozen-state Loss/gradient diagnosis."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
RUN=HERE/'runs/20261028_full_ports_taskloss_v1_seed0'
OUT=HERE/'runs/20261028_bbox_void_audit'
BASELINES={
    'E65':ROOT/'experiments/E65_geometry_messages/runs/20261027_geometry_message_seed0',
    'E68':ROOT/'experiments/E68_port_tokens/runs/20261028_full_ports_nosummary_v2_seed0',
    'E70':RUN,
}

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def member(r):return tuple(r['member'])

def geometry(p):
    """Exact axis-aligned rectangle union by x slabs; no raster approximation."""
    p=np.asarray(p,dtype=float);lo=p[:,:2];hi=lo+p[:,2:]
    xs=np.unique(np.r_[lo[:,0],hi[:,0]]);union=0.
    for left,right in zip(xs[:-1],xs[1:]):
        active=(lo[:,0]<right)&(hi[:,0]>left)
        y=np.column_stack((lo[active,1],hi[active,1]));y=y[np.argsort(y[:,0])]
        if not len(y):continue
        prev=np.r_[-np.inf,np.maximum.accumulate(y[:,1])[:-1]]
        length=np.maximum(0,y[:,1]-np.maximum(y[:,0],prev)).sum()
        union+=(right-left)*length
    area=p[:,2:].prod(1).sum();extent=hi.max(0)-lo.min(0);bbox=extent.prod()
    return {'A':float(area),'B':float(bbox),'U':float(union),
            'B_over_A':float(bbox/area),'U_over_A':float(union/area),
            'empty_over_A':float((bbox-union)/area),'empty_over_B':float((bbox-union)/bbox),
            'redundant_over_A':float((area-union)/area),
            'width':float(extent[0]),'height':float(extent[1])}

def summarize(records):
    keys=['B_over_A','U_over_A','empty_over_A','empty_over_B','redundant_over_A',
          'pair_overlap_over_A','bbox_over_reference','raw_hpwl_gap']
    return {k:{'mean':float(np.mean([r[k] for r in records])),
               'median':float(np.median([r[k] for r in records])),
               'p95':float(np.percentile([r[k] for r in records],95))} for k in keys}

def analyze(payload):
    label,r=payload;v=geometry(r['draft_positions']);m=r['raw_metrics']
    v.update(member=r['member'],index=r['index'],n=r['n'],
             pair_overlap_over_A=r['draft_diagnostics']['overlap_area_fraction'],
             bbox_over_reference=m['bbox_area']/m['bbox_area_baseline'],raw_hpwl_gap=m['hpwl_gap'])
    return label,v

def saved():
    OUT.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    feedback='这次的复杂重叠情况确实大幅度改善了，但似乎是以增大外界矩形面积为代价。而且模块并没有充分填补矩形内的空缺，就好像是通过牺牲矩形面积达成的重叠率下降。这显然不是我们想要的。'
    save(OUT/'feedback.json',{'date':'2026-10-28','model':'E70 best71','original_user_feedback':feedback,
         'interpretation_version':'bbox_void_v1','hypotheses':['E70 enlarges bbox versus saved E68/E65',
         'E70 has larger empty area','Overlap reduction is only a bbox-denominator artifact',
         'E70 compactness supplies no direct gradient to most interior blocks'],
         'status':'Post-hoc hypotheses; preserve original feedback; no ratings inferred'})
    # Analytic checks: disjoint, identical, containment and partial overlap.
    for p,expected in [([[0,0,1,1],[1,0,1,1]],2),([[0,0,1,1],[0,0,1,1]],1),
                       ([[0,0,3,3],[1,1,1,1]],9),([[0,0,2,2],[1,1,2,2]],7)]:
        assert abs(geometry(p)['U']-expected)<1e-10
    originals={label:read(path/'terminal/results.json')['records'] for label,path in BASELINES.items()}
    assert all({member(r) for r in v}=={member(r) for r in originals['E70']} for v in originals.values())
    values={k:[] for k in originals}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for label,row in pool.map(analyze,[(k,r) for k,v in originals.items() for r in v]):values[label].append(row)
    result={'date':'2026-10-28','workers':4,'scope':'Existing dev_monitor500 drafts only; no new inference',
            'formula':'A=sum block areas, B=bbox area, U=exact rectangle union; empty=B-U; redundant=A-U',
            'summary':{k:summarize(v) for k,v in values.items()},'paired':{},'records':values,'test_read':False}
    current={member(r):r for r in values['E70']}
    for label in ['E65','E68']:
        base={member(r):r for r in values[label]};pairs=[]
        for key,r in current.items():
            b=base[key];assert abs(r['A']/b['A']-1)<1e-5
            pairs.append({'member':list(key),'index':r['index'],'bbox_relative_change':r['B']/b['B']-1,
              **{k+'_delta':r[k]-b[k] for k in ['B_over_A','U_over_A','empty_over_A','empty_over_B','pair_overlap_over_A']}})
        result['paired'][label]={'count':len(pairs),'bbox_larger':sum(r['bbox_relative_change']>1e-8 for r in pairs),
          'empty_area_larger':sum(r['empty_over_A_delta']>1e-8 for r in pairs),
          'bbox_relative_change_mean':float(np.mean([r['bbox_relative_change'] for r in pairs])),
          'deltas':{k:float(np.mean([r[k] for r in pairs])) for k in pairs[0] if k.endswith('_delta')},'records':pairs}
    # Quantiles of E70/E68 bbox increase: a diagnostic sample, not a validation panel.
    ordered=sorted(result['paired']['E68']['records'],key=lambda r:r['bbox_relative_change'])
    chosen=[ordered[round(q*(len(ordered)-1))] for q in [.1,.3,.5,.7,.9]]
    result['five_probe_indices']=[r['index'] for r in chosen]
    result['elapsed_seconds']=time.perf_counter()-started
    save(OUT/'saved_geometry.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4.3),layout='constrained')
    labels=list(values)
    for ax,key,title in zip(axes,['B_over_A','empty_over_B','redundant_over_A'],
                            ['Bounding box / total block area','Empty space / bounding box','Redundant coverage / block area']):
        ax.boxplot([[r[key] for r in values[k]] for k in labels],tick_labels=labels,showfliers=False)
        ax.set_title(title,fontsize=10);ax.grid(axis='y',alpha=.25)
    fig.suptitle('Same 500 saved development drafts; exact rectangle union (no inference)')
    fig.savefig(OUT/'bbox_void.png',dpi=160);plt.close(fig)
    print(json.dumps({k:v for k,v in result.items() if k not in ['records','paired']},ensure_ascii=False))
    print('paired',json.dumps({k:{a:b for a,b in v.items() if a!='records'} for k,v in result['paired'].items()}))

def probe():
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    from .data import load_rows,collate
    from .batches import inputs_only,decode
    from .model import GeometryModel
    from .losses import components,WEIGHTS
    data=read(OUT/'saved_geometry.json');ids=data['five_probe_indices']
    rows=load_rows('dev_monitor500',ids,use_cache=False)
    checkpoint=RUN/'checkpoints/best.pt';sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=GeometryModel(**state['model_config']).eval();model.load_state_dict(state['model'])
    saved={r['index']:r for r in read(RUN/'terminal/results.json')['records']};records=[]
    for index,row in zip(ids,rows):
        assert list(row['member'])==saved[index]['member']
        g,_=collate([row],'cpu')
        with torch.no_grad():pred=model(inputs_only(g))
        pred=pred.detach().requires_grad_();loss=components(pred,g);p=(decode(pred,g)[0]*g['scale'][0]).detach().numpy()
        discrepancy=float(np.max(np.abs(p-np.array(saved[index]['draft_positions']))))
        center=pred[...,:2].detach();bboxcenter=((p[:,:2].min(0)+(p[:,:2]+p[:,2:]).max(0))/2)/float(g['scale'][0])
        radial=(center-torch.tensor(bboxcenter,dtype=center.dtype)[None,None,:])*g['mask'][...,:2]
        gradients={}
        for k,w in WEIGHTS.items():
            grad=torch.autograd.grad(w*loss[k].sum(),pred,retain_graph=True)[0];active=torch.linalg.vector_norm(grad[0,:,:2],dim=-1)>1e-9
            gradients[k]={'weighted_value':float(w*loss[k]),'weighted_state_gradient_norm':float(grad.norm()),
               'nonzero_center_gradient_blocks':int(active.sum()),'movable_blocks':int(g['mask'][0,:,0].sum()),
               'radial_expansion_derivative':float((grad[...,:2]*radial).sum())}
        counterfactuals=[]
        for alpha in [1.,.99,.98,.95,.90]:
            changed=pred.detach().clone();changed[...,:2]=center+(alpha-1)*radial
            with torch.no_grad():terms=components(changed,g);cp=(decode(changed,g)[0]*g['scale'][0]).numpy()
            counterfactuals.append({'center_scale':alpha,'terms':{k:float(v) for k,v in terms.items()},'geometry':geometry(cp)})
        records.append({'index':index,'member':list(row['member']),'n':row['graph']['n'],
          'max_saved_prediction_difference':discrepancy,'gradients':gradients,'counterfactuals':counterfactuals})
    result={'checkpoint_sha256':sha,'selection':'Five predeclared 10/30/50/70/90% quantiles of saved bbox increase versus E68',
      'records':records,'weights':WEIGHTS,'threads':4,'test_read':False,'training_updates':0,
      'limits':'Output-state gradients and center-only compression; WH, preplaced geometry preserved by decode. Not an optimizer update or trained alternative; five adaptive dev cases only.'}
    save(OUT/'five_case_probe.json',result)
    print(json.dumps({'checkpoint_sha256':sha,'cases':len(records),
       'indices':ids,'saved_prediction_max_difference':max(r['max_saved_prediction_difference'] for r in records),
       'records':[{'index':r['index'],'n':r['n'],'compactness_gradient':r['gradients']['compactness'],
          'compression':[{ 'alpha':c['center_scale'],'total':c['terms']['total'],
             'compactness':c['terms']['compactness'],'overlap':c['terms']['intersection_area'],
             'B_over_A':c['geometry']['B_over_A'],'empty_over_B':c['geometry']['empty_over_B']} for c in r['counterfactuals']]} for r in records]}))

def gallery42():
    previous=ROOT/'history/20261026_flash_pro_consolidation/experiments/E51_module_overlap/runs/20261025_reference_E51_E42_draft100/e42_drafts_local'
    data=read(OUT/'saved_geometry.json');manifest=read(RUN/'draft_comparison100/manifest.json')
    current={r['index']:r for r in data['records']['E70']}
    metric={r['index']:r for r in read(ROOT/'experiments/E42_joint_regularization/runs/20261023_rectangle_w96d12_ab_seed0/terminal/results.json')['records']}
    def one(case):
        old=read(previous/f'case_{case["id"]:04d}.json');r=current[case['id']]
        assert old['member']==r['member'] and old['checkpoint_sha256']=='12878108e9a58b8c195b314ba6586686c2a6e2796c1f4b6443e102b4db4224a6'
        p=np.array(old['positions']);v=geometry(p);m=metric[case['id']]['raw_metrics']
        assert metric[case['id']]['member']==r['member'] and abs(v['A']/r['A']-1)<1e-5
        lo=np.maximum(p[:,None,:2],p[None,:,:2]);hi=np.minimum((p[:,:2]+p[:,2:])[:,None,:],(p[:,:2]+p[:,2:])[None,:,:])
        overlap=np.triu(np.maximum(0,hi-lo).prod(-1),1).sum()/v['A']
        v.update(index=case['id'],member=r['member'],n=r['n'],pair_overlap_over_A=float(overlap),
           bbox_over_reference=m['bbox_area']/m['bbox_area_baseline'],raw_hpwl_gap=m['hpwl_gap'])
        return v
    with ThreadPoolExecutor(max_workers=4) as pool:old=list(pool.map(one,manifest['cases']))
    ids={r['index'] for r in old};values={k:[r for r in v if r['index'] in ids] for k,v in data['records'].items()};values['E42']=old
    oldmap={r['index']:r for r in old};new=values['E70']
    result={'scope':'Same fixed gallery100; saved E42 drafts only; no inference',
        'summary':{k:summarize(v) for k,v in values.items()},'records':values,
        'paired_E70_E42':{'count':len(new),'bbox_larger':sum(r['B']>oldmap[r['index']]['B'] for r in new),
           'empty_area_larger':sum(r['empty_over_A']>oldmap[r['index']]['empty_over_A'] for r in new),
           'bbox_relative_increase_mean':float(np.mean([r['B']/oldmap[r['index']]['B']-1 for r in new]))},'test_read':False}
    save(OUT/'gallery100_E42.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='records'}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--probe',action='store_true');parser.add_argument('--gallery42',action='store_true');args=parser.parse_args()
    if args.probe:probe()
    elif args.gallery42:gallery42()
    else:saved()
