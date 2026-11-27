"""Step4: five blind left/right comparisons, prepared from saved evidence."""
import os
os.environ['MPLBACKEND']='Agg'
import random
from pathlib import Path
import numpy as np
from .orderliness import OUT,OLD,ROOT,REVIEW,read,save,sha,geometry,pileup
from visualization.layouts import plot_case,geometry_details


def overlap_fraction(p):
    lo=p[:,:2];hi=lo+p[:,2:];intersection=np.maximum(np.minimum(hi[:,None],hi[None])-np.maximum(lo[:,None],lo[None]),0.).prod(-1)
    return float((intersection.sum()-np.trace(intersection))/2/p[:,2:].prod(1).sum())


def perturb(p,c,seed):
    rng=np.random.default_rng(seed);free=c[:,1]==0;local=np.sqrt(p[:,2:].prod(1));centers=p[:,:2]+p[:,2:]/2
    groups={}
    cuts=np.quantile(centers[:,0],[.25,.5,.75])
    for i in np.flatnonzero(free):
        key=('specified_group',int(c[i,3])) if c[i,3]>0 else ('spatial_band',int(np.searchsorted(cuts,centers[i,0])))
        groups.setdefault(key,[]).append(i)
    coherent=p.copy()
    for ids in groups.values():
        direction=rng.uniform(.1,1.,2);direction/=direction.sum()
        coherent[ids,:2]+=.05*local[ids].min()*direction
    target_overlap=overlap_fraction(coherent)
    target_move=float((np.abs(coherent[:,:2]-p[:,:2]).sum(1)/local)[free].mean())
    trials=[]
    for trial in range(128):
        direction=rng.normal(size=(len(p),2));direction/=np.abs(direction).sum(1)[:,None]
        amount=.05*local*rng.uniform(.5,1.,len(p));amount[~free]=0.
        delta=direction*amount[:,None]
        for _ in range(4):
            candidate=p.copy();candidate[:,:2]=np.maximum(0.,p[:,:2]+delta)
            avg=float((np.abs(candidate[:,:2]-p[:,:2]).sum(1)/local)[free].mean())
            delta*=target_move/max(avg,1e-12)
            load=np.abs(delta).sum(1)/local
            delta*=np.minimum(1.,.05/np.maximum(load,1e-12))[:,None]
        candidate=p.copy();candidate[:,:2]=np.maximum(0.,p[:,:2]+delta)
        avg=float((np.abs(candidate[:,:2]-p[:,:2]).sum(1)/local)[free].mean())
        difference=abs(overlap_fraction(candidate)-target_overlap)
        trials.append((difference+abs(avg-target_move),difference,avg,trial,candidate))
    _,difference,mean,trial,independent=min(trials,key=lambda x:x[0])
    return coherent,independent,{'seed':seed,'candidate_count':128,'selected_by':'closest total overlap fraction and free-module mean movement only; no visual-score selection',
        'coherent_overlap_fraction':target_overlap,'independent_overlap_fraction':overlap_fraction(independent),
        'absolute_overlap_fraction_difference':difference,'matched_within_0_001':difference<=.001,
        'coherent_mean_move_free':target_move,'independent_mean_move_free':mean,'selected_trial':trial,
        'max_move_cap_local':.05,'areas_and_shapes_unchanged':True,'preplaced_unchanged':True}


def main():
    if not (OUT/'stage3_summary.json').exists():raise RuntimeError('Complete step3 before preparing step4')
    folder=OUT/'comparison_review';folder.mkdir(exist_ok=True);(folder/'images').mkdir(exist_ok=True)
    (OUT/'counterfactual_local').mkdir(exist_ok=True)
    if (folder/'manifest.json').exists():
        print('Existing immutable comparison round retained.');return
    pairs=[]
    for key in ('case_0063','case_0076','case_0602'):
        row=read(OLD/'cases_local'/f'{key}.json');p=np.asarray(read(ROOT/row['source_draft'])['raw_positions']);c=np.asarray(row['constraints_for_plot'])
        k=next(i for i,v in enumerate(row['probe_curve']) if v['max_move_local']==.05 and v['max_log_shape']==.05)
        probe=row['probe_curve'][k];changed=np.asarray(probe['positions'])
        pairs.append({'source_case':key,'n':len(p),'constraints':c,'layouts':[p,changed],
            'coordinate_sources':[row['source_draft']+'#raw_positions',(OLD/f'cases_local/{key}.json').relative_to(ROOT).as_posix()+f'#probe_curve/{k}/positions'],
            'kind':'original_vs_bounded_joint_adjustment','metrics':probe['strict']})
    for seed,key in ((101,'case_0088'),(202,'case_0538')):
        row=read(OLD/'cases_local'/f'{key}.json');p=np.asarray(read(ROOT/row['source_draft'])['raw_positions']);c=np.asarray(row['constraints_for_plot'])
        coherent,independent,matching=perturb(p,c,seed)
        paths=[]
        for label,positions in [('coherent',coherent),('independent',independent)]:
            file=OUT/'counterfactual_local'/f'{key}_{label}.json'
            save(file,{'source_draft':row['source_draft'],'kind':label,'positions':positions.tolist(),'matching':matching})
            paths.append(file.relative_to(ROOT).as_posix()+'#positions')
        pairs.append({'source_case':key,'n':len(p),'constraints':c,'layouts':[coherent,independent],
            'coordinate_sources':paths,'kind':'coherent_vs_independent_small_motion','metrics':matching})
    rng=random.Random(20261024);cases=[]
    for i,pair in enumerate(pairs,1):
        swap=rng.choice([False,True]);indices=[1,0] if swap else [0,1];left,right=[pair['layouts'][k].tolist() for k in indices]
        c=pair['constraints'].tolist();name=f'pair_{i:02d}'
        adapted={'panel':'Visual comparison','case_id':name,'constraints':c,
            'caption':'Same input and module IDs. Left/right order randomized independently for each pair.',
            'variants':{'left':{'positions':left,'soft_details':geometry_details(left,c)},'right':{'positions':right,'soft_details':geometry_details(right,c)}}}
        image=folder/'images'/f'{name}.png'
        axes=plot_case(adapted,image,stages=(('left','A'),('right','B')),show_metrics=False)
        cases.append({'id':name,'n':pair['n'],'member':None,'image':f'images/{name}.png','image_sha256':sha(image),'axes_bounds':axes['axes_bounds'],
            'auxiliary':{'来源类型':pair['kind'],'原例':pair['source_case'],'A来源':pair['coordinate_sources'][indices[0]],'B来源':pair['coordinate_sources'][indices[1]],
                         '数值核对':pair['metrics'],'说明':'辅助资料会暴露变体身份；建议先完成观察再展开'},
            'provenance':{'case':pair['source_case'],'kind':pair['kind'],'swapped':swap,'coordinate_sources':pair['coordinate_sources']}})
    manifest={'schema_version':1,'session_id':'E42_orderliness_pairwise_v2','title':'整齐程度 · 五组前后对照',
        'display_note':'每组输入和模块编号一致，A/B顺序已随机固定。前三组比较小幅联合调整，后两组比较相近重叠量的协同/独立微扰。无需补完上一轮100张。',
        'view_stages':[{'key':'left','label':'放大 A'},{'key':'right','label':'放大 B'}],
        'image_description':'同一输入的A和B两幅布局对照',
        'rubric':{'version':2,'target':'pairwise_orderliness','target_label':'先判断哪幅整体更整齐',
            'prompt':'A 和 B，哪幅更整齐？','instructions':'1 A更整齐 · 2 差不多 · 3 B更整齐。下方两个观察维度可选填；选完后打主分会一起保存。',
            'scale':{'1':'A 更整齐','2':'差不多','3':'B 更整齐'},
            'questions':{
                'repairability':{'prompt':'哪边更容易小幅调整到合法？','scale':{'1':'A','2':'相近','3':'B'}},
                'whitespace':{'prompt':'哪边的留白更合理？','scale':{'1':'A','2':'相近','3':'B'}}}},
        'selection':{'seed':20261024,'count':5,'criterion':'fixed three joint-witness cases and two high-rated same-input perturbation cases; exploratory, not score validation'},
        'cases':cases}
    save(folder/'manifest.json',manifest)
    save(OUT/'stage4_preparation.json',{'status':'awaiting_human_feedback','count':5,'manifest_sha256':sha(folder/'manifest.json'),
        'source_sha256':sha(__file__),'human_ratings_inserted_by_assistant':0,
        'matching':[pair['metrics'] for pair in pairs if pair['kind'].startswith('coherent')],
        'old_scores_unchanged':True,'new_rubric_version':2,'new_sealed_members_accessed':False})
    print('Five pairwise comparison images ready; human feedback is pending.')


if __name__=='__main__':main()
