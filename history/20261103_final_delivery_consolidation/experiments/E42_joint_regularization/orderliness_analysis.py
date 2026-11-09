"""Saved evidence for the first two orderliness questions; no network calls."""
import numpy as np
from scipy.stats import spearmanr
from .orderliness import OUT,OLD,REVIEW,ROOT,HERE,read,save,geometry,verify


def pair_bound(p,c):
    centers=p[:,:2]+p[:,2:]/2;local=np.sqrt(p[:,2:].prod(1));pre=c[:,1]!=0
    worst=(0.,None,None)
    for i in range(len(p)):
        for j in range(i+1,len(p)):
            choices=[]
            for a in (0,1):
                half=(p[i,a+2]+p[j,a+2])/2
                for l,r in ((i,j),(j,i)):
                    needed=half-(centers[r,a]-centers[l,a])
                    if needed<=0:choices.append((0.,l,r,a));continue
                    def possible(t):
                        left=0. if pre[l] else min(p[l,a],t*local[l])
                        right=0. if pre[r] else t*local[r]
                        return left+right>=needed
                    if not possible(100.):continue
                    lo,hi=0.,100.
                    for _ in range(55):
                        mid=(lo+hi)/2
                        if possible(mid):hi=mid
                        else:lo=mid
                    choices.append((hi,l,r,a))
            bound,_,_,_=min(choices) if choices else (float('inf'),0,0,0)
            if bound>worst[0]:worst=(bound,[i,j],min(choices) if choices else None)
    b,ids,choice=worst
    return {'lower_bound_max_local_move':float(b),'blocking_pair':ids,
            'best_axis_for_pair':list(choice[1:]) if choice else None,
            'scope':'necessary lower bound with original fixed shapes, nonnegative coordinates and preplaced anchors; no bbox/wire or relative-order restriction'}


def main():
    from datasets.public_member_inputs import public_member_inputs
    rows=[read(p) for p in (OUT/'repair_local').glob('*.json')]
    manifest=read(REVIEW/'manifest.json');lookup={r['id']:r for r in manifest['cases']}
    finalfolder=HERE/'runs/20261024_final_test1000_seed20261024';config=read(finalfolder/'config.json')
    detail=[]
    for row in rows:
        p=np.asarray(read(ROOT/row['source_draft'])['raw_positions']);c=np.asarray(row['constraints'])
        bound=pair_bound(p,c)
        sourceindex=lookup[row['id']]['source_index'];saved=read(finalfolder/f'cases/{sourceindex:04d}.json')
        inputs=public_member_inputs(config['records'][sourceindex],'local_test',authorize_local_test=True)
        final=verify(p,np.asarray(saved['positions'],float),inputs)
        detail.append({'id':row['id'],'score':row['annotation'].get('score'),'n':row['n'],'necessary_pair_bound':bound,
                       'saved_production_final_witness':final,
                       'note':'Saved production solution is a constructive endpoint, not a minimal repair; shapes may change and quality costs are shown explicitly'})
    rated=[r for r in detail if r['score']]
    summary={'count':len(detail),'rated_count':len(rated),'by_grade':{},'detail':detail,
             'thresholds':[.02,.05,.1,.2,.5],
             'key_scope':'fixed-shape bounds do not rule out tiny coordinated shape changes; capped model certificates are narrower than global layout feasibility'}
    for s in (1,2,3):
        group=[r for r in rated if r['score']==s]
        summary['by_grade'][str(s)]={'n':len(group),'median_pair_lower_bound':float(np.median([r['necessary_pair_bound']['lower_bound_max_local_move'] for r in group])),
            'pair_lower_bound_exceeds_half_local_side':sum(r['necessary_pair_bound']['lower_bound_max_local_move']>.5+1e-6 for r in group),
            'median_production_mean_move_free':float(np.median([r['saved_production_final_witness']['mean_move_free'] for r in group])),
            'median_production_max_move':float(np.median([r['saved_production_final_witness']['max_move'] for r in group]))}
    savedrho=float(spearmanr([r['saved_production_final_witness']['mean_move_free'] for r in rated],[r['score'] for r in rated]).statistic)
    summary['production_mean_move_score_spearman']=savedrho
    summary['repair_trial_counts']={}
    for arm in ('fixed_relation','flexible_relation','joint_small_shape'):
        trials=[next(t for t in r['trials'] if t['name']==arm) for r in rows]
        summary['repair_trial_counts'][arm]={'hard_witness':sum(t.get('verification',{}).get('hard_legal',False) for t in trials),
            'guarded_witness':sum(t.get('verification',{}).get('hard_legal',False) and t.get('verification',{}).get('quality_guard',False) for t in trials),
            'infeasible_within_stated_subproblem':sum(t['solver']['status'] in (2,'PrimalInfeasible') for t in trials),
            'unknown':sum(t['solver']['status'] not in (0,2,'Solved','AlmostSolved','PrimalInfeasible') for t in trials)}
    save(OUT/'stage1_summary.json',summary)
    # Fixed calipers from plan. No adaptive widening or assignment by outcome.
    candidates=[]
    for a in rows:
        if not a['annotation'].get('score'):continue
        for b in rows:
            if a['id']>=b['id'] or not b['annotation'].get('score') or a['annotation']['score']==b['annotation']['score']:continue
            dn=abs(a['n']-b['n']);do=abs(a['pileup']['pair_overlap_fraction']-b['pileup']['pair_overlap_fraction'])
            dp=abs(a['input_features']['preplaced_fraction']-b['input_features']['preplaced_fraction'])
            if dn<=20 and do<=.01 and dp<=.05:
                cost=dn/20+do/.01+dp/.05
                candidates.append({'ids':[a['id'],b['id']],'scores':[a['annotation']['score'],b['annotation']['score']],
                    'module_counts':[a['n'],b['n']],'pair_overlap_fractions':[a['pileup']['pair_overlap_fraction'],b['pileup']['pair_overlap_fraction']],
                    'preplaced_fractions':[a['input_features']['preplaced_fraction'],b['input_features']['preplaced_fraction']],
                    'coverage_energies':[a['pileup']['coverage_energy'],b['pileup']['coverage_energy']],
                    'complex_energies':[a['pileup']['complex_energy_10pct'],b['pileup']['complex_energy_10pct']],
                    'matching_cost':cost})
    disjoint=[];used=set()
    for pair in sorted(candidates,key=lambda p:(p['matching_cost'],p['ids'])):
        if not (set(pair['ids'])&used):disjoint.append(pair);used.update(pair['ids'])
    stage2={'calipers':{'n':20,'overlap_fraction':.01,'preplaced_fraction':.05},'eligible_pairs':candidates,
            'disjoint_pairs':disjoint,'no_caliper_relaxation':True,
            'interpretation':'matching on three observables cannot remove netlist, shapes, grouping and family confounds; insufficient pairs are a result, not a reason to widen after looking'}
    save(OUT/'stage2_matching.json',stage2)
    print({'stage1':summary['by_grade'],'repair_trials':summary['repair_trial_counts'],'production_mean_move_rho':savedrho})
    print({'stage2_eligible':len(candidates),'stage2_disjoint':len(disjoint),'pairs':disjoint})


if __name__=='__main__':main()
