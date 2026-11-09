"""Source, weight, geometry and artifact audit; never repeat a terminal run."""
import sys,json,csv,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from experiments.E83_strategy_attribution.io import HERE,sha,save,sources
from experiments.E83_strategy_attribution.prepare import parents,read
from experiments.E83_strategy_attribution.data import load_rows
from experiments.E83_strategy_attribution.analysis import geometry,OLD

def plain(x):
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [plain(v) for v in x]
    return x

def main():
    begin=time.perf_counter();plan=read(HERE/'plan.json');parent=parents()
    source=torch.load(ROOT/parent['F81']['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
    raw={};terminal={};tensor_checks={};immutable={}
    for arm in plan['arms']:
        folder=HERE/'runs'/arm['name'];frozen=read(folder/'frozen.json')
        assert frozen['sources']==sources() and frozen['parents']==parent
        assert sha(folder/'composition.pt')==frozen['checkpoint_sha256']
        state=torch.load(folder/'composition.pt',map_location='cpu',weights_only=False,mmap=True)
        placer=torch.load(ROOT/parent[arm['placer']]['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
        assert state['model_config']['learned_residuals']==arm['learned_residuals']
        fkeys=[k for k in state['model'] if k.startswith('refiner.')];pkeys=[k for k in state['model'] if k.startswith('placer.')]
        assert all(torch.equal(state['model'][k],source['model'][k]) for k in fkeys)
        assert all(torch.equal(state['model'][k],placer['model'][k[len('placer.'):]]) for k in pkeys)
        tensor_checks[arm['name']]={'F_tensors':len(fkeys),'placer_tensors':len(pkeys),'all_bit_exact':True,
          'active_learned_F':arm['learned_residuals'],'composition_sha256':frozen['checkpoint_sha256']}
        raw[arm['name']]=read(folder/'full_dev.json')['records'];terminal[arm['name']]=read(folder/'terminal/results.json')['records']
        assert len(raw[arm['name']])==2000 and len(terminal[arm['name']])==500
        assert not any('error' in r for r in terminal[arm['name']])
        for p in (folder/'full_dev.json',folder/'terminal/results.json',folder/'composition.pt'):immutable[p.relative_to(ROOT).as_posix()]=sha(p)
    result=read(HERE/'runs/analysis.json')
    for p,digest in result['evidence_sha256'].items():assert sha(ROOT/p)==digest
    # Authenticated old P74-view loss should match this new frozen composition.
    audit=ROOT/'experiments/E81_multisource_refinement/runs/20261029_placer_corrector_audit'
    denominators=list(csv.DictReader((audit/'area_denominators.csv').open(encoding='utf-8-sig')))
    oldraw=read(OLD['E81']/'full_dev.json')['records'];newraw=raw['P74_frozen_F81']
    expected=np.array([r['loss_P74_true_intersection'] for r in oldraw])*np.array([float(r['graph_loss_physical_area'])/float(r['public_area']) for r in denominators])
    errors=abs(expected-np.array([r['overlap_area_fraction'] for r in newraw]));assert errors.max()<2e-6
    rows=load_rows('dev_monitor500');spec_checks={}
    for name,rr in terminal.items():
        for row,r in zip(rows,rr):
            assert list(row['member'])==r['member'];inp=row['case'].solver_inputs;c=np.asarray(inp[5]);s=np.asarray(inp[6]);p=np.array(r['draft_positions'])
            locked=c[:,:2].any(1);pre=c[:,1]!=0;area=np.where(locked,s[:,2:].prod(1),inp[1])
            assert np.array_equal(p[locked,2:],s[locked,2:]) and np.array_equal(p[pre],s[pre])
            assert np.allclose(p[:,2:].prod(1),area,rtol=3e-6,atol=2e-5)
        spec_checks[name]={'count':500,'area_fixed_preplaced_preserved':True}
    manifest=read(HERE/'runs/gallery_manifest.json');fixed=[r['id'] for r in read(OLD['E81']/'draft_comparison100/manifest.json')['cases']]
    assert manifest['fixed100']==fixed and len(manifest['cases'])==100
    images=0
    for r in manifest['cases']:
        axes=[]
        for name,v in r['models'].items():
            assert sha(HERE/'runs'/v['image'])==v['sha256'];assert len(v['axes_bounds'])==4;axes.append(v['axes_bounds']);images+=1
        assert np.allclose(axes[0],axes[1],rtol=0,atol=1e-12)
    assert images==200
    old81={r['index']:r for r in read(OLD['E81']/'terminal/results.json')['records']}
    source74={r['index']:r for r in read(OLD['E74']/'terminal/results.json')['records']}
    cases={}
    for index in (450,164,443):
        cases[index]={}
        for name,rr in [('E81',list(old81.values())),*terminal.items()]:
            r=next(r for r in rr if r['index']==index);trace=r.get('trace',[])
            cases[index][name]={'Q':r['metrics']['cost'],'raw_overlap':r['draft_diagnostics']['overlap_area_fraction'],
              'rescue':r.get('rescue_used'),'failed_shape_candidates':r.get('failed_shape_candidate_count'),'shape_candidates':r.get('shape_candidate_count'),
              'hpwl_gap':r['metrics'].get('hpwl_gap'),'bbox_gap':r['metrics'].get('area_gap'),'grouping':r['metrics'].get('grouping_violations'),
              'phases':[t.get('phase') for t in trace]}
    report={'status':'numeric_and_artifact_audit_passed_pending_visual','tensor_identity':tensor_checks,'specs':spec_checks,
      'P74_full_F81_old_loss_max_abs_overlap_error':float(errors.max()),'gallery_images_verified':images,'fixed100_exact':True,
      'historical_hashes_unchanged':True,'source_frame_proxy_all500_inside':all(v['frame_containment']['count']==500 for v in result['models'].values()),
      'case_diagnostics':cases,'immutable_evidence_sha256':immutable,'model_forward_calls':0,'solver_calls':0,'new_Q_evaluations':0,
      'seconds':time.perf_counter()-begin,'test_read':False}
    save(HERE/'runs/audit_saved.json',plain(report));print(json.dumps(plain({'tensor_checks':tensor_checks,'cases':cases}),ensure_ascii=True,indent=2),flush=True)

if __name__=='__main__':main()
