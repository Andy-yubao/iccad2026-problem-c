"""Five unique train members, synthetic arithmetic, guarded updates and exact resume."""
import copy,gc,json,time
import numpy as np
import torch
from .io import HERE,ROOT,save,sources,sha,checkpoint
from .initialization import load,verify_protected,configure_trainability
from .model import GeometryModel
from .arms import model_config
from .data import load_rows,collate
from .batches import inputs_only,decode,sample,collate as basic_collate
from .simple_loss import components,public_components,PUBLIC
from .loss import losses,per_case
from .cuda_step import CapturedStep
from .train import seed_all,parser,run,rng_state,restore_rng,PlateauController

IDS=[1683,10027,5474,10140,15001]
def verify():
    v=json.loads((HERE/'runs/preflight/results.json').read_text(encoding='utf-8'))
    if v['status']!='passed' or v['sources']!=sources():raise ValueError('Passing current-source five-case preflight required')
    verify_protected()
    registration=json.loads((HERE/'runs/registered_loss.json').read_text(encoding='utf-8'))
    for path,digest in registration['protected_E87_checkpoints'].items():
        if sha(ROOT/path)!=digest:raise ValueError('Protected E87 checkpoint changed')
    from .loss_config import load_coefficients
    load_coefficients(HERE/'runs/registered_loss.json')
    return v

def same(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:same(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    else:assert a==b

def clear():
    from . import data
    data._resident.clear();gc.collect();torch.cuda.empty_cache()

def arithmetic():
    from .ports import public_ports
    from .metrics import module_overlap
    from scoring.official import official_evaluator
    def make(rects,groups=None):
        ref=torch.tensor(rects,dtype=torch.float32);n=len(ref);c=np.zeros((n,5),dtype=np.int64)
        if groups:c[:,3]=groups
        inp=(n,ref[:,2:].prod(-1).numpy(),np.empty((0,3)),np.empty((0,3)),np.empty((0,2)),c,np.full((n,4),-1.))
        row=sample(inp,ref);row.update(public_ports(inp,row['scale']))
        item=dict(graph=row,case=type('Fixture',(),{'solver_inputs':inp})(),member=('synthetic',0))
        g,_=collate([item],'cuda');return g,g['target'].clone().requires_grad_()
    g,s=make([[0,0,1,1],[.9,0,1,1]])
    v=components(s,g,dict(overlap=1.,task=1.));assert .0999<float(v['module_load_mean'])<.1001
    shifted=s.detach().clone();shifted[:,1,0]+=.01/g['scale'][0]
    assert float(components(shifted,g,dict(overlap=1.,task=1.))['overlap_total'])<float(v['overlap_total'])
    g,s=make([[0,0,1,1],[.2,0,1,1],[.2,0,1,1]])
    p=(decode(s,g)[0]*g['scale'][0]).detach().cpu().numpy();r=module_overlap(p)
    assert r['N10']>0 and r['covered_fractions'][0]<1
    np.testing.assert_allclose(r['covered_fractions'][0],.8,atol=2e-6)
    assert float(components(s,g,dict(overlap=1.,task=1.))['module_load_max'])>1
    g,s=make([[0,0,1,1],[1,1,1,1]],[1,1]);assert float(components(s,g,dict(overlap=1.,task=1.))['grouping'])>0
    g,s=make([[0,0,1,1],[1,0,1,1]],[1,1]);assert float(components(s,g,dict(overlap=1.,task=1.))['grouping'])<1e-8
    return dict(threshold_continuous=True,union_vs_pair_sum_distinguished=True,group_edge_vs_corner=True)

def main():
    begin=time.perf_counter();
    from .loss_config import load_coefficients
    load_coefficients(HERE/'runs/registered_loss.json')
    session=HERE/('runs/preflight/'+time.strftime('%Y%m%d_%H%M%S'));seed_all(0);protected=verify_protected();fixtures=arithmetic()
    rows=load_rows('train',IDS);raw=load_rows('train',IDS,use_cache=False)
    for a,b in zip(rows,raw):
        assert a['member']==b['member']
        for key,value in a['graph'].items():
            if isinstance(value,torch.Tensor):torch.testing.assert_close(value,b['graph'][key],rtol=0,atol=0)
    from .dataset_view import ReferenceLayouts
    for role in ('test','local_test','tail_test'):
        try:ReferenceLayouts(role)
        except PermissionError:pass
        else:raise AssertionError('Sealed role accepted')
    batch=[rows[i%5] for i in range(32)];g,_=collate(batch,'cuda')
    model=configure_trainability(GeometryModel(**model_config('simple')).cuda()).eval();parent=load(model,model_config('simple'))
    assert set(inputs_only(g)).issubset(set(('features','adjacency','mib','group','mask','area','known','specs','constraints','scale','valid')))
    try:model(g)
    except ValueError:pass
    else:raise AssertionError('Model accepted labels/full-port Loss tensors')
    with torch.no_grad():pred=model(inputs_only(g))
    state=pred.detach().requires_grad_();v=per_case(state,g,'simple')
    from .resident import ResidentRows
    padded=ResidentRows(batch,'cuda',max_nodes=120).batch(list(range(32)))
    with torch.no_grad():padded_state=model(inputs_only(padded));padded_values=per_case(padded_state,padded,'simple')
    for key in v:torch.testing.assert_close(padded_values[key],v[key],rtol=4e-5,atol=3e-5)
    from .loss_config import FIXED
    torch.testing.assert_close(v['total'],v['reference']+FIXED['overlap']*v['overlap_total']+FIXED['task']*v['task'])
    base=components(state,g,{k:x/3 for k,x in FIXED.items()})
    for key in v:
        if key!='total':torch.testing.assert_close(v[key],base[key],rtol=0,atol=0)
    torch.testing.assert_close(v['total']-v['reference'],3*(base['total']-base['reference']),rtol=1e-5,atol=1e-6)
    for name in ('simple_loss.py','model.py','features.py','batches.py','cuda_step.py','step_control.py','resident.py'):
        assert sha(HERE/name)==sha(ROOT/'experiments/E87_simple_supervised_placer'/name),name

    assert torch.equal(losses(state,g,arm='simple',epoch=0)['total'],losses(state,g,arm='simple',epoch=200)['total'])
    altered={**g,'target':g['target']+1};alt=per_case(state,altered,'simple')
    for k in ('overlap_total','task'):assert torch.equal(v[k],alt[k])
    assert not torch.equal(v['reference'],alt['reference'])
    try:public_components(state,{**{k:g[k] for k in PUBLIC},'target':g['target']})
    except ValueError:pass
    else:raise AssertionError('Public task Loss accepted labels')
    assert torch.isfinite(torch.autograd.grad(v['total'].mean(),state)[0]).all()
    # Per-case masks and arbitrary padding do not change the loss.
    for i,row in enumerate(rows):
        one,_=collate([row],'cuda')
        with torch.no_grad():out=model(inputs_only(one));solo=per_case(out,one,'simple')
        torch.testing.assert_close(solo['total'][0],v['total'][i],rtol=3e-5,atol=2e-5)
    official=__import__('scoring.official',fromlist=['official_evaluator']).official_evaluator();wire_error=0.
    pos=(decode(pred,g)*g['scale'][:,None,None]).cpu().numpy()
    for i,row in enumerate(rows):
        inp=row['case'].solver_inputs;n=inp[0];wl=official.calculate_hpwl_b2b(pos[i,:n].tolist(),inp[2])+official.calculate_hpwl_p2b(pos[i,:n].tolist(),inp[3],inp[4])
        w=sum(float(e[2]) for edges in (inp[2],inp[3]) for e in edges if float(e[2])>0)
        effective=float((g['area'][i]*g['valid'][i]).sum().sqrt()*g['scale'][i])
        wire_error=max(wire_error,abs(float(v['wirelength'][i])-wl/(effective*w)))
    assert wire_error<2e-5
    del state,v,alt,pred,model,parent,g,padded,padded_state,padded_values;clear()
    outputs=[]
    for arm in ('simple',):
        folder=session/f'{arm}_resume';whole=session/f'{arm}_whole'
        args=parser().parse_args(['--arm',arm,'--output',str(folder),'--max-hours','.25','--max-epochs','2'])
        run(args,implementation_check=True,check_rows=(batch,batch),stop_after_steps=1)
        first=torch.load(folder/'checkpoints/latest.pt',map_location='cpu',weights_only=False);clear()
        args.resume=True;run(args,implementation_check=True,check_rows=(batch,batch),stop_after_steps=2)
        resumed=torch.load(folder/'checkpoints/latest.pt',map_location='cpu',weights_only=False);clear()
        args.resume=False;args.output=whole;run(args,implementation_check=True,check_rows=(batch,batch),stop_after_steps=2)
        complete=torch.load(whole/'checkpoints/latest.pt',map_location='cpu',weights_only=False);clear()
        for key in ('model','optimizer','scheduler','graph_step_control'):same(resumed[key],complete[key])
        for key in ('torch','cuda'):same(resumed['rng'][key],complete['rng'][key])
        assert resumed['progress']['step']==2 and first['progress']['step']==1
        source=torch.load(ROOT/'experiments/E74_collision_32k_posttrain/runs/20261029_e42_topology_seed0/checkpoints/best.pt',map_location='cpu',weights_only=False)
        for key in source['model']:
            if '.local.' in key:assert torch.equal(source['model'][key],resumed['model'][key])
        outputs.append(dict(arm=arm,updates=2,dropout_adam_resume_bit_exact=True))
        del first,resumed,complete,source;clear()
    # Independently exercise actual nonfinite rejection and no-update capture.
    seed_all(0);m=configure_trainability(GeometryModel(**model_config('simple')).cuda()).train();load(m,model_config('simple'))
    g=ResidentRows(batch,'cuda',max_nodes=120).batch(list(range(32)))
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=0.)
    before={k:x.detach().clone() for k,x in m.state_dict().items()};step=CapturedStep(m,opt,g,'simple')
    assert all(torch.equal(before[k],x) for k,x in m.state_dict().items())
    step.inputs['target'].fill_(float('nan'))
    try:step.replay()
    except RuntimeError as e:assert 'Nonfinite' in str(e)
    else:raise AssertionError('Nonfinite Adam step accepted')
    assert all(torch.equal(before[k],x) for k,x in m.state_dict().items())
    assert verify_protected()==protected
    save(HERE/'runs/preflight/results.json',dict(status='passed',sources=sources(),ids=IDS,unique_real_train_cases=5,
        raw_cache_bit_exact=True,public_only_model=True,public_task_label_independent=True,fixed_auxiliary_x3=True,unweighted_components_bit_exact=True,core_model_loss_capture_sources_unchanged=True,
        fixed_total=True,padding_consistent=True,maximum120_padding_and_capture=True,actual_wirelength_max_error=wire_error,fixtures=fixtures,
        guarded_updates=outputs,no_warmup_update=True,nonfinite_rejected=True,parent_unchanged=True,
        seconds=time.perf_counter()-begin,formal_training_started=False,test_read=False))
    print('E90 five-case preflight passed',flush=True)

if __name__=='__main__':
    started=time.perf_counter()
    try:main()
    finally:
        from .budget import charge_preflight
        charge_preflight(time.perf_counter()-started)

