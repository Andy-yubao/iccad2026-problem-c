"""At most five unique training members; real CUDA/Adam/Dropout resume check."""
import gc,time,json,importlib.util,subprocess,sys
import torch
from .io import HERE,ROOT,save,sources
from .arms import model_config
from .initialization import load,verify_protected
from .model import GeometryModel,decode
from .data import load_rows,collate
from .batches import inputs_only
from .frame import predicted_frame
from .cooperation import spatial_centers
from .loss import losses,per_case
from .train import seed_all,parser,run
from .cuda_step import CapturedStep

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

def geometry(s,g,base):
    p=decode(s,g);lo,hi=predicted_frame(base,g);v=g['valid']
    assert bool((p[...,:2]>=lo[:,None]-3e-6).masked_fill(~v[...,None],True).all())
    assert bool((p[...,:2]+p[...,2:]<=hi[:,None]+3e-6).masked_fill(~v[...,None],True).all())
    torch.testing.assert_close(p[...,2:].prod(-1)[v],g['area'][v],rtol=4e-6,atol=3e-8)
    placed=(~g['mask'][...,0].bool())&v;locked=(~g['mask'][...,2].bool())&v
    assert torch.equal(p[placed],g['specs'][placed]);assert torch.equal(p[...,2:][locked],g['specs'][...,2:][locked])

def synthetic():
    g=dict(area=torch.ones(1,3,device='cuda'),valid=torch.ones(1,3,device='cuda',dtype=torch.bool),mask=torch.ones(1,3,3,device='cuda'),known=torch.zeros(1,3,3,device='cuda'),specs=torch.zeros(1,3,4,device='cuda'))
    s=torch.tensor([[[1.,.5,0.],[1.8,.5,0.],[2.8,.5,0.]]],device='cuda',requires_grad=True)
    frame=(torch.tensor([[0.,0.]],device='cuda'),torch.tensor([[4.,1.]],device='cuda'))
    one=spatial_centers(s,s[...,:2],g,frame,steps=1);eight=spatial_centers(s,s[...,:2],g,frame)
    assert float(one[0,2,0])==float(s[0,2,0]);assert float(eight[0,2,0])>float(s[0,2,0])+1e-4
    assert float((eight-s[...,:2]).abs().max())<=.150001
    grad=torch.autograd.grad(eight.square().sum(),s)[0];assert torch.isfinite(grad).all() and bool((grad!=0).any())
    # Simultaneous force is permutation equivariant when geometry has no ties.
    perm=torch.tensor([2,0,1],device='cuda');gg={k:v[:,perm] for k,v in g.items()}
    permuted=spatial_centers(s[:,perm],s[:,perm,:2],gg,frame)
    torch.testing.assert_close(permuted,eight[:,perm],rtol=1e-6,atol=1e-6)
    return dict(neighbour_pressure_propagates_beyond_initial_collision=True,finite_ordinary_gradient=True,permutation_without_ties=True,steps=8,one=one.detach().cpu().tolist(),eight=eight.detach().cpu().tolist())

def main():
    begin=time.perf_counter();seed_all(0);protected=verify_protected();syn=synthetic()
    ids=json.loads((HERE/'plan.json').read_text(encoding='utf-8'))['preflight_train_ids']
    rows=load_rows('train',ids,use_cache=False);cached=load_rows('train',ids)
    for raw,packed in zip(rows,cached):
        assert raw['member']==packed['member']
        for k,v in raw['graph'].items():
            if isinstance(v,torch.Tensor):same(v,packed['graph'][k])
            else:assert v==packed['graph'][k]
    from .dataset_view import ReferenceLayouts
    for role in ('test','local_test','tail_test'):
        try:ReferenceLayouts(role);raise AssertionError('Sealed role accepted')
        except PermissionError:pass
    commands=json.loads(subprocess.check_output([sys.executable,'-B','-m','experiments.E85_cooperative_space_refinement.pipeline','--dry-run'],cwd=ROOT,text=True))
    for _,command in commands:assert importlib.util.find_spec(command[command.index('-m')+1]) is not None
    batch=[rows[i%5] for i in range(32)];g,_=collate(batch,'cuda');pub=inputs_only(g)
    initial={};records={}
    for arm in ('control','cooperative'):
        seed_all(0);model=GeometryModel(**model_config(arm)).cuda().eval();load(model,model_config(arm))
        weights={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if initial:
            for k,v in weights.items():same(v,initial[k])
        else:initial=weights
        assert sum(p.numel() for p in model.parameters())==6071486
        assert sum(p.numel() for p in model.parameters() if p.requires_grad)==241547
        with torch.no_grad():
            prediction=model(pub);geometry(prediction,g,model.placer(pub))
            v=per_case(prediction,g,arm);a=losses(prediction,g,None,arm,0);b=losses(prediction,g,None,arm,200)
            assert all(torch.equal(a[k],b[k]) for k in a)
        try:model(g);raise AssertionError('Labels accepted')
        except ValueError:pass
        records[arm]=[dict(train_index=index,total=float(v['total'][i]),I=float(v['true_intersection'][i])) for i,index in enumerate(ids)]
        if arm=='cooperative':
            for before,after in zip(records['control'],records['cooperative']):assert after['I']<=before['I']+2e-7
        model.train();assert not model.placer.training and model.refiner.training
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=0,weight_decay=3e-4)
        cap=CapturedStep(model,opt,g,arm);cap.replay()
        direct=losses(cap.prediction,cap.inputs,None,arm,0)
        for k in direct:torch.testing.assert_close(cap.values[k],direct[k],rtol=1e-5,atol=1e-6)
        assert all(p.grad is None for p in model.placer.parameters())
        assert all(torch.equal(v.cpu(),initial[k]) for k,v in model.state_dict().items())
        cap.inputs['target'].fill_(float('nan'))
        try:cap.replay();raise AssertionError('Nonfinite accepted')
        except RuntimeError as error:assert 'Nonfinite' in str(error)
        assert all(torch.equal(v.cpu(),initial[k]) for k,v in model.state_dict().items())
        del cap,opt,model,prediction,v,a,b,direct;clear()
        resumed=HERE/('runs/preflight/'+arm+'_resumed');whole=HERE/('runs/preflight/'+arm+'_whole')
        args=parser().parse_args(['--arm',arm,'--output',str(resumed),'--max-hours','.15','--max-epochs','2'])
        run(args,implementation_check=True,check_rows=(batch,batch),stop_after_steps=1)
        first=torch.load(resumed/'checkpoints/latest.pt',map_location='cpu',weights_only=False);clear()
        for stage in ('first','second','third'):assert any(not torch.equal(first['model'][k],initial[k]) for k in first['model'] if k.startswith('refiner.'+stage+'.'))
        args.resume=True;run(args,implementation_check=True,check_rows=(batch,batch),stop_after_steps=2)
        second=torch.load(resumed/'checkpoints/latest.pt',map_location='cpu',weights_only=False);clear()
        args.resume=False;args.output=whole;run(args,implementation_check=True,check_rows=(batch,batch),stop_after_steps=2)
        full=torch.load(whole/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
        for key in ('model','optimizer','scheduler','graph_step_control'):same(second[key],full[key])
        for key in ('torch','cuda'):same(second['rng'][key],full['rng'][key])
        assert second['progress']['step']==full['progress']['step']==2
        assert second['elapsed_seconds']>first['elapsed_seconds']
        for k in full['model']:
            if k.startswith('placer.'):same(full['model'][k],initial[k])
        model=GeometryModel(**model_config(arm)).cuda().eval();model.load_state_dict(full['model'])
        with torch.no_grad():geometry(model(pub),g,model.placer(pub))
        del model,first,second,full;clear()
    assert verify_protected()==protected
    save(HERE/'runs/preflight/results.json',dict(status='passed',indices=ids,unique_train_cases=5,synthetic=syn,initial_metrics=records,matched_initial_weights_and_parameter_count=True,frozen_E74_bit_exact=True,public_only=True,fixed_Loss_epoch_invariant=True,cache_parity=True,test_roles_rejected=True,area_specs_frame_after_real_updates=True,CUDA_Loss_consistent=True,nonfinite_rejected_before_update=True,real_Dropout_Adam_RNG_cursor_resume_bit_exact_both_arms=True,sources=sources(),seconds=time.perf_counter()-begin,test_read=False,formal_training_started=False))
    print('E85 FIVE-CASE PREFLIGHT PASSED',flush=True)
if __name__=='__main__':main()
