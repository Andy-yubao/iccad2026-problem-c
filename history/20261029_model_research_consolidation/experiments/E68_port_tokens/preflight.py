"""Five unique training cases only; short implementation updates, no formal training."""
import gc
import hashlib
import json
import time
import torch
from .io import HERE,ROOT,save,sha,sources
from .initialization import verify_protected,metadata
from .arms import ARMS,model_config
from .model import GeometryModel
from .base_model import GeometryModel as OriginalModel
from .batches import inputs_only,collate as batch_collate
from .data import load_rows
from .resident import ResidentRows
from .loss import losses
from .loss_reference import components as original_components,WEIGHTS
from .cuda_step import CapturedStep
from .train import seed_all,parser,run,selection


def same(a,b,path='state'):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b),path
    elif isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for k in a:same(a[k],b[k],path+'/'+str(k))
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):same(x,y,path+'/'+str(i))
    else:assert a==b,path


def clear():
    from . import data
    data._resident.clear();gc.collect();torch.cuda.empty_cache()


def check_arm(arm,batch):
    seed_all(0);model=GeometryModel(**model_config(arm)).cuda().eval();pack=ResidentRows(batch,'cuda',120)
    g=pack.batch(list(range(32)));public=inputs_only(g)
    from .frozen_batches import collate as frozen_collate
    old_batch=frozen_collate([r['graph'] for r in batch],list(range(32)),'cuda')
    new_batch=batch_collate([r['graph'] for r in batch],list(range(32)),'cuda')
    for k in old_batch:assert torch.equal(old_batch[k],new_batch[k]),'Frozen batch changed: '+k
    pw=g['pin_weights'];mean=(pw[...,None]*g['pin_coordinates']).sum(2)/pw.sum(2,keepdim=True).clamp_min(1e-6)
    torch.testing.assert_close(mean,g['features'][...,15:17],rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(pw.sum(2).log1p(),g['features'][...,14],rtol=1e-5,atol=1e-6)
    base=OriginalModel(width=96,layers=12,rectangle=True,ffn_dropout=.05).cuda().eval()
    base.input[0]=torch.nn.Linear(14,96).cuda()
    base.load_state_dict({k:v for k,v in model.state_dict().items() if not k.startswith('port_readers.')})
    base_public={k:v for k,v in public.items() if not k.startswith('pin_')}
    with torch.no_grad():
        actual=model(public);expected=base(base_public)
        assert torch.equal(actual,expected),'Zero projection must exactly preserve base forward'
        poisoned={**g,'features':g['features'].clone()};poisoned['features'][...,14:]=1e8
        assert torch.equal(actual,model(inputs_only(poisoned))),'Removed summaries must have no route into the network'
    try:model({**public,'features':g['features']})
    except ValueError:pass
    else:raise AssertionError('Network must reject legacy17-channel inputs')
    # Verify that row normalization did not discard the original b2b magnitudes.
    for i,row in enumerate(batch[:5]):
        n=row['graph']['n'];raw=torch.zeros(n,n,device='cuda')
        for a,b,w in row['case'].solver_inputs[2]:
            if float(w)>0:raw[int(a),int(b)]+=float(w);raw[int(b),int(a)]+=float(w)
        reconstructed=g['adjacency'][i,:n,:n]*g['features'][i,:n,13].expm1()[:,None]
        torch.testing.assert_close(raw,reconstructed,rtol=1e-5,atol=1e-5)
    try:model({**public,'target':g['target']})
    except ValueError:pass
    else:raise AssertionError('Supervision inputs must be rejected')
    state=actual.detach().requires_grad_();values=losses(state,g,None,arm)
    old=original_components(state,g);target=sum(WEIGHTS[k]*old[k].mean() for k in WEIGHTS)
    grad=torch.autograd.grad(values['total'],state,retain_graph=True)[0];oldgrad=torch.autograd.grad(target,state)[0]
    torch.testing.assert_close(values['total'],target,rtol=1e-6,atol=1e-6)
    torch.testing.assert_close(grad,oldgrad,rtol=1e-5,atol=1e-6)
    assert selection(values) is values['total']
    assert torch.equal(values['total'],losses(state,g,None,arm,200)['total'])
    # Exercise nonlinear token reader with nonzero output weights.
    branch=model.port_readers[0]
    with torch.no_grad():
        assert public['features'].shape[-1]==14 and model.input[0].in_features==14
        branch.output.weight.fill_(.001);hidden=model.input(public['features'])
        a=branch(hidden,g,state.detach());branch.mode='full' if branch.mode=='centroid' else 'centroid'
        b=branch(hidden,g,state.detach());sensitivity=float((a-b).abs().max());assert sensitivity>1e-8
        branch.mode=model.port_mode
        stronger={**g,'pin_weights':g['pin_weights']*2}
        weight_sensitivity=float((branch(hidden,g,state.detach())-branch(hidden,stronger,state.detach())).abs().max())
        assert weight_sensitivity>1e-8,'Absolute port weights must remain observable'
        permutation=torch.arange(g['pin_coordinates'].shape[2]-1,-1,-1,device='cuda')
        perm={**g,'pin_coordinates':g['pin_coordinates'][:,:,permutation],
              'pin_valid':g['pin_valid'][:,:,permutation],'pin_weights':g['pin_weights'][:,:,permutation]}
        torch.testing.assert_close(branch(hidden,g,state.detach()),branch(hidden,perm,state.detach()),rtol=2e-5,atol=1e-7)
        none={**g,'pin_weights':torch.zeros_like(g['pin_weights'])}
        assert not branch(hidden,none,state.detach()).count_nonzero()
        padded={**g,'pin_coordinates':torch.nn.functional.pad(g['pin_coordinates'],(0,0,0,3)),
                'pin_weights':torch.nn.functional.pad(g['pin_weights'],(0,3)),
                'pin_valid':torch.nn.functional.pad(g['pin_valid'],(0,3))}
        torch.testing.assert_close(branch(hidden,g,state.detach()),branch(hidden,padded,state.detach()),rtol=2e-5,atol=1e-7)
        branch.output.weight.zero_()
    # Public-only eager inference matches the batched preparation on the same five.
    from .inference import graph as inference_graph
    for i in range(5):
        row=batch[i];single={k:v.cuda() for k,v in inference_graph(row['case'].solver_inputs).items()}
        with torch.no_grad():pred=model(single)
        torch.testing.assert_close(pred[0],actual[i,:row['graph']['n']],rtol=5e-5,atol=1e-5)
    initial={k:v.clone() for k,v in model.state_dict().items()};optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=3e-4)
    initial_digest=hashlib.sha256(b''.join(v.cpu().numpy().tobytes() for v in initial.values())).hexdigest()
    captured=CapturedStep(model,optimizer,g,arm);assert not optimizer.state
    assert all(torch.equal(v,initial[k]) for k,v in model.state_dict().items())
    captured.inputs['target'].fill_(float('nan'))
    try:captured.replay()
    except RuntimeError as e:assert 'Nonfinite' in str(e)
    else:raise AssertionError('Nonfinite loss must prevent Adam')
    assert not optimizer.state;captured.inputs['target'].copy_(g['target']);captured.replay()
    reference={k:v.clone() for k,v in model.state_dict().items()}
    assert any(not torch.equal(v,initial[k]) for k,v in reference.items() if k.startswith('port_readers.'))
    model.load_state_dict(initial);eager=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=3e-4)
    eager.zero_grad(set_to_none=True);losses(model(public),g,None,arm)['total'].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True,foreach=True);eager.step()
    delta=max(float((v-reference[k]).abs().max()) for k,v in model.state_dict().items());assert delta<1e-6,delta
    return {'parameters':sum(p.numel() for p in model.parameters()),'neutral_initial_forward_bit_exact':True,
            'initial_weights_sha256':initial_digest,'frozen_batch_semantics_bit_exact':True,
            'loss_abs_delta':float((values['total']-target).abs()),'gradient_max_abs_delta':float((grad-oldgrad).abs().max()),
            'full_vs_centroid_reader_max_abs_delta':sensitivity,'capture_eager_Adam_max_abs_delta':delta,
            'input_channels':14,'legacy_pin_summaries_excluded':True,'absolute_weight_sensitivity':weight_sensitivity,
            'b2b_weights_reconstructible':True,'poisoned_removed_summaries_have_no_effect':True,
            'port_permutation_padding_and_no_pin':True,'public_inference_matches_batch':True,
            'nonfinite_blocks_Adam':True,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30}


def main():
    final=HERE/'runs/preflight/results.json'
    if final.exists():
        prior=json.loads(final.read_text())
        if prior['status']=='passed' and prior['sources']==sources():print('Current E68 preflight already passed.');return
    started=time.perf_counter();seed_all(0);protected=verify_protected();metadata()
    assert sha(HERE/'loss_reference.py')==sha(ROOT/'experiments/E42_joint_regularization/losses.py')
    indices=[1683,10027,5474,9793,1814];rows=load_rows('train',indices,use_cache=False);batch=[rows[i%5] for i in range(32)]
    suffix=str(time.time_ns());checks={}
    for arm in ARMS:
        checks[arm]=check_arm(arm,batch);clear()
        args=parser().parse_args(['--arm',arm,'--output',str(HERE/f'runs/preflight_{suffix}_{arm}_resumed'),
                                 '--max-hours','.1','--max-epochs','2'])
        run(args,implementation_check=True,check_rows=(batch,batch),stop_after_steps=1)
        first=torch.load(args.output/'checkpoints/latest.pt',map_location='cpu',weights_only=False);clear()
        args.resume=True;run(args,implementation_check=True,check_rows=(batch,batch),stop_after_steps=2)
        resumed=torch.load(args.output/'checkpoints/latest.pt',map_location='cpu',weights_only=False);clear()
        args.resume=False;args.output=HERE/f'runs/preflight_{suffix}_{arm}_uninterrupted'
        run(args,implementation_check=True,check_rows=(batch,batch),stop_after_steps=2)
        uninterrupted=torch.load(args.output/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
        assert first['progress']['step']==1 and resumed['progress']['step']==uninterrupted['progress']['step']==2
        for k in ('model','optimizer','scheduler'):same(resumed[k],uninterrupted[k],k)
        for k in ('torch','cuda'):same(resumed['rng'][k],uninterrupted['rng'][k],k)
        assert resumed['elapsed_seconds']>first['elapsed_seconds']
        assert first['identity']==resumed['identity']==uninterrupted['identity']
        checks[arm]['Dropout_Adam_resume_bit_exact']=True
        del first,resumed,uninterrupted;clear()
    assert verify_protected()==protected
    save(final,{'status':'passed','unique_train_cases':5,'indices':indices,'arms':checks,'sources':sources(),
                'formal_training_started':False,'protected_checkpoints_unchanged':protected,'test_read':False,
                'seconds':time.perf_counter()-started})
    print('E68 v2 five-case combined input/Loss/Adam/resume preflight passed; no formal training started.',flush=True)


if __name__=='__main__':main()
