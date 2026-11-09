"""Same five members: public batching, actual updates, restore and VRAM."""
import copy,gc,time
import torch
import numpy as np
from .runtime import seed,rng,restore_rng,CONFIG,checked_update
from .model import GeometryModel
from .initialization import load
from .packed import PublicPacker
from .step_engine import PackedStep
from .io import HERE,checkpoint

def cpu_state(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_state(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_state(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_state(v) for v in value)
    return copy.deepcopy(value)

def exact(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a.cpu(),b.cpu())
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:exact(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):exact(x,y)
    else:assert a==b

def run(rows):
    from .data import collate,inputs_only
    seed();model=GeometryModel(**CONFIG).cuda();load(model);model.eval()
    with torch.no_grad():draft=model.draft(inputs_only(collate(rows,'cuda'))).cpu()
    for i,row in enumerate(rows):row['draft']=draft[i,:row['graph']['n']].clone()
    selected=[rows[i%5] for i in range(32)];packer=PublicPacker(rows[0])
    g,draft,_=packer.gpu(selected);reference=inputs_only(collate(selected,'cuda'))
    exact(g,reference);del reference
    initial=cpu_state(model.state_dict());state=rng()
    opt=torch.optim.AdamW(model.corrector.parameters(),lr=.001,weight_decay=.0003)
    first,_=checked_update(model,opt,g,draft);expected=cpu_state(model.state_dict())
    model.load_state_dict(initial);opt=torch.optim.AdamW(model.corrector.parameters(),lr=.001,weight_decay=.0003)
    restore_rng(state);engine=PackedStep(model,opt);values,_,_=engine.replay(g,draft)
    loss_error=max(abs(first[k]-float(v.detach().cpu())) for k,v in values.items())
    param_error=max(float((v.detach().cpu()-expected[k]).abs().max()) for k,v in model.state_dict().items())
    assert loss_error<2e-6 and param_error<2e-6
    print('Engine: packed/reference update parity',loss_error,param_error,flush=True)
    cpu,source,_=packer.pack(selected);large={k:v.cuda() for k,v in cpu.items()};source=source.cuda()
    torch.cuda.reset_peak_memory_stats();engine.replay(large,source)
    checkpoint(HERE/'runs/preflight_bbox_v3/engine_step1.pt',dict(model=cpu_state(model.state_dict()),optimizer=cpu_state(opt.state_dict()),rng=rng(),cursor=32))
    engine.replay(large,source)
    whole=cpu_state(model.state_dict());moments=cpu_state(opt.state_dict());random=rng()
    peak=torch.cuda.max_memory_allocated()/2**20;reserved=torch.cuda.max_memory_reserved()/2**20
    print('Engine: worst32x120x512 actual updates passed; MiB',peak,reserved,flush=True)
    saved=torch.load(HERE/'runs/preflight_bbox_v3/engine_step1.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);restore_rng(saved['rng'])
    engine.replay(large,source);exact(whole,model.state_dict());exact(moments,opt.state_dict());exact(random,rng())
    del large,source;gc.collect();torch.cuda.empty_cache()
    print('Engine: exact large-attention Adam/Dropout/RNG restore passed',flush=True)
    torch.cuda.synchronize();begin=time.perf_counter()
    for _ in range(4):
        g,draft,_=packer.gpu(selected);engine.replay(g,draft)
    torch.cuda.synchronize();packed_ms=(time.perf_counter()-begin)*250
    torch.cuda.synchronize();begin=time.perf_counter()
    for _ in range(4):
        ref=inputs_only(collate(selected,'cuda'));old_draft=torch.zeros(32,ref['valid'].shape[1],3,device='cuda')
        for i,row in enumerate(selected):old_draft[i,:row['graph']['n']]=row['draft'].cuda()
        checked_update(model,opt,ref,old_draft)
    torch.cuda.synchronize();eager_ms=(time.perf_counter()-begin)*250
    before=cpu_state(model.state_dict());before_opt=cpu_state(opt.state_dict())
    bad=draft.clone();bad[0,0,0]=float('nan')
    try:engine.replay(g,bad);raise AssertionError('Accepted nonfinite source')
    except RuntimeError:pass
    exact(before,model.state_dict());exact(before_opt,opt.state_dict())
    return dict(shape='dynamic batch32, max120modules/512ports',public_only_pinned_batch=True,
        public_collate_bit_exact=True,loss_max_abs_difference=loss_error,parameter_max_abs_difference=param_error,
        exact_large_attention_resume_model_Adam_RNG=True,serialized_cursor=32,
        nonfinite_rejected_before_Adam=True,peak_allocated_MiB=peak,peak_reserved_MiB=reserved,
        reference_ms_per_step=eager_ms,packed_ms_per_step=packed_ms,
        microbenchmark_scope='four updates each including packing; same five cases; not full epoch speed',
        worst_shape_valid_port_max=207,stress_shape_padded_ports=512,attention_recompute_above_tokens=448)
