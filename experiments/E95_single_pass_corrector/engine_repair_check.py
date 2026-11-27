"""Stopped weights/Adam: five-case attention, graph update/restore and timing."""
import copy,gc,json,time
import torch
from .io import HERE,save,checkpoint,sha
from .runtime import CONFIG,seed,rng,restore_rng,charge
from .model import GeometryModel
from .data import load_rows,collate,inputs_only
from .loss import per_case
from .packed import PublicPacker
from .step_engine import PackedStep
from .graph_step import CapturedStep
from .engine_check import cpu_state,exact

FOLDER=HERE/'runs/20261101_engine_repair'
OLD=HERE/'runs/20261101_direct_bbox846_seed0'

def backend(model,name):
    for layer in model.corrector.layers:layer.attention_backend=name

def update_with_grad(model,opt,g,draft):
    model.train();opt.zero_grad(set_to_none=True)
    out=model.corrector(g,draft);values=per_case(out,g,draft)
    values['total'].mean().backward()
    gradients={k:p.grad.detach().cpu().clone() for k,p in model.corrector.named_parameters() if p.grad is not None}
    torch.nn.utils.clip_grad_norm_(model.corrector.parameters(),1.,foreach=True);opt.step()
    return out.detach().cpu(),{k:float(v.mean().detach().cpu()) for k,v in values.items()},gradients

def main():
    seed();saved=torch.load(OLD/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
    assert saved['status']=='user_stop' and not saved['finished']
    ids=json.loads((HERE/'plan.json').read_text(encoding='utf-8'))['preflight_train_ids']
    rows=load_rows('train',ids);config=dict(CONFIG,attention_backend='manual')
    model=GeometryModel(**config).cuda();model.load_state_dict(saved['model'])
    model.eval()
    with torch.no_grad():draft=model.draft(inputs_only(collate(rows,'cuda'))).cpu()
    for i,row in enumerate(rows):row['draft']=draft[i,:row['graph']['n']].clone()
    selected=[rows[i%5] for i in range(32)];packer=PublicPacker(rows[0],p=480)
    host,source,_=packer.pack(selected);g={k:v.cuda() for k,v in host.items()};draft=source.cuda()
    def optimizer():
        opt=torch.optim.AdamW(model.corrector.parameters(),lr=.001,weight_decay=.0003)
        opt.load_state_dict(copy.deepcopy(saved['optimizer']));return opt
    def reset(name):
        model.load_state_dict(saved['model']);backend(model,name);opt=optimizer();restore_rng(saved['rng']);return opt
    opt=reset('manual');old_out,old_values,old_grads=update_with_grad(model,opt,g,draft)
    old_after=cpu_state(model.state_dict());del opt;gc.collect();torch.cuda.empty_cache()
    opt=reset('efficient');out,values,grads=update_with_grad(model,opt,g,draft)
    grad_delta=sum(float((grads[k]-old_grads[k]).double().square().sum()) for k in grads)**.5
    grad_base=sum(float(old_grads[k].double().square().sum()) for k in grads)**.5
    output_error=float((out-old_out).abs().max());loss_error=max(abs(values[k]-old_values[k]) for k in values)
    update_error=max(float((v.detach().cpu()-old_after[k]).abs().max()) for k,v in model.state_dict().items())
    assert output_error<1e-4 and loss_error<2e-5 and grad_delta/max(grad_base,1e-12)<1e-3 and update_error<2e-5
    efficient_after=cpu_state(model.state_dict());efficient_rng=rng();efficient_opt=cpu_state(opt.state_dict())
    print('Manual/efficient actual update:',output_error,loss_error,grad_delta/grad_base,update_error,flush=True)
    del opt,model,grads,old_grads;gc.collect();torch.cuda.empty_cache()
    # A fresh module avoids reusing leaf-gradient stream metadata from the
    # eager control. Formal resume likewise constructs a fresh module, then
    # loads the preserved tensors and Adam without any prior eager backward.
    model=GeometryModel(**dict(CONFIG,attention_backend='efficient')).cuda()
    opt=reset('efficient');before=cpu_state(model.state_dict());before_opt=cpu_state(opt.state_dict())
    engine=CapturedStep(model,opt,g,draft)
    exact(before,model.state_dict());exact(before_opt,opt.state_dict())
    print('Capture complete, no model or Adam changes; MiB',torch.cuda.memory_reserved()/2**20,flush=True)
    restore_rng(saved['rng']);engine.replay()
    graph_error=max(float((v.detach().cpu()-efficient_after[k]).abs().max()) for k,v in model.state_dict().items())
    assert graph_error<2e-6
    exact(efficient_opt,opt.state_dict());exact(efficient_rng,rng())
    checkpoint(FOLDER/'graph_step1.pt',dict(model=cpu_state(model.state_dict()),optimizer=cpu_state(opt.state_dict()),rng=rng(),progress=saved['progress']))
    engine.replay();whole=cpu_state(model.state_dict());whole_opt=cpu_state(opt.state_dict());whole_rng=rng()
    del engine,opt,model;gc.collect();torch.cuda.empty_cache()
    first=torch.load(FOLDER/'graph_step1.pt',map_location='cpu',weights_only=False)
    model=GeometryModel(**dict(CONFIG,attention_backend='efficient')).cuda();model.load_state_dict(first['model'])
    opt=optimizer();opt.load_state_dict(first['optimizer'])
    engine=CapturedStep(model,opt,g,draft);restore_rng(first['rng']);engine.replay()
    exact(whole,model.state_dict());exact(whole_opt,opt.state_dict());exact(whole_rng,rng());exact(saved['progress'],first['progress'])
    print('Fresh graph reconstruction resumes next model/Adam/RNG bit-exactly',flush=True)
    torch.cuda.synchronize();begin=time.perf_counter()
    for _ in range(4):
        h,d,_=packer.pack(selected);engine.replay(h,d)
    torch.cuda.synchronize();graph_ms=(time.perf_counter()-begin)*250
    peak=torch.cuda.max_memory_allocated()/2**20;reserved=torch.cuda.max_memory_reserved()/2**20
    before=cpu_state(model.state_dict());before_opt=cpu_state(opt.state_dict());bad=draft.clone();bad[0,0,0]=float('nan')
    try:engine.replay(draft=bad);raise AssertionError('Accepted nonfinite draft')
    except RuntimeError:pass
    exact(before,model.state_dict());exact(before_opt,opt.state_dict())
    del engine;gc.collect();torch.cuda.empty_cache()
    opt=reset('manual');eager=PackedStep(model,opt);natural=PublicPacker(rows[0])
    ng,nd,_=natural.gpu(selected);eager.replay(ng,nd)
    torch.cuda.synchronize();begin=time.perf_counter()
    for _ in range(4):
        ng,nd,_=natural.gpu(selected);eager.replay(ng,nd)
    torch.cuda.synchronize();natural_ms=(time.perf_counter()-begin)*250
    # Full480-port padding stresses actual train bounds, with the same five cases.
    eager.replay(g,draft);torch.cuda.synchronize();begin=time.perf_counter()
    for _ in range(4):
        h,d,_=packer.pack(selected);eager.replay({k:v.cuda() for k,v in h.items()},d.cuda())
    torch.cuda.synchronize();full_ms=(time.perf_counter()-begin)*250
    result=dict(status='passed',unique_train_cases=5,indices=ids,old_checkpoint_sha256=sha(OLD/'checkpoints/latest.pt'),
        stopped_step=saved['progress']['step'],shape=[32,120,480],attention_backend='FP32 efficient SDPA; no AMP/TF32',
        manual_efficient_output_max_abs=output_error,manual_efficient_loss_max_abs=loss_error,
        manual_efficient_gradient_relative_l2=grad_delta/grad_base,manual_efficient_update_max_abs=update_error,
        efficient_eager_graph_update_max_abs=graph_error,exact_new_graph_Adam_RNG_restore=True,serialized_progress_preserved=True,
        warmup_capture_no_optimizer_updates=True,nonfinite_rejected_before_Adam=True,
        peak_allocated_MiB=peak,peak_reserved_MiB=reserved,
        reference_natural_ms=natural_ms,reference_full_padding_ms=full_ms,graph_full_padding_ms=graph_ms,
        natural_to_graph_speedup=natural_ms/graph_ms,full_to_graph_speedup=full_ms/graph_ms,
        timing_scope='four updates including public packing/transfers; five repeated train cases; not complete epoch speed',
        numerical_migration_required=True,reason='efficient reduction order and fixed padding change rounding/Dropout stream consumption; no claim of old-path bit-exact continuation',
        no_formal_optimizer_updates=True,test_read=False)
    save(FOLDER/'check.json',result);print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    started=time.perf_counter()
    try:main()
    finally:charge(time.perf_counter()-started)
