"""Five train examples; no solver calls or terminal evaluations."""
import time
import numpy as np
import torch
from .io import HERE,ROOT,sha,save,sources
from .prepare import parents,load_parent,build,read
from .data import load_rows,collate
from .batches import inputs_only
from .model import GeometryModel,decode
from .native_model import GeometryModel as Native
from .frame import predicted_frame
from .loss import source_terms

def main():
    begin=time.perf_counter();torch.set_num_threads(4)
    plan=read(HERE/'plan.json');meta=parents();src=load_parent('F81')
    for file,digest in src['identity']['sources'].items():
        if sha(ROOT/file)!=digest:raise ValueError('Frozen E81 production source changed: '+file)
    ids=plan['preflight_train_ids'];rows=load_rows('train',ids,use_cache=False);cached=load_rows('train',ids)
    for a,b in zip(rows,cached):
        assert a['member']==b['member']
        for k,v in a['graph'].items():
            if isinstance(v,torch.Tensor):assert torch.equal(v,b['graph'][k]),k
            else:assert v==b['graph'][k]
    native=Native(**src['model_config']).eval();native.load_state_dict(src['model'],strict=True)
    # Reconstruct both native E81 views before performing either intervention.
    full42,c42=build({'placer':'P42','learned_residuals':True})
    full74,c74=build({'placer':'P74','learned_residuals':True})
    ops,cops=build(plan['arms'][0]);records=[]
    before={k:v.clone() for k,v in ops.state_dict().items()}
    def forbid(*args):raise AssertionError('Ops-only unexpectedly called a learned correction head')
    handles=[module.register_forward_pre_hook(forbid) for module in (ops.refiner.first,ops.refiner.second,ops.refiner.third)]
    # Independent no-residual reference: zero all terminal head weights in a
    # copied native correction package, retaining operators and source geometry.
    zero=GeometryModel(**{**cops,'learned_residuals':True});zero.load_state_dict(ops.state_dict())
    for module in (zero.refiner.first,zero.refiner.second,zero.refiner.third):
        module.output[-1].weight.data.zero_();module.output[-1].bias.data.zero_()
    with torch.no_grad():
        for index,row in zip(ids,rows):
            g,_=collate([row],'cpu');pub=inputs_only(g)
            expected=native(pub)
            torch.testing.assert_close(full42(pub),expected[...,:3],rtol=0,atol=0)
            torch.testing.assert_close(full74(pub),expected[...,3:],rtol=0,atol=0)
            torch.testing.assert_close(ops(pub),zero(pub),rtol=0,atol=0)
            item={'train_index':index,'member':list(row['member'])}
            for name,model in (('ops',ops),('P74F81',full74)):
                state=model(pub);assert torch.isfinite(state).all();p=decode(state,g)
                lo,hi=predicted_frame(model.placer(pub),pub)
                assert (p[...,:2]>=lo[:,None]-3e-6).all() and (p[...,:2]+p[...,2:]<=hi[:,None]+3e-6).all()
                torch.testing.assert_close(p[...,2:].prod(-1),g['area'],rtol=3e-6,atol=2e-8)
                placed=~g['mask'][...,0].bool();locked=~g['mask'][...,2].bool()
                assert torch.equal(p[placed],g['specs'][placed])
                assert torch.equal(p[...,2:][locked],g['specs'][...,2:][locked])
                values=source_terms(state,g);assert all(torch.isfinite(v).all() for v in values.values())
                item[name+'_fixed_loss']=float(values['total'][0])
                try:model(g);raise AssertionError('Labels accepted')
                except ValueError:pass
                model.train();assert not any(m.training for m in model.modules())
                assert not any(p.requires_grad for p in model.parameters())
            records.append(item)
        g,_=collate(rows,'cpu');pub=inputs_only(g)
        cpu={n:m(pub).clone() for n,m in (('ops',ops),('P74F81',full74))}
        gpu={k:v.cuda() for k,v in pub.items()};native.cuda();expected_gpu=native(gpu)
        zero.cuda();numeric={}
        for name,model in (('ops',ops),('P74F81',full74)):
            model.cuda();on_gpu=model(gpu);actual=on_gpu.cpu()
            expected=zero(gpu) if name=='ops' else expected_gpu[...,3:]
            # Exact same-device parity is the intervention correctness check.
            torch.testing.assert_close(on_gpu,expected,rtol=0,atol=0)
            physical_delta=float((decode(on_gpu,gpu).cpu()-decode(cpu[name],pub)).abs().max())
            numeric[name]={'max_state_CPU_GPU_delta':float((actual-cpu[name]).abs().max()),
                'max_physical_CPU_GPU_delta_over_total_area_sqrt':physical_delta,'same_device_native_or_zero_parity_bit_exact':True}
            save(HERE/'runs/preflight_numeric_diagnostic.json',numeric)
            # State log-aspect close to zero makes relative error misleading;
            # enforce the original frame tolerance on physical geometry instead.
            assert physical_delta<3e-6,numeric
            model.cpu()
        assert all(torch.equal(v,before[k]) for k,v in ops.state_dict().items())
    for handle in handles:handle.remove()
    from .dataset_view import ReferenceLayouts
    for role in ('test','local_test','tail_test'):
        try:ReferenceLayouts(role);raise AssertionError('Sealed role accepted')
        except PermissionError:pass
    save(HERE/'runs/preflight.json',{'status':'passed','sources':sources(),'parents':meta,'train_ids':ids,'records':records,
      'native_P42_P74_outputs_bit_exact':True,'ops_equals_zero_terminal_heads_bit_exact':True,
      'no_learned_F_calls_in_ops':True,'all_weights_unchanged':True,'CUDA_and_public_spec_checks':True,
      'CPU_GPU_numeric_diagnostic':numeric,'solver_calls':0,'training_updates':0,'test_read':False,'seconds':time.perf_counter()-begin})
    print('E83 five-case frozen-intervention preflight PASSED',flush=True)

if __name__=='__main__':main()
