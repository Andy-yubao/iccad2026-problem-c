"""Five training members: semantic transposition, symmetric Loss, real update."""
import ast
import time
import traceback
import numpy as np
import torch
from ..io import save
from ..batches import sample, inputs_only
from ..ports import public_ports
from ..resident import ResidentRows
from ..loss import per_case
from .io import OWNER, sources
from .data import load_rows, register_resident
from .augmentation import axis_swap, boundary_swap, flags_for
from .train import seed_all, GeometryModel, MODEL, optimizer_for, build_capture

def main():
    started=time.perf_counter()
    result=dict(status='error',unique_train_cases=5,dev_payloads=0,test_read=False,postprocessing_calls=0)
    try:
        for path in OWNER.glob('*.py'):
            ast.parse(path.read_text(encoding='utf-8'))
        seed_all(0)
        rows=load_rows('train',[1683,10027,5474,10140,15001])
        fixed_shape_modules=sum(int(((r['graph']['constraints'][:,0]!=0)&(r['graph']['constraints'][:,1]==0)).sum()) for r in rows)
        preplaced_modules=sum(int((r['graph']['constraints'][:,1]!=0).sum()) for r in rows)
        assert fixed_shape_modules>0 and preplaced_modules>0, 'Fixture must cover fixed-shape and preplaced modules'
        pack=ResidentRows(rows,'cuda',max_nodes=120,max_pins=146)
        register_resident(pack)
        batch=pack.batch([i%5 for i in range(32)])
        batch['axis_swap']=torch.ones(32,device='cuda',dtype=torch.bool)
        swapped=axis_swap(batch)
        restored=axis_swap(swapped)
        for key in batch:
            torch.testing.assert_close(restored[key],batch[key],rtol=0,atol=0)
        # Rebuild independently from transformed public solver inputs and target.
        max_error=0.
        for i,row in enumerate(rows):
            inp=row['case'].solver_inputs
            transformed=[int(inp[0])]+[np.array(x,copy=True) for x in inp[1:]]
            transformed[4][:,:2]=transformed[4][:,[1,0]]
            transformed[5][:,4]=boundary_swap(torch.as_tensor(transformed[5][:,4]).long()).numpy()
            transformed[6]=transformed[6][:,[1,0,3,2]]
            reference=row['graph']['reference']*row['graph']['scale']
            rebuilt=sample(tuple(transformed),reference[:,[1,0,3,2]])
            rebuilt.update(public_ports(tuple(transformed),rebuilt['scale']))
            n=rebuilt['n']
            for key in ('features','mask','area','known','target','specs','reference','constraints',
                        'adjacency','mib','group','pin_coordinates','pin_weights','pin_valid','wire_b2b'):
                value=rebuilt[key].cuda()
                actual=swapped[key][i]
                actual=actual[tuple(slice(0,x) for x in value.shape)]
                torch.testing.assert_close(actual,value,rtol=2e-5,atol=2e-6)
                if value.is_floating_point():
                    max_error=max(max_error,float((actual-value).abs().max()))
        state=batch['target']+.01
        swapped_state=torch.stack((state[...,1],state[...,0],-state[...,2]),-1)
        old=per_case(state,batch)
        new=per_case(swapped_state,swapped)
        for key in old:
            torch.testing.assert_close(old[key],new[key],rtol=3e-5,atol=3e-6)
        model=GeometryModel(**MODEL).cuda()
        assert sum(p.numel() for p in model.parameters())==1471539
        assert all(layer.ff_dropout.p==.10 and layer.residual_dropout.p==.05 for layer in model.layers)
        opt=optimizer_for(model)
        cap=build_capture(model,opt,pack)
        ids=np.array([i%5 for i in range(32)])
        # Exercise both paths even if five randomly drawn flags happen to coincide.
        cap.inputs['axis_swap'].copy_(torch.as_tensor(np.arange(32)%2==0,device='cuda'))
        assert 0<int(cap.inputs['axis_swap'].sum())<32
        before=model.layers[0].local.weight.detach().clone()
        values,norm=cap.replay()
        assert not torch.equal(before,model.layers[0].local.weight)
        assert all(int(v['step'])==1 for v in opt.state.values())
        assert cap.projection.steps==1
        result.update(status='passed',sources=sources(),actual_updates=1,
            reconstruction_max_error=max_error,axis_swap_twice_exact=True,loss_axis_symmetric=True,
            parameters=1471539,ffn_dropout=.10,residual_dropout=.05,
            swapped_in_batch=int(cap.inputs['axis_swap'].sum()),loss=float(values['total']),
            fixed_shape_modules_checked=fixed_shape_modules,preplaced_modules_checked=preplaced_modules,
            gradient_norm=float(norm),elapsed_seconds=time.perf_counter()-started)
    except BaseException as exc:
        result.update(error=repr(exc),traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-started)
        raise
    finally:
        save(OWNER/'preflight.json',result)
        print({k:v for k,v in result.items() if k!='sources'},flush=True)

if __name__=='__main__':
    main()
