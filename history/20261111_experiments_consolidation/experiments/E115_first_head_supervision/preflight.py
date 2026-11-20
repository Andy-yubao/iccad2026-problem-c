"""Same five train cases; common initialization and one disposable guarded update."""
import json
import time
import traceback
import torch
from .io import OWNER,save,sources
from .train import MODEL,seed_all,optimizer_for
from .model import GeometryModel
from .reference_model import GeometryModel as ReferenceModel
from .data import load_rows
from .resident import ResidentRows
from .batches import inputs_only
from .loss import per_case,base_per_case
from .cuda_step import CapturedStep
from visualization.layer_capture import observe_readouts

def main():
    started=time.perf_counter()
    result=dict(status='error',unique_train_cases=5,dev_payloads=0,test_read=False,actual_updates=0)
    try:
        plan=json.loads((OWNER/'plan.json').read_text(encoding='utf-8'))
        rows=load_rows('train',plan['preflight_train_indices'])
        pack=ResidentRows(rows,'cuda',max_nodes=120,max_pins=max(r['graph']['pin_coordinates'].shape[1] for r in rows))
        batch=pack.batch([i%5 for i in range(32)])
        seed_all(0)
        reference=ReferenceModel(width=64,layers=12,rectangle=True,ffn_dropout=.1,residual_dropout=.05)
        old=reference.state_dict();expected_rng=torch.get_rng_state().clone()
        seed_all(0);model=GeometryModel(**MODEL)
        assert torch.equal(torch.get_rng_state(),expected_rng)
        current=model.state_dict();checked=0
        for name,value in old.items():
            if name.startswith('layers.') and int(name.split('.')[1])>=9:continue
            assert torch.equal(current[name],value),name
            checked+=1
        assert sum(p.numel() for p in model.parameters())==609682
        assert sum(p.numel() for p in model.first_head.parameters())==148995
        del reference,old,current
        model=model.cuda().eval()
        with torch.no_grad(),observe_readouts(model) as captured:output=model(inputs_only(batch))
        assert output.shape==(9,32,120,3) and len(captured['states'])==10
        assert torch.equal(captured['states'][1],output[0]) and torch.equal(captured['states'][-1],output[-1])
        with torch.no_grad():
            objective=per_case(output,batch,'first9')['total']
            explicit=.2*base_per_case(output[0],batch)['total']+.8*base_per_case(output[8],batch)['total']
            assert torch.allclose(objective,explicit,rtol=1e-6,atol=1e-6)
        model.train();optimizer=optimizer_for(model)
        first=model.layers[0].qkv.weight.detach().clone();last=model.layers[-1].qkv.weight.detach().clone()
        residual=model.first_head.residual[2].weight.detach().clone()
        head=model.head[0].weight.detach().clone()
        torch.cuda.reset_peak_memory_stats()
        cap=CapturedStep(model,optimizer,batch,'first9')
        cap.inputs['axis_swap'].copy_(torch.arange(32,device='cuda')%2==0)
        values,norm=cap.replay();torch.cuda.synchronize()
        assert not torch.equal(first,model.layers[0].qkv.weight)
        assert not torch.equal(last,model.layers[-1].qkv.weight)
        assert not torch.equal(residual,model.first_head.residual[2].weight)
        assert not torch.equal(head,model.head[0].weight)
        assert cap.projection.steps==1 and all(int(s['step'])==1 for s in optimizer.state.values())
        result.update(status='passed',sources=sources(),actual_updates=1,
            actual_updates_scope='Disposable five-member implementation check only, not the formal run',
            parameters=609682,first_head_parameters=148995,common_initial_tensors_exact=checked,
            global_initialization_rng_preserved=True,real_readouts=10,
            only_layers1_and9_supervised=True,first_last_blocks_and_both_heads_updated=True,
            total=float(values['total']),terminal_original_loss=float(values['final_total']),gradient_norm=float(norm),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            resume_tested=False)
    except BaseException as error:
        result.update(error=repr(error),traceback=traceback.format_exc());raise
    finally:
        result['elapsed_seconds']=time.perf_counter()-started
        save(OWNER/'preflight.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('sources','traceback')}),flush=True)

if __name__=='__main__':main()
