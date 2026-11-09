"""Five original train cases; one captured real update with original safeguards."""
import json,time,traceback
import torch
from .io import OWNER,save,sources
from .data import load_rows,register_resident
from .resident import ResidentRows
from .reference_model import GeometryModel as AModel
from .train import seed_all,GeometryModel,MODEL,optimizer_for,build_capture


def main():
    start=time.perf_counter();result=dict(status='error',unique_train_cases=5,dev_payloads=0,test_read=False,actual_updates=0)
    try:
        plan=json.loads((OWNER/'plan.json').read_text())
        seed_all(0);rows=load_rows('train',plan['preflight_train_indices'])
        pack=ResidentRows(rows,'cuda',max_nodes=120,max_pins=146);register_resident(pack)
        original=dict(MODEL);original.pop('critical_hidden');original['layers']=12
        seed_all(0);reference=AModel(**original).cuda()
        old=reference.state_dict();reference_rng=torch.get_rng_state().clone()
        seed_all(0);model=GeometryModel(**MODEL).cuda()
        assert torch.equal(torch.get_rng_state(),reference_rng),'New critical modules changed global initialization RNG'
        current=model.state_dict();kind=plan['critical_subnetwork']['kind']
        checked=0
        for key,value in old.items():
            if key.startswith('layers.') and int(key.split('.')[1])>=8:continue
            if kind=='geometry' and '.bias.' in key and key.startswith('layers.'):continue
            if kind=='head' and key.startswith('head.'):continue
            assert key in current and torch.equal(value,current[key]),key
            checked+=1
        assert sum(p.numel() for p in model.parameters())==plan['parameters']
        assert abs(plan['parameters']/1471539-1)<.004
        del reference,old,current
        optimizer=optimizer_for(model)
        first=model.layers[0].local.weight.detach().clone();last=model.layers[-1].local.weight.detach().clone()
        critical=model.layers[0].bias if kind=='geometry' else model.head
        critical_before=critical.residual[2].weight.detach().clone()
        cap=build_capture(model,optimizer,pack)
        cap.inputs['axis_swap'].copy_(torch.arange(32,device='cuda')%2==0)
        values,norm=cap.replay();torch.cuda.synchronize()
        assert not torch.equal(first,model.layers[0].local.weight)
        assert not torch.equal(last,model.layers[-1].local.weight)
        assert not torch.equal(critical_before,critical.residual[2].weight)
        assert cap.projection.steps==1 and all(int(s['step'])==1 for s in optimizer.state.values())
        result.update(status='passed',sources=sources(),actual_updates=1,common_A_initial_tensors_exact=checked,
            original_global_initialization_rng_preserved=True,critical_residual_updated=True,
            first_and_last_blocks_updated=True,parameters=plan['parameters'],outer_depth=8,
            total_loss=float(values['total']),terminal_original_loss=float(values['final_total']),
            gradient_norm=float(norm),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),elapsed_seconds=time.perf_counter()-start,
            resume_tested=False)
    except BaseException as error:
        result.update(error=repr(error),traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-start);raise
    finally:
        save(OWNER/'preflight.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('sources','traceback')}),flush=True)


if __name__=='__main__':main()
