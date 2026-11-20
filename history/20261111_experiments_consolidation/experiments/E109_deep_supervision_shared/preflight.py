"""Five original train instances; one real captured update, no resume rehearsal."""
import json
import time
import traceback
import torch
from .io import save, OWNER, sources
from .batches import inputs_only
from .resident import ResidentRows
from .loss import per_case
from .reference_model import GeometryModel as HistoricalModel
from .train import seed_all, GeometryModel, MODEL, optimizer_for, build_capture
from .data import load_rows, register_resident

def main():
    started=time.perf_counter()
    result=dict(status='error', unique_train_cases=5, dev_payloads=0, test_read=False, actual_updates=0)
    try:
        plan=json.loads((OWNER/'plan.json').read_text(encoding='utf-8'))
        seed_all(0)
        rows=load_rows('train',plan['preflight_train_indices'])
        pack=ResidentRows(rows,'cuda',max_nodes=120,max_pins=146)
        register_resident(pack)
        seed_all(0)
        reference=HistoricalModel(**MODEL).cuda()
        initial={k:v.detach().clone() for k,v in reference.state_dict().items()}
        seed_all(0)
        model=GeometryModel(**MODEL).cuda()
        assert all(torch.equal(value,model.state_dict()[key]) for key,value in initial.items())
        if plan['readout']['independent']:
            for head in model.readout_heads:
                assert all(torch.equal(a,b) and a.data_ptr()!=b.data_ptr() for a,b in zip(head.parameters(),model.head.parameters()))
        assert sum(p.numel() for p in model.parameters())==plan['parameters']
        batch=pack.batch([i%5 for i in range(32)])
        model.train();reference.train()
        rng=torch.cuda.get_rng_state_all()
        with torch.no_grad():
            old=reference(inputs_only(batch))
            torch.cuda.set_rng_state_all(rng)
            states=model(inputs_only(batch))
            assert torch.equal(states[-1],old), 'Treatment changed initial forward arithmetic'
            values=per_case(states,batch)
            assert states.shape[:2]==(12,32)
            assert torch.isfinite(values['total']).all()
        torch.cuda.set_rng_state_all(rng)
        del reference,initial,states,old,values
        optimizer=optimizer_for(model)
        before_first=model.layers[0].local.weight.detach().clone()
        before_last=model.layers[-1].local.weight.detach().clone()
        cap=build_capture(model,optimizer,pack)
        cap.inputs['axis_swap'].copy_(torch.arange(32,device='cuda')%2==0)
        values,norm=cap.replay()
        torch.cuda.synchronize()
        assert not torch.equal(before_first,model.layers[0].local.weight)
        assert not torch.equal(before_last,model.layers[-1].local.weight)
        assert cap.projection.steps==1
        assert all(int(state['step'])==1 for state in optimizer.state.values())
        result.update(status='passed',sources=sources(),actual_updates=1,
            parameters=plan['parameters'],baseline_initial_weights_and_final_forward_exact=True,
            head_storage_independent=plan['readout']['independent'],first_and_last_block_updated=True,
            loss=float(values['total']),final_loss=float(values['final_total']),
            per_layer_loss=[float(values[f'layer{i:02d}_total']) for i in range(1,13)],
            gradient_norm=float(norm),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),elapsed_seconds=time.perf_counter()-started)
    except BaseException as error:
        result.update(error=repr(error),traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-started)
        raise
    finally:
        save(OWNER/'preflight.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('sources','traceback')},ensure_ascii=False),flush=True)

if __name__=='__main__': main()
