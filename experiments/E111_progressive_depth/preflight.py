"""One genuine update per active-depth mode on the same five train instances."""
import gc,json,time,traceback
import torch
from .io import OWNER,save,sources
from .data import load_rows,register_resident
from .resident import ResidentRows
from .model import configure_stage
from .train import seed_all,GeometryModel,MODEL,optimizer_for,build_capture

def main():
    start=time.perf_counter(); result=dict(status='error',unique_train_cases=5,test_read=False,dev_payloads=0,stages=[])
    try:
        plan=json.loads((OWNER/'plan.json').read_text())
        seed_all(0)
        rows=load_rows('train',plan['preflight_train_indices'])
        for depth in (4,8,12):
            seed_all(0)
            model=GeometryModel(**MODEL).cuda();configure_stage(model,depth)
            optimizer=optimizer_for(model)
            frozen={k:p.detach().clone() for k,p in model.named_parameters() if not p.requires_grad}
            new=model.layers[depth-4].local.weight.detach().clone()
            head=model.head[2].weight.detach().clone()
            pack=ResidentRows(rows,'cuda',max_nodes=120,max_pins=146)
            cap=build_capture(model,optimizer,pack)
            cap.inputs['axis_swap'].copy_(torch.arange(32,device='cuda')%2==0)
            values,norm=cap.replay();torch.cuda.synchronize()
            assert all(torch.equal(v,dict(model.named_parameters())[k]) for k,v in frozen.items())
            assert not torch.equal(new,model.layers[depth-4].local.weight)
            assert not torch.equal(head,model.head[2].weight)
            assert cap.projection.steps==1
            result['stages'].append(dict(active_depth=depth,actual_updates=1,all_frozen_parameters_unchanged=True,
                new_block_updated=True,shared_head_updated=True,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                loss=float(values['total']),gradient_norm=float(norm),peak_allocated_bytes=torch.cuda.max_memory_allocated()))
            del cap,optimizer,model,frozen,pack;gc.collect();torch.cuda.empty_cache()
        result.update(status='passed',sources=sources(),elapsed_seconds=time.perf_counter()-start,
            resume_tested=False,stage_handoff_rehearsed=False)
    except BaseException as error:
        result.update(error=repr(error),traceback=traceback.format_exc());raise
    finally:
        save(OWNER/'preflight.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('sources','traceback')},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
