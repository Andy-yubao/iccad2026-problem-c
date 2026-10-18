"""Bounded five-dev-case implementation checks, never a full training run."""
import argparse
import copy
import io
import time
import json
import torch
import numpy as np
from datasets.worksets import workset_manifest,LayoutReader
from .panels import read as read_panels
from .records import save,provenance,snapshot,run_path
from .batches import sample,collate,inputs_only,decode
from .inference import prepare_inference
from .model import GeometryModel
from .losses import all_components,objective_loss
from .occupancy import exact_diagnostics,components as occupancy_components
from .train import VARIANTS,seed_all


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');args=p.parse_args();begin=time.monotonic()
    out=run_path(args.output);out.mkdir(parents=True)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);seed_all(0)
    panels=read_panels();reader=LayoutReader(workset_manifest(),'dev_quality')
    ids=[reader.members.index((r['shard'],r['sample_index'])) for r in panels['dev5']]
    pairs=[reader[i] for i in ids];samples=[sample(*pair) for pair in pairs]
    g=collate(samples,list(range(5)),args.device);public=inputs_only(g)
    config={'case_members':[reader.members[i] for i in ids],'real_dev_cases':5,'test_payload_reads':0,**provenance()}
    save(out/'config.json',config);snapshot(out,config['source_sha256'])
    checks=[];bench={};state=g['target'];predicted=decode(state,g)
    locked=(~g['mask'][...,2].bool())&g['valid'];pre=(~g['mask'][...,0].bool())&g['valid']
    assert torch.equal(predicted[locked][:,2:],g['specs'][locked][:,2:])
    assert torch.equal(predicted[pre],g['specs'][pre])
    inp,reference=pairs[0];alter=reference.clone();alter[:,:2]+=17
    altered=sample(inp,alter)
    for key in inputs_only(collate([samples[0]],[0],'cpu')):
        assert torch.equal(inputs_only(collate([samples[0]],[0],'cpu'))[key],inputs_only(collate([altered],[0],'cpu'))[key]),key
    inference=prepare_inference(inp,'cpu');training=inputs_only(collate([samples[0]],[0],'cpu'))
    for k in inference:assert torch.equal(inference[k],training[k]),k
    tiled=exact_diagnostics([[0,0,1,1],[1,0,1,1],[0,1,1,1],[1,1,1,1]])
    triple=exact_diagnostics([[0,0,1,1],[0,0,1,1],[0,0,1,1]])
    assert tiled['empty_fraction']==0 and tiled['overlap_pair_count']==0
    assert abs(triple['union_area']-1)<1e-12 and triple['max_multiplicity']==3
    assert abs(triple['overlap_excess_area_fraction']-2/3)<1e-12
    assert triple['largest_conflict_component']==3 and triple['triple_covered_area_fraction']==1
    # Direct finite directional gradient: widening one gap in a legal row should
    # worsen void/compactness; compressing an overlap should worsen overlap area.
    sg={'area':torch.ones(1,2,device=args.device),'valid':torch.ones(1,2,dtype=torch.bool,device=args.device),
        'mask':torch.ones(1,2,3,device=args.device),'specs':torch.zeros(1,2,4,device=args.device)}
    separated=torch.tensor([[[.5,.5,0.],[2.,.5,0.]]],device=args.device,requires_grad=True)
    sv=occupancy_components(separated,sg);(sv['void']+sv['compactness']).sum().backward()
    assert separated.grad[0,1,0]>0,separated.grad
    overlapped=torch.tensor([[[.5,.5,0.],[1.,.5,0.]]],device=args.device,requires_grad=True)
    occupancy_components(overlapped,sg)['overlap_area'].sum().backward()
    assert overlapped.grad[0,1,0]<0,overlapped.grad
    for name,(arch,objective) in VARIANTS.items():
        if time.monotonic()-begin>480:raise TimeoutError('Preflight short-check guard')
        seed_all(0);model=GeometryModel(arch).to(args.device).eval()
        if args.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            batched=model(public)
            singles=torch.cat([model(inputs_only(collate(samples,[i],args.device)))[0,:samples[i]['n']] for i in range(5)])
            joined=torch.cat([batched[i,:samples[i]['n']] for i in range(5)])
            error=float((singles-joined).abs().max());assert error<1e-4,(name,error)
            perm=torch.arange(g['valid'].shape[1]-1,-1,-1,device=args.device)
            shuffled={k:(v[:,perm][:,:,perm] if k in ('adjacency','mib','group') else v[:,perm] if v.ndim>=2 else v) for k,v in public.items()}
            perm_error=float((model(shuffled)-batched[:,perm]).abs().max());assert perm_error<1e-4,(name,perm_error)
        try:model(g)
        except ValueError:pass
        else:raise AssertionError('Network accepted supervision')
        opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=4)
        start=time.monotonic()
        for step in range(3):
            model.train();opt.zero_grad(set_to_none=True);values=all_components(model(public),g)
            loss=objective_loss(values,objective,3)
            assert torch.isfinite(loss);loss.backward()
            assert all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if args.device.startswith('cuda'):torch.cuda.synchronize()
        step_seconds=(time.monotonic()-start)/3
        # Serialize and restore optimizer/scheduler and generator states; the
        # next completed step must match an uninterrupted copy exactly.
        stream=io.BytesIO();torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),
            'rng_torch':torch.get_rng_state(),'rng_python':__import__('random').getstate(),
            'rng_cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},stream);stream.seek(0)
        saved=torch.load(stream,map_location=args.device,weights_only=False)
        restored=GeometryModel(arch).to(args.device);restored.load_state_dict(saved['model'])
        restored_opt=torch.optim.AdamW(restored.parameters(),lr=1e-3,weight_decay=1e-4);restored_opt.load_state_dict(saved['optimizer'])
        restored_sched=torch.optim.lr_scheduler.ReduceLROnPlateau(restored_opt,patience=4);restored_sched.load_state_dict(saved['scheduler'])
        for candidate,optimizer in ((model,opt),(restored,restored_opt)):
            candidate.train();optimizer.zero_grad(set_to_none=True)
            objective_loss(all_components(candidate(public),g),objective,3).backward()
            torch.nn.utils.clip_grad_norm_(candidate.parameters(),1.);optimizer.step()
        restore_error=max(float((a-b).abs().max()) for a,b in zip(model.parameters(),restored.parameters()))
        assert restore_error==0,(name,restore_error)
        model.eval();start=time.monotonic()
        with torch.no_grad():
            for _ in range(10):model(public)
        if args.device.startswith('cuda'):torch.cuda.synchronize()
        bench[name]={'parameters':sum(v.numel() for v in model.parameters()),'batch5_train_step_seconds':step_seconds,
            'batch5_forward_seconds':(time.monotonic()-start)/10,'loss':float(loss.detach()),
            'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0}
        checks.append({'variant':name,'batched_vs_single_max_abs':error,'permutation_max_abs':perm_error,'resume_step_max_abs':restore_error,
                       'output_channels':3,'finite_gradients':True})
        del model,opt,restored,restored_opt
    result={'status':'passed','seconds':time.monotonic()-begin,'real_dev_cases':5,'optimizer_steps_per_variant':4,
        'test_payload_reads':0,'checks':checks,'benchmark':bench,'public_inference_matches_training':True,
        'labels_excluded':True,'immutable_specs_preserved':True,'exact_geometry_synthetic_checks':True,
        'loss_directional_gradient_checks':True,'training_loop_integration':'not executed; only bounded in-memory gradient/restore checks'}
    save(out/'results.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
