"""Fixed-panel, read-only component/teacher comparison; no optimizer or solver."""
from pathlib import Path
import sys,json,gc,time
ROOT=Path.cwd();sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import torch
from experiments.E101_final_c5_solver.common import HERE,RUN,read,save,seed,sha
from experiments.E101_final_c5_solver.data import Data,Packer,public
from experiments.E101_final_c5_solver.model import DeploymentModel
from experiments.E101_final_c5_solver.nn.base_model import decode
from experiments.E101_final_c5_solver.placer.loss import per_case as l_loss
from experiments.E101_final_c5_solver import loss_f
from draft_metrics import module_overlap
OUT=Path(__file__).parent
ARCHIVE=ROOT/'history/20261103_final_delivery_consolidation/experiments/E97_gpu_hybrid_solver'
OLD_TEACHER=ARCHIVE/'runs/20261101_public_teacher'
OLD=ARCHIVE/'runs/20261102_round2_candidate5_countweighted_seed0'

def old_labels(role,members):
    parent='train' if role.startswith('train') else 'dev_monitor500'
    config=read(OLD_TEACHER/'config.json');all_members=config['members'][parent];lookup={tuple(m):i for i,m in enumerate(all_members)}
    requested=[lookup[tuple(m)] for m in members];need=set(requested);found={}
    for entry in read(OLD_TEACHER/parent/'manifest.json')['parts']:
        if not need.intersection(entry['indices']):continue
        path=OLD_TEACHER/entry['path'];assert sha(path)==entry['sha256'];part=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        for i,row in zip(part['indices'],part['rows']):
            if i in need:found[i]=row
    assert len(found)==len(requested)
    return [found[i] for i in requested]

def append_losses(destination,key,values,count):
    target=destination.setdefault(key,{})
    for name,value in values.items():target.setdefault(name,[]).extend(value[:count].detach().cpu().tolist())

@torch.inference_mode()
def main():
    # Read completion state only; test outcomes/payloads never enter this audit.
    assert read(RUN/'state.json')['status']=='complete','Wait until timed final evaluation finishes'
    seed();begin=time.perf_counter();models={};hashes={}
    for label,path in [('old',HERE/'assets/c5_baseline.pt'),('new',HERE/'delivery/model.pt')]:
        saved=torch.load(path,map_location='cpu',weights_only=False);model=DeploymentModel(**saved['model_config']).cuda().eval();model.load_state_dict(saved['model']);model.requires_grad_(False);models[label]=model;hashes[label]=sha(path);del saved
    selected=read(RUN/'frozen_selection.json');assert hashes['new']==selected['checkpoint_sha256']
    l_saved=torch.load(RUN/'L/checkpoints/best.pt',map_location='cpu',weights_only=False);f_saved=torch.load(RUN/'F/checkpoints/best.pt',map_location='cpu',weights_only=False)
    assert all(torch.equal(v.cpu(),models['new'].placer.state_dict()[k].cpu()) for k,v in l_saved['model'].items())
    assert all(torch.equal(v.cpu(),models['new'].refiner.state_dict()[k].cpu()) for k,v in f_saved['model'].items());del l_saved,f_saved
    summaries={};all_geometry={};percase={};deployment_check={};cached_base_check={};raw_dev=None
    for role in ('train_monitor500','dev_monitor500'):
        data=Data(role);pack=Packer(data);labels=old_labels(role,data.members);teacher={k:np.load(RUN/'teacher'/data.parent/(k+'.npy'),mmap_mode='r') for k in ('base','teacher','usable')}
        losses={};positions={k:[] for k in ('old_L','new_L','oldL_oldF','newL_oldF','newL_newF','oldL_newF')};cache_errors=[];label_differences=[]
        for start in range(0,500,32):
            ids=list(range(start,min(start+32,500)));g,count=pack.gpu(ids);inp=public(g);padded_ids=ids+[ids[-1]]*(32-count)
            old_target=torch.zeros(32,120,3,device='cuda');old_usable=torch.zeros(32,device='cuda');cached_base=torch.from_numpy(np.array(teacher['base'][data.indices[padded_ids]],copy=True)).cuda()
            for j,i in enumerate(padded_ids):
                row=labels[i];assert row['member']==data.members[i];old_target[j,:row['target'].shape[0]]=row['target'].cuda();old_usable[j]=float(row['usable'])
            new_target=torch.from_numpy(np.array(teacher['teacher'][data.indices[padded_ids]],copy=True)).cuda();new_usable=torch.from_numpy(np.array(teacher['usable'][data.indices[padded_ids]],copy=True)).cuda()
            base_old=models['old'].placer(inp);base_new=models['new'].placer(inp)
            cache_errors.extend((base_new-cached_base).abs()[:count].amax((1,2)).cpu().tolist())
            state=dict(old_L=base_old,new_L=base_new,oldL_oldF=models['old'].refiner(base_old,inp),newL_oldF=models['old'].refiner(base_new,inp),newL_newF=models['new'].refiner(base_new,inp),oldL_newF=models['new'].refiner(base_old,inp))
            for name in ('old_L','new_L'):append_losses(losses,'L/'+name,l_loss(state[name],g,'topology'),count)
            for name,value in state.items():
                append_losses(losses,'F_old_teacher/'+name,loss_f.per_case(value,g,old_target,old_usable),count)
                append_losses(losses,'F_new_teacher/'+name,loss_f.per_case(value,g,new_target,new_usable),count)
                physical=(decode(value,g)*g['scale'][:,None,None]).cpu().double().numpy()
                for j,i in enumerate(ids):positions[name].append(physical[j,:int(data.arrays['n'][data.indices[i]])].copy())
            for name,target,usable in [('old',old_target,old_usable),('new',new_target,new_usable)]:
                append_losses(losses,'teacher_nominal_reconstruction/'+name,loss_f.per_case(target,g,target,usable),count)
            mask=g['mask']*g['valid'][...,None];local=g['area'].clamp_min(1e-12).sqrt()
            difference=((new_target[...,:2]-old_target[...,:2])/local[...,None]).abs();value=(difference*mask[...,:2]).sum((1,2))/mask[...,:2].sum((1,2)).clamp_min(1)
            label_differences.extend(value[:count].cpu().tolist())
            print(role,'Loss',start+count,'/500',flush=True)
        cached_base_check[role]=dict(max_abs=max(cache_errors),mean_abs_max=float(np.mean(cache_errors)),teacher_center_mean_abs_change_module_scale=float(np.mean(label_differences)))
        summaries[role]={k:{n:float(np.mean(v)) for n,v in values.items()} for k,values in losses.items()};percase[role]=losses
        if role=='dev_monitor500':
            errors={name:[] for name in ('old_L','new_L','oldL_oldF','newL_newF')}
            for i in range(500):
                a=read(OLD/'evaluation/terminal'/f'{i:04d}.json');b=read(RUN/'evaluation/dev_monitor500/cases'/f'{i:04d}.json');assert a['member']==b['member']==data.members[i];scale=float(data.arrays['scale'][data.indices[i]])
                for name,record,key in [('old_L',a,'L_positions'),('new_L',b,'L_positions'),('oldL_oldF',a,'draft_positions'),('newL_newF',b,'draft_positions')]:
                    errors[name].append(float(np.max(np.abs(positions[name][i]-np.array(record[key])))/scale))
                    # Exact saved deployment geometry is preferred for actual-pipeline metrics.
                    positions[name][i]=np.array(record[key],dtype=float)
            deployment_check={k:dict(max_normalized_error=max(v),mean_normalized_error=float(np.mean(v))) for k,v in errors.items()}
            raw_dev=positions
        save(OUT/'loss_summary.json',dict(panels=summaries,cached_base_check=cached_base_check,deployment_check=deployment_check,model_SHA256=hashes,selected_L_F_weights_match=True,no_optimizer_or_solver=True,test_used=False))
        save(OUT/'loss_percase.json',percase)
        if role=='dev_monitor500':
            metrics={}
            for name,layouts in positions.items():
                rows=[]
                for i,p in enumerate(layouts):
                    row=module_overlap(p);row.update(index=i,n=len(p),module_pass10_fraction=1-row['N10']/len(p));rows.append(row)
                metrics[name]=dict(mean={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ('index','n')},global_worst_module=max(r['max_covered_fraction'] for r in rows),worst_case=max(rows,key=lambda r:r['max_covered_fraction'])['index'])
                print('Geometry',name,'done',flush=True)
            all_geometry[role]=metrics;save(OUT/'geometry_summary.json',all_geometry)
    save(OUT/'completed.json',dict(status='complete',elapsed_seconds=time.perf_counter()-begin,panels=['train_monitor500','dev_monitor500'],model_SHA256=hashes,no_new_optimizer_or_solver_or_test_reads=True))
    print(json.dumps(dict(loss_summary=summaries,cached_base_check=cached_base_check,deployment_check=deployment_check,geometry=all_geometry),ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as e:save(OUT/'error.json',dict(error=repr(e),scope='diagnostic only; final solver and weights unchanged'));raise
