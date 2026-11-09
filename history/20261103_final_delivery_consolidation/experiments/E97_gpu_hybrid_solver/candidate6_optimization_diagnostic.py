"""Original-five train activation/gradient audit; no optimization or final Q calls."""
import json
from pathlib import Path
from .candidate6 import HERE,torch,IDS,seed,teacher_rows,load_rows,attach,Packer,per_case,cpu,sha
from .deep_geometry import DeploymentModel
from .train_io.io import save

def main():
    seed();labels=teacher_rows('train');rows=attach(load_rows('train',IDS),labels,IDS);packer=Packer(rows[0]);g,d,t,u,_=packer.gpu([rows[i%5] for i in range(32)])
    root=HERE/'runs/20261102_candidate6_deep_geometry_seed0/checkpoints'
    paths={'untrained':root/'untrained.pt','first_preflight_Adam':HERE/'runs/20261102_candidate6_preflight/step1.pt','best':root/'best.pt','latest':root/'latest.pt'}
    initial=torch.load(paths['untrained'],map_location='cpu',weights_only=False)['model'];results={}
    for label,path in paths.items():
        saved=torch.load(path,map_location='cpu',weights_only=False);cfg=saved.get('model_config') or torch.load(root/'untrained.pt',map_location='cpu',weights_only=False)['model_config']
        model=DeploymentModel(**cfg).cuda().eval();model.load_state_dict(saved['model'],strict=True);model.requires_grad_(False);model.refiner.adapter.requires_grad_(True)
        mask=g['valid'];features={};handles=[]
        def hook(name):
            def record(module,args,value):
                a=value.detach()[mask];features.setdefault(name,[]).append(dict(rms=float(a.square().mean().sqrt().cpu()),max_abs=float(a.abs().max().cpu()),positive_fraction=float((a>0).float().mean().cpu()),zero_fraction=float((a==0).float().mean().cpu()),node_std_mean=float(a.std(dim=0).mean().cpu())))
            return record
        for index,layer in enumerate(model.refiner.adapter.layers):handles.append(layer.register_forward_hook(hook('block'+str(index))))
        handles.append(model.refiner.adapter.head[0].register_forward_hook(hook('readout_pre_SiLU')));handles.append(model.refiner.adapter.head[1].register_forward_hook(hook('readout_post_SiLU')))
        prediction=model.refiner(d,g);loss=per_case(prediction,g,t,u)['total'].mean();loss.backward()
        gradients={k:dict(norm=float(v.grad.norm().cpu()),nonzero_fraction=float((v.grad!=0).float().mean().cpu())) for k,v in model.refiner.adapter.named_parameters() if v.grad is not None and k in ['input.0.weight','layers.0.qkv.weight','head.0.weight','head.0.bias','head.2.weight','head.2.bias']}
        change={k:float((v.cpu()-initial['refiner.adapter.'+k]).norm()) for k,v in model.refiner.adapter.state_dict().items() if k in ['input.0.weight','layers.0.qkv.weight','head.0.weight','head.2.weight','head.2.bias']}
        with torch.no_grad():
            base=model.refiner.draft(d,g);local=g['area'].clamp_min(1e-12).sqrt();unit=torch.stack((local,local,torch.ones_like(local)),-1);correction=((prediction-base)/unit)[mask]
            detail=dict(local_scaled_output_rms=float(correction.square().mean().sqrt().cpu()),local_scaled_output_max=float(correction.abs().max().cpu()),new_input5_columns_norm=float(model.refiner.adapter.input[0].weight[:,17:].norm().cpu()))
        results[label]=dict(checkpoint_sha256=sha(path),loss=float(loss.detach().cpu()),features=features,gradients=gradients,parameter_change_from_initial=change,**detail)
        for h in handles:h.remove()
        del model,prediction,saved,loss
        torch.cuda.empty_cache()
    save(HERE/'runs/candidate6_optimization_diagnostic.json',dict(status='complete',ids=IDS,records=results,no_optimizer_updates=True,no_Q_or_dev_calls=True,formal_slots_consumed=0,test_read=False))
    for name,row in results.items():print(name,json.dumps({k:v for k,v in row.items() if k not in ['features','checkpoint_sha256']}), 'head_features',json.dumps(row['features']['readout_pre_SiLU']),flush=True)

if __name__=='__main__':main()
