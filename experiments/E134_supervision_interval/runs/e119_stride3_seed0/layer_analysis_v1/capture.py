"""Observe frozen forward states, then evaluate Loss and local eval gradients."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[_key]='1'
import gc
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from experiments.E134_supervision_interval.model import GeometryModel
from experiments.E134_supervision_interval.evaluation.run import public_graph
from experiments.E134_supervision_interval.batches import decode
from experiments.E134_supervision_interval.loss import base_per_case,A,B
from experiments.E134_supervision_interval.data import load_rows
from experiments.E134_supervision_interval.resident import ResidentRows
from experiments.E134_supervision_interval.batches import inputs_only

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[4];RUN=HERE.parent;OWNER=HERE.parents[2]
HIDDEN_NAMES=['encoded','global1','S0',*[f'S{i}' for i in range(1,13)]]
BLOCK_NAMES=HIDDEN_NAMES[1:]
LOSS_NAMES=['total','position','shape','reference_weighted','overlap_weighted','wire_compact_weighted',
    'grouping_weighted','boundary_weighted','public_shape_weighted','public_preplaced_position_weighted',
    'penetration','module_load_mean','module_max_excess10','wirelength','compactness']

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def beat(activity,**fields):save(HERE/'capture_state.json',dict(activity=activity,**fields))

class ObservedPredictor:
    def __init__(self,checkpoint,inputs):
        checkpoint=torch.load(checkpoint,map_location='cpu',weights_only=False)
        self.model=GeometryModel(**checkpoint['model_config']).cuda().eval();self.model.load_state_dict(checkpoint['model'])
        self.g={k:v.cuda() for k,v in public_graph(inputs).items()}
        self.handles=[]
        self.handles.append(self.model.input.register_forward_hook(lambda module,args,value:self.h.__setitem__('encoded',value)))
        self.handles.append(self.model.head.register_forward_hook(lambda module,args,value:self.readouts.append(value)))
        blocks=[('global1',self.model.initial_blocks[0]),('S0',self.model.initial_blocks[1]),*[(f'S{i+1}',block) for i,block in enumerate(self.model.layers)]]
        for name,block in blocks:
            def observe(module,args,value,name=name):self.h[name]=value;self.block_input[name]=args[0]
            self.handles.append(block.register_forward_hook(observe))
            self.handles.append(block.qkv.register_forward_hook(lambda module,args,value,name=name:self.qkv.__setitem__(name,value)))
            if name not in ('global1','S0'):
                self.handles.append(block.bias.register_forward_hook(lambda module,args,value,name=name:self.bias.__setitem__(name,value)))
        def forward():
            self.h={};self.block_input={};self.qkv={};self.bias={};self.readouts=[]
            self.states=self.model(self.g)
            if len(self.readouts)!=13:raise ValueError('Existing shared head must emit S0..S12')
            return decode(self.states[-1],self.g)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):forward()
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=stream):self.output=forward()
        torch.cuda.synchronize()

    def predict(self,inputs):
        for key,value in public_graph(inputs).items():self.g[key].copy_(value)
        self.graph.replay()
        return (self.output[0,:int(inputs[0])]*self.g['scale'][0]).cpu().numpy().astype(float)

    def diagnostics(self):
        valid=self.g['valid'][0].float();n=valid.sum();w=64
        hidden=torch.stack([self.h[name][0] for name in HIDDEN_NAMES])
        rms=((hidden.square()*valid[None,:,None]).sum((1,2))/(n*w)).sqrt()
        centered=(hidden-(hidden*valid[None,:,None]).sum(1,keepdim=True)/n)*valid[None,:,None]
        covariance=centered.transpose(1,2)@centered/n
        trace=covariance.diagonal(dim1=-2,dim2=-1).sum(-1)
        rank=trace.square()/covariance.square().sum((1,2)).clamp_min(1e-20)
        block_input=torch.stack([self.block_input[name][0] for name in BLOCK_NAMES]);out=hidden[1:]
        norm=lambda value:((value.square()*valid[None,:,None]).sum((1,2))/(n*w)).sqrt()
        residual=norm(out-block_input)/norm(block_input).clamp_min(1e-12)
        injection=norm(block_input-hidden[:-1])/rms[:-1].clamp_min(1e-12)
        qkv=torch.stack([self.qkv[name][0] for name in BLOCK_NAMES])
        q,k,v=qkv.view(14,120,3,4,16).permute(2,0,3,1,4).unbind(0)
        zero=torch.zeros_like(self.bias['S1'][0])
        bias=torch.stack([zero if name in ('global1','S0') else self.bias[name][0] for name in BLOCK_NAMES]).permute(0,3,1,2)
        scores=(q@k.transpose(-2,-1))/4+bias
        scores=scores.masked_fill(~self.g['valid'][0][None,None,None,:],-1e4)
        attention=scores.softmax(-1)
        query_mean=lambda value:(value*valid[None,None,:]).sum((1,2))/(4*n)
        entropy=query_mean(-(attention*attention.clamp_min(1e-12).log()).sum(-1))/n.log()
        own=query_mean(attention.diagonal(dim1=-2,dim2=-1))
        wire=query_mean((attention*(self.g['adjacency'][0]>0)[None,None,:,:]).sum(-1))
        group=query_mean((attention*(self.g['group'][0]>0)[None,None,:,:]).sum(-1))
        return dict(hidden_rms=rms.cpu().numpy(),hidden_rank=rank.cpu().numpy(),block_residual=residual.cpu().numpy(),
            state_injection=injection.cpu().numpy(),attention=np.stack([x.cpu().numpy() for x in (entropy,own,wire,group)],-1))

def gradients(checkpoint,rows,pack):
    # Forward diagnostics allocate inference-only transfer buffers. Gradient
    # diagnostics must use fresh ordinary buffers, never reuse those tensors.
    pack=ResidentRows(rows,'cuda',max_nodes=120)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=GeometryModel(**saved['model_config']).cuda().eval();model.load_state_dict(saved['model'])
    named=list(model.named_parameters());parameters=[p for _,p in named]
    aggregate=[[torch.zeros_like(p) for p in parameters] for _ in range(4)]
    for start in range(0,500,32):
        batch=pack.batch(range(start,min(start+32,500)));states=model(inputs_only(batch))
        losses=[base_per_case(states[k-1],batch)['total'].sum()/500 for k in (3,6,9,12)]
        for j,value in enumerate(losses):
            grads=torch.autograd.grad(value,parameters,allow_unused=True,retain_graph=j<3)
            for total,grad in zip(aggregate[j],grads):
                if grad is not None:total.add_(grad)
        beat('frozen eval-gradient diagnostic',completed=min(start+32,500),total=500)
        if (start//32)%4==0:print('Gradients',min(start+32,500),'/500',flush=True)
    names=[name for name,_ in named];vectors=[np.concatenate([v.detach().cpu().numpy().ravel() for v in group]) for group in aggregate]
    slices={};offset=0;counts={}
    for name,p in named:
        if name.startswith('layers.'):group='geometry'+str(int(name.split('.')[1])+1)
        elif name.startswith('initial_blocks.'):group='global'+str(int(name.split('.')[1])+1)
        else:group=name.split('.')[0]
        slices.setdefault(group,[]).extend(range(offset,offset+p.numel()));counts[group]=counts.get(group,0)+p.numel();offset+=p.numel()
    weights=[.2/3,.2/3,.2/3,.8]
    norms={group:[float(np.linalg.norm(vector[ids])) for vector in vectors] for group,ids in slices.items()}
    subsets=dict(frontend=[i for group in ('input','global1','global2') for i in slices[group]],
        head=slices['head'],first_three=[i for group in ('geometry1','geometry2','geometry3') for i in slices[group]])
    cosines={}
    for group,ids in subsets.items():
        parts=np.stack([v[ids] for v in vectors]).astype(float);norm=np.linalg.norm(parts,axis=1)
        cosines[group]=(parts@parts.T/np.maximum(norm[:,None]*norm[None,:],1e-30)).tolist()
    if any(not torch.equal(value.cpu(),saved['model'][name]) for name,value in model.state_dict().items()):raise ValueError('Gradient diagnostic changed weights')
    save(HERE/'gradients.json',dict(supervised_layers=[3,6,9,12],weights=weights,parameter_counts=counts,
        dev_mean_gradient_norms=norms,weighted_gradient_norms={group:[x*w for x,w in zip(values,weights)] for group,values in norms.items()},
        cosine=cosines,mode='frozen eval-mode full dev500 derivatives; no optimizer or update',model_unchanged=True))
    np.savez_compressed(HERE/'gradient_vectors.npz',**{f'S{k}':v for k,v in zip((3,6,9,12),vectors)})

def main():
    from datasets.random_reference import RandomReference
    if (HERE/'manifest.json').exists():raise FileExistsError('Preserve prior analysis; no implicit rerun')
    started=time.perf_counter();plan=read(HERE/'plan.json');checkpoint=RUN/'checkpoints/best.pt'
    if sha(checkpoint)!=plan['checkpoint_sha256']:raise ValueError('Wrong baseline')
    gate=read(OWNER/'preflight.json')
    if any(sha(ROOT/name)!=digest for name,digest in gate['sources'].items()):raise ValueError('Frozen baseline sources changed')
    dataset=RandomReference('dev_monitor500');evaluation=RUN/'evaluation_no_mib_dev500'
    prior=ROOT/'experiments/E136_frontend_F_linear/runs/statistics_dev500_v1/manifest.json'
    prior_manifest=read(prior)
    if prior_manifest['models'][0]['checkpoint_sha256']!=plan['checkpoint_sha256']:raise ValueError('Prior metrics have another baseline')
    if prior_manifest['sources']['scoring/draft_geometry.py']!=sha(ROOT/'scoring/draft_geometry.py'):raise ValueError('Geometry definition changed')
    manifest=dict(checkpoint_sha256=sha(checkpoint),plan_sha256=sha(HERE/'plan.json'),evaluation=evaluation.relative_to(ROOT).as_posix(),
        members=[dataset.records[i]['member'] for i in dataset.indices],source_gate_sha256=sha(OWNER/'preflight.json'),
        hidden_names=HIDDEN_NAMES,block_names=BLOCK_NAMES,loss_names=LOSS_NAMES,prior_statistics_manifest_sha256=sha(prior),
        training=False,postprocessing_calls=0,Q_calls=0,test_read=False)
    save(HERE/'manifest.json',manifest)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    try:
        states=np.zeros((500,13,120,3),np.float32);positions=np.zeros((500,13,120,4),np.float64)
        reference=np.zeros((500,120,4),np.float64);areas=np.zeros((500,120),np.float64);flags=np.zeros((500,120,5),np.int64)
        hidden={name:np.zeros((500,120,64),np.float32) for name in HIDDEN_NAMES};features=np.zeros((500,120,16),np.float32)
        diagnostics=[]
        with torch.inference_mode():
            predictor=ObservedPredictor(checkpoint,dataset[0]['solver_inputs'])
            for i in range(500):
                item=dataset[i];record=read(evaluation/f'cases/{i:04d}.json');n=record['n']
                actual=predictor.predict(item['solver_inputs']);np.testing.assert_array_equal(actual,np.asarray(record['draft_positions']))
                states[i]=torch.stack(predictor.readouts)[:,0].cpu().numpy()
                for k,state in enumerate(predictor.readouts):positions[i,k,:n]=(decode(state,predictor.g)[0,:n]*predictor.g['scale'][0]).cpu().numpy().astype(float)
                np.testing.assert_array_equal(positions[i,12,:n],actual)
                for name in HIDDEN_NAMES:hidden[name][i]=predictor.h[name][0].cpu().numpy()
                features[i]=predictor.g['features'][0].cpu().numpy()
                reference[i,:n]=item['supervision']['reference_xywh'];areas[i,:n]=item['solver_inputs'][1];flags[i,:n]=item['solver_inputs'][5]
                d=predictor.diagnostics();diagnostics.append({k:v.tolist() for k,v in d.items()})
                if (i+1)%25==0:beat('capturing actual S0..S12 and hidden/attention',completed=i+1,total=500);print('Capture',i+1,'/500',flush=True)
        for name,value in [('states',states),('positions',positions),('reference',reference),('areas',areas),('flags',flags),('features',features)]:np.save(HERE/(name+'.npy'),value,allow_pickle=False)
        (HERE/'hidden').mkdir()
        for name,value in hidden.items():np.save(HERE/'hidden'/f'{name}.npy',value,allow_pickle=False)
        save(HERE/'hidden_attention.json',diagnostics)
        old_gallery=read(RUN/'visualization_dev15_loss/data.json')
        for case in old_gallery['cases']:
            np.testing.assert_array_equal(positions[case['dev_index'],1:,:case['n']],np.asarray(case['layers']))
        save(HERE/'arrays_ready.json',dict(status='passed',cases=500,states=13,old_gallery_all_layers_bitwise_equal=True,terminal_bitwise_equal=True))
        del predictor,hidden,features,diagnostics;gc.collect();torch.cuda.empty_cache()
        rows=load_rows('dev_monitor500');pack=ResidentRows(rows,'cuda',max_nodes=120)
        loss_array=np.zeros((500,13,len(LOSS_NAMES)),np.float64)
        with torch.inference_mode():
            for start in range(0,500,32):
                ids=list(range(start,min(start+32,500)));batch=pack.batch(ids)
                raw=torch.as_tensor(states[ids],device='cuda')
                for layer in range(13):
                    values=base_per_case(raw[:,layer],batch)
                    loss_array[ids,layer]=torch.stack([values[name] for name in LOSS_NAMES],-1).cpu().double().numpy()
                beat('all-layer common Loss',completed=ids[-1]+1,total=500)
        np.save(HERE/'losses.npy',loss_array,allow_pickle=False)
        frozen=read(OWNER/'evaluation/frozen_e119_stride3.json')
        if abs(loss_array[:,12,0].mean()-frozen['own_new_final_dev_loss'])>5e-5:raise ValueError('Terminal mean Loss differs')
        gradients(checkpoint,rows,pack)
        if sha(checkpoint)!=plan['checkpoint_sha256']:raise ValueError('Checkpoint changed')
        save(HERE/'capture_state.json',dict(status='complete',seconds=time.perf_counter()-started,cases=500,states=13,training=False,postprocessing_calls=0))
        print('Frozen capture, all-layer Loss and gradient diagnostics complete.',flush=True)
    except BaseException as error:
        save(HERE/'capture_state.json',dict(status='failed',error=repr(error),traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
