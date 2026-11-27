"""Five public train inputs, batch-one latency; no solver, labels, or training."""
from pathlib import Path
import datetime, gc, hashlib, json, subprocess, time
import numpy as np
import torch
from datasets.paths import ROOT
from datasets.reference_layouts import registry
from datasets.public_member_inputs import public_member_inputs
from .nn85.model import DeploymentModel
from .nn85.base_model import decode as decode85
from .nn85.inference import graph
from .nn85.frame import contain, predicted_frame
from .nn85.mib_projection import coherent_shapes
from .nn85.cooperation import spatial_centers
from .nn96.model import GeometryModel
from .nn96.base_model import decode as decode96

HERE=Path(__file__).resolve().parent
FOLDER=HERE/'runs/20261101_draft_latency'
IDS=[1683,10027,5474,10140,15001]

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def stats(values):
    a=np.asarray(values)*1000
    return dict(mean_ms=float(a.mean()),median_ms=float(np.median(a)),p95_ms=float(np.percentile(a,95)),max_ms=float(a.max()),repetitions=len(a))

def public_graph(inputs,full):
    g=graph(inputs)
    if not full:return g
    n,_,b2b,p2b,pins,_,_=inputs
    pins=torch.as_tensor(pins,dtype=torch.float32)
    xy=pins[:,:2]/g['scale'][0]
    pw=torch.zeros(int(n),len(pins));bw=torch.zeros(int(n),int(n))
    for original,port in ((b2b,False),(p2b,True)):
        edges=torch.as_tensor(original,dtype=torch.float32)
        edges=edges[~(edges==-1).all(1)]
        if not len(edges):continue
        a,b=edges[:,0].long(),edges[:,1].long();w=edges[:,2]
        if port:pw.index_put_((b,a),w,accumulate=True)
        else:
            bw.index_put_((a,b),w,accumulate=True);bw.index_put_((b,a),w,accumulate=True)
    g.update(port_xy=xy[None],port_valid=torch.ones(1,len(pins),dtype=torch.bool),p2b_weights=pw[None],b2b_weights=bw[None])
    return g

def returned(state,g,inputs,decode):
    p=(decode(state,g)[0]*g['scale'][0]).cpu().numpy().astype(float)
    c,s=np.asarray(inputs[5]),np.asarray(inputs[6]);locked=c[:,:2].any(1)
    p[locked,2:]=s[locked,2:];p[c[:,1]!=0]=s[c[:,1]!=0]
    return p

def timed(fn,device,repeats):
    walls=[];stream=[]
    for _ in range(repeats):
        if device=='cuda':
            torch.cuda.synchronize();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
        start=time.perf_counter();fn()
        if device=='cuda':b.record();torch.cuda.synchronize()
        walls.append(time.perf_counter()-start)
        if device=='cuda':stream.append(a.elapsed_time(b)/1000)
    return dict(wall=stats(walls),cuda_stream=stats(stream) if stream else None),walls

class Captured:
    def __init__(self,fn,g):
        self.g={k:v.clone() for k,v in g.items()};s=torch.cuda.Stream();s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):fn(self.g)
        torch.cuda.current_stream().wait_stream(s);torch.cuda.synchronize()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=s):self.output=fn(self.g)
        torch.cuda.synchronize()
    def replay(self):self.graph.replay();return self.output

def split85(model,g,device):
    events=[]
    def run(name,fn):
        if device=='cuda':
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();value=fn();b.record();events.append((name,a,b))
        else:
            start=time.perf_counter();value=fn();events.append((name,time.perf_counter()-start,None))
        return value
    base=run('L_neural',lambda:model.placer(g));f=model.refiner
    local=g['area'].clamp_min(1e-12).sqrt();bounds=.15*torch.stack((local,local,torch.ones_like(local)),-1)
    first=run('F1_neural',lambda:f.first(base,g))
    def frame1():
        frame=predicted_frame(base,g)
        old=base+first.tanh()*bounds*g['mask']*g['valid'][...,None]
        return contain(old,g,frame),frame
    anchor,frame=run('frame1_fixed',frame1)
    second=run('F2_neural',lambda:f.second(anchor,g))
    source=run('frame2_MIB_fixed',lambda:coherent_shapes(contain(anchor+second.tanh()*bounds*g['mask']*g['valid'][...,None],g,frame),g,frame))
    third=run('F3_neural',lambda:f.third(source,g))
    def final():
        proposed=source[...,:2]+third.tanh()*.15*local[...,None]*g['mask'][...,:2]*g['valid'][...,None]
        centers=spatial_centers(source,proposed,g,frame,.15)
        return torch.cat((centers,source[...,2:3]),-1)*g['valid'][...,None]
    result=run('eight_steps_fixed',final)
    if device=='cuda':torch.cuda.synchronize()
    return result,{name:(a.elapsed_time(b) if device=='cuda' else a*1000) for name,a,b in events}

@torch.inference_mode()
def main():
    begin=time.perf_counter();FOLDER.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(0);torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    plan=json.loads((HERE/'plan.json').read_text(encoding='utf-8'));repeats=plan['stage0']['repetitions'];warmups=plan['stage0']['warmups']
    records=registry()['sets']['train']['records'];cases=[public_member_inputs(records[i],'train') for i in IDS]
    path85=ROOT/'experiments/E85_cooperative_space_refinement/runs/20261030_P74_F81_cooperative_seed0/checkpoints/best.pt'
    path96=ROOT/'experiments/E96_port_compression/runs/20261101_centroid_bbox05_seed0/checkpoints/best.pt'
    expected=json.loads((HERE/'provenance.json').read_text(encoding='utf-8'))['protected_checkpoints']
    assert sha(path85)==expected['E85'] and sha(path96)==expected['E96']
    saved85=torch.load(path85,map_location='cpu',weights_only=False);saved96=torch.load(path96,map_location='cpu',weights_only=False)
    metadata=dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),device=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,utilization.gpu,power.draw','--format=csv,noheader'],text=True).strip(),torch_version=torch.__version__,threads=2,batch=1,precision='FP32 no AMP/TF32',repeats=repeats,warmups=warmups,train_ids=IDS,members=[records[i]['member'] for i in IDS],cold_start_excluded=True,files_loading_excluded=True,quality_not_evaluated=True,test_read=False,source_sha256={p.relative_to(HERE).as_posix():sha(p) for p in HERE.rglob('*.py') if 'runs' not in p.parts})
    save(FOLDER/'registration.json',metadata)
    results=[];reference={};pooled={};parity=[]
    for device in ('cpu','cuda'):
        model85=DeploymentModel(**saved85['model_config']).to(device).eval();model85.load_state_dict(saved85['model'],strict=True)
        config96=dict(saved96['model_config']);config96['attention_backend']='manual' if device=='cpu' else 'efficient'
        model96=GeometryModel(**config96).to(device).eval();model96.load_state_dict(saved96['model'],strict=True)
        models=[('L74',model85.placer,False,decode85),('E85_full_draft',model85,False,decode85),('L90',model96.draft,True,decode96),('E96_full_draft',model96,True,decode96)]
        for ci,inputs in enumerate(cases):
            for name,fn,full,decode in models:
                gh=public_graph(inputs,full);g={k:v.to(device) for k,v in gh.items()}
                for _ in range(warmups):fn(g)
                state=fn(g);positions=returned(state,g,inputs,decode)
                if device=='cpu':reference[name,ci]=positions
                else:
                    error=float(np.max(np.abs(positions-reference[name,ci])));scale=float(gh['scale'][0]);normalized=error/scale
                    parity.append(dict(model=name,train_id=IDS[ci],CPU_GPU_max_abs=error,normalized_max_abs=normalized))
                    if normalized>2e-4:raise AssertionError(f'Unexpected CPU/GPU drift {name} {normalized}')
                resident,resident_raw=timed(lambda:fn(g),device,repeats)
                def end_to_end():
                    host=public_graph(inputs,full);batch={k:v.to(device) for k,v in host.items()}
                    return returned(fn(batch),batch,inputs,decode)
                for _ in range(warmups):end_to_end()
                e2e,e2e_raw=timed(end_to_end,device,repeats)
                value=dict(model=name,device=device,path='eager',train_id=IDS[ci],n=int(inputs[0]),ports=len(inputs[4]),resident_forward=resident,public_to_draft=e2e)
                if name=='E85_full_draft':
                    staged,stage_times=split85(model85,g,device)
                    torch.testing.assert_close(staged,state,rtol=0,atol=0)
                    value['stage_trace_ms']=stage_times
                results.append(value);pooled.setdefault((name,device,'eager'),[]).extend(e2e_raw)
                print(name,device,IDS[ci],'resident_ms',resident['wall']['median_ms'],'public_to_draft_ms',e2e['wall']['median_ms'],flush=True)
                save(FOLDER/'progress.json',dict(records=results,CPU_GPU_parity=parity,finished=False,test_read=False))
                if device=='cuda':
                    capture_start=time.perf_counter();cap=Captured(fn,g);capture_seconds=time.perf_counter()-capture_start
                    captured=cap.replay();torch.cuda.synchronize();torch.testing.assert_close(captured,state,rtol=1e-5,atol=2e-6)
                    cr,cr_raw=timed(cap.replay,device,repeats)
                    def captured_e2e():
                        host=public_graph(inputs,full)
                        for k,v in host.items():cap.g[k].copy_(v)
                        return returned(cap.replay(),cap.g,inputs,decode)
                    for _ in range(warmups):captured_e2e()
                    ce,ce_raw=timed(captured_e2e,device,repeats)
                    results.append(dict(model=name,device=device,path='cuda_graph',train_id=IDS[ci],n=int(inputs[0]),ports=len(inputs[4]),resident_forward=cr,public_to_draft=ce,capture_seconds_excluded=capture_seconds))
                    pooled.setdefault((name,device,'cuda_graph'),[]).extend(ce_raw)
                    print(name,device,'graph',IDS[ci],'resident_ms',cr['wall']['median_ms'],'public_to_draft_ms',ce['wall']['median_ms'],flush=True)
                    del cap;gc.collect()
        del model85,model96;gc.collect()
        if device=='cuda':torch.cuda.empty_cache()
    aggregate=[dict(model=name,device=device,path=path,public_to_draft=stats(values)) for (name,device,path),values in pooled.items()]
    result=dict(metadata=metadata,status='passed',scope='five exposed train problems repeated20; deployment measurement, not full-dev or final legal solve evidence',records=results,aggregate=aggregate,CPU_GPU_parity=parity,elapsed_seconds=time.perf_counter()-begin,test_read=False)
    save(FOLDER/'results.json',result)
    print(json.dumps(aggregate,indent=2),flush=True)

if __name__=='__main__':main()
