"""One reusable graph per model at public max120/480, on the same five inputs."""
from pathlib import Path
import datetime,json,time,hashlib,gc
import numpy as np
from .latency import torch,ROOT,HERE,IDS,public_graph,public_member_inputs,registry,save,stats,Captured,timed,DeploymentModel,GeometryModel,decode85,decode96,sha

FOLDER=HERE/'runs/20261101_fixed_shape_latency'

def padded(inputs,full):
    g=public_graph(inputs,full);n=int(inputs[0]);p=len(inputs[4])
    if n>120 or p>480:raise ValueError('Outside registered public bounds')
    result={}
    for k,v in g.items():
        if k=='scale':result[k]=v;continue
        shape=list(v.shape);shape[1]=480 if k in ('port_xy','port_valid') else 120
        if k in ('adjacency','group','mib','b2b_weights'):shape[2]=120
        elif k=='p2b_weights':shape[2]=480
        result[k]=torch.zeros(shape,dtype=v.dtype)
        result[k][tuple(slice(0,s) for s in v.shape)].copy_(v)
    return result

def returned(state,g,inputs,decode):
    p=(decode(state,g)[0,:int(inputs[0])]*g['scale'][0]).cpu().numpy().astype(float)
    c,s=np.asarray(inputs[5]),np.asarray(inputs[6]);locked=c[:,:2].any(1)
    p[locked,2:]=s[locked,2:];p[c[:,1]!=0]=s[c[:,1]!=0]
    return p

@torch.inference_mode()
def main():
    begin=time.perf_counter();FOLDER.mkdir(exist_ok=False)
    torch.manual_seed(0);torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    study=json.loads((HERE/'study.json').read_text(encoding='utf-8'))
    if datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime.fromisoformat(study['deadline_utc']):raise RuntimeError('Study deadline reached')
    records=registry()['sets']['train']['records'];cases=[public_member_inputs(records[i],'train') for i in IDS]
    specs=[('E85_full_draft','experiments/E85_cooperative_space_refinement/runs/20261030_P74_F81_cooperative_seed0/checkpoints/best.pt',DeploymentModel,False,decode85),('E96_full_draft','experiments/E96_port_compression/runs/20261101_centroid_bbox05_seed0/checkpoints/best.pt',GeometryModel,True,decode96)]
    save(FOLDER/'registration.json',dict(public_shape=[1,120,480],new_inputs=0,train_ids=IDS,repetitions=20,warmups=3,changes='one cached graph/model reused across all five distinct public shapes; CPU padding and all transfers included; deployment gap check, not additional coefficient candidates',source_sha256=sha(Path(__file__)),test_read=False))
    results=[];aggregate=[]
    for name,relative,constructor,full,decode in specs:
        loaded=torch.load(ROOT/relative,map_location='cpu',weights_only=False);model=constructor(**loaded['model_config']).cuda().eval();model.load_state_dict(loaded['model'],strict=True)
        g={k:v.cuda() for k,v in padded(cases[0],full).items()};start=time.perf_counter();cap=Captured(model,g);capture_seconds=time.perf_counter()-start
        wall_all=[];resident_all=[]
        for index,inputs in zip(IDS,cases):
            natural={k:v.cuda() for k,v in public_graph(inputs,full).items()};expected=returned(model(natural),natural,inputs,decode)
            host=padded(inputs,full)
            for k,v in host.items():cap.g[k].copy_(v)
            actual=returned(cap.replay(),cap.g,inputs,decode)
            error=float(np.max(np.abs(expected-actual)));normalized=error/float(host['scale'][0])
            assert normalized<2e-4,(name,index,normalized)
            for _ in range(3):cap.replay()
            resident,resident_raw=timed(cap.replay,'cuda',20)
            def e2e():
                h=padded(inputs,full)
                for k,v in h.items():cap.g[k].copy_(v)
                return returned(cap.replay(),cap.g,inputs,decode)
            for _ in range(3):e2e()
            measured,raw=timed(e2e,'cuda',20);wall_all.extend(raw);resident_all.extend(resident_raw)
            results.append(dict(model=name,train_id=index,n=int(inputs[0]),resident_forward=resident,public_to_draft=measured,natural_fixed_normalized_max_abs=normalized))
            print(name,index,'fixed120 public-to-draft mean/P95 ms',measured['wall']['mean_ms'],measured['wall']['p95_ms'],flush=True)
        aggregate.append(dict(model=name,public_to_draft=stats(wall_all),resident_forward=stats(resident_all),capture_seconds_excluded=capture_seconds,checkpoint_sha256=sha(ROOT/relative)))
        del cap,model;gc.collect();torch.cuda.empty_cache()
    save(FOLDER/'results.json',dict(status='passed',aggregate=aggregate,records=results,elapsed_seconds=time.perf_counter()-begin,batch_size=1,unique_inputs=5,shape=[120,480],no_solver=True,test_read=False))
    print(json.dumps(aggregate,indent=2),flush=True)

if __name__=='__main__':main()
