"""Frozen-model structural audit and explicit-feedback ablations, never training."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

OWNER=Path(__file__).resolve().parent
ROOT=OWNER.parents[1]
RUN=OWNER/'runs/e119_stride3_seed0'
PRIOR=RUN/'layer_analysis_v1'
OUT=RUN/'unsupervised_audit_v1'
LAYERS=list(range(3,13))
UNSUP=[4,5,7,8,10,11]

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def finite(value):return float(value) if np.isfinite(value) else None
def check():
    plan=read(OUT/'plan.json');manifest=read(PRIOR/'manifest.json')
    assert sha(RUN/'checkpoints/best.pt')==plan['checkpoint_sha256']==manifest['checkpoint_sha256']
    assert all(sha(ROOT/name)==digest for name,digest in read(OWNER/'preflight.json')['sources'].items())
    return plan,manifest
def persistent(good):
    stable=np.logical_and.accumulate(np.asarray(good,bool)[::-1],axis=0)[::-1]
    return int(np.flatnonzero(stable)[0])+3 if stable.any() else None

def largest_empty(p):
    from numba import njit
    global _empty_kernel
    if '_empty_kernel' not in globals():
        @njit
        def kernel(occupied,x,y):
            ny,nx=occupied.shape
            heights=np.zeros(nx);best=0.;rect=np.zeros(4)
            stack_i=np.empty(nx+1,np.int64);stack_h=np.empty(nx+1)
            for row in range(ny):
                for col in range(nx):
                    heights[col]=0. if occupied[row,col] else heights[col]+y[row+1]-y[row]
                size=0
                for col in range(nx+1):
                    h=heights[col] if col<nx else 0.
                    left=col
                    while size>0 and stack_h[size-1]>h:
                        size-=1;left=stack_i[size];old=stack_h[size]
                        area=(x[col]-x[left])*old
                        if area>best:
                            best=area;rect[0]=x[left];rect[1]=y[row+1]-old;rect[2]=x[col]-x[left];rect[3]=old
                    if h>0 and (size==0 or stack_h[size-1]<h):
                        stack_i[size]=left;stack_h[size]=h;size+=1
            return best,rect
        _empty_kernel=kernel
    lo=p[:,:2];hi=lo+p[:,2:]
    x=np.unique(np.r_[lo[:,0],hi[:,0]]);y=np.unique(np.r_[lo[:,1],hi[:,1]])
    a=np.searchsorted(x,lo[:,0]);b=np.searchsorted(x,hi[:,0])
    c=np.searchsorted(y,lo[:,1]);d=np.searchsorted(y,hi[:,1])
    diff=np.zeros((len(x),len(y)),np.int32)
    for xx,yy,v in ((a,c,1),(b,c,-1),(a,d,-1),(b,d,1)):np.add.at(diff,(xx,yy),v)
    occupied=(diff.cumsum(0).cumsum(1)[:-1,:-1]>0).T.copy()
    area,rect=_empty_kernel(occupied,x,y)
    return dict(area=float(area),bbox_fraction=float(area/((x[-1]-x[0])*(y[-1]-y[0]))),rectangle=rect.tolist())

def blockers(p,i,j,foreign,areas):
    center=p[:,:2]+p[:,2:]/2
    a=center[i];v=center[j]-a
    inset=.02*np.sqrt(areas[foreign])[:,None]
    lo=p[foreign,:2]+inset;hi=p[foreign,:2]+p[foreign,2:]-inset
    lower=np.full(len(foreign),.15);upper=np.full(len(foreign),.85)
    for axis in range(2):
        if abs(v[axis])<1e-12:
            valid=(a[axis]>lo[:,axis])&(a[axis]<hi[:,axis])
            upper=np.where(valid,upper,-1.)
        else:
            t1=(lo[:,axis]-a[axis])/v[axis];t2=(hi[:,axis]-a[axis])/v[axis]
            lower=np.maximum(lower,np.minimum(t1,t2));upper=np.minimum(upper,np.maximum(t1,t2))
    return upper>lower

def boundary(p,flags,areas):
    lo=p[:,:2];hi=lo+p[:,2:];center=lo+p[:,2:]/2
    bblo=lo.min(0);bbhi=hi.max(0);local=np.sqrt(areas)
    result=[]
    for i in np.flatnonzero(flags[:,4]):
        for bit,axis,outward in ((1,0,-1),(2,0,1),(4,1,1),(8,1,-1)):
            if not flags[i,4]&bit:continue
            tangent=1-axis
            gap=(lo[i,axis]-bblo[axis] if outward<0 else bbhi[axis]-hi[i,axis])/local[i]
            overlap=np.minimum(hi[i,tangent],hi[:,tangent])-np.maximum(lo[i,tangent],lo[:,tangent])
            farther=(center[:,axis]-center[i,axis])*outward>.05*local[i]
            shadow=farther&(overlap>=.1*np.minimum(p[i,2+tangent],p[:,2+tangent]))
            shadow[i]=False
            result.append(dict(module=int(i),bit=bit,gap=float(gap),shadow=int(shadow.sum())))
    return result

def pair_motion(p,areas,flags):
    n=len(areas);center=p[:,:,:2]+p[:,:,2:]/2
    scale=(np.sqrt(areas)[:,None]+np.sqrt(areas)[None,:])/2
    d=center[:,:,None,:]-center[:,None,:,:]
    half=(p[:,:,None,2:]+p[:,None,:,2:])/2
    pairs=np.triu(np.ones((n,n),bool),1)&~((flags[:,1]!=0)[:,None]&(flags[:,1]!=0)[None,:])
    dist=np.linalg.norm(d[0]/scale[...,None],axis=-1);np.fill_diagonal(dist,np.inf)
    order=np.argsort(dist,axis=1,kind='stable')[:,:min(6,n-1)]
    near=np.zeros((n,n),bool);near[np.arange(n)[:,None],order]=True
    near=(near|near.T)&pairs
    clear=np.where(np.abs(d)-half>.02*scale[None,:,:,None],np.sign(d),0)
    last=clear[0].copy();pending=np.zeros((n,n,2),bool)
    output=[];events=[]
    for k in range(1,10):
        old=d[k-1];new=d[k]
        orderflip=(old*new<0)&(np.abs(old)>.05*scale[...,None])&(np.abs(new)>.05*scale[...,None])
        eligible=(np.abs(old)>.05*scale[...,None])&near[...,None]
        for axis in range(2):
            cross=old[...,axis]*new[...,axis]<0
            t=np.divide(old[...,axis],old[...,axis]-new[...,axis],out=np.zeros_like(scale),where=cross)
            orth=1-axis
            rel=old[...,orth]+t*(new[...,orth]-old[...,orth])
            width=half[k-1,...,orth]+t*(half[k,...,orth]-half[k-1,...,orth])
            through=cross&(width-np.abs(rel)>.02*scale)&pairs
            pending[...,axis]|=through
        exchange=(last*clear[k]<0)&pending&pairs[...,None]
        for i,j,axis in np.argwhere(exchange):events.append(dict(layer=k+3,i=int(i),j=int(j),axis=int(axis),local=bool(near[i,j])))
        resolved=clear[k]!=0
        last=np.where(resolved,clear[k],last);pending=np.where(resolved,False,pending)
        movement=(center[k]-center[k-1]);raw=np.linalg.norm(movement,axis=-1)/np.sqrt(areas)
        relative=np.linalg.norm(movement-np.median(movement,axis=0),axis=-1)/np.sqrt(areas)
        changed_nodes=np.any(orderflip&near[...,None],axis=(1,2))|np.any(orderflip&near[...,None],axis=(0,2))
        full_nodes=np.any(exchange,axis=(1,2))|np.any(exchange,axis=(0,2))
        output.append(dict(layer=k+3,local_pair_axes=int(eligible.sum()),
            local_clear_order_reversals=int((orderflip&near[...,None]).sum()),
            local_clear_order_reversal_rate=float((orderflip&near[...,None]).sum()/max(eligible.sum(),1)),
            full_side_exchanges=int(exchange.sum()),local_full_side_exchanges=int((exchange&near[...,None]).sum()),
            modules_in_full_exchange=int(full_nodes.sum()),modules_with_order_change=float(changed_nodes.mean()),
            raw_move_mean=float(raw.mean()),translation_removed_move_mean=float(relative.mean()),
            large_move025=float((relative>=.25).mean()),large_move05=float((relative>=.5).mean()),large_move1=float((relative>=1.).mean())))
    cumulative=(d[0]*d[-1]<0)&(np.abs(d[0])>.05*scale[...,None])&(np.abs(d[-1])>.05*scale[...,None])&near[...,None]
    return output,events,dict(local_final_order_reversal_rate=float(cumulative.sum()/max((near[...,None]&(np.abs(d[0])>.05*scale[...,None])).sum(),1)))

def observe():
    from scoring.draft_geometry import contact_distances,minimum_tree
    plan,manifest=check();started=time.perf_counter()
    assert not (OUT/'observations.json').exists()
    positions=np.load(PRIOR/'positions.npy',mmap_mode='r')
    flags=np.load(PRIOR/'flags.npy');areas=np.load(PRIOR/'areas.npy')
    loss=np.load(PRIOR/'losses.npy')
    results=[];(OUT/'cases').mkdir(exist_ok=True)
    for index in range(500):
        old=read(PRIOR/f'cases/{index:04d}.json');n=old['n']
        assert old['member']==manifest['members'][index]
        p=np.asarray(positions[index,3:,:n]);a=areas[index,:n];f=flags[index,:n]
        motion,exchanges,cumulative=pair_motion(p,a,f)
        void=[largest_empty(value) for value in p]
        ref=np.load(PRIOR/'reference.npy',mmap_mode='r')[index,:n]
        refvoid=largest_empty(ref)
        bound=[boundary(value,f,a) for value in p]
        assert all([(r['module'],r['bit']) for r in value]==[(r['module'],r['bit']) for r in bound[0]] for value in bound)
        bitems=[]
        for j,item in enumerate(bound[0]):
            gaps=[value[j]['gap'] for value in bound];shadows=[value[j]['shadow'] for value in bound]
            bitems.append(dict(module=item['module'],bit=item['bit'],gaps=gaps,shadows=shadows,
                initial_bad=bool(gaps[0]>.1),initial_bad_blocked=bool(gaps[0]>.1 and shadows[0]>0),
                repair_layers={str(t):persistent(np.array(gaps)<=t) for t in (.01,.02,.05)}))
        distances=[contact_distances(value)[0] for value in p]
        gedges=[]
        for group in sorted(set(f[:,3])-{0}):
            members=np.flatnonzero(f[:,3]==group);foreign=np.flatnonzero(f[:,3]!=group)
            if len(members)<2:continue
            for u,v,_ in minimum_tree(distances[0][np.ix_(members,members)]):
                i,j=int(members[u]),int(members[v])
                block=np.stack([blockers(value,i,j,foreign,a) for value in p])
                count=block.sum(1);cost=np.array([d[i,j] for d in distances])
                initial=block[0];bad=bool(count[0]>0 and cost[0]>.05)
                gedges.append(dict(group=int(group),i=i,j=j,initial_bad=bad,
                    contact=cost.tolist(),blocker_counts=count.tolist(),
                    original_blocker_counts=block[:,initial].sum(1).tolist(),
                    new_blocker_counts=block[:,~initial].sum(1).tolist(),
                    initial_blockers=foreign[initial].tolist(),
                    all_clear_layer=persistent(count==0),
                    clear_and_contact_halved_layer=persistent((count==0)&(cost<=cost[0]/2)),
                    original_clear_layer=persistent(block[:,initial].sum(1)==0) if initial.any() else None))
        windows=[]
        centers=p[:,:,:2]+p[:,:,2:]/2
        for anchor in (3,6,9):
            k=anchor-3;delta=centers[k:k+4]-centers[k]
            delta-=np.median(delta,axis=1)[:,None,:]
            move=np.linalg.norm(delta,axis=-1)/np.sqrt(a)[None,:]
            peak=move[1:3].max(0);large=peak>=.5
            undone=large&(move[3]<=.5*peak)
            windows.append(dict(anchor=anchor,end=anchor+3,large_excursion_modules=int(large.sum()),
                undone_modules=int(undone.sum()),n=n,
                loss_sacrifice=bool(loss[index,anchor+1:anchor+3,0].max()>loss[index,anchor,0] and loss[index,anchor+3,0]<loss[index,anchor,0])))
        row=dict(index=index,member=old['member'],n=n,motion=motion,exchanges=exchanges,cumulative=cumulative,
            void=void,reference_void=refvoid,boundary=bitems,group_edges=gedges,windows=windows,
            geometry=[value['geometry'] for value in old['layers'][3:]])
        save(OUT/f'cases/{index:04d}.json',row)
        results.append(row)
        if (index+1)%50==0:
            print('Observed',index+1,'/500',flush=True)
            save(OUT/'observe_status.json',dict(stage='observations',completed=index+1))
    save(OUT/'observations.json',dict(cases=500,seconds=time.perf_counter()-started,
        positions_sha256=sha(PRIOR/'positions.npy'),source_sha256=sha(Path(__file__)),plan_sha256=sha(OUT/'plan.json')))
    print('Observation audit complete',flush=True)

def intervene():
    import torch
    from .model import GeometryModel
    from .evaluation.run import public_graph
    from .data import load_rows
    from .resident import ResidentRows
    from .loss import base_per_case
    from datasets.random_reference import RandomReference
    plan,manifest=check();started=time.perf_counter()
    assert not (OUT/'interventions.json').exists()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    saved=torch.load(RUN/'checkpoints/best.pt',map_location='cpu',weights_only=False)
    model=GeometryModel(**saved['model_config']).cuda().eval();model.load_state_dict(saved['model'])
    ds=RandomReference('dev_monitor500')
    assert [ds.records[i]['member'] for i in ds.indices]==manifest['members']
    graphs=[public_graph(ds[i]['solver_inputs']) for i in range(500)]
    g={key:value.cuda() for key,value in graphs[0].items()}
    old=np.load(PRIOR/'states.npy',mmap_mode='r');oldloss=np.load(PRIOR/'losses.npy')
    modes={'original':('none',[]),'skip45':('skip',[4,5]),'skip78':('skip',[7,8]),'skip1011':('skip',[10,11]),
        'hold45':('hold',[4,5]),'hold78':('hold',[7,8]),'hold1011':('hold',[10,11]),'hold_all':('hold',UNSUP)}
    results={};pack=None
    with torch.inference_mode():
        for name,(kind,targets) in modes.items():
            def forward():
                h=model.input(g['features'])*g['valid'][...,None]
                for block in model.initial_blocks:h=block(h,g)
                state=model.head(h);readouts=[state]
                for index,block in enumerate(model.layers,1):
                    if kind=='skip' and index in targets:
                        readouts.append(state);continue
                    h=block(h+model.state_embed(state*g['mask']),g,state)
                    candidate=model.head(h);readouts.append(candidate)
                    if not (kind=='hold' and index in targets):state=candidate
                return torch.stack(readouts)
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):forward()
            torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):outputs=forward()
            states=np.zeros((500,13,120,3),np.float32)
            prefix=min(targets) if targets else 13
            for i,cpu in enumerate(graphs):
                for key,value in cpu.items():g[key].copy_(value)
                graph.replay();states[i]=outputs[:,0].cpu().numpy()
                np.testing.assert_array_equal(states[i,:prefix],old[i,:prefix])
            del graph,outputs
            np.save(OUT/f'{name}_states.npy',states,allow_pickle=False)
            if pack is None:pack=ResidentRows(load_rows('dev_monitor500'),'cuda',max_nodes=120)
            names=manifest['loss_names'];losses=np.zeros((500,13,len(names)),np.float64)
            for start in range(0,500,32):
                ids=list(range(start,min(start+32,500)));batch=pack.batch(ids)
                raw=torch.as_tensor(states[ids],device='cuda')
                for layer in (3,6,9,12):
                    values=base_per_case(raw[:,layer],batch)
                    losses[ids,layer]=torch.stack([values[key] for key in names],-1).cpu().double().numpy()
            if name=='original':np.testing.assert_array_equal(losses[:,[3,6,9,12]],oldloss[:,[3,6,9,12]])
            assert np.isfinite(losses).all()
            np.save(OUT/f'{name}_losses.npy',losses,allow_pickle=False)
            results[name]=dict(kind=kind,targets=targets,mean_final_loss=float(losses[:,12,0].mean()),
                prefix_bitwise_unchanged=True)
            print(name,results[name]['mean_final_loss'],flush=True)
            save(OUT/'intervene_status.json',dict(completed=list(results),total=len(modes)))
    assert all(torch.equal(v.cpu(),saved['model'][key]) for key,v in model.state_dict().items())
    check()
    save(OUT/'interventions.json',dict(modes=results,seconds=time.perf_counter()-started,
        weights_unchanged=True,original_all_states_and_supervised_losses_bitwise_equal=True,
        source_sha256=sha(Path(__file__)),training=False,postprocessing_calls=0,Q_calls=0,test_read=False))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['observe','intervene'])
    args=parser.parse_args()
    (observe if args.mode=='observe' else intervene)()

if __name__=='__main__':main()
