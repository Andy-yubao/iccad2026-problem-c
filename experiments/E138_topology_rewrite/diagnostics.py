"""Post-selection structural diagnostics; no optimizer, new Q, or test reads."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import json
import signal
import time
import numpy as np
import torch
from .io import HERE,ROOT,save,sha,exclusive
from .loss_config import ARMS
from .evaluation.run import Predictor
from .batches import decode
from .data import load_rows
from .resident import ResidentRows
from .loss import base_per_case
from .run_queue import verify

def read(path):return json.loads(path.read_text(encoding='utf-8'))

def relation_metrics(trajectory,reference,baseline3,areas,flags):
    n=len(areas);scale=(np.sqrt(areas)[:,None]+np.sqrt(areas)[None,:])/2
    pairs=np.triu(np.ones((n,n),bool),1)&~((flags[:,1]!=0)[:,None]&(flags[:,1]!=0)[None,:])
    def coded(p):
        centers=p[:,:2]+p[:,2:]/2;d=centers[:,None]-centers[None,:];half=(p[:,None,2:]+p[None,:,2:])/2
        return np.where(np.abs(d)-half>.02*scale[...,None],np.sign(d),0),d,half
    refcode,refd,_=coded(reference);basecode,_,_=coded(baseline3);owncode,_,_=coded(trajectory[3])
    basebad=pairs[...,None]&(refcode!=0)&(basecode==-refcode)
    ownbad=pairs[...,None]&(refcode!=0)&(owncode==-refcode)
    metrics={}
    for name,mask in [('fixed_baseline_S3_wrong',basebad),('own_S3_wrong',ownbad)]:
        strong=[];direction=[]
        for p in trajectory:
            code,d,_=coded(p)
            strong.append(int(((code==refcode)&mask).sum()))
            direction.append(int(((np.sign(d)==np.sign(refd))&mask).sum()))
        metrics[name]=dict(count=int(mask.sum()),strong_repaired=strong,direction_repaired=direction)
    last,d0,h0=coded(trajectory[3]);pending=np.zeros_like(last,dtype=bool);exchanges=[]
    for layer in range(4,13):
        code,d,h=coded(trajectory[layer])
        for axis in range(2):
            cross=d0[...,axis]*d[...,axis]<0
            t=np.divide(d0[...,axis],d0[...,axis]-d[...,axis],out=np.zeros_like(scale),where=cross)
            orth=1-axis;dd=d0[...,orth]+t*(d[...,orth]-d0[...,orth]);hh=h0[...,orth]+t*(h[...,orth]-h0[...,orth])
            pending[...,axis]|=cross&(hh-np.abs(dd)>.02*scale)&pairs
        event=(last*code<0)&pending&pairs[...,None]
        exchanges.append(int(event.sum()))
        resolved=code!=0;last=np.where(resolved,code,last);pending=np.where(resolved,False,pending)
        d0,h0=d,h
    metrics['complete_side_exchanges_by_arrival_S4_to_S12']=exchanges
    return metrics

def main():
    from datasets.random_reference import RandomReference
    from scoring.draft_geometry import evaluate_geometry
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=ARMS,required=True);args=parser.parse_args()
    verify();run=HERE/'runs'/f'{args.arm}_seed0';out=run/'structural_audit'
    frozen=read(HERE/'evaluation'/f'frozen_{args.arm}.json');checkpoint=run/'checkpoints/best.pt'
    assert sha(checkpoint)==frozen['checkpoint_sha256']
    assert read(run/'evaluation_no_mib_dev500/results.json')['status']=='complete'
    if (out/'results.json').exists():
        assert read(out/'results.json')['checkpoint_sha256']==frozen['checkpoint_sha256'];return
    out.mkdir(exist_ok=True);(out/'cases').mkdir(exist_ok=True)
    requested=[False];signal.signal(signal.SIGINT,lambda *_:requested.__setitem__(0,True))
    def halted():return requested[0] or any((HERE/name).exists() for name in ('STOP','QUEUE_STOP','PAUSE_REQUEST')) or (run/'STOP').exists()
    def beat(status,completed,**fields):save(out/'state.json',dict(status=status,completed=completed,total=500,pid=os.getpid(),**fields))
    with exclusive(out/'diagnostics.lock'):
        try:
            torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            ds=RandomReference('dev_monitor500');members=read(HERE/'evaluation/plan.json')['members']
            assert [ds.records[i]['member'] for i in ds.indices]==members
            baseline=ROOT/'experiments/E134_supervision_interval/runs/e119_stride3_seed0/layer_analysis_v1'
            baseline_manifest=read(baseline/'manifest.json')
            control=read(HERE/'plan.json')['baseline']
            assert sha(baseline/'manifest.json')==control['trajectory_manifest_sha256']
            assert sha(baseline/'positions.npy')==control['trajectory_sha256']
            assert baseline_manifest['members']==members
            assert baseline_manifest['checkpoint_sha256']==read(HERE/'plan.json')['baseline']['checkpoint_sha256']
            bp=np.load(baseline/'positions.npy',mmap_mode='r')
            predictor=None;records=[]
            with torch.inference_mode():
                for i in range(500):
                    if halted():beat('paused',len(records));return
                    path=out/f'cases/{i:04d}.json'
                    if path.exists():
                        row=read(path);assert row['member']==members[i] and row['checkpoint_sha256']==frozen['checkpoint_sha256'];records.append(row);continue
                    beat('running',i,stage='capturing true layer outputs')
                    item=ds[i];inputs=item['solver_inputs'];n=int(inputs[0])
                    if predictor is None:predictor=Predictor(checkpoint,inputs)
                    final=predictor.predict(inputs)
                    old=read(run/f'evaluation_no_mib_dev500/cases/{i:04d}.json')
                    np.testing.assert_array_equal(final,np.asarray(old['draft_positions']))
                    trajectory=(decode(predictor.states,predictor.g)[:,0,:n]*predictor.g['scale'][0]).cpu().numpy().astype(float)
                    raw=predictor.states[:,0].cpu().numpy()
                    ref=np.asarray(item['supervision']['reference_xywh'],float);flags=np.asarray(inputs[5]);areas=np.asarray(inputs[1])
                    relation=relation_metrics(trajectory,ref,np.asarray(bp[i,3,:n]),areas,flags)
                    geometry,_,_=evaluate_geometry(final,inputs,details=False)
                    row=dict(index=i,member=members[i],checkpoint_sha256=frozen['checkpoint_sha256'],n=n,
                        states=raw.tolist(),positions=trajectory.tolist(),relation=relation,geometry=geometry)
                    save(path,row);records.append(row)
                    if (i+1)%50==0:print(args.arm,'structure',i+1,'/500',flush=True)
                if halted():beat('paused',500);return
                beat('running',500,stage='all-layer complete Loss')
                pack=ResidentRows(load_rows('dev_monitor500'),'cuda',max_nodes=120)
                raw=np.array([r['states'] for r in records],np.float32)
                names=['total','reference_weighted','overlap_weighted','wire_compact_weighted','grouping_weighted','boundary_weighted','public_shape_weighted','public_preplaced_position_weighted']
                loss=np.zeros((500,13,len(names)))
                for start in range(0,500,32):
                    if halted():beat('paused',500);return
                    ids=list(range(start,min(start+32,500)));batch=pack.batch(ids);states=torch.as_tensor(raw[ids],device='cuda')
                    for layer in range(13):
                        values=base_per_case(states[:,layer],batch)
                        loss[ids,layer]=torch.stack([values[name] for name in names],-1).cpu().double().numpy()
                np.save(out/'layer_losses.npy',loss,allow_pickle=False)
            relations={}
            for key in ('fixed_baseline_S3_wrong','own_S3_wrong'):
                selected=[r['relation'][key] for r in records];den=sum(v['count'] for v in selected)
                relations[key]=dict(count=den,cases=sum(v['count']>0 for v in selected),
                    strong_repaired_by_layer=[sum(v['strong_repaired'][k] for v in selected) for k in range(13)],
                    strong_repair_fraction_S12=sum(v['strong_repaired'][12] for v in selected)/den if den else None,
                    direction_repair_fraction_S12=sum(v['direction_repaired'][12] for v in selected)/den if den else None)
            result=dict(status='complete',count=500,checkpoint_sha256=frozen['checkpoint_sha256'],
                layer_loss_names=names,layer_loss_means=loss.mean(0).tolist(),strong_relations=relations,
                mean_complete_side_exchanges=float(np.mean([sum(r['relation']['complete_side_exchanges_by_arrival_S4_to_S12']) for r in records])),
                final_geometry_means={key:float(np.mean([r['geometry'][key] for r in records])) for key in
                    ('duplicate_coverage','bbox_over_public_area','whitespace_bbox','group_contact_bottleneck','boundary_gap_local')},
                original_evaluation_outputs_bitwise_equal=True,test_read=False,training=False,postprocessing_calls=0,Q_calls=0,
                note='Reference strong-order fidelity is diagnostic, not unique good topology; no Q/step/branch selection.')
            assert sha(checkpoint)==frozen['checkpoint_sha256']
            save(out/'results.json',result)
            (out/'report.md').write_text('# '+args.arm+' 结构诊断\n\n```json\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n```\n',encoding='utf-8')
            beat('complete',500)
        except BaseException as error:
            beat('error',len(locals().get('records',[])),error=repr(error));raise

if __name__=='__main__':main()
