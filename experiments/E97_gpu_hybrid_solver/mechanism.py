"""One registered four-arm, five-train probe of fewer-step repair usefulness."""
import collections,datetime,gc,json,time
import numpy as np
from .latency import torch,ROOT,HERE,IDS,registry,public_member_inputs,Captured,DeploymentModel,decode85,save,sha
from .latency_fixed import padded,returned
from .nn85.frame import predicted_frame,contain
from .nn85.mib_projection import coherent_shapes
from .nn85.cooperation import spatial_centers
from .repair85.solver import optimize,raw_measures
from .repair85.canonical_native import warmup
from .repair85.backend.frozen.packing import prepare_problem
from .repair85.backend.frozen.feasible import Objective
from scoring.checks import independent_hard_checks
from datasets.reference_layouts import public_hard_reference

FOLDER=HERE/'runs/20261101_slim_repair_probe'
FULL=dict(relaxed=False,mode=0,adaptive=True,fast_budget=False,late_extension=True,plateau=True,robust_seeds=True,certified=True,quick_project=True,quadratic_project=True,incumbent_project=True,snap=1e-4)
SLIM=dict(iterations=3,seconds=.18,adaptive=False,late_extension=False,certified=False,robust_seeds=False,quick_project=False,quadratic_project=True,incumbent_project=False,projection=True,snap=1e-4)

class OpsOnly(torch.nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,g):
        base=self.model.placer(g);frame=predicted_frame(base,g)
        source=coherent_shapes(contain(contain(base,g,frame),g,frame),g,frame)
        centers=spatial_centers(source,source[...,:2],g,frame,.15)
        return torch.cat((centers,source[...,2:3]),-1)*g['valid'][...,None]

@torch.inference_mode()
def main():
    begin=time.perf_counter();FOLDER.mkdir(exist_ok=False)
    torch.manual_seed(0);torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    study=json.loads((HERE/'study.json').read_text(encoding='utf-8'));deadline=datetime.datetime.fromisoformat(study['deadline_utc'])
    frozen=ROOT/'experiments/E85_cooperative_space_refinement/runs/20261030_P74_F81_cooperative_seed0/checkpoints/best.pt'
    saved=torch.load(frozen,map_location='cpu',weights_only=False);model=DeploymentModel(**saved['model_config']).cuda().eval();model.load_state_dict(saved['model'],strict=True)
    entries=registry()['sets']['train']['records'];inputs=[public_member_inputs(entries[i],'train') for i in IDS]
    start=time.perf_counter();warmup();native_init=time.perf_counter()-start
    g={k:v.cuda() for k,v in padded(inputs[0],False).items()};caps={}
    start=time.perf_counter()
    caps['E85']=Captured(model,g);caps['ops_only']=Captured(OpsOnly(model).eval(),g)
    setup=time.perf_counter()-start
    save(FOLDER/'registration.json',dict(train_ids=IDS,FULL=FULL,SLIM=SLIM,checkpoint_sha256=sha(frozen),source_sha256=sha(__import__('pathlib').Path(__file__)),native_initialization_seconds=native_init,GPU_capture_seconds=setup,official_Q_evaluated=False,test_read=False))
    records=[]
    for i,raw_input in zip(IDS,inputs):
        for network,settings,label in [('ops_only',FULL,'full'),('E85',FULL,'full'),('ops_only',SLIM,'slim3'),('E85',SLIM,'slim3')]:
            if (HERE/'runs/STOP').exists() or datetime.datetime.now(datetime.timezone.utc)>=deadline:
                save(FOLDER/'results.json',dict(status='stopped',records=records,test_read=False));return
            torch.cuda.synchronize();start=time.perf_counter()
            host=padded(raw_input,False);cap=caps[network]
            for k,v in host.items():cap.g[k].copy_(v)
            draft=returned(cap.replay(),cap.g,raw_input,decode85);draft_seconds=time.perf_counter()-start
            final,trace=optimize(draft,raw_input,**settings)
            hard=independent_hard_checks(final.tolist(),np.asarray(raw_input[1]),np.asarray(raw_input[5]),public_hard_reference(raw_input[1],raw_input[5],raw_input[6]))
            hard['nonnegative']=bool((final[:,:2]>=0).all())
            total=time.perf_counter()-start
            assert hard['is_feasible'] and hard['nonnegative']
            p=prepare_problem(*raw_input);obj=Objective(p);area,wire=raw_measures(final,p)
            phases=collections.Counter(x.get('phase','unknown') for x in trace['trace'])
            result=dict(train_id=i,member=entries[i]['member'],n=int(raw_input[0]),network=network,repair=label,draft_seconds=draft_seconds,repair_seconds=trace['seconds'],complete_seconds=total,hard_legal=True,public_proxy=obj(final),bbox_over_total_area=area/obj.area_scale,wire_over_public_scale=wire/obj.wire_scale,soft_violations=obj.violations(final),phase_counts=dict(phases),fallback=trace['row_retained'],positions=final.tolist(),draft_positions=draft.tolist(),trace=trace['trace'])
            records.append(result)
            print(i,network,label,'complete_ms',round(total*1000,2),'public_proxy',round(result['public_proxy'],6),'soft',result['soft_violations'],'phases',dict(phases),'fallback',result['fallback'],flush=True)
            save(FOLDER/'progress.json',dict(records=records,test_read=False))
    aggregate=[]
    for network in ('ops_only','E85'):
        for label in ('full','slim3'):
            rows=[r for r in records if r['network']==network and r['repair']==label]
            aggregate.append(dict(network=network,repair=label,count=len(rows),mean_complete_ms=float(np.mean([r['complete_seconds'] for r in rows])*1000),max_complete_ms=max(r['complete_seconds'] for r in rows)*1000,mean_public_proxy=float(np.mean([r['public_proxy'] for r in rows])),fallbacks=sum(r['fallback'] for r in rows),hard_legal=sum(r['hard_legal'] for r in rows)))
    save(FOLDER/'results.json',dict(status='passed',records=records,aggregate=aggregate,elapsed_seconds=time.perf_counter()-begin,scope='five-train mechanism only; public proxy is not Q; no model/coefficients selected by dev',test_read=False))
    print(json.dumps(aggregate,indent=2),flush=True)

if __name__=='__main__':main()
