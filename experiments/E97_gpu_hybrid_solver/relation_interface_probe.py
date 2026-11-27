"""Registered five-train relation-input mechanism; oracle used only in this probe."""
import datetime,json,time
import numpy as np
from .candidate2 import HERE,ROOT,torch,teacher_rows,attach,seed,IDS,decode85
from .student import DeploymentModel
from .train_io.data import load_rows,collate
from .latency import public_member_inputs,registry,sha
from .mechanism import SLIM,independent_hard_checks,public_hard_reference,warmup,Objective,prepare_problem
from .repair85.fast_geometry import topology
from .solver_relations import optimize
from .train_io.io import save

def main():
    folder=HERE/'runs/20261102_relation_interface_probe';folder.mkdir(exist_ok=False);begin=time.perf_counter();seed()
    deadline=datetime.datetime.fromisoformat(json.loads((HERE/'study.json').read_text(encoding='utf-8'))['deadline_utc'])
    labels=teacher_rows('train');rows=attach(load_rows('train',IDS),labels,IDS);entries=registry()['sets']['train']['records'];warmup()
    saved=torch.load(HERE/'runs/20261102_candidate1_distill_seed0/checkpoints/best.pt',map_location='cpu',weights_only=False);model=DeploymentModel(**saved['model_config']).eval();model.load_state_dict(saved['model'],strict=True);results=[]
    with torch.inference_mode():
        for index,row in zip(IDS,rows):
            if (HERE/'runs/STOP').exists() or datetime.datetime.now(datetime.timezone.utc)>=deadline:raise RuntimeError('STOP/deadline')
            g=collate([row],'cpu');base=row['draft'][None];state=model.refiner(base,g);n=row['graph']['n'];scale=g['scale'][:,None,None]
            draft=(decode85(state,g)*scale)[0,:n].double().numpy();target=(decode85(row['teacher'][None],g)*scale)[0,:n].double().numpy()
            inputs=public_member_inputs(entries[index],'train');teacher_edges=topology(target,np.asarray(inputs[5]),array=True)
            for arm,edge_input in [('native',None),('teacher_edges',teacher_edges)]:
                final,trace=optimize(draft,inputs,**SLIM,relations_override=edge_input)
                hard=independent_hard_checks(final.tolist(),np.asarray(inputs[1]),np.asarray(inputs[5]),public_hard_reference(inputs[1],inputs[5],inputs[6]));assert hard['is_feasible']
                p=prepare_problem(*inputs);result=dict(train_id=index,n=n,arm=arm,final_legal=True,fallback=trace['row_retained'],repair_seconds=trace['seconds'],public_proxy=Objective(p)(final),trace=trace['trace']);results.append(result)
                print(index,arm,'legal',True,'fallback',trace['row_retained'],'proxy',result['public_proxy'],flush=True)
    save(folder/'results.json',dict(status='complete',records=results,SLIM=SLIM,seconds=time.perf_counter()-begin,source_sha256=sha(__import__('pathlib').Path(__file__)),test_read=False))

if __name__=='__main__':main()
