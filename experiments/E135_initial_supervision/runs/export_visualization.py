"""Read-only E135 E119 galleries and all-layer new Loss; no training or repair."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import torch
from datasets.random_reference import RandomReference
from visualization.style import normalize
from experiments.E135_initial_supervision.evaluation.run import Predictor
from experiments.E135_initial_supervision.batches import decode,inputs_only
from experiments.E135_initial_supervision.data import load_rows
from experiments.E135_initial_supervision.resident import ResidentRows
from experiments.E135_initial_supervision.loss import base_per_case,frontend_components
from experiments.E135_initial_supervision.loss_config import layer_weights

RUNS=Path(__file__).resolve().parent
OWNER=RUNS.parent
ROOT=OWNER.parents[1]
ARMS=('e119_A','e119_B','e119_C')

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def main():
    dataset=RandomReference('dev_monitor500')
    registrations={}
    union=set()
    for arm in ARMS:
        run=RUNS/f'{arm}_seed0';evaluation=run/'evaluation_no_mib_dev500'
        result=read(evaluation/'results.json')
        checkpoint=run/'checkpoints/best.pt';digest=sha(checkpoint)
        if result['status']!='complete' or result['count']!=500 or digest!=result['checkpoint_sha256']:
            raise ValueError('Completed frozen E119 evaluation required')
        rows=[read(evaluation/'cases'/f'{i:04d}.json') for i in range(500)]
        for i,row in enumerate(rows):
            if row['index']!=i or row['member']!=dataset.records[dataset.indices[i]]['member'] or row['checkpoint_sha256']!=digest:
                raise ValueError('Evaluation member/index/checkpoint mismatch')
        worst=sorted(range(500),key=lambda i:(-rows[i]['metrics']['cost'],i))[:5]
        union.update(worst)
        registrations[arm]=dict(run=run,evaluation=evaluation,result=result,checkpoint=checkpoint,digest=digest,rows=rows,worst=worst)
    ordinary=random.Random(20261114).sample([i for i in range(500) if i not in union],10)
    # All selections are frozen before reading payloads, cached tensors or inference.
    for arm,r in registrations.items():
        folder=r['run']/'visualization_dev15_loss'
        if folder.exists():raise FileExistsError('Preserve existing visualization; no automatic overwrite/retry')
        folder.mkdir()
        r['folder']=folder;r['indices']=ordinary+r['worst']
        save(folder/'selection.json',dict(arm=arm,checkpoint_sha256=r['digest'],best_epoch=r['result']['best_epoch'],seed=20261114,
            ordinary_indices=ordinary,worst_indices=r['worst'],
            ordinary_rule='Same ten random dev members in all three galleries; exclude union of all three own worst5 sets',
            worst_rule='Own saved final per-case Q descending; stable dev index ascending tie-break',
            members=[dict(index=i,member=r['rows'][i]['member'],evaluation_sha256=sha(r['evaluation']/'cases'/f'{i:04d}.json')) for i in r['indices']],
            loss='Per-case per-layer public-priority-v1 base Loss; before supervision-layer weighting; actual raw states, including unsupervised layers',
            training=False,postprocessing_calls=0,Q_calls=0,test_read=False))
        save(folder/'state.json',dict(status='capturing',completed=0,count=15))
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    for arm,r in registrations.items():
        cases=[]
        try:
            # Reuse compatible persistent CPU loss-port/target tensors; no static rebuild.
            rows=load_rows('dev_monitor500',r['indices'])
            if any(list(row['member'])!=r['rows'][i]['member'] for row,i in zip(rows,r['indices'])):
                raise ValueError('Cached loss tensors have different members')
            items=[dataset[i] for i in r['indices']]
            pack=ResidentRows(rows,'cuda',max_nodes=120)
            weights=layer_weights(arm)
            with torch.inference_mode():
                predictor=Predictor(r['checkpoint'],items[0]['solver_inputs'])
                model_before={k:v.detach().clone() for k,v in predictor.model.state_dict().items()}
                for j,(i,item) in enumerate(zip(r['indices'],items)):
                    row=r['rows'][i];n=row['n']
                    terminal=predictor.predict(item['solver_inputs'])
                    np.testing.assert_array_equal(terminal,np.asarray(row['draft_positions']))
                    batch=pack.batch([j])
                    for key,value in inputs_only(batch).items():
                        if not torch.equal(value,predictor.g[key]):raise ValueError('Loss/public inference inputs differ: '+key)
                    layers=[];losses=[]
                    for depth,state in enumerate(predictor.states):
                        positions=(decode(state,predictor.g)[0,:n]*predictor.g['scale'][0]).cpu().numpy().astype(float).tolist()
                        values=base_per_case(state,batch)
                        values={key:float(value[0]) for key,value in values.items()}
                        keys=('reference_weighted','overlap_weighted','wire_compact_weighted','grouping_weighted','boundary_weighted','public_shape_weighted','public_preplaced_position_weighted')
                        np.testing.assert_allclose(values['total'],sum(values[k] for k in keys),rtol=1e-6,atol=1e-6)
                        values.update(supervision_weight=weights.get(depth,0.),directly_supervised=depth in weights,
                            objective_name='full',objective_total=values['total'])
                        if arm=='e119_C' and depth==0:
                            front={key:float(value[0]) for key,value in frontend_components(state,batch).items()}
                            values.update(objective_name='F',objective_total=front['total'],frontend_components=front)
                        layers.append(positions);losses.append(values)
                    np.testing.assert_array_equal(layers[-1],np.asarray(row['draft_positions']))
                    rank=j+1 if j<10 else j-9
                    label=f'{"普通例" if j<10 else "最坏第"} {rank}{"" if j<10 else " 名"} · dev {i:04d}'
                    case=normalize(dict(label=label,group='ordinary' if j<10 else 'worst',dev_index=i,member=row['member'],n=n,
                        constraints=item['solver_inputs'][5].tolist(),reference=item['supervision']['reference_xywh'].tolist(),
                        layers=layers,layer_ids=list(range(13)),layer_losses=losses,final=row['positions'],
                        metrics=dict(Q=row['metrics']['cost'],hard_legal=bool(row['metrics']['is_feasible']),complete_ms=row['complete_seconds']*1000)))
                    cases.append(case)
                    save(r['folder']/'state.json',dict(status='capturing',completed=j+1,count=15))
                    print(arm,j+1,'/15',i,'L12 Loss',losses[-1]['total'],flush=True)
                if not all(torch.equal(v,model_before[k]) for k,v in predictor.model.state_dict().items()):raise ValueError('Model changed')
            if sha(r['checkpoint'])!=r['digest']:raise ValueError('Checkpoint changed')
            interval={'e119_A':'A · S0完整Loss/均分','e119_B':'B · S0完整Loss/线性增权','e119_C':'C · S0四项F/均分'}[arm]
            save(r['folder']/'data.json',dict(title=f'E135 · E119 {interval} · best{r["result"]["best_epoch"]}',
                note='10个相同普通例＋各自最终Q最坏5例；真实S0–S12共13个状态。完整Loss为当前例、当前层的共同诊断，未乘层权重；C的S0实际监督目标F另列。后处理复用原结果。',
                identity=dict(arm=arm,best_epoch=r['result']['best_epoch'],checkpoint_sha256=r['digest'],
                    source_evaluation=r['evaluation'].relative_to(ROOT).as_posix(),layer_weights=weights,
                    loss_definition='public_priority_v1; per-case base total, unweighted across depth',
                    layer_ids=list(range(13)),terminal_drafts_bitwise_equal=True,training=False,postprocessing_calls=0,Q_calls=0,test_read=False),cases=cases))
            save(r['folder']/'state.json',dict(status='complete',completed=15,count=15,actual_layers=13,
                layer_loss_count=195,terminal_drafts_bitwise_equal=True,model_state_unchanged=True,checkpoint_unchanged=True,
                loss_inputs_exactly_match_inference=True,loss_components_sum_verified=True,existing_cache_reused=True,
                training=False,postprocessing_calls=0,Q_calls=0,test_read=False))
        except BaseException as error:
            save(r['folder']/'state.json',dict(status='failed',completed=len(cases),error=repr(error),automatic_retry=False))
            raise

if __name__=='__main__':main()
