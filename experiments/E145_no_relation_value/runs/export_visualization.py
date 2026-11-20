"""E145 requested5+5+5 gallery; capture readouts only, never train/repair."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
import copy
import gc
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import torch
from datasets.random_reference import RandomReference
from visualization.style import normalize
from experiments.E145_no_relation_value.evaluation.run import Predictor as OriginalPredictor, public_graph
from experiments.E145_no_relation_value.model import GeometryModel
from experiments.E145_no_relation_value.batches import decode,inputs_only
from experiments.E145_no_relation_value.data import load_rows
from experiments.E145_no_relation_value.resident import ResidentRows
from experiments.E145_no_relation_value.loss import base_per_case,TOTAL_COMPONENTS
from experiments.E145_no_relation_value.loss_config import layer_weights,NO_OVERLAP_LAYERS

RUNS=Path(__file__).resolve().parent
ROOT=RUNS.parents[2]
ARMS=('e145_no_relation_value',)
SEED=20261117

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

class Predictor(OriginalPredictor):
    """Identical prediction path with all thirteen real shared-head outputs captured."""
    def __init__(self,checkpoint,inputs):
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        self.model=GeometryModel(**state['model_config']).cuda().eval()
        self.model.load_state_dict(state['model'],strict=True)
        self.g={k:v.cuda() for k,v in public_graph(inputs).items()}
        heads=[]
        hook=self.model.head.register_forward_hook(lambda module,args,value: heads.append(value))
        def prediction():
            heads.clear()
            self.states=self.model(self.g)
            assert len(heads)==13
            self.all_states=torch.stack(heads,0)
            return decode(self.states[-1],self.g)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):prediction()
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=stream):self.output=prediction()
        torch.cuda.synchronize()
        hook.remove()

def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dataset=RandomReference('dev_monitor500');registrations={};extremes=set()
    for arm in ARMS:
        run=RUNS/f'{arm}_seed0';evaluation=run/'evaluation_no_mib_dev500'
        result=read(evaluation/'results.json');checkpoint=run/'checkpoints/best.pt';digest=sha(checkpoint)
        assert result['status']=='complete' and result['count']==500 and result['checkpoint_sha256']==digest
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        best=state['progress']['best_epoch'];assert best==result['best_epoch']
        entry=next(h for h in state['progress']['history'] if h['epoch']==best)
        loss=entry['dev_monitor']['per_case_total'];assert len(loss)==500
        panels=read(run/'panels.json');members=[dataset.records[i]['member'] for i in dataset.indices]
        assert panels['dev_monitor_members']==members
        rows=[read(evaluation/'cases'/f'{i:04d}.json') for i in range(500)]
        assert all(r['index']==i and r['member']==members[i] and r['checkpoint_sha256']==digest for i,r in enumerate(rows))
        worst_loss=sorted(range(500),key=lambda i:(-loss[i],i))[:5]
        worst_q=sorted(range(500),key=lambda i:(-rows[i]['metrics']['cost'],i))[:5]
        extremes.update(worst_loss+worst_q)
        registrations[arm]=dict(run=run,result=result,checkpoint=checkpoint,digest=digest,rows=rows,loss=loss,
            worst_loss=worst_loss,worst_q=worst_q)
        del state
    ordinary=[457, 77, 13, 125, 107]
    assert not set(ordinary)&extremes
    for arm,r in registrations.items():
        folder=r['run']/'visualization_normal5_loss5_q5'
        if folder.exists():raise FileExistsError('Preserve existing gallery: '+str(folder))
        groups=[('普通例',ordinary),('Loss最差',r['worst_loss']),('Q最差',r['worst_q'])]
        entries=[dict(category=name,rank=j+1,index=i) for name,indices in groups for j,i in enumerate(indices)]
        selected=sorted({v['index'] for v in entries});folder.mkdir()
        save(folder/'selection.json',dict(arm=arm,seed=SEED,ordinary=ordinary,worst_loss=r['worst_loss'],worst_Q=r['worst_q'],
            entries=entries,duplicates_preserved=True,
            normal_rule='Same five cases for all four models; seed20261117; excludes union of four own worst5 Loss/Q sets',
            loss_rule='Saved per-case own aggregate dev Loss at this frozen own-Loss-best; descending, dev index tie-break',
            Q_rule='Saved final postprocessed per-case Q descending, dev index tie-break',
            checkpoint_sha256=r['digest'],best_epoch=r['result']['best_epoch'],training=False,postprocessing_calls=0,Q_calls=0,test_read=False))
        save(folder/'state.json',dict(status='capturing',completed=0,unique_count=len(selected),display_count=15))
        rows=load_rows('dev_monitor500',selected);lookup={i:j for j,i in enumerate(selected)}
        assert all(list(row['member'])==r['rows'][i]['member'] for row,i in zip(rows,selected))
        pack=ResidentRows(rows,'cpu',max_nodes=120);captured={};weights=layer_weights(arm)
        with torch.inference_mode():
            first=dataset[selected[0]];predictor=Predictor(r['checkpoint'],first['solver_inputs'])
            before={k:v.detach().cpu().clone() for k,v in predictor.model.state_dict().items()}
            for j,i in enumerate(selected):
                item=dataset[i];row=r['rows'][i];n=row['n']
                terminal=predictor.predict(item['solver_inputs'])
                np.testing.assert_array_equal(terminal,np.asarray(row['draft_positions']))
                batch=pack.batch([lookup[i]])
                for key,value in inputs_only(batch).items():
                    assert torch.equal(value,predictor.g[key].cpu()),key
                layers=(decode(predictor.all_states,predictor.g)[:,0,:n]*predictor.g['scale'][0]).cpu().numpy().astype(float)
                np.testing.assert_array_equal(layers[-1],np.asarray(row['draft_positions']))
                states=predictor.all_states.cpu();losses=[]
                for depth,state in enumerate(states,0):
                    values=base_per_case(state,batch)
                    v={k:float(value[0]) for k,value in values.items()}
                    extra=0.
                    v['total']+=extra
                    objective=sum(v[k] for k in TOTAL_COMPONENTS if k!='overlap_weighted' or depth not in NO_OVERLAP_LAYERS)+extra
                    v.update(supervision_weight=weights.get(depth,0),directly_supervised=depth in weights,
                        objective_total=objective,objective_name='完整Loss' if depth not in NO_OVERLAP_LAYERS else '去overlap项')
                    losses.append(v)
                captured[i]=dict(dev_index=i,member=row['member'],n=n,constraints=item['solver_inputs'][5].tolist(),
                    reference=item['supervision']['reference_xywh'].tolist(),layers=layers.tolist(),layer_ids=list(range(13)),
                    layer_losses=losses,final=row['positions'],
                    objective_note='最差Loss按该组best权重保存的逐题整体训练目标选取；Q按原后处理结果选取。滑块展示真实S0–S12，逐层Loss未乘层权重。',
                    metrics=dict(Q=row['metrics']['cost'],hard_legal=bool(row['metrics']['is_feasible']),complete_ms=row['complete_seconds']*1000))
                normalize(captured[i])
                save(folder/'state.json',dict(status='capturing',completed=j+1,unique_count=len(selected),display_count=15))
                print(f'{arm} captured {j+1}/{len(selected)} dev{i:04d}',flush=True)
            assert all(torch.equal(v.cpu(),before[k]) for k,v in predictor.model.state_dict().items())
        cases=[]
        for entry in entries:
            i=entry['index'];c=copy.deepcopy(captured[i]);c.update(group=entry['category'],
                label=f"{entry['category']} {entry['rank']} · dev {i:04d} · own Loss {r['loss'][i]:.6f} · Q {r['rows'][i]['metrics']['cost']:.6f}")
            cases.append(c)
        assert len(cases)==15 and sha(r['checkpoint'])==r['digest']
        title='去掉关系value ψ'
        save(folder/'data.json',dict(title=f'E145 · {title} · best{r["result"]["best_epoch"]}',
            note='5例正常例 + 5例Loss最差 + 5例Q最差；各类别独立保留，重复样例重复展示。左Reference，中栏真实S0–S12，右原后处理结果。',
            identity=dict(arm=arm,checkpoint_sha256=r['digest'],layer_ids=list(range(13)),
                own_loss_ranking='saved best dev500 aggregate per_case_total',layer_diagnostics='CPU FP32 on exact GPU readout states',
                terminal_drafts_bitwise_equal=True,postprocessing_reused=True,training=False,Q_calls=0,postprocessing_calls=0,test_read=False),cases=cases))
        save(folder/'state.json',dict(status='complete',display_count=15,unique_count=len(selected),actual_layers=13,
            category_counts={'normal':5,'worst_loss':5,'worst_Q':5},duplicates_preserved=True,
            terminal_drafts_bitwise_equal=True,model_unchanged=True,checkpoint_unchanged=True,training=False,
            Q_calls=0,postprocessing_calls=0,test_read=False))
        del predictor,before,pack,rows,states,batch,values;gc.collect();torch.cuda.empty_cache()
        print('Saved '+str(folder/'data.json'),flush=True)

if __name__=='__main__':main()
