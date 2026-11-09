"""Read-only true-head traces on the original five selected training members."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from ..io import ROOT,HERE,save,sha
from ..model import GeometryModel
from ..batches import inputs_only,decode
from ..data import load_rows,collate
from visualization.layer_comparison import measures
from visualization.layer_capture import observe_readouts
from visualization.layer_budget_viewer import publish_candidate

def main():
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    baseline=json.loads((ROOT/'experiments/E98_loss_weight_learning/runs/20261106_layer_diagnosis5/trace.json').read_text(encoding='utf-8'))
    indices=[c['train_index'] for c in baseline['cases']]
    rows=load_rows('train',indices)
    plans=json.loads((HERE/'plan.json').read_text(encoding='utf-8'))
    runs=([f'20261107_depth{d:02d}_seed0' for d in (4,8,12)] if plans['experiment']=='E111_progressive_depth' else [plans['run']])
    for run_name in runs:
        run=HERE/'runs'/run_name;checkpoint=run/'checkpoints/best.pt';out=run/'layer_trace5_20261107'
        digest=sha(checkpoint)
        if (out/'trace.json').exists():
            previous=json.loads((out/'trace.json').read_text(encoding='utf-8'))
            assert previous['identity']['checkpoint_sha256']==digest
            print('Identical five-case trace reused',out,flush=True);continue
        if out.exists():raise FileExistsError('Partial trace requires review; no automatic overwrite')
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model=GeometryModel(**saved['model_config']).eval();model.load_state_dict(saved['model'],strict=True)
        depth=saved.get('active_depth',saved['model_config']['layers'])
        if hasattr(model,'active_depth'):model.active_depth=depth
        before={k:v.detach().clone() for k,v in model.state_dict().items()}
        identity=dict(experiment=plans['experiment'],checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=digest,
            active_depth=depth,best_epoch=saved['progress']['best_epoch'],indices=indices,
            model_source_sha256=sha(HERE/'model.py'),diagnostic_source_sha256=sha(Path(__file__)),
            inference='CPU FP32 eval/no_grad batch1; original registered inputs retained',
            direct_supervision='all eight block readouts; half terminal plus half preceding mean',
            test_read=False,training=False,postprocessing_calls=0,Q_calls=0)
        save(out/'frozen.json',identity);cases=[]
        with torch.no_grad():
            for i,row in enumerate(rows):
                g,_=collate([row],'cpu');g=inputs_only(g)
                with observe_readouts(model) as observed:
                    output=model(g)
                states,hidden=observed['states'],observed['hidden']
                terminal=output[-1] if output.ndim==4 else output
                assert len(states)==depth+1 and torch.equal(states[-1],terminal)
                plain=model(g);plain=plain[-1] if plain.ndim==4 else plain
                assert torch.equal(terminal,plain),'Tracing altered the real output'
                scale=float(g['scale'][0]);reference=(row['graph']['reference']*row['graph']['scale']).numpy()
                frames=[];previous=None
                free=np.asarray(row['case'].solver_inputs[5])[:,1]==0
                for k,state in enumerate(states):
                    positions=decode(state,g)[0].numpy().astype(np.float64)*scale
                    center=positions[:,:2]+positions[:,2:]/2
                    metrics=measures(positions)
                    metrics['center_step_mean']=float(np.linalg.norm((center-previous)[free],axis=1).mean()/scale) if k and free.any() else 0.
                    metrics['hidden_rms']=float(hidden[k].double().square().mean().sqrt())
                    frames.append(dict(depth=k,positions=positions.tolist(),metrics=metrics));previous=center
                assert list(row['member'])==baseline['cases'][i]['member']
                cases.append(dict(id=baseline['cases'][i]['id'],train_index=indices[i],member=list(row['member']),n=len(reference),
                    reference=reference.tolist(),constraints=np.asarray(row['case'].solver_inputs[5]).tolist(),scale=scale,stages=frames))
        assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
        assert sha(checkpoint)==digest
        save(out/'trace.json',dict(identity=identity,cases=cases))
        save(out/'verification.json',dict(status='complete',unique_train_cases=5,head_calls=depth+1,
            hooked_and_plain_terminal_bitwise_equal=True,parameters_unchanged=True,checkpoint_unchanged=True,
            training=False,test_read=False,postprocessing_calls=0,Q_calls=0))
        publish_candidate(ROOT, HERE, plans, out/'trace.json')
        print('Saved and published',plans['experiment'],run_name,'five cases',depth+1,'readouts each',flush=True)

if __name__=='__main__':main()
