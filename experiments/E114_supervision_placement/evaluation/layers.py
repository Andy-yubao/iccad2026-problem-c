"""Read-only real-head traces for the same original E114 five training members."""
import json
import numpy as np
import torch
from ..io import ROOT,HERE,save,sha
from ..model import GeometryModel
from ..data import load_rows,collate
from ..batches import inputs_only,decode
from visualization.layer_capture import observe_readouts
from visualization.layer_comparison import measures
from visualization.supervision_layer_viewer import publish_pair

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    plan=json.loads((HERE/'plan.json').read_text(encoding='utf-8'))
    evaluation=json.loads((HERE/'evaluation/plan.json').read_text(encoding='utf-8'))
    indices=plan['preflight_train_indices'];rows=load_rows('train',indices)
    models={}
    for arm,key,label in [('terminal','baseline','末层监督'),('dense','dense','密集监督')]:
        run=HERE/'runs'/f'{arm}_seed0';out=run/'layer_trace5';checkpoint=run/'checkpoints/best.pt'
        expected=evaluation['models'][arm]
        assert sha(checkpoint)==expected['checkpoint_sha256']
        if out.exists():raise FileExistsError('Existing/partial trace must be reviewed before rerunning')
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model=GeometryModel(**saved['model_config']).eval();model.load_state_dict(saved['model'],strict=True)
        identity=dict(experiment='E114_supervision_placement',arm=arm,checkpoint=expected['checkpoint'],
            checkpoint_sha256=expected['checkpoint_sha256'],best_epoch=expected['best_epoch'],
            active_depth=12,indices=indices,inference='CPU FP32 eval/no_grad batch1; not a GPU timing measurement',
            direct_supervision='layer12 only' if arm=='terminal' else 'layers1..12, .8 terminal + .2 auxiliary mean',
            test_read=False,training=False,postprocessing_calls=0,Q_calls=0)
        before={k:v.detach().clone() for k,v in model.state_dict().items()};cases=[]
        with torch.no_grad():
            for i,row in enumerate(rows):
                batch,_=collate([row],'cpu');g=inputs_only(batch)
                with observe_readouts(model) as captured:output=model(g)
                plain=model(g)
                assert torch.equal(output,plain) and len(captured['states'])==13
                assert torch.equal(captured['states'][-1],output[-1])
                scale=float(g['scale'][0]);reference=(row['graph']['reference']*row['graph']['scale']).numpy()
                frames=[];previous=None
                for d,(state,hidden) in enumerate(zip(captured['states'],captured['hidden'])):
                    positions=decode(state,g)[0].numpy().astype(float)*scale
                    center=positions[:,:2]+positions[:,2:]/2;metrics=measures(positions)
                    metrics['center_step_mean']=float(np.linalg.norm(center-previous,axis=1).mean()/scale) if d else 0.
                    metrics['hidden_rms']=float(hidden.double().square().mean().sqrt())
                    frames.append(dict(depth=d,positions=positions.tolist(),metrics=metrics));previous=center
                cases.append(dict(id=f'case{i+1:02d}',train_index=indices[i],member=list(row['member']),n=len(reference),
                    reference=reference.tolist(),constraints=row['graph']['constraints'].tolist(),scale=scale,stages=frames))
        assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
        assert sha(checkpoint)==expected['checkpoint_sha256']
        save(out/'trace.json',dict(identity=identity,cases=cases))
        save(out/'verification.json',dict(status='complete',unique_train_cases=5,head_calls=13,
            hooked_plain_outputs_bitwise_equal=True,parameters_unchanged=True,checkpoint_unchanged=True,
            training=False,test_read=False,postprocessing_calls=0,Q_calls=0))
        models[key]=dict(label=label+' · 64×12 · 共享头',short_label=arm,depth=12,parameters=612307,identity=identity,cases=cases)
    folder=ROOT/'visualization/E114_supervision/20261107_two_models'
    publish_pair(models,folder)
    print('Published same-five layer comparison:',folder,flush=True)

if __name__=='__main__':main()
