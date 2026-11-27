"""Same five frozen cases: decode/reference consistency, no update or Q solve."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
import torch
import torch.nn.functional as F
from experiments.E92_two_stage_placer.io import ROOT,HERE,save,sha
from experiments.E92_two_stage_placer.model import GeometryModel
from experiments.E92_two_stage_placer.arms import model_config
from experiments.E92_two_stage_placer.data import load_rows
from experiments.E92_two_stage_placer.resident import ResidentRows
from experiments.E92_two_stage_placer.batches import inputs_only,decode
from experiments.E92_two_stage_placer.simple_loss import average
from experiments.E92_two_stage_placer.qualification.metrics import module_overlap

OUT=Path(__file__).resolve().parent
torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
data=json.loads((OUT/'offline.json').read_text(encoding='utf-8'));ids=[r['index'] for r in data['probe_cases']]
rows=load_rows('dev_monitor500',ids);pack=ResidentRows(rows,'cuda',max_nodes=120)
paths={'E90':ROOT/'experiments/E90_auxiliary_weight_placer/runs/20261031_aux3_seed0/checkpoints/best.pt',
       'E92':HERE/'runs/20261031_two_stage32k_seed0/checkpoints/F_best.pt'}
before={name:sha(path) for name,path in paths.items()};records=[]
with torch.no_grad():
    for name,path in paths.items():
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        model=GeometryModel(**model_config('simple')).cuda().eval();model.load_state_dict(checkpoint['model']);del checkpoint
        for i,row in enumerate(rows):
            g=pack.batch([i]);n=row['graph']['n'];state=model(inputs_only(g));p=decode(state,g)
            center=p[...,:2]+p[...,2:]/2;local=g['area'].clamp_min(1e-12).sqrt();mask=g['mask']*g['valid'][...,None]
            latent=average(F.smooth_l1_loss(state[...,:2]/local[...,None],g['target'][...,:2]/local[...,None],reduction='none',beta=.1),mask[...,:2],(1,2))
            actual=average(F.smooth_l1_loss(center/local[...,None],g['target'][...,:2]/local[...,None],reduction='none',beta=.1),mask[...,:2],(1,2))
            shifted=(((center-state[...,:2]).abs()/local[...,None]).amax(-1)>1e-5)&g['valid']&g['mask'][...,0].bool()
            original=(g['reference'][0,:n]*g['scale'][0]).cpu().numpy()
            teacher=(decode(g['target'],g)[0,:n]*g['scale'][0]).cpu().numpy()
            original_stats=module_overlap(original);teacher_stats=module_overlap(teacher)
            records.append({'model':name,'index':ids[i],'n':n,'latent_position_loss':float(latent),'decoded_center_position_loss':float(actual),
                'clamped_mutable_blocks':int(shifted.sum()),'max_center_shift_local':float(((center-state[...,:2]).abs()/local[...,None]).masked_fill(~mask[...,:2].bool(),0).max()),
                'original_reference_N10':original_stats['N10'],'decoded_reference_N10':teacher_stats['N10'],
                'original_reference_worst_coverage':original_stats['max_covered_fraction'],'decoded_reference_worst_coverage':teacher_stats['max_covered_fraction']})
        del model;torch.cuda.empty_cache()
after={name:sha(path) for name,path in paths.items()};assert before==after
save(OUT/'geometry_probe.json',{'records':records,'unique_cases':5,'training_updates':0,'new_Q_calls':0,'test_read':False,'checkpoint_hashes_unchanged':True})
print(json.dumps(records,ensure_ascii=False,indent=2))
