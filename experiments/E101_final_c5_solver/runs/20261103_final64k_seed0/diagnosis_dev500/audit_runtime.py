"""Follow up observed batch32/serial draft differences; raw networks only."""
from pathlib import Path
import sys,json,gc
ROOT=Path.cwd();sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from experiments.E101_final_c5_solver.common import HERE,RUN,read,save,seed
from experiments.E101_final_c5_solver.data import Data,Packer,public
from experiments.E101_final_c5_solver.model import DeploymentModel
from experiments.E101_final_c5_solver.nn.base_model import decode
from experiments.E101_final_c5_solver.solver import Captured
from scipy.stats import spearmanr
OUT=Path(__file__).parent;OLD=ROOT/'history/20261103_final_delivery_consolidation/experiments/E97_gpu_hybrid_solver/runs/20261102_round2_candidate5_countweighted_seed0'

@torch.inference_mode()
def main():
    assert read(RUN/'state.json')['status']=='complete';seed();models={}
    for name,path in [('old',HERE/'assets/c5_baseline.pt'),('new',HERE/'delivery/model.pt')]:
        s=torch.load(path,map_location='cpu',weights_only=False);m=DeploymentModel(**s['model_config']).cuda().eval();m.load_state_dict(s['model']);models[name]=m;del s
    data=Data('dev_monitor500');packer=Packer(data,False);states={k:[] for k in ('old_L','new_L','oldL_oldF','newL_oldF','newL_newF','oldL_newF')};rects={k:[] for k in states}
    records_old=[read(OLD/'evaluation/terminal'/f'{i:04d}.json') for i in range(500)];records_new=[read(RUN/'evaluation/dev_monitor500/cases'/f'{i:04d}.json') for i in range(500)]
    for start in range(0,500,32):
        ids=list(range(start,min(start+32,500)));g,count=packer.gpu(ids);base_old=models['old'].placer(g);base_new=models['new'].placer(g)
        outputs=dict(old_L=base_old,new_L=base_new,oldL_oldF=models['old'].refiner(base_old,g),newL_oldF=models['old'].refiner(base_new,g),newL_newF=models['new'].refiner(base_new,g),oldL_newF=models['new'].refiner(base_old,g))
        for key,value in outputs.items():
            states[key].extend(value[:count].cpu().numpy());physical=(decode(value,g)*g['scale'][:,None,None]).cpu().double().numpy()
            rects[key].extend(physical[:count])
    np.savez_compressed(OUT/'raw_dev500_batch32.npz',**{k:np.asarray(v) for k,v in states.items()})
    errors={};details={}
    for key,source,field in [('old_L',records_old,'L_positions'),('new_L',records_new,'L_positions'),('oldL_oldF',records_old,'draft_positions'),('newL_newF',records_new,'draft_positions')]:
        v=[]
        for i,r in enumerate(source):
            n=r['n'];scale=float(data.arrays['scale'][data.indices[i]]);v.append(float(np.abs(rects[key][i][:n]-np.array(r[field])).max()/scale))
        errors[key]=v;rank=np.argsort(-np.array(v));details[key]=dict(max_normalized_error=max(v),mean_normalized_error=float(np.mean(v)),above1e4=int((np.array(v)>1e-4).sum()),above1e3=int((np.array(v)>1e-3).sum()),worst5=[dict(index=int(i),error=v[i]) for i in rank[:5]])
    ids=[x['index'] for x in details['newL_newF']['worst5']];checks=[];m=models['new']
    for i in ids:
        host,_=packer.gpu([i]);g={k:v[:1].clone() for k,v in host.items()};b=m.placer(g);f=m.refiner(b,g);p=(decode(f,g)*g['scale'][:,None,None])[0].cpu().double().numpy();scale=float(data.arrays['scale'][data.indices[i]]);n=records_new[i]['n']
        cap=Captured(lambda gg:decode(m(gg),gg)*gg['scale'][:,None,None],g);captured=cap.replay()[0].cpu().double().numpy()
        frozen_base=torch.from_numpy(np.asarray(states['new_L'][i])[None]).cuda();same_base=m.refiner(frozen_base,g);fixed_base_p=(decode(same_base,g)*g['scale'][:,None,None])[0].cpu().double().numpy()
        checks.append(dict(index=i,eager1_vs_saved=float(np.max(np.abs(p[:n]-np.array(records_new[i]['draft_positions'])))/scale),captured1_vs_eager1=float(np.max(np.abs(captured[:n]-p[:n]))/scale),batch32_base_vs_batch1_base=float(np.max(np.abs(np.asarray(states['new_L'][i])-b[0].cpu().numpy()))),same_base_F_batch1_vs_F_batch32=float(np.max(np.abs(fixed_base_p[:n]-np.asarray(rects['newL_newF'][i])[:n]))/scale)))
        del cap;gc.collect();torch.cuda.empty_cache()
    qdelta=np.array([b['metrics']['cost']-a['metrics']['cost'] for a,b in zip(records_old,records_new)])
    save(OUT/'runtime_consistency.json',dict(batch32_vs_saved_serial=details,selected5_serial_checks=checks,spearman_newF_runtime_error_vs_Q_delta=float(spearmanr(errors['newL_newF'],qdelta).statistic),model_or_solver_changes=False,no_optimizer_or_solver_calls=True,test_used=False))
    print(json.dumps(dict(summary=details,selected5=checks),ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
