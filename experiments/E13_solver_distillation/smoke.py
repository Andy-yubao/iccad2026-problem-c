"""Five existing train-pilot cases only: gradient/resume/decode/repair plumbing."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from .io import ROOT,HERE,save,sha,Worker
from .data import load_bank,collate
from .student import Student
from .loss import losses
from .train import seed_all,rng_state,restore_rng,atomic_checkpoint,sources
from .decode import decode
from .backend.frozen.packing import prepare_problem
from .backend.frozen.solver import hard_valid
from datasets.worksets import LayoutReader,workset_manifest
from datasets.cases import _build_case
from scoring.evaluation import evaluate_case
from .pipeline import Pipeline

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--device',default='cuda');args=ap.parse_args()
    begin=time.perf_counter();args.output.mkdir(parents=True,exist_ok=False);(args.output/'checkpoints').mkdir()
    seed_all(0);bank=ROOT/'experiments/E12_distillation_readiness/runs/20261018_teacher_trainpilot100_seed0'
    samples,bank_id,quality=load_bank(bank,'train',smoke=True);assert len(samples)==5
    model=Student().to(args.device);model.initialize();g,t=collate(samples,list(range(5)),args.device)
    model.eval()
    with torch.no_grad():
        old=model.backbone(g);new=model(g,False)['geometry'];mask=g['mask'].bool()
        assert torch.equal(old[mask],new[mask])
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    def step(net,opt):
        net.train();opt.zero_grad(set_to_none=True);v,_=losses(net(g,True),g,t,True)
        assert torch.isfinite(v['total']);v['total'].backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
        opt.step();return float(v['total'].detach())
    first=step(model,optimizer)
    # Geometry-only backward is also exercised; restore the same teacher initialization afterward.
    geo=Student().to(args.device);geo.initialize();geo_values,_=losses(geo(g,False),g,t,False)
    geo_values['total'].backward();assert torch.isfinite(geo.backbone.head[-1].weight.grad).all();del geo
    for _ in range(3):step(model,optimizer)
    cp={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':rng_state(),'arm':'structure','smoke':True}
    atomic_checkpoint(args.output/'checkpoints/latest.pt',cp)
    uninterrupted_loss=step(model,optimizer);expected={k:v.detach().clone() for k,v in model.state_dict().items()}
    saved=torch.load(args.output/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
    resumed=Student().to(args.device);resumed.load_state_dict(saved['model'])
    resumed_opt=torch.optim.AdamW(resumed.parameters(),lr=1e-4,weight_decay=1e-4);resumed_opt.load_state_dict(saved['optimizer']);restore_rng(saved['rng'])
    resumed_loss=step(resumed,resumed_opt)
    assert uninterrupted_loss==resumed_loss and all(torch.equal(expected[k],v) for k,v in resumed.state_dict().items())
    for _ in range(11):last=step(resumed,resumed_opt)
    atomic_checkpoint(args.output/'checkpoints/best.pt',{'model':resumed.state_dict(),'arm':'structure','smoke':True})
    del model,optimizer,resumed,resumed_opt,expected
    records=json.loads((bank/'results.json').read_text(encoding='utf-8'))['records'];lookup={r['case_id']:r for r in records}
    reader=LayoutReader(workset_manifest(),'train_monitor');index={pair:i for i,pair in enumerate(reader.members)}
    worker=Worker(args.output/'oracle_worker.log');rows=[]
    try:
        for sample in samples:
            raw=reader.case(index[sample['member']]);p=prepare_problem(*raw.solver_inputs);record=lookup[raw.case_id]
            with np.load(bank/record['label_file'],allow_pickle=False) as z:
                out={'geometry':z['geometry'],'boundary':np.where(sample['targets']['boundary'].numpy(),20.,-20.),
                    'mib':np.where(z['mib_same'],20.,-20.),'contact':np.full((*z['separation'].shape,5),-20.)}
            np.put_along_axis(out['contact'],sample['targets']['contact'].numpy()[...,None],20.,axis=-1)
            seed,cert,stats=decode(out,p,True)
            response=worker.call({'inputs':[int(raw.solver_inputs[0]),*[np.asarray(v).tolist() for v in raw.solver_inputs[1:]]],
              'draft':seed.tolist(),'guidance':cert,'joint':True,'polished':True,'seconds':.35})
            assert 'error' not in response and hard_valid(response['positions'],p)
            case=_build_case(raw.case_id,raw.source,raw.solver_inputs[1:6],raw.reference_positions,raw.stored_metrics)
            ev=evaluate_case(case,response['positions'],response['solve_seconds'])
            assert ev['metrics']['is_feasible'] and ev['metrics']['cost']<=record['evaluation']['metrics']['cost']+.002
            rows.append({'case_id':raw.case_id,'perfect_head_decoder':stats,'Q_delta_vs_teacher':ev['metrics']['cost']-record['evaluation']['metrics']['cost'],
               'perfect_head_final':ev,'repair':response['stats']})
    finally:worker.close()
    pipeline=Pipeline(args.output/'checkpoints/best.pt',args.device,args.output/'student_worker.log')
    try:
        for sample,row in zip(samples,rows):
            raw=reader.case(index[sample['member']]);final,stats=pipeline.solve(raw.solver_inputs)
            assert hard_valid(final,prepare_problem(*raw.solver_inputs)) and not stats['teacher_information_used']
            row['actual_short_step_pipeline']={'stats':stats,'positions':final.tolist()}
    finally:pipeline.close()
    save(args.output/'results.json',{'status':'passed','elapsed_seconds':time.perf_counter()-begin,'unique_cases':5,
      'checkpoint_resume_bitwise_equal':True,'pretrained_geometry_preserved':True,'geometry_backward_finite':True,
      'first_training_loss':first,'last_training_loss':last,'train_label_audit':quality,'rows':rows,
      'source_hashes':sources(),'pilot_identity':bank_id,'device':args.device,
      'scope':'16 tiny diagnostic updates on five train-pilot cases; not a formal trained student or generalization result.'})
    print('Five-case implementation check passed; diagnostic loss:',first,'->',last,flush=True)

if __name__=='__main__':main()
