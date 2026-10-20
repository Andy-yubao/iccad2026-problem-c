"""Five-case cache equivalence, same-gradient continuation and no-repair checks."""
import argparse
import copy
import json
from pathlib import Path
import time
from unittest.mock import patch
import numpy as np
import torch
from datasets.quality_layouts import QualityLayouts
from scoring.evaluation import evaluate_case
from .cache import build_role, PackedRows
from .data import load_rows, collate, prepare
from .io import HERE, save, checkpoint, sources
from .loss import losses
from .model import GeometryModel, decode
from .train import ARMS, run, seed_all


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--device',default='cuda')
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    seed_all(0)
    before=time.perf_counter()
    original=[load_rows('train',range(3),use_cache=False),load_rows('dev_monitor500',range(2),use_cache=False)]
    old_seconds=time.perf_counter()-before
    archived=[];before=time.perf_counter()
    for role,count in (('train',3),('dev_monitor500',2)):
        dataset=QualityLayouts(role)
        archived.append([prepare(dataset.archive_item(i)) for i in range(count)])
    archive_seconds=time.perf_counter()-before
    result={'unique_members':[list(r['member']) for group in original for r in group],
        'legacy_prepare_five_seconds':old_seconds,'archive_prepare_five_seconds':archive_seconds,
        'formal_training_started':False,'test_read':False,'tail_test_read':False,'postprocessing_calls':0}
    for role,old,new in zip(('train','dev_monitor500'),original,archived):
        for a,b in zip(old,new):
            for key in a['graph']:assert torch.equal(a['graph'][key],b['graph'][key]),(role,key)
            for key in ('geometry','b2b','p2b','pins'):
                assert torch.equal(a['target'][key],b['target'][key]),(role,key)
            assert a['target']['wire_anchor']==b['target']['wire_anchor']
            assert a['target']['area_anchor']==b['target']['area_anchor']
            # Public-spec-only references produce exactly the same official
            # target score; no original free reference coordinate is necessary.
            # Compare scores for one common generated/decoded candidate.
            graph,target=collate([a],args.device)
            candidate=(decode(target[0]['geometry'][None],graph)[0]*graph['scale'][0]).cpu().double().numpy()
            c,s=np.asarray(a['case'].solver_inputs[5]),np.asarray(a['case'].solver_inputs[6])
            candidate[c[:,:2].any(1),2:]=s[c[:,:2].any(1),2:];candidate[c[:,1]!=0]=s[c[:,1]!=0]
            scores=[evaluate_case(row['case'],candidate.tolist(),0)['metrics'] for row in (a,b)]
            assert scores[0]==scores[1]
    root=args.output/'cache'
    build_role('train',root/'train',indices=range(3),workers=2,chunk_size=2)
    build_role('dev',root/'dev',indices=[next(i for i,r in enumerate(QualityLayouts('dev').records)
        if r['member']==m['member']) for m in QualityLayouts('dev_monitor500').records[:2]],workers=2,chunk_size=2)
    before=time.perf_counter()
    packed=[PackedRows('train',range(3),cache_root=root,allow_pilot=True),
            PackedRows('dev_monitor500',range(2),cache_root=root,allow_pilot=True)]
    result['mmap_open_five_seconds']=time.perf_counter()-before
    for old,cache in zip(original,packed):
        for i,a in enumerate(old):
            b=cache[i]
            for key in a['graph']:assert torch.equal(a['graph'][key],b['graph'][key]),key
            for key in ('geometry','b2b','p2b','pins'):assert torch.equal(a['target'][key],b['target'][key]),key
            for key in ('wire_anchor','area_anchor'):assert a['target'][key]==b['target'][key],key
    graph,targets=collate(original[0]+original[1],args.device)
    result['arms']={}
    for arm in ARMS:
        seed_all(0);model=GeometryModel(192 if arm=='curriculum_large' else 96,4).to(args.device)
        values=losses(model(graph),graph,targets,arm,20,include_quality=arm!='supervised_small')
        values['total'].backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        result['arms'][arm]={'finite_backward':True,'parameters':sum(p.numel() for p in model.parameters())}
    seed_all(0);a=GeometryModel().to(args.device);b=copy.deepcopy(a)
    oa=torch.optim.AdamW(a.parameters(),lr=1e-4,weight_decay=1e-4)
    ob=torch.optim.AdamW(b.parameters(),lr=1e-4,weight_decay=1e-4)
    losses(a(graph),graph,targets,'supervised_small',6,include_quality=True)['total'].backward()
    losses(b(graph),graph,targets,'supervised_small',6,include_quality=False)['total'].backward()
    assert all(torch.equal(pa.grad,pb.grad) for pa,pb in zip(a.parameters(),b.parameters()))
    for model,opt in ((a,oa),(b,ob)):
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
    assert all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters()))
    result['pure_supervision_gradient_and_update_identical']=True
    # If any hidden repair remains in the controller, the five-case run fails.
    config=argparse.Namespace(arm='curriculum_small',output=args.output/'controller',device=args.device,
        seed=0,max_hours=.1,max_epochs=1,batch_size=2,resume=False)
    def forbidden(*a,**kw):raise AssertionError('Postprocessing is forbidden during training/validation')
    with patch('experiments.E15_quality_curriculum.evaluation.quality_monitor',forbidden), \
         patch('experiments.E15_quality_curriculum.evaluation.optimize',forbidden), \
         patch('experiments.E15_quality_curriculum.evaluation.polish',forbidden):
        run(config,implementation_check=True,stop_after_steps=1,check_rows=packed)
        first=torch.load(config.output/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
        assert first['status']=='user_stop' and first['progress']['cursor']==2
        config.resume=True;run(config,implementation_check=True,check_rows=packed)
    final=torch.load(config.output/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
    assert final['finished'] and final['progress']['epoch']==1 and final['progress']['step']==2
    assert final['identity']['sources']==sources()
    assert (config.output/'checkpoints/best.pt').exists()
    result.update(cache_values_identical=True,archive_values_identical=True,
        public_spec_scoring_identical=True,controller_stop_resume_passed=True,passed=True)
    save(args.output/'results.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
