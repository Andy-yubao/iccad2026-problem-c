"""Five fixed train members: cache equality, real discarded updates and scorer."""
from pathlib import Path
import argparse,importlib,json,subprocess,sys,os
import numpy as np
import torch

BUNDLE=Path(__file__).resolve().parent
WORKSPACE=BUNDLE.parents[3]
IDS=[0,15999,16000,31999,63999]

def stage_check(stage,device):
    sys.path.insert(0,str(Path.cwd()))
    name='E42_joint_regularization' if stage=='E42' else 'E74_collision_32k_posttrain'
    module=lambda n:importlib.import_module('experiments.'+name+'.'+n)
    train=module('train');train.seed_all(0)
    data=module('data');cache=module('cache');rows=data.load_rows('train');selected=[rows[i] for i in IDS]
    assert len(rows)==64000
    exact=cache.load_rows('train',IDS);expected=module('batches').collate([r['graph'] for r in exact],list(range(5)),'cpu')
    actual,_=data.collate(selected,'cpu')
    assert expected.keys()==actual.keys()
    for key in actual:assert torch.equal(actual[key],expected[key]),key
    if device=='cpu':return dict(status='passed',stage=stage,cache_bitwise_equal=True,unique_members=5,members=[list(r['member']) for r in selected],actual_optimizer_updates=0)
    cfg=module('arms').model_config('rectangle_ab' if stage=='E42' else 'topology');model=module('model').GeometryModel(**cfg).cuda()
    if stage=='E74':
        state=torch.load(BUNDLE/'historical/E42/best.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(module('widen').widen(state['model'],model));module('initialization').configure_trainability(model)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=.0003)
    before=[p.detach().clone() for p in model.parameters()]
    ids=IDS+[IDS[-1]]*27
    if stage=='E74':
        pack=module('resident').ResidentRows(rows,'cuda',120);data.register_resident(pack);g=pack.batch(ids)
        engine=module('cuda_step').CapturedStep(model,opt,g,'topology');values,norm=engine.replay(g)
    else:
        model.train();g,_=data.collate([rows[i] for i in ids],'cuda');values=module('loss').losses(model(module('batches').inputs_only(g)),g,None,'rectangle_ab',0.,include_quality=False,start_epoch=6)
        assert bool(torch.isfinite(torch.stack(list(values.values()))).all())
        values['total'].backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
    delta=max(float((p.detach()-q).abs().max()) for p,q in zip(model.parameters(),before));assert delta>0 and np.isfinite(delta)
    return dict(status='passed',stage=stage,cache_bitwise_equal=True,unique_members=5,members=[list(r['member']) for r in selected],actual_optimizer_updates=1,discarded=True,max_parameter_change=delta,loss=float(values['total'].detach()),test_read=False)

def f_check(device):
    sys.path.insert(0,str(Path.cwd()))
    from experiments.E101_final_c5_solver.common import seed,HERE,RUN,snapshot,read,sha
    from experiments.E101_final_c5_solver.data import Data,Packer,public
    from scoring.official import official_evaluator
    official=official_evaluator();assert official.SHAPELY_AVAILABLE
    names=('iccad2026_evaluate','litetestLoader','liteLoader','lite_dataset','lite_dataset_test','prime_dataset','visualize','cost','utils')
    for name in names:assert Path(sys.modules[name].__file__).resolve().is_relative_to(Path.cwd()/'third_party/FloorSet')
    seed();data=Data('train');assert len(data)==64000
    if device=='cpu':return dict(status='passed',stage='F',train_count=len(data),official_import_closure=True,test_read=False)
    import prepare_F_labels as labels
    from experiments.E101_final_c5_solver.train import baseline,configure,Engine
    from experiments.E101_final_c5_solver import loss_f
    from datasets.worksets import FrozenMemberReader
    from datasets.cases import _build_case
    from datasets.reference_layouts import public_hard_reference
    from scoring.evaluation import evaluate_case
    from experiments.E101_final_c5_solver.solver import Service
    seed();g,count=Packer(data,supervision=False).gpu(IDS);teacher=labels.teacher_model()
    with torch.no_grad():
        base=teacher.placer(public(g));physical=(labels.decode(teacher.refiner(base,public(g)),public(g))*g['scale'][:,None,None]).cpu().double().numpy();scales=g['scale'].cpu().numpy()
    target=torch.zeros((5,120,3));usable=torch.zeros(5);inputs=labels.PublicRows();labels.init_worker()
    for k,index in enumerate(IDS):
        raw=inputs.get(index);n=int(raw[0]);p=physical[k,:n].copy();c,s=np.asarray(raw[5]),np.asarray(raw[6]);locked=c[:,:2].any(1)
        p[locked,2:]=s[locked,2:];p[c[:,1]!=0]=s[c[:,1]!=0]
        _,t,u=labels.solve((index,raw,p,float(scales[k])));target[k,:n]=torch.from_numpy(t);usable[k]=u
    del teacher
    model,_=baseline();model=model.cuda().eval()
    with torch.no_grad():g['base']=model.placer(public(g)).detach()
    part=configure('F',model);pad=list(range(5))+[4]*27;g['teacher']=target[pad].cuda();g['usable']=usable[pad].cuda();loss_f.MAX_GROUP_SIZE=data.meta['max_public_group']
    opt=torch.optim.AdamW(part.parameters(),lr=.00025,weight_decay=.0003);before=[p.detach().clone() for p in part.parameters()]
    engine=Engine('F',part,opt,g);value=engine.step(g);delta=max(float((p.detach()-q).abs().max()) for p,q in zip(part.parameters(),before));assert delta>0
    del engine,model,part,opt,g
    members=[data.members[i] for i in IDS];reader=FrozenMemberReader(members,'train',identity='64k-five-member-implementation-check')
    first=reader.case(0);service=Service(HERE/'assets/c5_baseline.pt',first.solver_inputs);legal=0
    for i in range(5):
        raw=reader.case(i);positions,diagnostic,inputs=service.solve(raw.solver_inputs)
        case=_build_case(raw.case_id,raw.source,tuple(inputs[1:6]),raw.reference_positions,raw.stored_metrics);case.solver_inputs=inputs;case.reference_positions=public_hard_reference(inputs[1],inputs[5],inputs[6])
        scored=evaluate_case(case,positions,diagnostic['complete_seconds']);assert np.isfinite(scored['metrics']['cost']);assert scored['metrics']['is_feasible'];legal+=1
    snapshot()
    return dict(status='passed',stage='F',unique_members=5,members=members,original_E85_teacher_cases=5,actual_optimizer_updates=1,discarded=True,max_parameter_change=delta,loss=value,complete_solver_and_official_score_cases=5,hard_legal=legal,official_import_closure=True,test_read=False)

def main(cpu_only=False):
    results=[]
    for stage,sub in [('E42','s42'),('E74','s74'),('F','sf')]:
        device='cpu' if cpu_only else 'cuda';path=BUNDLE/('preflight_'+stage+'_'+device+'.json')
        p=subprocess.run([sys.executable,'-u','-B',str(Path(__file__)),'--stage',stage,'--device',device,'--output',str(path)],cwd=BUNDLE/sub)
        assert p.returncode==0,f'{stage} short check failed; no formal training started'
        results.append(json.loads(path.read_text(encoding='utf-8')))
    path=BUNDLE/('cpu_preflight.json' if cpu_only else 'preflight.json')
    path.write_text(json.dumps(dict(status='passed',unique_train_indices=IDS,checks=results,test_read=False),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--cpu-only',action='store_true');ap.add_argument('--stage',choices=['E42','E74','F']);ap.add_argument('--device',default='cuda');ap.add_argument('--output');args=ap.parse_args()
    if args.stage:
        result=f_check(args.device) if args.stage=='F' else stage_check(args.stage,args.device)
        Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(args.stage,result['status'],flush=True)
    else:main(args.cpu_only)
