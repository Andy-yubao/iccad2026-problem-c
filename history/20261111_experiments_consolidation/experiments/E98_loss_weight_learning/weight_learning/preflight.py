"""Five existing train members: coefficients, inherited Adam, real updates/resume."""
import ast,copy,gc,json,time
import numpy as np
import torch
from ..io import save,sha
from ..model import GeometryModel
from ..train import seed_all,rng_state,restore_rng
from ..loss import per_case as original_loss
from ..data import load_rows,register_resident
from ..resident import ResidentRows
from ..batches import inputs_only
from ..preflight import same
from .engine import per_case
from .run import OWNER,PLAN,ROOT,MODEL,sources,verify_parent,restore_branch,evaluator,choose

def main():
    started=time.perf_counter();result=dict(status='error',test_read=False)
    try:
        for path in OWNER.glob('*.py'):ast.parse(path.read_text(encoding='utf-8'))
        before=sha(ROOT/PLAN['parent_checkpoint']);parent=verify_parent()
        assert len(PLAN['feedback_panel'])==24 and len({tuple(r['member']) for r in PLAN['feedback_panel']})==24
        train_records=evaluator.registry()['sets']['train']['records']
        assert all(train_records[r['train_index']]['member']==r['member'] for r in PLAN['feedback_panel'])
        co=PLAN['coefficients'];base=np.asarray(list(co['control'].values()))
        for arm in ('plus','minus'):
            assert set(co[arm])==set(co['control']) and min(co[arm].values())>0
            assert abs(sum(co[arm].values())-base.sum())<1e-12
        evaluator.setup();rows=load_rows('train',PLAN['preflight_train_indices'])
        pack=ResidentRows(rows,'cuda',max_nodes=120,max_pins=146);register_resident(pack)
        batch=pack.batch([i%5 for i in range(32)])
        model,opt,cap=restore_branch(parent,pack,co['control']);model.eval()
        with torch.no_grad():
            predicted=model(inputs_only(batch));old=original_loss(predicted,batch);new=per_case(predicted,batch,co['control'])
        for key in old:torch.testing.assert_close(old[key],new[key],rtol=0,atol=0)
        assert {int(v['step']) for v in opt.state.values()}=={29000}
        result['parent_epoch']=58;result['parent_adam_age']=29000;result['control_loss_exact']=True
        del model,opt,cap;gc.collect();torch.cuda.empty_cache()

        def trajectory(arm,resume=False):
            model,opt,cap=restore_branch(parent,pack,co[arm]);initial=copy.deepcopy(model.state_dict())
            progress=dict(epoch=0,cursor=0,step=0,order=np.random.default_rng(58).permutation(5))
            for i in range(4):
                model.train();cap.replay(batch);progress['step']+=1;progress['cursor']+=32
                if resume and i==1:
                    state=dict(model={k:v.cpu().clone() for k,v in model.state_dict().items()},
                         optimizer=copy.deepcopy(opt.state_dict()),rng=rng_state(),projection=cap.projection.summary(),
                         progress=copy.deepcopy(progress),coefficients=co[arm],elapsed_seconds=17.5)
                    del cap,opt,model;gc.collect();torch.cuda.empty_cache()
                    model,opt,cap=restore_branch(state,pack,co[arm]);progress=state['progress']
                    same(state['rng'],rng_state());assert state['elapsed_seconds']==17.5
            assert cap.projection.steps==29004
            assert all(int(v['step'])==29004 for v in opt.state.values()), (arm,sorted({int(v['step']) for v in opt.state.values()}))
            assert all(int(v['step'])==29000 for v in parent['optimizer']['state'].values()), 'Parent CPU Adam step mutated'
            for layer in range(12):
                name=f'layers.{layer}.local.weight';assert not torch.equal(initial[name],model.state_dict()[name])
            final=dict(model={k:v.cpu().clone() for k,v in model.state_dict().items()},optimizer=copy.deepcopy(opt.state_dict()),
                 rng=rng_state(),projection=cap.projection.summary(),progress=progress)
            if resume:
                saved_model=copy.deepcopy(model.state_dict());saved_opt=copy.deepcopy(opt.state_dict());saved_projection=cap.projection.summary()
                bad={k:v.clone() for k,v in batch.items()};bad['target'].fill_(float('nan'))
                try:cap.replay(bad)
                except RuntimeError:pass
                else:raise AssertionError('Invalid weighted update accepted')
                same(saved_model,model.state_dict());same(saved_opt,opt.state_dict());same(saved_projection,cap.projection.summary())
            del cap,opt,model,initial;gc.collect();torch.cuda.empty_cache();return final
        control=trajectory('control');plus=trajectory('plus');resumed=trajectory('plus',True)
        same(plus,resumed)
        assert any(not torch.equal(control['model'][key],plus['model'][key]) for key in plus['model'])
        same(control['rng'],plus['rng'])
        result['actual_updates']='all twelve local branches updated; weighted vs control states differ with matched RNG; plus4 equals2+restore+2 exactly including Adam/projection/cursor'
        with torch.inference_mode(),evaluator.threadpool_limits(limits=1):
            torch.set_num_threads(1);evaluator.warmup();predictor=evaluator.Predictor(rows[0]['case'].solver_inputs)
            predictor.model.load_state_dict(plus['model'])
            hard=[]
            for row in rows:
                draft,final,stats,legal,wall,draft_time=evaluator.request(predictor,row['case'].solver_inputs)
                assert legal['is_feasible'];hard.append(dict(member=list(row['member']),hard_legal=True,seconds=wall))
        result['feedback_interface_five_train_only']=hard
        # Selection-rule arithmetic, no solver queries or writes of real selected.json.
        margin=PLAN['minimum_absolute_gain_per_repeat']
        for name in ('control','plus','minus'):assert name in PLAN['coefficients']
        assert sha(ROOT/PLAN['parent_checkpoint'])==before
        result.update(status='passed',sources=sources(),parent_sha256=before,parameters=1471539,
             new_dev_payloads=0,new_test_payloads=0,unique_train_cases=5)
    except BaseException as exc:
        result['error']=repr(exc);raise
    finally:
        result['elapsed_seconds']=time.perf_counter()-started
        save(OWNER/'preflight.json',result)
        print(json.dumps({k:v for k,v in result.items() if k!='sources'},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
