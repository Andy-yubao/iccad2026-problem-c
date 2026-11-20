"""Five train cases: new schedule, fresh Adam, two LR updates, and repair interface."""
import ast
import gc
import time
import traceback
import torch
from ..io import ROOT, sha, save
from ..model import GeometryModel
from ..data import load_rows, register_resident
from ..resident import ResidentRows
from ..loss import per_case as baseline_loss
from ..batches import inputs_only
from ..qualification import evaluate as evaluator
from ..weight_learning.engine import per_case
from .common import OWNER, RUN, PLAN_PATH, read, sources, coefficients, PhaseSchedule, q_summary
from .run import make_model

def main():
    started = time.perf_counter()
    result = dict(status='error',test_read=False,unique_train_cases=5,dev_payloads=0)
    try:
        for path in OWNER.glob('*.py'):
            ast.parse(path.read_text(encoding='utf-8'))
        plan = read(PLAN_PATH)
        parent_path = ROOT/plan['initial_checkpoint']
        assert sha(parent_path)==plan['initial_sha256']
        parent = torch.load(parent_path,map_location='cpu',weights_only=False)
        assert len(set(plan['feedback_pool_train_indices']))==1000
        for spec in plan['rounds']:
            panel = spec['feedback_train_indices']
            assert len(set(panel))==100 and set(panel)<=set(plan['feedback_pool_train_indices'])
        sched = PhaseSchedule()
        transitions = []
        for epoch in range(9):
            stop = sched.step(1.,epoch)
            transitions.append(dict(epoch=epoch,lr=sched.lr,stop=stop))
        assert all(t['lr']==1e-4 for t in transitions[:5])
        assert transitions[5]['lr']==5e-5 and not any(t['stop'] for t in transitions[:8])
        assert transitions[8]['stop']
        result['flat_loss_schedule'] = transitions
        evaluator.setup()
        torch.set_num_threads(4)
        rows = load_rows('train',plan['implementation_shortcheck_indices'])
        pack = ResidentRows(rows,'cuda',max_nodes=120,max_pins=146)
        register_resident(pack)
        batch = pack.batch([i%5 for i in range(32)])
        center = plan['initial_coefficients']
        direction = plan['rounds'][0]['direction']
        vectors = dict(control=center,
             plus=coefficients(center,direction,1,plan['log_perturbation']),
             minus=coefficients(center,direction,-1,plan['log_perturbation']))
        updates = []
        final_state = None
        for arm,co in vectors.items():
            assert min(co.values())>0 and abs(sum(co.values())-sum(center.values()))<1e-12
            model,opt,cap,sched = make_model(parent,pack,co)
            assert not opt.state and opt.param_groups[0]['lr']==1e-4
            model.eval()
            with torch.no_grad():
                out = model(inputs_only(batch))
                values = per_case(out,batch,co)
                if arm=='control':
                    original = baseline_loss(out,batch)
                    for key in original:
                        torch.testing.assert_close(values[key],original[key],rtol=0,atol=0)
            model.train()
            before = model.layers[0].local.weight.detach().clone()
            cap.replay(batch)
            assert all(int(v['step'])==1 for v in opt.state.values())
            assert not torch.equal(before,model.layers[0].local.weight)
            for group in opt.param_groups:
                group['lr']=5e-5
            cap.replay(batch)
            assert all(int(v['step'])==2 for v in opt.state.values())
            updates.append(dict(arm=arm,actual_updates=2,lrs=[1e-4,5e-5],adam_fresh=True,
                 gradient_norm=float(cap.norm),loss=float(cap.values['total'])))
            if arm=='plus':
                final_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
            del model,opt,cap,before
            gc.collect()
            torch.cuda.empty_cache()
        assert {int(v['step']) for v in parent['optimizer']['state'].values()}=={29000}
        result['updates'] = updates
        torch.set_num_threads(1)
        with torch.inference_mode(),evaluator.threadpool_limits(limits=1):
            evaluator.warmup()
            predictor = evaluator.Predictor(rows[0]['case'].solver_inputs)
            predictor.model.load_state_dict(final_state)
            records = []
            from scoring.evaluation import evaluate_case
            for row in rows:
                draft,final,stats,hard,wall,draft_time = evaluator.request(predictor,row['case'].solver_inputs)
                scored = evaluate_case(row['case'],final,wall)
                assert hard['is_feasible'] and scored['metrics']['is_feasible']
                records.append(dict(metrics=scored['metrics'],n=int(row['case'].block_count),
                     complete_seconds=wall,fallback=stats['row_retained']))
        result['five_train_feedback_interface'] = q_summary(records)
        assert sha(parent_path)==plan['initial_sha256']
        result.update(status='passed',sources=sources(),initial_checkpoint_sha256=sha(parent_path),
             actual_updates_total=6,engine='existing E98 FP32 CUDA backward graph; original AdamW outside graph; pinned CPU minibatches; no numerical path change',
             elapsed_seconds=time.perf_counter()-started)
    except BaseException as exc:
        result.update(error=repr(exc),traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-started)
        raise
    finally:
        save(RUN/'preflight.json',result)
        print({k:v for k,v in result.items() if k!='sources'},flush=True)

if __name__=='__main__':
    main()
