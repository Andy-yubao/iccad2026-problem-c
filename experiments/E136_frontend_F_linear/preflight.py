"""Necessary five-case check and provenance-based reuse of verified resume engine."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import ast
import hashlib
import json
import time
import traceback
import torch
from . import train
from .io import HERE,ROOT,save,sha,sources
from .model import GeometryModel
from .data import load_rows,cache_identity
from .resident import ResidentRows
from .batches import inputs_only
from .loss import per_case,base_per_case,frontend_components
from .loss_config import ARMS,layer_weights
from .cuda_step import CapturedStep

PARENT=ROOT/'experiments/E135_initial_supervision'

def read(path):return json.loads(path.read_text(encoding='utf-8'))

def mapped(text,provenance):
    for old,new in provenance['identity_mapping'].items():text=text.replace(old,new)
    return text

def inherited_engine():
    provenance=read(HERE/'provenance.json');gate=read(PARENT/'preflight.json')
    if sha(PARENT/'preflight.json')!=provenance['parent_preflight_sha256']:raise ValueError('Parent proof changed')
    if gate['status']!='passed' or not gate['ctrl_c_resume_verified']:raise ValueError('Parent resume proof missing')
    for name,digest in gate['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Parent proof sources changed: '+name)
    if not all(all(r['bitwise_equal'].values()) for r in gate['resume_checks']):raise ValueError('Parent resume equality failed')
    if not gate['queue_signal']['explicit_resume_advances'] or not gate['queue_signal']['no_next_stage_after_pause']:
        raise ValueError('Parent controller proof missing')
    if not gate['evaluation_signal']['first_committed_file_unchanged']:raise ValueError('Parent evaluation proof missing')
    checked=[]
    for name,original in provenance['copied_originals'].items():
        parent=ROOT/original['parent'];local=HERE/name
        if sha(parent)!=original['sha256']:raise ValueError('Copied parent changed')
        expected=mapped(parent.read_text(encoding='utf-8'),provenance)
        actual=local.read_text(encoding='utf-8')
        if name=='run_queue.py':
            # Only final comparison reporting changes; controller, launch, freeze and pause logic stay exact.
            def controller(text):
                tree=ast.parse(text)
                tree.body=[n for n in tree.body if not isinstance(n,ast.FunctionDef) or n.name!='write_comparison']
                return ast.dump(tree,include_attributes=False)
            equal=controller(expected)==controller(actual)
        else:equal=expected==actual
        if not equal:raise ValueError('Unexpected numerical/continuation source change: '+name)
        checked.append(name)
    return dict(mode='inherited E135 proof after identity-normalized runtime source equality',
        parent_preflight_sha256=provenance['parent_preflight_sha256'],checked_files=checked,
        parent_training_resume_arms=[r['arm'] for r in gate['resume_checks']],
        queue_resume_verified=True,evaluation_committed_prefix_verified=True,
        new_interruption_simulations=0,new_draft_updates_in_this_check=0)

def cache_coverage(plan):
    from datasets.random_reference import RandomReference
    identity=cache_identity(plan)
    key=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if key!=plan['required_cache_key']:raise ValueError('Shared cache identity changed')
    base=ROOT/'datasets/static_tensor_cache_v1'/key
    if read(base/'identity.json')!=identity:raise ValueError('Cache producer mismatch')
    report={}
    for role in ('train','dev_monitor500'):
        dataset=RandomReference(role);blocks={}
        for index in dataset.indices:blocks.setdefault(index//256,[]).append(index)
        for block,ids in blocks.items():
            path=base/dataset.role/f'{block:04d}.pt';entry=read(path.with_suffix('.json'))
            if not path.is_file() or entry['identity']!=key:raise ValueError('Missing cached block')
            if any(entry['members'][str(i)]!=dataset.records[i]['member'] for i in ids):raise ValueError('Cached member changed')
        report[role]=dict(members=len(dataset.indices),chunks=len(blocks),metadata_only=True)
    return dict(key=key,roles=report)

def main():
    started=time.perf_counter();result=dict(status='preparing',unique_train_cases=5,
        formal_training_started=False,formal_optimizer_updates=0,dev_payloads=0,test_payloads=0,
        postprocessing_calls=0,Q_calls=0,ctrl_c_resume_verified=False)
    try:
        for path in HERE.rglob('*.py'):ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
        plan=read(HERE/'plan.json');auth=read(HERE/'authorization.json')
        if auth['training_enabled'] or auth['allowed_arms'] or (HERE/'runs').exists():raise ValueError('Formal run during preparation')
        result['continuation_proof']=inherited_engine()
        result['cache']=cache_coverage(plan)
        for control in plan['comparison_controls']:
            if sha(ROOT/control['result'])!=control['results_sha256']:raise ValueError('Frozen control result changed')
        arm=ARMS[0];weights=layer_weights(arm)
        from experiments.E135_initial_supervision.loss_config import layer_weights as previous_weights
        if weights!=previous_weights('e119_B') or {str(k):v for k,v in weights.items()}!=plan['arms'][arm]['layer_weights']:
            raise ValueError('D must use exact B weights')
        if abs(sum(weights.values())-1)>1e-12 or weights[12]!=.8 or abs(weights[9]/weights[0]-2)>1e-12:raise ValueError('Weight registration mismatch')
        train.seed_all(0);model=GeometryModel(**train.MODEL)
        digest=hashlib.sha256(b''.join(t.contiguous().numpy().tobytes() for t in model.state_dict().values())).hexdigest()
        parent_gate=read(PARENT/'preflight.json')
        if digest!=parent_gate['arms'][0]['initial_state_sha256']:raise ValueError('Scratch initialization changed')
        if sum(p.numel() for p in model.parameters())!=plan['parameters']['e119']:raise ValueError('Parameter count changed')
        rows=load_rows('train',plan['preflight_train_indices'])
        if len({tuple(r['member']) for r in rows})!=5:raise ValueError('Wrong short-check members')
        pack=ResidentRows(rows,'cuda',max_nodes=120);batch=pack.batch([i%5 for i in range(32)])
        public=inputs_only(batch)
        if set(public)&{'reference','target','pin_coordinates'}:raise ValueError('Label input leak')
        if not (public['mask'][public['valid']]==1).all() or not (public['constraints'][...,2]==0).all():raise ValueError('Prediction/MIB boundary changed')
        model.cuda().eval()
        with torch.inference_mode():
            output=model(public)
            if output.shape[0]!=13:raise ValueError('S0..S12 readouts required')
            actual=per_case(output,batch,arm)
            expected=sum(weights[k]*base_per_case(output[k],batch)['total'] for k in (3,6,9,12))+weights[0]*frontend_components(output[0],batch)['total']
            torch.testing.assert_close(actual['total'],expected,rtol=0,atol=0)
            torch.testing.assert_close(actual['final_total'],base_per_case(output[12],batch)['total'],rtol=0,atol=0)
        model.train();optimizer=train.optimizer_for(model)
        before={k:t.detach().clone() for k,t in model.state_dict().items()}
        cap=CapturedStep(model,optimizer,batch,arm)
        if any(not torch.equal(v,before[k]) for k,v in model.state_dict().items()):raise ValueError('Capture changed parameters')
        values,norm=cap.replay();torch.cuda.synchronize()
        if cap.projection.steps!=1 or torch.equal(model.head[0].weight,before['head.0.weight']):raise ValueError('No guarded update')
        for layer in (model.initial_blocks[0],model.layers[0],model.layers[-1]):
            if not layer.qkv.weight.grad.abs().sum()>0:raise ValueError('Missing frontend/backbone gradient')
        result.update(status='passed',ctrl_c_resume_verified=True,sources=sources(),
            arm=arm,weights=weights,initial_state_sha256=digest,parameters=plan['parameters']['e119'],
            disposable_optimizer_updates=1,disposable_loss=float(values['total']),gradient_norm=float(norm),
            objective_matches_C_F_and_B_weights=True,training_model_not_reused=True,
            scope='Short numerical update only; continuation inherited by source proof, no repeated interruption simulation')
        save(HERE/'preparation_complete.json',dict(status='ready_to_start',arm=arm,
            automatic_postprocessing_prepared=True,continuation_mechanism_inherited=True,
            real_dev_payloads=0,test_payloads=0,real_postprocessing_calls=0))
    except BaseException as error:
        result.update(status='failed',error=repr(error),traceback=traceback.format_exc());raise
    finally:
        result['elapsed_seconds']=time.perf_counter()-started;save(HERE/'preflight.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('sources','continuation_proof','cache','traceback')},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
