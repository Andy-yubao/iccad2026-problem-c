"""Reuse five train cases: learned-score/tree parity, callback cost and legality."""
from .resources import configure
RESOURCE=configure()
import json
import time
import numpy as np
import torch
from .data import load_rows
from .cases import WEIGHTS,predict,graph
from .model import GeometryModel
from .relations import RelationPolicy,encode_positions
from .sparse import SparsePairPolicy,SparseContacts,group_pairs
from .policy import LearnedRelations
from .batches import decode
from .backend.frozen.packing import prepare_problem
from .fast_geometry import topology
from .solver import optimize
from .canonical_native import warmup
from .native_reduce import warmup as reduce_warmup
from .io import HERE,ROOT,save,sha,sources,snapshot
from scoring.evaluation import evaluate_case

PARENT=ROOT/'experiments/E32_learned_relation_feedback'
CHECKPOINT=PARENT/'runs/20261022_replay_seed0/checkpoints/best.pt'
STEADY=dict(relaxed=False,mode=0,adaptive=True,fast_budget=False,late_extension=True,plateau=True,robust_seeds=True,certified=True,quick_project=True,quadratic_project=True,incumbent_project=True,snap=1e-4,native_reduce=True)

def load_models(checkpoint,example,draft):
    state=torch.load(checkpoint,map_location='cpu',weights_only=False);dense=RelationPolicy(**state['model_config']).eval();dense.load_state_dict(state['model'])
    sparse=SparsePairPolicy(**state['model_config']).eval();sparse.load_state_dict(state['model'])
    if any(not torch.equal(dense.state_dict()[k],sparse.state_dict()[k]) for k in dense.state_dict()):raise ValueError('Sparse parameter mismatch')
    g=graph(example);d=encode_positions(torch.as_tensor(draft,dtype=torch.float32)[None]/g['scale'][:,None,None],g);i,j=group_pairs(example)
    if not len(i):i=np.array([1]);j=np.array([0])
    with torch.inference_mode():
        dense=torch.jit.optimize_for_inference(torch.jit.freeze(torch.jit.trace(dense,(g,d,d),strict=False)))
        sparse=torch.jit.optimize_for_inference(torch.jit.freeze(torch.jit.trace(sparse,(g,d,d,torch.as_tensor(i),torch.as_tensor(j)),strict=False)))
    return dense,sparse

def main():
    started=time.perf_counter();finished=json.loads((PARENT/'runs/20261022_replay_seed0/results.json').read_text())
    if finished['status'] not in ('early_stop','max_epochs','budget'):raise ValueError('Parent weights not frozen')
    warmup();reduce_warmup();members=json.loads((PARENT/'runs/20261022_interface_check5/config.json').read_text())['members'];allrows=load_rows('train');mapping={tuple(r['member']):r for r in allrows};rows=[mapping[tuple(m)] for m in members]
    original=torch.load(WEIGHTS,map_location='cpu',weights_only=False);base=GeometryModel(**original['model_config']).eval();base.load_state_dict(original['model'])
    drafts=[predict(base,row['case'].solver_inputs)[0] for row in rows];dense,sparse=load_models(CHECKPOINT,rows[0]['case'].solver_inputs,drafts[0])
    folder=PARENT/'runs/prepared_relation_replay_v1/train';manifest=json.loads((folder/'manifest.json').read_text());needed={tuple(m) for m in members};states={}
    for part in manifest['parts']:
        chunk_members=manifest['members'][part['start']:part['start']+part['count']]
        if not any(tuple(m) in needed for m in chunk_members):continue
        path=folder/part['name'];st=path.stat()
        if (st.st_size,st.st_mtime_ns)!=(part['size'],part['mtime_ns']):raise ValueError('Replay state changed')
        value=torch.load(path,map_location='cpu',weights_only=False)
        for member,state in zip(value['members'],value['states']):
            if tuple(member) in needed:states[tuple(member)]=state
    records=[];timing={'dense':[],'sparse':[]};maximum=0.;trees_equal=True;fallbacks=0
    for row,draft in zip(rows,drafts):
        inputs=row['case'].solver_inputs;p=prepare_problem(*inputs);g=graph(inputs);initial=encode_positions(torch.as_tensor(draft,dtype=torch.float32)[None]/g['scale'][:,None,None],g);i,j=group_pairs(inputs);local=[]
        for slot,state in enumerate(states[tuple(row['member'])]):
            current=(decode(state[None],g)[0]*g['scale'][0]).numpy().astype(float);encoded=encode_positions(torch.as_tensor(current,dtype=torch.float32)[None]/g['scale'][:,None,None],g)
            if len(i):
                with torch.inference_mode():a=dense(g,initial,encoded)[0,i,j];b=sparse(g,initial,encoded,torch.as_tensor(i),torch.as_tensor(j))[0]
                difference=float((a-b).abs().max());maximum=max(maximum,difference)
            else:difference=0.
            full=topology(current,p.constraints,array=True);callbacks={'dense':LearnedRelations(dense,inputs,draft,feedback=True,axis=False),'sparse':SparseContacts(sparse,dense,inputs,draft,feedback=True)}
            for callback in callbacks.values():callback(current,p,full)
            for repetition in range(8):
                results={}
                for method in (('dense','sparse') if repetition%2==0 else ('sparse','dense')):
                    callback=callbacks[method];callback.scores=None;callback.calls=0;begin=time.perf_counter();results[method]=callback(current,p,full);timing[method].append(time.perf_counter()-begin)
                equal=np.array_equal(np.asarray(results['dense'][1]).reshape(-1,3),np.asarray(results['sparse'][1]).reshape(-1,3));trees_equal=trees_equal and equal
                fallbacks+=int(results['sparse'][2]['dense_fallback'])
            local.append({'slot':slot,'group_pairs':len(i),'score_max_abs_difference':difference})
        complete={}
        for method in ('dense','sparse'):
            callback=LearnedRelations(dense,inputs,draft,feedback=True,axis=False) if method=='dense' else SparseContacts(sparse,dense,inputs,draft,feedback=True)
            x,stats=optimize(draft,inputs,**STEADY,relation_policy=callback);complete[method]={'positions':x.tolist(),'trace':stats['trace'],'metrics':evaluate_case(row['case'],x.tolist(),stats['seconds'])['metrics'],'seconds':stats['seconds']}
        records.append({'member':row['member'],'n':p.n,'states':local,'complete':complete});print('sparse parity',p.n,'tree equal',trees_equal,flush=True)
    report={'summary':{'score_max_abs_difference':maximum,'trees_equal':trees_equal,'mean_dense_call_seconds':float(np.mean(timing['dense'])),'mean_sparse_call_seconds':float(np.mean(timing['sparse'])),
        'callback_speedup':float(np.mean(timing['dense'])/np.mean(timing['sparse'])),'dense_fallback_calls':fallbacks,'hard_legal':{m:sum(r['complete'][m]['metrics']['is_feasible'] for r in records) for m in ('dense','sparse')},'repeated_calls_per_method':len(timing['dense'])},
        'records':records,'checkpoint_sha256':sha(CHECKPOINT),'seconds':time.perf_counter()-started,'test_read':False}
    report['continue_to_terminal']=bool(trees_equal and maximum<=1e-4 and report['summary']['callback_speedup']>=1.2 and all(v==5 for v in report['summary']['hard_legal'].values()))
    output=HERE/'runs/20261022_sparse_check5';output.mkdir(parents=True,exist_ok=False);identity={'sources':sources(),'checkpoint_sha256':sha(CHECKPOINT),'resource':RESOURCE,'members':members}
    save(output/'config.json',identity);snapshot(output/'source_snapshot.zip',identity);save(output/'results.json',report);print(json.dumps(report['summary']),flush=True)

if __name__=='__main__':main()
