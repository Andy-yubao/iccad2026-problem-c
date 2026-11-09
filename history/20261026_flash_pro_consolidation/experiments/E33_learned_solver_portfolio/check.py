"""Original five train cases only: same draft, public gate, expert legality and SQLite."""
from .resources import configure
RESOURCE=configure()
import json
import sqlite3
import time
import zlib
import numpy as np
import torch
from .data import load_rows
from .cases import predict
from .prepare import load_experts,expert,packed,ACTION_COUNT
from .routing import prefix,statistics,Gate,objective
from .inference import ConsensusGate
from .canonical_native import warmup
from .native_reduce import warmup as reduce_warmup
from .io import HERE,ROOT,save,sources,snapshot
from scoring.evaluation import evaluate_case

def main():
    started=time.perf_counter();torch.manual_seed(7);torch.cuda.manual_seed_all(7);torch.use_deterministic_algorithms(True);warmup();reduce_warmup()
    members=json.loads((ROOT/'experiments/E32_learned_relation_feedback/runs/20261022_interface_check5/config.json').read_text())['members'];allrows=load_rows('train');mapping={tuple(r['member']):r for r in allrows};rows=[mapping[tuple(m)] for m in members]
    base,geometry,relation=load_experts(rows[0]['case'].solver_inputs);features=[];records=[]
    for row in rows:
        inputs=row['case'].solver_inputs;draft,h,_=prefix(base,inputs);ordinary,_=predict(base,inputs);assert np.array_equal(draft,ordinary)
        x=statistics(inputs,draft,h);features.append(x);record={'member':row['member'],'n':int(inputs[0]),'draft_exact_parity':True,'experts':{}}
        for action in range(ACTION_COUNT):
            p,stats,seconds=expert(action,row['case'],base,geometry,relation);metrics=evaluate_case(row['case'],p.tolist(),seconds)['metrics'];assert metrics['is_feasible']
            record['experts'][str(action)]={'seconds':seconds,'metrics':metrics,'positions':p.tolist()}
        records.append(record);print('expert gate check',record['n'],flush=True)
    assert len({len(f) for f in features})==1
    gate=Gate(len(features[0]),actions=ACTION_COUNT).cuda();x=torch.tensor(np.stack(features),device='cuda');q=torch.tensor([[r['experts'][str(a)]['metrics']['cost'] for a in range(ACTION_COUNT)] for r in records],device='cuda');t=torch.tensor([[r['experts'][str(a)]['seconds'] for a in range(ACTION_COUNT)] for r in records],device='cuda')
    ids=torch.arange(32,device='cuda')%5;loss,_,_=objective(gate,x[ids],q[ids],t[ids]);loss.mean().backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in gate.parameters())
    import copy
    optimizer=torch.optim.AdamW(gate.parameters(),lr=1e-3);optimizer.step();saved_model=copy.deepcopy(gate.state_dict());saved_optimizer=copy.deepcopy(optimizer.state_dict())
    def update(model,optim):
        optim.zero_grad(set_to_none=True);values,_,_=objective(model,x[ids],q[ids],t[ids]);values.mean().backward();optim.step()
    update(gate,optimizer);expected=copy.deepcopy(gate.state_dict());restored_gate=Gate(len(features[0])).cuda();restored_gate.load_state_dict(saved_model);restored_optimizer=torch.optim.AdamW(restored_gate.parameters(),lr=1e-3);restored_optimizer.load_state_dict(saved_optimizer);update(restored_gate,restored_optimizer)
    assert all(torch.equal(expected[key],restored_gate.state_dict()[key]) for key in expected)
    class Constant(torch.nn.Module):
        def __init__(self,action):super().__init__();self.action=action
        def forward(self,value):
            logits=torch.zeros(len(value),ACTION_COUNT,device=value.device);logits[:,self.action]=8.;return logits,torch.zeros_like(logits)
    agree=ConsensusGate([Constant(1),Constant(1),Constant(1)])(x[:1])[0].softmax(-1);disagree=ConsensusGate([Constant(1),Constant(0),Constant(1)])(x[:1])[0].softmax(-1)
    assert int(agree.argmax(1)[0])==1 and float(agree.max())>.9 and float(disagree.max())<.5
    con=sqlite3.connect(':memory:');con.execute('CREATE TABLE layouts (id INTEGER PRIMARY KEY, positions BLOB)');value=records[0]['experts']['0']['positions'];con.execute('INSERT INTO layouts VALUES (?,?)',(0,packed(value)));con.commit()
    restored=json.loads(zlib.decompress(con.execute('SELECT positions FROM layouts WHERE id=0').fetchone()[0]));assert restored==value
    try:con.execute('INSERT INTO layouts VALUES (?,?)',(0,packed(value)))
    except sqlite3.IntegrityError:pass
    else:raise ValueError('Duplicate layout overwrote evidence')
    con.close();folder=HERE/'runs/20261022_interface_check5';folder.mkdir(parents=True,exist_ok=False);identity={'sources':sources(),'members':members,'resource':RESOURCE}
    save(folder/'config.json',identity);snapshot(folder/'source_snapshot.zip',identity);save(folder/'results.json',{'records':records,'feature_dimensions':len(features[0]),'gate_parameters':sum(p.numel() for p in gate.parameters()),'batch_size':32,'same_five_repeated_for_batch':True,
        'finite_gradients':True,'exact_gate_optimizer_restore':True,'consensus_disagreement_to_E25_check':True,'exact_frozen_draft_parity':True,'sqlite_layout_roundtrip_and_duplicate_protection':True,'seconds':time.perf_counter()-started,'test_read':False})
    print('Five-case portfolio interface passed',flush=True)

if __name__=='__main__':main()
