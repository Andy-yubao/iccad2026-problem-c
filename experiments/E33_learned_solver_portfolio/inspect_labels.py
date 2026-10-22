"""Train-counterfactual headroom only; neither trained policy nor test evidence."""
from .resources import configure
configure()
import json
import numpy as np
import torch
from .prepare import load_matrix,FOLDER,DATABASE,ACTION_COUNT
from .routing import oracle_actions
from .io import save

def metrics(q,t,n):
    w=np.exp((n-n.max())/12);count=max(1,int(np.ceil(.05*len(q))))
    return {'count':len(q),'weighted_Q':float(np.average(q,weights=w)),'mean_Q':float(q.mean()),'cvar95_Q':float(np.sort(q)[-count:].mean()),'max_Q':float(q.max()),'seconds_mean':float(t.mean()),'seconds_p95':float(np.percentile(t,95)),'seconds_max':float(t.max())}

def main():
    import sqlite3
    rows=load_matrix();q=np.stack([r['q'] for r in rows]);t=np.stack([r['seconds'] for r in rows]);standalone=np.stack([r['standalone_seconds'] for r in rows]);n=np.asarray([r['n'] for r in rows])
    oracle=oracle_actions(torch.tensor(q),torch.tensor(t),torch.tensor(n)).numpy();k=np.arange(len(rows))
    quality_oracle=np.where(q<=q.min(1,keepdims=True)+.002,t,1e4).argmin(1)
    report={'scope':'train-only paired expert counterfactual; oracle consumes reference quality and is not deployable','experts':{str(a):metrics(q[:,a],standalone[:,a],n) for a in range(ACTION_COUNT)},
        'optimistic_oracle_with_prefix_cost':metrics(q[k,oracle],t[k,oracle],n),'quality_near_min_oracle':metrics(q[k,quality_oracle],t[k,quality_oracle],n),
        'oracle_rule':'module-importance quality regret + .05 seconds + .2 max(Q-1.2,0), per-case Q regret cap .05',
        'oracle_actions':np.bincount(oracle,minlength=ACTION_COUNT).tolist(),'complete_training_cases':len(rows),'test_read':False}
    con=sqlite3.connect(f'file:{DATABASE.as_posix()}?mode=ro',uri=True)
    feasible={(index,action):bool(json.loads(raw).get('is_feasible',False)) for index,action,raw in con.execute('SELECT idx,action,metrics FROM outcomes')}
    con.close()
    for action in range(ACTION_COUNT):report['experts'][str(action)]['hard_legal']=sum(feasible[(row['index'],action)] for row in rows)
    report['optimistic_oracle_with_prefix_cost']['hard_legal']=sum(feasible[(row['index'],int(action))] for row,action in zip(rows,oracle))
    report['quality_near_min_oracle']['hard_legal']=sum(feasible[(row['index'],int(action))] for row,action in zip(rows,quality_oracle))
    baseline=report['experts']['0'];ideal=report['optimistic_oracle_with_prefix_cost']
    report['continue_to_full_labels']=bool(ideal['weighted_Q']<baseline['weighted_Q']-.001 and ideal['seconds_mean']<baseline['seconds_mean']*.98)
    path=FOLDER/f'headroom_{len(rows)}.json'
    if path.exists():raise ValueError('Headroom evidence immutable')
    save(path,report);print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
