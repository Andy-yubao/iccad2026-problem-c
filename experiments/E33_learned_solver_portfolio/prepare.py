"""Train-only paired expert outcomes, all layouts retained in resumable SQLite."""
from .resources import configure
RESOURCE=configure()
import argparse
import datetime as dt
import importlib.metadata as metadata
import json
from pathlib import Path
import sqlite3
import sys
import time
import zlib
import numpy as np
import torch
from .cases import WEIGHTS,predict,graph,seed_all
from .model import GeometryModel
from .relations import RelationPolicy
from .policy import LearnedRelations
from .routing import prefix,statistics
from .data import load_rows
from .solver import optimize
from .canonical_native import warmup
from .native_reduce import warmup as reduce_warmup
from .io import HERE,ROOT,save,sha,sources,snapshot,exclusive
from .cache import verify
from datasets.reference_layouts import REGISTRY
from scoring.evaluation import evaluate_case

FOLDER=HERE/'runs/prepared_expert_outcomes_v1';DATABASE=FOLDER/'outcomes.sqlite3'
GEOMETRY=ROOT/'experiments/E30_stable_fast_geometry/runs/20261022_training_seed0/geometry_control/checkpoints/best.pt'
RELATION=ROOT/'experiments/E32_learned_relation_feedback/runs/20261022_replay_seed0/checkpoints/best.pt'
ACTION_COUNT=4
STEADY=dict(relaxed=False,mode=0,adaptive=True,fast_budget=False,late_extension=True,plateau=True,robust_seeds=True,certified=True,quick_project=True,quadratic_project=True,incumbent_project=True,snap=1e-4)

def train_order(rows):
    import hashlib
    grouped={}
    for i,row in enumerate(rows):grouped.setdefault(row['graph']['n'],[]).append(i)
    for ids in grouped.values():ids.sort(key=lambda i:hashlib.sha256(('E33.labels.v1:'+json.dumps(rows[i]['member'])).encode()).hexdigest())
    out=[]
    for depth in range(max(map(len,grouped.values()))):
        for n in sorted(grouped):
            if depth<len(grouped[n]):out.append(grouped[n][depth])
    return out

def load_experts(example):
    a=torch.load(WEIGHTS,map_location='cpu',weights_only=False);base=GeometryModel(**a['model_config']).eval();base.load_state_dict(a['model'])
    b=torch.load(GEOMETRY,map_location='cpu',weights_only=False);geometry=GeometryModel(**b['model_config']).eval();geometry.load_state_dict(b['model'])
    parent_result=json.loads((RELATION.parents[1]/'results.json').read_text())
    if parent_result['status'] not in ('early_stop','max_epochs','budget'):raise ValueError('Relation expert is not frozen')
    policy_state=torch.load(RELATION,map_location='cpu',weights_only=False)
    if policy_state['identity']['arm']!='replay' or policy_state['identity']['seed']!=0:raise ValueError('Registered relation expert changed')
    relation=RelationPolicy(**policy_state['model_config']).eval();relation.load_state_dict(policy_state['model'])
    from .relations import encode_positions
    g=graph(example);draft,_=predict(base,example);encoded=encode_positions(torch.as_tensor(draft,dtype=torch.float32)[None]/g['scale'][:,None,None],g)
    with torch.inference_mode():
        geometry=torch.jit.optimize_for_inference(torch.jit.freeze(torch.jit.trace(geometry,(g,),strict=False)))
        relation=torch.jit.optimize_for_inference(torch.jit.freeze(torch.jit.trace(relation,(g,encoded,encoded),strict=False)))
    return base,geometry,relation

def expert(action,case,base,geometry,relation=None,*,draft=None):
    started=time.perf_counter();inference=0.
    if draft is None:draft,inference=predict(geometry if action==2 else base,case.solver_inputs)
    policy=dict(STEADY)
    if action:policy['native_reduce']=True
    if action==1:policy['fast_budget']=True
    if action==2:policy['tail_refine']=True
    callback=LearnedRelations(relation,case.solver_inputs,draft,feedback=True,axis=True,contact=True) if action==3 else None
    x,stats=optimize(draft,case.solver_inputs,**policy,relation_policy=callback)
    stats['draft_inference_seconds']=inference
    return x,stats,time.perf_counter()-started

def packed(value):return zlib.compress(json.dumps(value,ensure_ascii=False,separators=(',',':')).encode(),level=3)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--limit',type=int,choices=(1000,16000),default=1000);ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    started=time.perf_counter();deadline=dt.datetime.fromisoformat('2026-10-22T19:39:00+08:00').timestamp();FOLDER.mkdir(parents=True,exist_ok=True)
    critical=['model.py','cases.py','features.py','batches.py','routing.py','prepare.py','solver.py','fast_geometry.py','projection.py','certify_polish.py','native_reduce.py','canonical_native.py','canonical.py','canonical_py.py','legal_seed.py','data.py','cache.py','resources.py','io.py','relations.py','policy.py']
    generation_sources={name:digest for name,digest in sources().items() if not name.startswith('experiments/') or '/backend/' in name or Path(name).name in critical}
    identity={'revision':'E33.experts.train_only.v1','actions':ACTION_COUNT,'sources':generation_sources,'weights':{'E25':sha(WEIGHTS),'E30':sha(GEOMETRY),'E32_relation':sha(RELATION)},
        'registry_sha256':sha(REGISTRY),'cache_identity':verify('train')['identity'],'versions':{'torch':str(torch.__version__),'numpy':np.__version__,'clarabel':metadata.version('clarabel')},
        'policy':STEADY,'action1':'native, fast_budget=True','action2':'E30+JIT+native+tail','action3':'E25 plain draft + frozen E32 replay full_feedback','max_seconds':21600,'resources':{k:v for k,v in RESOURCE.items() if k!='pid'},'test_read':False}
    with exclusive(FOLDER/'prepare.lock'):
        if DATABASE.exists() and not args.resume:raise ValueError('Use --resume for existing label data')
        con=sqlite3.connect(DATABASE);con.execute('PRAGMA journal_mode=WAL');con.execute('PRAGMA synchronous=FULL')
        con.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
        con.execute('CREATE TABLE IF NOT EXISTS cases (idx INTEGER PRIMARY KEY, member TEXT UNIQUE NOT NULL, n INTEGER NOT NULL, features BLOB NOT NULL, prefix_seconds REAL NOT NULL, feature_seconds REAL NOT NULL)')
        con.execute('CREATE TABLE IF NOT EXISTS outcomes (idx INTEGER NOT NULL, action INTEGER NOT NULL, seconds REAL NOT NULL, q REAL NOT NULL, inference_seconds REAL NOT NULL, metrics TEXT NOT NULL, positions BLOB NOT NULL, trace BLOB NOT NULL, PRIMARY KEY(idx,action))')
        found=con.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
        if found:
            if json.loads(found[0])!=identity:raise ValueError('Label-generation identity changed')
        else:
            con.execute('INSERT INTO metadata VALUES (?,?)',('identity',json.dumps(identity,sort_keys=True)));con.commit();save(FOLDER/'identity.json',identity);snapshot(FOLDER/'source_snapshot.zip',{'sources':sources()})
        completed_path=FOLDER/f'results_{args.limit}.json'
        if completed_path.exists() and json.loads(completed_path.read_text())['status'] in ('complete','pilot_complete'):
            con.close();print('Completed label stage reused',flush=True);return
        prior_row=con.execute("SELECT value FROM metadata WHERE key='elapsed'").fetchone();prior=float(prior_row[0]) if prior_row else 0.;elapsed=lambda:prior+time.perf_counter()-started
        seed_all(0);warmup();reduce_warmup();rows=load_rows('train');order=train_order(rows);base,geometry,relation=load_experts(rows[order[0]]['case'].solver_inputs)
        rng=np.random.default_rng(20261022);blocks=[rng.permutation(ACTION_COUNT).tolist() for _ in range((len(order)+ACTION_COUNT-1)//ACTION_COUNT)]
        for ordinal,index in enumerate(order[:args.limit]):
            if len(con.execute('SELECT action FROM outcomes WHERE idx=?',(index,)).fetchall())==ACTION_COUNT:continue
            if (FOLDER/'STOP').exists() or elapsed()>=21600 or time.time()>=deadline:
                save(FOLDER/'progress.json',{'status':'paused','ordinal':ordinal,'limit':args.limit,'elapsed_seconds':elapsed(),'test_read':False});break
            row=rows[index];case=row['case'];seed_all(index);draft,hidden,prefix_seconds=prefix(base,case.solver_inputs)
            feature_start=time.perf_counter();features=statistics(case.solver_inputs,draft,hidden);feature_seconds=time.perf_counter()-feature_start
            existing=con.execute('SELECT member FROM cases WHERE idx=?',(index,)).fetchone()
            if existing and json.loads(existing[0])!=list(row['member']):raise ValueError('Case identity changed')
            if not existing:con.execute('INSERT INTO cases VALUES (?,?,?,?,?,?)',(index,json.dumps(row['member']),case.block_count,features.tobytes(),prefix_seconds,feature_seconds));con.commit()
            order_actions=blocks[ordinal//ACTION_COUNT];shift=ordinal%ACTION_COUNT;order_actions=order_actions[shift:]+order_actions[:shift]
            for action in order_actions:
                if con.execute('SELECT 1 FROM outcomes WHERE idx=? AND action=?',(index,action)).fetchone():continue
                seed_all(index);action_start=time.perf_counter();positions=[]
                try:
                    x,stats,seconds=expert(action,case,base,geometry,relation);positions=x.tolist()
                    try:metrics=evaluate_case(case,positions,seconds)['metrics']
                    except Exception as error:metrics={'cost':10.,'is_feasible':False,'scoring_error':repr(error)}
                except Exception as error:
                    seconds=time.perf_counter()-action_start;metrics={'cost':10.,'is_feasible':False,'solver_error':repr(error)};stats={'trace':[{'phase':'exception','error':repr(error),'returned_layout':False}]}
                con.execute('INSERT INTO outcomes VALUES (?,?,?,?,?,?,?,?)',(index,action,seconds,float(metrics['cost']),stats.get('draft_inference_seconds',0.),json.dumps(metrics),packed(positions),packed(stats['trace'])))
                con.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',('elapsed',str(elapsed())));con.commit()
            if (ordinal+1)%25==0:
                save(FOLDER/'progress.json',{'status':'running','completed':ordinal+1,'limit':args.limit,'elapsed_seconds':elapsed(),'test_read':False})
            if (ordinal+1)%100==0:print('paired train',ordinal+1,'/',args.limit,'seconds',round(elapsed(),1),flush=True)
        count=con.execute('SELECT COUNT(*) FROM (SELECT idx FROM outcomes GROUP BY idx HAVING COUNT(*)=?)',(ACTION_COUNT,)).fetchone()[0]
        status='complete' if count==16000 else ('pilot_complete' if count>=1000 else 'paused')
        layouts=con.execute('SELECT COUNT(*) FROM outcomes').fetchone()[0]
        con.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',('elapsed',str(elapsed())));con.commit()
        if status=='complete':
            checkpoint_status=con.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone()
            if checkpoint_status[0]!=0:raise RuntimeError('Outcome WAL could not be safely checkpointed')
        con.close()
        if status=='complete':
            artifact=FOLDER/'manifest.json'
            if not artifact.exists():
                info=DATABASE.stat();save(artifact,{'identity_sha256':sha(FOLDER/'identity.json'),'database_sha256':sha(DATABASE),'size':info.st_size,'mtime_ns':info.st_mtime_ns,
                    'cases':16000,'actions':ACTION_COUNT,'layouts':16000*ACTION_COUNT,'test_read':False})
        save(FOLDER/f'results_{args.limit}.json',{'status':status,'complete_cases':count,'elapsed_seconds':elapsed(),'expert_layouts':layouts,'test_read':False})

def load_matrix():
    artifact=FOLDER/'manifest.json'
    if artifact.exists():
        manifest=json.loads(artifact.read_text());info=DATABASE.stat()
        if (info.st_size,info.st_mtime_ns)!=(manifest['size'],manifest['mtime_ns']):raise ValueError('Frozen outcome database changed')
    con=sqlite3.connect(f'file:{DATABASE.as_posix()}?mode=ro',uri=True);items=[]
    for index,member,n,features,prefix_seconds,feature_seconds in con.execute('SELECT idx,member,n,features,prefix_seconds,feature_seconds FROM cases ORDER BY idx'):
        outcomes=con.execute('SELECT action,q,seconds,inference_seconds FROM outcomes WHERE idx=? ORDER BY action',(index,)).fetchall()
        if len(outcomes)!=ACTION_COUNT:continue
        if [r[0] for r in outcomes]!=list(range(ACTION_COUNT)):raise ValueError('Outcome action identity changed')
        q=np.asarray([r[1] for r in outcomes]);standalone=np.asarray([r[2] for r in outcomes]);seconds=standalone.copy()
        for action in range(ACTION_COUNT):seconds[action]+=prefix_seconds if action==2 else prefix_seconds-outcomes[action][3]
        seconds+=feature_seconds
        items.append({'index':index,'member':json.loads(member),'n':n,'features':np.frombuffer(features,dtype=np.float32).copy(),'q':q,'seconds':seconds,
            'standalone_seconds':standalone,'prefix_seconds':prefix_seconds,'feature_seconds':feature_seconds})
    con.close();return items

if __name__=='__main__':main()
