"""Static feature inventory and saved-score contradictions; standard library only."""
from pathlib import Path
import ast
import hashlib
import json
import math
import sqlite3
import zlib

ROOT=Path(__file__).resolve().parents[4];HERE=ROOT/'experiments/E35_optimizer_attribution';OUT=Path(__file__).resolve().parent


def main():
    folder=HERE/'runs/20261022_attribution100';config=json.loads((folder/'config.json').read_text())
    tree=ast.parse((HERE/'reference_solver.py').read_text());f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='optimize')
    values={a.arg:ast.literal_eval(v) for a,v in zip(f.args.kwonlyargs,f.args.kw_defaults)};values.update(config['config'])
    switches={k:{'value':v,'source':'steady override' if k in config['config'] else 'frozen default'} for k,v in values.items() if isinstance(v,bool)}
    con=sqlite3.connect('file:'+(folder/'outcomes.sqlite3').resolve().as_posix()+'?mode=ro',uri=True)
    records=[json.loads(zlib.decompress(row[0])) for row in con.execute("SELECT payload FROM records WHERE arm='full'")];con.close()
    scores=json.loads((OUT/'layout_scores_local.json').read_text());bad=[];penalties={}
    for r in records:
        local=scores[str(r['index'])]
        for e in r['events']:
            if e['kind']=='acceptance' and e['depth']==1 and e['updated'] and e['before'] in local and e['after'] in local:
                before,after=local[e['before']],local[e['after']];dq=after['cost']-before['cost']
                if dq>1e-8:bad.append({'index':r['index'],'member':r['member'],'n':r['n'],'module':e['module'],'proxy_gain':e['before_proxy']-e['after_proxy'],'Q_regression':dq,'before_Q':before['cost'],'after_Q':after['cost'],'before_soft':before.get('total_soft_violations'),'after_soft':after.get('total_soft_violations')})
            if e['kind']=='penalty_terms':
                p=penalties.setdefault(e['module'],{'calls':0,'boundary_nonempty_calls':0,'contact_nonempty_calls':0,'boundary_cases':set(),'contact_cases':set(),'reweighted_calls':0});p['calls']+=1
                if e['boundary_terms']>0:p['boundary_nonempty_calls']+=1;p['boundary_cases'].add(r['index'])
                if e['contact_terms']>0:p['contact_nonempty_calls']+=1;p['contact_cases'].add(r['index'])
                p['reweighted_calls']+=e.get('reweighted',False)
    for p in penalties.values():p['boundary_cases']=len(p['boundary_cases']);p['contact_cases']=len(p['contact_cases'])
    bad.sort(key=lambda r:-r['Q_regression'])
    result={'switches':switches,'enabled_boolean_count':sum(s['value'] for s in switches.values()),'disabled_boolean_count':sum(not s['value'] for s in switches.values()),
        'note':'Dormant switches were not active in E25 steady; absence is not an importance ranking. Fixed/preplaced/area/nonnegative/no-overlap invariants remain enforced.',
        'penalty_term_opportunities_all_recursive_calls':penalties,'proxy_improved_Q_worsened_updates':bad,'regression_gt_1e_4':sum(r['Q_regression']>1e-4 for r in bad),
        'regression_gt_1e_3':sum(r['Q_regression']>1e-3 for r in bad),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'test_read':False}
    (OUT/'inventory.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'enabled':result['enabled_boolean_count'],'disabled':result['disabled_boolean_count'],'penalties':penalties,'contradictions':len(bad),'gt_1e_4':result['regression_gt_1e_4'],'gt_1e_3':result['regression_gt_1e_3'],'worst':bad[:3]},ensure_ascii=False))


if __name__=='__main__':main()
