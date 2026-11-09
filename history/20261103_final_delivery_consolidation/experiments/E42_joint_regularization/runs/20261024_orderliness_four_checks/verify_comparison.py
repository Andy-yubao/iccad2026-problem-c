"""Two-panel rubric and optional dimensions: disposable-copy persistence checks."""
from pathlib import Path
import sys,json,shutil
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
from visualization.review import ReviewStore
HERE=Path(__file__).resolve().parent;QA=ROOT/'.cache/orderliness_pairwise_qa'
assert QA.resolve().is_relative_to((ROOT/'.cache').resolve())
QA.mkdir(exist_ok=False);(QA/'images').mkdir()
m=json.loads((HERE/'comparison_review/manifest.json').read_text(encoding='utf-8'));m['session_id']='disposable_qa';m['title']='临时验证副本';m['cases']=m['cases'][:1]
for row in m['cases']:shutil.copy2(HERE/'comparison_review'/row['image'],QA/row['image'])
(QA/'manifest.json').write_text(json.dumps(m,ensure_ascii=False),encoding='utf-8')
store=ReviewStore(QA);key=m['cases'][0]['id']
result=store.update({'action':'rating','expected_revision':0,'id':key,'score':1,'note':'临时核验','tags':['观察'],
                     'dimensions':{'repairability':3,'whitespace':None}})
assert result['ratings'][key]['dimensions']=={'repairability':3,'whitespace':None}
try:
    store.update({'action':'rating','expected_revision':1,'id':key,'score':2,'dimensions':{'unknown':1}})
    raise AssertionError('Unknown dimension accepted')
except ValueError:pass
store.db.close();store=ReviewStore(QA);assert store.state()==result;store.db.close()
# A fresh disposable state is left for UI actions; the real five-pair state is untouched.
for f in QA.glob('ratings.sqlite3*'):f.unlink()
(HERE/'comparison_verification.json').write_text(json.dumps({'backend_passed':True,'optional_answers_saved_and_restored':True,
     'unknown_dimension_rejected':True,'rubric_version':2,'actual_human_ratings_inserted':0,'browser':'pending'},indent=2)+'\n',encoding='utf-8')
print('PASS: comparison rubric dimensions round-trip; disposable UI fixture ready')
