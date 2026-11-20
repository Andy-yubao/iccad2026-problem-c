"""User-authorized single recovery; preserve original evidence and exact scientific scope."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
RECOVERY=HERE/'recovery1_20261109'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not RECOVERY.exists(),'Recovery already prepared; inspect instead of retry'
original=read(HERE/'plan.json')
assert read(HERE/'output/state.json')['completed']==0
assert not list((HERE/'output').glob('cases/*.json'))
RECOVERY.mkdir()
archive=RECOVERY/'original_evidence';archive.mkdir()
for filename in ('run.py','prepare.py','plan.json','launch.ps1','launch.json','interruption_verification.json','stdout.log','stderr.log'):
    p=HERE/filename
    if p.exists():shutil.copyfile(p,archive/filename)
shutil.copytree(HERE/'output',archive/'output')
worker=(HERE/'run.py').read_text(encoding='utf-8').replace('ROOT=HERE.parents[2]','ROOT=HERE.parents[3]\nORIGINAL=HERE.parent')
worker=worker.replace("if (HERE/'STOP').exists():raise InterruptedError('STOP before dispatch')",
    "if (HERE/'STOP').exists() or (ORIGINAL/'STOP').exists():raise InterruptedError('STOP before dispatch')")
worker=worker.replace("if requested[0] or (HERE/'STOP').exists() or (out/'STOP').exists():",
    "if requested[0] or (HERE/'STOP').exists() or (ORIGINAL/'STOP').exists() or (out/'STOP').exists():")
start=worker.index('            items=[]\n')
end=worker.index("            save(out/'frozen.json'",start)
worker=worker[:start]+'''            def load_item(i):
                check();record=plan['records'][i]
                beat(stage='loading_existing_authorized_test_NPZ',loading_index=i,loaded_payloads=i)
                assert {k:v for k,v in record.items() if k!='frozen_test_index'}==reg['sets']['test']['records'][record['frozen_test_index']]
                path=(folder/record['payload']).resolve()
                assert path.is_relative_to(folder/'test') and sha(path)==ledger[record['id']]
                with np.load(path,allow_pickle=False) as raw:
                    inputs=(len(raw['areas']),*[raw[k].copy() for k in ('areas','b2b','p2b','pins','constraints','specs')])
                    supervision={k:raw[k].copy() for k in ('reference_xywh','stored_metrics')}
                item=dict(member=record['member'],solver_inputs=inputs,supervision=supervision)
                beat(stage='building_scoring_case_from_existing_reference',loading_index=i,loaded_payloads=i+1)
                case=_build_case('authorized_random_test:'+str(item['member']),dict(member=item['member'],role='authorized_test500'),
                    [torch.as_tensor(v).clone() for v in inputs[1:6]],supervision['reference_xywh'].tolist(),supervision['stored_metrics'])
                np.testing.assert_array_equal(case.solver_inputs[6],inputs[6])
                return item,case
            first_item,first_case=load_item(0)
''' +worker[end:]
worker=worker.replace("Predictor(ROOT/plan['checkpoint'],items[0]['solver_inputs'])","Predictor(ROOT/plan['checkpoint'],first_item['solver_inputs'])")
worker=worker.replace('existing_NPZ_loaded=500','existing_NPZ_loaded_at_initialization=1')
worker=worker.replace('for i,(item,case) in enumerate(zip(items,cases)):',
    'for i in range(500):\n                    item,case=(first_item,first_case) if i==0 else load_item(i)')
worker=worker.replace("'stage='evaluating_frozen_test500_original_budget'", "'stage='evaluating_frozen_test500_original_budget'")
worker=worker.replace("out/'frozen.json',dict(plan_sha256", "out/'frozen.json',dict(recovery_authorized=True,original_plan_sha256=plan['original_plan_sha256'],plan_sha256")
assert 'items[' not in worker and 'zip(items' not in worker
ast.parse(worker);(RECOVERY/'run.py').write_text(worker,encoding='utf-8')
supervisor='''"""Console-free supervisor; one worker, explicit status, no retry."""
import hashlib,json,os,subprocess,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+'.'+str(os.getpid())+'.tmp')
    temp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\\n',encoding='utf-8')
    for attempt in range(30):
        try:os.replace(temp,p);return
        except PermissionError:
            if attempt==29:raise
            time.sleep(.05)
state=dict(status='starting',supervisor_pid=os.getpid(),started_at=now(),automatic_retry=False,training=False,test_read=True)
def beat(**kw):state.update(kw);save(HERE/'supervisor_state.json',dict(**state,updated_at=now()))
try:
    if (HERE/'supervisor_state.json').exists():raise FileExistsError('Recovery already dispatched; no retry')
    beat()
    plan=read(HERE/'plan.json')
    for name,digest in plan['sources'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,('Source changed',name)
    assert not (HERE/'STOP').exists() and not (HERE.parent/'STOP').exists()
    with (HERE/'stdout.log').open('x',encoding='utf-8') as out,(HERE/'stderr.log').open('x',encoding='utf-8') as err:
        child=subprocess.Popen([str(ROOT/'.conda/envs/iccad2026/python.exe'),'-B','-u',str(HERE/'run.py')],
            cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,close_fds=True,
            creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP)
        beat(status='running',worker_pid=child.pid)
        while child.poll() is None:
            progress=HERE/'output/state.json'
            if progress.exists():
                try:
                    p=read(progress);beat(completed=p['completed'],stage=p['stage'],worker_updated_at=p['updated_at'])
                except (OSError,json.JSONDecodeError):pass
            else:beat(stage='worker_imports_and_identity_verification')
            time.sleep(3)
    result=HERE/'output/results.json'
    if child.returncode==0 and result.exists() and read(result)['status']=='complete':
        beat(status='complete',completed=500,returncode=0,completed_at=now())
    else:raise RuntimeError('Worker terminated: returncode='+str(child.returncode)+'; no retry')
except BaseException as error:
    beat(status='needs_attention',error=repr(error),traceback=traceback.format_exc())
    sys.exit(1)
'''
ast.parse(supervisor);(RECOVERY/'supervisor.py').write_text(supervisor,encoding='utf-8')
# The worker log is owned by the supervisor, so remove the old self-redirection wrapper.
worker=worker[:worker.index("if __name__=='__main__':")]+"if __name__=='__main__':\n    main()\n"
ast.parse(worker);(RECOVERY/'run.py').write_text(worker,encoding='utf-8')
plan=dict(original)
plan['sources']=dict(original['sources'])
plan.update(recovery_authorized=True,recovery_authorization='User explicitly requested fix and restart after interruption, 2026-11-09',
    original_plan_sha256=sha(HERE/'plan.json'),attempt=1,automatic_retry=False,
    engineering_changes=['pythonw console-free supervisor','CREATE_NO_WINDOW|CREATE_NEW_PROCESS_GROUP worker with redirected stdin/stdout/stderr',
        'stream one existing NPZ/scoring case at a time; per-case load heartbeat; unchanged inference/solver/scoring/order'])
for p in (RECOVERY/'run.py',RECOVERY/'supervisor.py',HERE/'prepare_recovery1.py'):
    plan['sources'][p.relative_to(ROOT).as_posix()]=sha(p)
assert plan['records']==original['records'] and plan['sampling']==original['sampling']
for key in ('checkpoint_sha256','budget_sha256','registry_sha256','payload_checksums_sha256'):assert plan[key]==original[key]
for name,digest in plan['sources'].items():assert sha(ROOT/name)==digest
(RECOVERY/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(RECOVERY/'preparation.json').write_text(json.dumps(dict(status='ready',static_parse_passed=True,original_completed=0,
    exact_frozen_members_preserved=True,checkpoint_and_original_budget_unchanged=True,
    original_evidence_sha256={p.relative_to(archive).as_posix():sha(p) for p in archive.rglob('*') if p.is_file()},
    separate_preflight_test_inference_or_solver_calls=0,automatic_retry=False),indent=2)+'\n',encoding='utf-8')
print('Single recovery ready; original evidence archived, same500/checkpoint/budget, no-console supervisor, streaming progress, no retry.')
