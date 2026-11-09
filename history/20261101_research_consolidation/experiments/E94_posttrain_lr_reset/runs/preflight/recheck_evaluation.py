"""Five saved results only; explicit pre-launch evaluation-report fix registration."""
import ast,json,time
from experiments.E94_posttrain_lr_reset.io import ROOT,HERE,save,sources,snapshot
from experiments.E94_posttrain_lr_reset.qualification.common import HOME,evaluator_sources,registration
from experiments.E94_posttrain_lr_reset.qualification.compare import draft_geometry,paired
begin=time.perf_counter();path=HERE/'runs/preflight/results.json';value=json.loads(path.read_text(encoding='utf-8'))
assert value['sources']==sources();previous=value['evaluation_sources'];reg=registration()
for p in HOME.rglob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
source=ROOT/reg['previous_results']['E93_C'];records=json.loads(source.read_text(encoding='utf-8'))['records']
expected=json.loads((source.parents[1]/'geometry500.json').read_text(encoding='utf-8'))['records']
for i in range(5):
    actual=draft_geometry(records[i]);old=expected[i]
    for key in ('N10','module_pass10_fraction','max_covered_fraction','bbox_over_A'):assert abs(actual[key]-old[key])<1e-12
    assert 0<=actual['empty_fraction_of_bbox']<=1
zero=paired(records,records);assert zero['weighted_Q_delta']==0 and zero['worker_paired_95pct_interval']==[0.,0.]
current=evaluator_sources();changed=[k for k in current if current[k]!=previous.get(k)]
assert changed==['experiments/E94_posttrain_lr_reset/qualification/compare.py']
value.update(evaluation_sources=current,evaluation_report_recheck={'status':'passed','reason':'derive bbox_over_A from existing module_overlap result; worker draft_diagnostics has no bbox key',
    'changed_files':changed,'previous_sha256':{k:previous[k] for k in changed},'new_sha256':{k:current[k] for k in changed},
    'saved_cases':5,'new_model_inference':0,'postprocessing_calls':0,'self_paired_Q_interval':[0.,0.]})
save(path,value);snapshot(HERE/'runs/preflight/source_snapshot.zip',{'sources':{**value['sources'],**current}})
from experiments.E94_posttrain_lr_reset.budget import charge_preflight,historical_charged_seconds
charge_preflight(time.perf_counter()-begin)
authorization=json.loads((HERE/'runs/authorization.json').read_text(encoding='utf-8'));authorization['historical_charged_seconds_readonly']=historical_charged_seconds();save(HERE/'runs/authorization.json',authorization)
from experiments.E94_posttrain_lr_reset.preflight import verify
verify();print('Evaluation-only five saved cases, zero-delta paired interval and complete final identities passed; no repeated NN/solver calls')
