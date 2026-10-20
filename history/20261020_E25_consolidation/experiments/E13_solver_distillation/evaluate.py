"""Full dev reference diagnostics or fixed dev100 actual-student repair evaluation."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
from .io import ROOT,HERE,save,sha
from .train import seed_all,sources
from .student import Student
from .pipeline import Pipeline
from .backend.frozen.inference import prepare_inference
from .backend.frozen.packing import prepare_problem
from datasets.worksets import LayoutReader,workset_manifest
from datasets.cases import _build_case
from scoring.evaluation import evaluate_case

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--phase',choices=['supervised','repair'],required=True);ap.add_argument('--device',default='cuda')
    ap.add_argument('--teacher-bank',type=Path);ap.add_argument('--oracle',choices=['geometry','structure'])
    ap.add_argument('--resume',action='store_true');ap.add_argument('--max-seconds',type=float,default=1200)
    args=ap.parse_args();begin=time.perf_counter();seed_all(0)
    checkpoint_header=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if checkpoint_header.get('smoke') or checkpoint_header.get('identity',{}).get('implementation_check'):
        raise ValueError('Five-case diagnostic checkpoints cannot enter formal evaluation')
    if args.oracle and (args.phase!='repair' or args.teacher_bank is None):raise ValueError('Oracle mode requires repair phase and explicit teacher bank')
    reader=LayoutReader(workset_manifest(),'dev' if args.phase=='supervised' else 'dev_quality')
    teacher_records={};bank_identity=None
    if args.teacher_bank:
        cfg=json.loads((args.teacher_bank/'config.json').read_text(encoding='utf-8'))['identity']
        state=json.loads((args.teacher_bank/'results.json').read_text(encoding='utf-8'))
        if cfg['split']!='dev_monitor' or cfg['pilot'] or state['status']!='completed' or state['error_count']:raise ValueError('Expected complete fixed dev_monitor teacher bank')
        if cfg['workset_sha256']!=sha(ROOT/'protocol/workset_v2.json'):raise ValueError('Teacher bank workset mismatch')
        teacher_records={r['case_id']:r for r in state['records']};bank_identity=sha(args.teacher_bank/'config.json')
    identity={'sources':sources(),'checkpoint_sha256':sha(args.checkpoint),'phase':args.phase,'device':args.device,
      'oracle':args.oracle,'teacher_bank_config_sha256':bank_identity,'members':reader.members,'max_seconds':args.max_seconds,
      'split_sha256':sha(ROOT/'protocol/split_v1.json'),'workset_sha256':sha(ROOT/'protocol/workset_v2.json'),'local_test_read':False}
    # JSON normalizes tuples in membership.
    identity=json.loads(json.dumps(identity))
    if args.resume:
        if json.loads((args.output/'config.json').read_text(encoding='utf-8'))['identity']!=identity:raise ValueError('Evaluation resume identity mismatch')
        prior=json.loads((args.output/'results.json').read_text(encoding='utf-8'))['elapsed_seconds'] if (args.output/'results.json').exists() else 0.
    else:
        args.output.mkdir(parents=True,exist_ok=False);(args.output/'cases').mkdir()
        save(args.output/'config.json',{'identity':identity,'command':[sys.executable,*sys.argv],
           'scope':'Reference diagnostics are not teacher-supervised validation. Oracle runs are never actual student results.'})
        prior=0.
    save(args.output/'results.json',{'status':'preparing','elapsed_seconds':prior,'completed_count':0,'local_test_read':False})
    records={int(p.stem):json.loads(p.read_text(encoding='utf-8')) for p in (args.output/'cases').glob('*.json')}
    if args.phase=='repair':pipeline=Pipeline(args.checkpoint,args.device,args.output/'worker.log');model=None
    else:
        state=torch.load(args.checkpoint,map_location='cpu',weights_only=False);model=Student().to(args.device)
        model.load_state_dict(state['model']);model.eval();pipeline=None
    status='running';warmed=False
    def summary():
        rows=list(records.values());out={'status':status,'completed_count':len(rows),'requested_count':len(reader),
          'elapsed_seconds':prior+time.perf_counter()-begin,'error_count':sum('error' in r for r in rows),
          'teacher_information_used':bool(args.oracle),'local_test_read':False}
        if args.phase=='repair' and rows:
            out.update(Q_mean=float(np.mean([r['final']['metrics']['cost'] for r in rows])),
                hard_legal_count=sum(r['final']['metrics']['is_feasible'] for r in rows),
                row_retained_count=sum(r.get('stats',{}).get('repair',{}).get('row_retained',False) for r in rows),
                total_seconds_mean=float(np.mean([r['total_seconds'] for r in rows])),
                total_seconds_p95=float(np.quantile([r['total_seconds'] for r in rows],.95)))
        elif rows:
            out['reference_center_local_rmse']=(sum(r.get('center_sq',0.) for r in rows)/max(1,sum(r.get('center_count',0) for r in rows)))**.5
            out['reference_logratio_rmse']=(sum(r.get('ratio_sq',0.) for r in rows)/max(1,sum(r.get('ratio_count',0) for r in rows)))**.5
            out['scope']='Full frozen dev2500 original-reference diagnostics; targets differ from teacher-supervised training.'
        return out
    try:
        for index in range(len(reader)):
            if index in records:continue
            if prior+time.perf_counter()-begin>=args.max_seconds or (args.output/'STOP').exists():status='stopped';break
            raw=reader.case(index);row={'case_id':raw.case_id,'index':index};start=time.perf_counter()
            try:
                if args.phase=='supervised':
                    g=prepare_inference(raw.solver_inputs,args.device)
                    with torch.no_grad():prediction=model(g,False)['geometry'][0].cpu().numpy()
                    p=prepare_problem(*raw.solver_inputs);ref=np.asarray(raw.reference_positions)
                    center=(prediction[:,:2]*p.scale-(ref[:,:2]+ref[:,2:]/2))/np.sqrt(p.areas)[:,None]
                    ratio=prediction[:,2]-np.log(ref[:,2]/ref[:,3]);free=~(p.fixed|p.preplaced);movable=~p.preplaced
                    row.update(center_sq=float((center[movable]**2).sum()),center_count=int(movable.sum()*2),
                      ratio_sq=float((ratio[free]**2).sum()),ratio_count=int(free.sum()),seconds=time.perf_counter()-start)
                else:
                    teacher=None;tr=teacher_records.get(raw.case_id)
                    if args.oracle:
                        if tr is None:raise ValueError('Missing oracle teacher member')
                        path=args.teacher_bank/tr['label_file']
                        if sha(path)!=tr['label_sha256']:raise ValueError('Changed oracle label')
                        with np.load(path,allow_pickle=False) as z:teacher={k:z[k] for k in ('geometry','positions')}
                    if not warmed:
                        warm_begin=time.perf_counter();pipeline.solve(raw.solver_inputs,args.oracle,teacher);warmed=True
                        row['warmup_seconds_excluded']=time.perf_counter()-warm_begin
                    final,stats=pipeline.solve(raw.solver_inputs,args.oracle,teacher)
                    case=_build_case(raw.case_id,raw.source,raw.solver_inputs[1:6],raw.reference_positions,raw.stored_metrics)
                    row.update(stats=stats,total_seconds=stats['end_to_end_seconds'],
                      raw=evaluate_case(case,stats['draft'],stats['inference_seconds']+stats['decode_seconds']),
                      final=evaluate_case(case,final,stats['end_to_end_seconds']))
                    if tr:row['teacher_Q']=tr['evaluation']['metrics']['cost']
            except Exception as exc:
                row.update(error=repr(exc),total_seconds=time.perf_counter()-start,
                  final={'metrics':{'is_feasible':False,'cost':10.}})
            records[index]=row;save(args.output/'cases'/f'{index:04d}.json',row);save(args.output/'results.json',summary())
            if (index+1)%25==0:print(args.phase,index+1,len(reader),flush=True)
        if len(records)==len(reader):status='completed'
    finally:
        if pipeline:pipeline.close()
        save(args.output/'results.json',summary())

if __name__=='__main__':main()
