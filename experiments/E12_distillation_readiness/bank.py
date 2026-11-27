"""Resumable train-only teacher bank, or dev_monitor evaluation labels.

STOP file stops after the current case. Resume preserves the cumulative budget.
The pilot is one preselected train_monitor case per worker; no result selection.
"""
import argparse,json,os,random,sys,time,uuid,zipfile
from pathlib import Path
from .io import ROOT,HERE,np,sha,save
from .teacher_source import Teacher,CHECKPOINT_SHA256
from .targets import SCHEMA,encode,validate
from .backend.frozen.packing import prepare_problem
from .backend.frozen.solver import hard_valid
from datasets.worksets import LayoutReader,workset_manifest
from datasets.cases import _build_case
from scoring.evaluation import evaluate_case

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--split',choices=['train','dev_monitor'],default='train')
    ap.add_argument('--pilot',action='store_true');ap.add_argument('--max-seconds',type=float,required=True)
    ap.add_argument('--resume',action='store_true');ap.add_argument('--device',default='cpu')
    args=ap.parse_args();begin=time.perf_counter()
    if args.max_seconds<=0:raise ValueError('Positive cumulative budget required')
    if args.pilot and args.split!='train':raise ValueError('Pilot is train-only')
    import torch
    torch.set_num_threads(2);torch.manual_seed(0);random.seed(0);np.random.seed(0)
    torch.use_deterministic_algorithms(True)
    reader=LayoutReader(workset_manifest(),'train_monitor' if args.pilot else args.split)
    indices=list(range(len(reader)))
    if args.pilot:
        seen=set();indices=[]
        for i,(shard,row) in enumerate(reader.members):
            worker=shard.split('/')[0]
            if worker not in seen:indices.append(i);seen.add(worker)
        assert len(indices)==100
    sources=sorted(HERE.glob('*.py'))+sorted((HERE/'backend').rglob('*.py'))
    identity={'schema':SCHEMA,'split':args.split,'pilot':args.pilot,'device':args.device,'seed':0,'threads':2,
      'members':[list(reader.members[i]) for i in indices],
      'checkpoint_sha256':CHECKPOINT_SHA256,'split_sha256':sha(ROOT/'protocol/split_v1.json'),
      'workset_sha256':sha(ROOT/'protocol/workset_v2.json'),
      'sources':{p.relative_to(HERE).as_posix():sha(p) for p in sources},'max_seconds':args.max_seconds,
      'training':False,'local_test_read':False,'reference_free_geometry_in_solver':False}
    if args.resume:
        old=json.loads((args.output/'config.json').read_text(encoding='utf-8'))
        if old['identity']!=identity:raise ValueError('Resume identity/budget mismatch; preserve original bank')
        state=json.loads((args.output/'results.json').read_text(encoding='utf-8'));elapsed=state['elapsed_seconds']
    else:
        args.output.mkdir(parents=True,exist_ok=False);(args.output/'labels').mkdir()
        save(args.output/'config.json',{'identity':identity,'command':[sys.executable,*sys.argv]})
        with zipfile.ZipFile(args.output/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
            for p in sources:z.write(p,p.relative_to(HERE).as_posix())
        state={'records':[],'elapsed_seconds':0.};elapsed=0.
        save(args.output/'results.json',state)
    done={r['index']:r for r in state['records']};status='running';teacher=None
    # Fail closed on missing or changed completed labels before resuming.
    for r in done.values():
        if r.get('label_file') and sha(args.output/r['label_file'])!=r['label_sha256']:raise ValueError('Saved label changed')
    for index in indices:
        if index in done:continue
        if elapsed+time.perf_counter()-begin>=args.max_seconds or (args.output/'STOP').exists():status='stopped';break
        if teacher is None:teacher=Teacher(args.device)
        raw=reader.case(index);p=prepare_problem(*raw.solver_inputs);start=time.perf_counter()
        record={'index':index,'case_id':raw.case_id,'member':list(reader.members[index])}
        try:
            x,stats=teacher.solve(raw.solver_inputs);assert hard_valid(x,p)
            case=_build_case(raw.case_id,raw.source,raw.solver_inputs[1:6],raw.reference_positions,raw.stored_metrics)
            result=evaluate_case(case,x,stats['end_to_end_seconds']);result.pop('positions')
            target=encode(x,p);validate(target,p)
            if not result['metrics']['is_feasible']:raise ValueError('Official hard guard rejected teacher')
            name=f'labels/{index:05d}.npz';path=args.output/name;temporary=path.with_suffix('.'+uuid.uuid4().hex+'.tmp')
            try:
                with temporary.open('wb') as f:np.savez_compressed(f,**target)
                os.replace(temporary,path)
            finally:
                if temporary.exists():temporary.unlink()
            record.update(label_file=name,label_sha256=sha(path),teacher_stats=stats,evaluation=result)
        except Exception as exc:record['error']=repr(exc)
        record['elapsed_seconds']=time.perf_counter()-start;done[index]=record
        state={'status':status,'elapsed_seconds':elapsed+time.perf_counter()-begin,'records':list(done.values()),
          'requested_count':len(indices),'completed_count':len(done),'label_count':sum('label_file' in r for r in done.values()),
          'error_count':sum('error' in r for r in done.values()),'schema':SCHEMA}
        save(args.output/'results.json',state)
        print(f'{len(done)}/{len(indices)}',raw.case_id,record.get('error','saved'),flush=True)
    if len(done)==len(indices):status='completed' if not any('error' in r for r in done.values()) else 'completed_with_errors'
    state.update(status=status,elapsed_seconds=elapsed+time.perf_counter()-begin);save(args.output/'results.json',state)

if __name__=='__main__':main()
