"""Train-only reference-action coverage and resumable masked imitation scorer.

No training is launched by the development run. Teacher-forced states and the
nearest representable action are train supervision, never inference inputs.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
import torch
from torch import nn
from datasets.paths import dataset_paths
from datasets.cases import allowed_target_positions
from experiments.masked_construction.construction import Problem, candidate_pool, features


def scorer_model():
    return nn.Sequential(nn.Linear(6,32),nn.Tanh(),nn.Linear(32,1))


def load_scorer(path):
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    model=scorer_model()
    model.load_state_dict(checkpoint['model'])
    model.eval()
    def score(f):
        with torch.no_grad():
            return model(torch.tensor(f,dtype=torch.float32)).numpy().ravel()
    return score


def members(path):
    manifest=json.loads(path.read_text(encoding='utf-8'))
    split_path=ROOT/'protocol/split_v1.json'
    if manifest['split_sha256'] != hashlib.sha256(split_path.read_bytes()).hexdigest():
        raise ValueError('Shared manifest frozen split mismatch')
    split=json.loads(split_path.read_text(encoding='utf-8'))
    train=set(split['shards']['train'])
    result=[]
    for entry in manifest['selected_shards']:
        if entry['shard'] not in train:
            raise ValueError('Only frozen train members accepted')
        for row in entry['sample_indices']:
            if type(row) is not int or not 0 <= row < 112:
                raise ValueError('Invalid row')
            result.append((entry['shard'],row))
    if len(result)!=manifest['sample_count'] or len(set(result))!=len(result) or len(result)>20000:
        raise ValueError('Invalid member count/duplicate')
    return result


def load_train_case(shard,row):
    root=dataset_paths()['training_root']
    path=(root/shard).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Invalid train path')
    raw=torch.load(path,map_location='cpu',weights_only=False)
    blocks=raw[0][row]
    ref=raw[5][row][:,[2,3,0,1]].numpy()
    constraints=blocks[:,1:]
    inputs=(len(blocks),blocks[:,0],raw[1][row],raw[2][row],raw[3][row],constraints,
            allowed_target_positions(ref,constraints))
    return Problem(*inputs),ref


def actions(p,ref):
    state=dict(p.initial)
    for i in p.order:
        pool,mask=candidate_pool(p,i,state)
        valid=pool[mask]
        if not len(valid):
            raise RuntimeError('Teacher-forced reference prefix violates shelf invariant')
        # Distance is fixed before training: center L2/scale + log-shape L2.
        centers=valid[:,:2]+valid[:,2:]/2
        target_center=ref[i,:2]+ref[i,2:]/2
        center_error=np.linalg.norm(centers-target_center,axis=1)/p.scale
        shape_error=np.linalg.norm(np.log(valid[:,2:]/ref[i,2:]),axis=1)
        total_error=center_error+shape_error
        choice=int(np.argmin(total_error))
        yield features(p,i,valid,state),choice,dict(block=i, candidate_count=len(valid),
             normalized_center_error=float(center_error[choice]), log_shape_error=float(shape_error[choice]),
             representation_error=float(total_error[choice]), exact=bool(total_error[choice]<1e-5))
        state[i]=ref[i].copy()


def write(path,value):
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_checkpoint(path,value):
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        torch.save(value,temporary)
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


class BudgetLedger:
    """Charge attempts before updates; replay never refunds consumed resources.

    An unclean exit leaves an active wall-clock lease. Resume conservatively
    charges elapsed time through recovery (including downtime), capped at the
    original time limit. Normal SIGINT closes the lease after saving state.
    """
    def __init__(self,path,identity,max_seconds,prior_seconds=0.,prior_batches=0,resume=False):
        self.path,self.identity,self.max_seconds=path,identity,max_seconds
        self.prior_seconds,self.batches=prior_seconds,prior_batches
        if resume:
            if not path.is_file():
                raise ValueError('Resume requires budget ledger; old checkpoints have no reliable ledger')
            old=json.loads(path.read_text(encoding='utf-8'))
            if old['identity']!=identity:
                raise ValueError('Budget ledger identity mismatch')
            charged=old['seconds']
            if old['active']:
                charged=max(charged,old['session_base_seconds']+max(0.,time.time()-old['session_started_unix']))
            self.prior_seconds=max(prior_seconds,min(max_seconds,charged))
            self.batches=max(prior_batches,old['charged_batches'])
        self.started=time.perf_counter()
        self.started_unix=time.time()
        self.persist()

    def seconds(self):
        return self.prior_seconds+time.perf_counter()-self.started

    def persist(self,active=True):
        write(self.path,dict(identity=self.identity,seconds=self.seconds(),charged_batches=self.batches,
              active=active,session_base_seconds=self.prior_seconds,session_started_unix=self.started_unix,
              recovery_policy='active unclean lease charges wall time through resume, including downtime, capped at limit'))

    def charge_batch(self):
        self.batches+=1
        self.persist()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=['coverage','train'],required=True)
    ap.add_argument('--manifest',type=Path,default=ROOT/'experiments/graph_proposal/train_members.json')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--max-layouts',type=int,default=1)
    ap.add_argument('--member-offset',type=int,default=0,help='Flat shared manifest offset; diagnostic smoke uses 6516')
    ap.add_argument('--max-seconds',type=float,default=3600.)
    ap.add_argument('--max-batches',type=int,default=100000)
    ap.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--legacy-protocol',action='store_true',help='Historical reproduction only; new training must adopt workset_v2 and validation controls')
    args=ap.parse_args()
    if not args.legacy_protocol:
        ap.error('This historical trainer requires upgrade to workset_v2 and validation controls; --legacy-protocol is only for explicit historical reproduction.')
    if not 0 < args.max_seconds <= 3600 or args.max_batches<=0 or not 1<=args.max_layouts<=20000:
        ap.error('Budget: 1..20000 layouts, <=3600 seconds cumulative, positive batch cap')
    if args.mode=='coverage' and args.max_layouts>5:
        ap.error('Coverage diagnostics cap at five train layouts')
    if args.resume and args.mode!='train':
        ap.error('Only training checkpoints can resume')
    torch.set_num_threads(1)
    torch.manual_seed(0)
    if args.member_offset < 0:
        ap.error('Nonnegative member offset required')
    selected=members(args.manifest)[args.member_offset:args.member_offset+args.max_layouts]
    if not selected:
        ap.error('Member slice is empty')
    source_hash=hashlib.sha256(Path(__file__).read_bytes()+(Path(__file__).parent/'construction.py').read_bytes()).hexdigest()
    manifest_hash=hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    args.output.mkdir(parents=True,exist_ok=args.resume)
    config=dict(command=subprocess.list2cmdline([sys.executable,*sys.argv]),
                utc=datetime.now(timezone.utc).isoformat(), mode=args.mode, seed=0, torch=torch.__version__,
                python=sys.version, device=args.device, threads=1, max_seconds=args.max_seconds,
                max_batches=args.max_batches, members=selected, manifest_sha256=manifest_hash,
                source_sha256=source_hash, data_paths={k:str(v) for k,v in dataset_paths().items()},
                supervision='train-only teacher-forced prefix; nearest representable action; equal logits CE',
                budget_scope='cumulative across resume; resource stop is not convergence')
    invocation=len(list(args.output.glob('config*.json')))
    write(args.output/f'config_{invocation}.json',config)
    model=scorer_model().to(args.device)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    cursor=0; batch_in_layout=0; batches=0; prior_seconds=0.; diagnostics=[]; stop='member_limit'
    identity={key:config[key] for key in ('source_sha256','manifest_sha256','device','data_paths',
                                        'torch','python','threads','max_seconds','max_batches','mode')}
    identity['members_sha256']=hashlib.sha256(json.dumps(selected).encode()).hexdigest()
    checkpoint=args.output/'checkpoints'/'latest.pt'
    if args.mode=='train':
        checkpoint.parent.mkdir(exist_ok=True)
    if args.resume:
        saved=torch.load(checkpoint,map_location=args.device,weights_only=False)
        if saved.get('resume_identity')!=identity or saved['members']!=selected:
            raise ValueError('Resume requires identical source, members, device, data paths, Python/Torch and budgets; historical pre-ledger smoke checkpoint is intentionally incompatible')
        model.load_state_dict(saved['model']); opt.load_state_dict(saved['optimizer'])
        cursor=saved['cursor']; batch_in_layout=saved['batch_in_layout']; batches=saved['batches']
        prior_seconds=saved['seconds']; diagnostics=saved['diagnostics']
    budget=BudgetLedger(args.output/'budget_ledger.json',identity,args.max_seconds,prior_seconds,batches,args.resume)
    last_save=time.perf_counter()
    interrupted=False
    def request_stop(signum,frame):
        nonlocal interrupted
        interrupted=True
    old_sigint=signal.signal(signal.SIGINT,request_stop)
    def save():
        atomic_checkpoint(checkpoint,dict(model=model.state_dict(),optimizer=opt.state_dict(),cursor=cursor,
                        batch_in_layout=batch_in_layout,batches=budget.batches,seconds=budget.seconds(),
                        source_sha256=source_hash,manifest_sha256=manifest_hash,members=selected,
                        diagnostics=diagnostics,resume_identity=identity))
    try:
        if args.mode=='train':
            save()
        while cursor<len(selected):
            if interrupted:
                stop='user_interrupt'; break
            if budget.seconds()>=args.max_seconds or budget.batches>=args.max_batches:
                stop='resource_budget'; break
            shard,row=selected[cursor]
            p,ref=load_train_case(shard,row)
            completed=True
            for action_index,(f,target,diag) in enumerate(actions(p,ref)):
                if action_index<batch_in_layout:
                    continue
                if interrupted:
                    stop='user_interrupt'; completed=False; break
                if budget.seconds()>=args.max_seconds or budget.batches>=args.max_batches:
                    stop='resource_budget'; completed=False; break
                budget.charge_batch()
                if args.mode=='train':
                    logits=model(torch.tensor(f,dtype=torch.float32,device=args.device)).flatten()
                    loss=nn.functional.cross_entropy(logits[None,:],torch.tensor([target],device=args.device))
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite imitation loss')
                    opt.zero_grad(); loss.backward(); opt.step()
                    diag['loss']=float(loss.detach())
                if args.mode=='coverage':
                    diagnostics.append(dict(shard=shard,row=row,**diag))
                batch_in_layout=action_index+1
                budget.persist()
                if args.mode=='train' and time.perf_counter()-last_save>30:
                    save(); last_save=time.perf_counter()
            if not completed:
                break
            cursor+=1; batch_in_layout=0
            if args.mode=='train':
                save()
    finally:
        try:
            if args.mode=='train':
                save()
            budget.persist(active=False)
        finally:
            signal.signal(signal.SIGINT,old_sigint)
    elapsed=budget.seconds()
    errors=[d['representation_error'] for d in diagnostics]
    write(args.output/'results.json',dict(status='completed' if cursor==len(selected) else 'resource_stopped',
         stop_reason=stop,processed_layouts=cursor,batches=budget.batches,cumulative_seconds=elapsed,
         candidate_coverage_mean_error=float(np.mean(errors)) if errors else None,
         exact_action_fraction=sum(d['exact'] for d in diagnostics)/len(diagnostics) if diagnostics else None,
         action_diagnostics=diagnostics,trained_model=args.mode=='train',
         caveat='Teacher-forced coverage is optimistic relative to rollout; no dev reference used'))
    print((args.output/'results.json').read_text(encoding='utf-8'))


if __name__=='__main__':
    main()
