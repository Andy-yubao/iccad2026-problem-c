"""Frozen own-Loss best: one-call primary; repeated calls are diagnostics only."""
import argparse,json,time,os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import multiprocessing as mp
import numpy as np
import torch
from .io import HERE,ROOT,save,sha,sources,exclusive
from .runtime import OUTPUT,CONFIG,seed,charge
from .data import load_rows,collate,inputs_only
from .model import GeometryModel
from .loss import per_case,configuration
from .metrics import module_overlap
from .diagnostics import geometry_checks
from scoring.evaluation import evaluate_case
from scoring.official import official_evaluator

STEADY=dict(relaxed=False,mode=0,adaptive=True,fast_budget=False,late_extension=True,
    plateau=True,robust_seeds=True,certified=True,quick_project=True,
    quadratic_project=True,incumbent_project=True,snap=1e-4)

def read(path):return json.loads(path.read_text(encoding='utf-8'))

def plain_positions(state,g,row,i):
    from .base_model import decode
    n=row['graph']['n'];p=(decode(state,g)[i,:n]*g['scale'][i]).cpu().double().numpy()
    c,s=np.asarray(row['case'].solver_inputs[5]),np.asarray(row['case'].solver_inputs[6]);locked=c[:,:2].any(1)
    p[locked,2:]=s[locked,2:];p[c[:,1]!=0]=s[c[:,1]!=0]
    return p

def record(state,g,draft,row,i,values):
    p=plain_positions(state,g,row,i);_,hard,geom=geometry_checks(p,row['case'].solver_inputs)
    return dict(positions=p.tolist(),loss={k:float(v[i]) for k,v in values.items()},
        diagnostics=module_overlap(p)|geom,raw_hard_legal=bool(hard['is_feasible']))

@torch.no_grad()
def raw(model,output,halt):
    rows=load_rows('dev');monitor=load_rows('dev_monitor500');monitor_ids={tuple(r['member']) for r in monitor}
    auxiliary={tuple(r['member']) for r in monitor[:100]};folder=output/'evaluation/raw';folder.mkdir(parents=True,exist_ok=True)
    model.eval()
    for start in range(0,len(rows),32):
        if halt():return None
        path=folder/f'{start:04d}.json'
        selected=rows[start:start+32]
        if path.exists():
            if [r['member'] for r in read(path)]!=[list(r['member']) for r in selected]:raise ValueError('Raw member drift')
            continue
        g=inputs_only(collate(selected,'cuda'));torch.cuda.synchronize();begin=time.perf_counter();draft=model.draft(g)
        torch.cuda.synchronize();placer_seconds=time.perf_counter()-begin
        begin=time.perf_counter();out=model.corrector(g,draft);torch.cuda.synchronize();f_seconds=time.perf_counter()-begin
        baseline=per_case(draft,g,draft);values=per_case(out,g,draft)
        chunk=[]
        for i,row in enumerate(selected):
            chunk.append(dict(index=start+i,member=list(row['member']),n=row['graph']['n'],
                monitor=tuple(row['member']) in monitor_ids,placer=record(draft,g,draft,row,i,baseline),
                corrected=record(out,g,draft,row,i,values),
                placer_seconds_amortized=placer_seconds/len(selected),F_seconds_amortized=f_seconds/len(selected)))
        aid=[i for i,r in enumerate(selected) if tuple(r['member']) in auxiliary]
        if aid:
            # Same F and no accept/reject loop. Compare all totals against the
            # ORIGINAL draft intent so loss values across calls are comparable.
            ag={k:v[aid] for k,v in g.items()};origin=draft[aid];current=out[aid];aux=[{} for _ in aid]
            elapsed=0.
            for call in range(2,9):
                begin=time.perf_counter();current=model.corrector(ag,current);torch.cuda.synchronize();elapsed+=time.perf_counter()-begin
                if call in (2,4,8):
                    av=per_case(current,ag,origin)
                    for j,i in enumerate(aid):
                        aux[j][str(call)]=record(current,ag,origin,selected[i],j,av)
                        aux[j][str(call)]['additional_F_seconds_amortized']=elapsed/len(aid)
            for j,i in enumerate(aid):chunk[i]['auxiliary']=aux[j]
        save(path,chunk)
        print('one-call raw',min(start+32,len(rows)),'/',len(rows),flush=True)
    return [r for start in range(0,len(rows),32) for r in read(folder/f'{start:04d}.json')]

def init_worker():
    seed(0,2)
    from .canonical_native import warmup
    begin=time.perf_counter();warmup()
    global worker_startup
    worker_startup=time.perf_counter()-begin

def solve(job):
    index,case,positions,network_seconds=job
    from .solver import optimize
    seed(index,2);begin=time.perf_counter()
    result,stats=optimize(np.asarray(positions),case.solver_inputs,**STEADY)
    seconds=time.perf_counter()-begin
    scored=evaluate_case(case,result,seconds+network_seconds)
    return dict(index=index,positions=result.tolist(),metrics=scored['metrics'],independent_hard_checks=scored['independent_hard_checks'],
        postprocess_seconds=seconds,network_seconds_amortized=network_seconds,
        solve_seconds_mixed_device_estimate=seconds+network_seconds,worker_startup_seconds=worker_startup,
        row_fallback=bool(stats['row_retained']),trace=stats.get('trace',[]),worker_pid=os.getpid())

def existing_source_terminal():
    from .initialization import parent
    base=ROOT/'experiments/E90_auxiliary_weight_placer/runs/20261031_aux3_seed0/terminal'
    config=read(base/'config.json')
    if config['checkpoint_sha256']!=parent()['sha256'] or config['steady']!=STEADY:
        raise ValueError('Existing E90 terminal identity/protocol mismatch')
    return base/'results.json'

def terminal(records,output,halt,identity_corrector=False):
    rows=load_rows('dev_monitor500');lookup={tuple(r['member']):r for r in records};folder=output/'evaluation/terminal';folder.mkdir(exist_ok=True)
    missing=[i for i in range(500) if not (folder/f'{i:04d}.json').exists()]
    if identity_corrector:
        prior_path=existing_source_terminal();old={tuple(r['member']):r for r in read(prior_path)['records']}
        for i,row in enumerate(rows):
            key=tuple(row['member']);r=lookup[key]
            if r['corrected']['positions']!=r['placer']['positions']:
                raise ValueError('Zero head did not preserve exact source geometry')
            value=dict(old[key]);value.update(index=i,terminal_result_reused=True,source=str(prior_path.relative_to(ROOT)),
                source_sha256=sha(prior_path),network_seconds_amortized=r['placer_seconds_amortized']+r['F_seconds_amortized'],
                source_draft_max_abs_difference=float(np.abs(np.asarray(old[key]['draft_positions'])-np.asarray(r['placer']['positions'])).max()),
                Q_scope='historical E90 reused for identity F; CPU vs GPU numerical path may differ; no newly measured F Q',
                time_scope='historical source solve; no claim of new timing')
            if i in missing:save(folder/f'{i:04d}.json',value)
        return [read(folder/f'{i:04d}.json') for i in range(500)]
    if missing:
        with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=init_worker) as pool:
            pending={};todo=iter(missing)
            def submit():
                if halt():return
                i=next(todo,None)
                if i is None:return
                r=lookup[tuple(rows[i]['member'])];seconds=r['placer_seconds_amortized']+r['F_seconds_amortized']
                pending[pool.submit(solve,(i,rows[i]['case'],r['corrected']['positions'],seconds))]=i
            for _ in range(4):submit()
            while pending:
                done,_=wait(pending,timeout=1,return_when=FIRST_COMPLETED)
                for future in done:
                    i=pending.pop(future);value=future.result();value['member']=list(rows[i]['member']);save(folder/f'{i:04d}.json',value)
                    submit()
    return [read(folder/f'{i:04d}.json') for i in range(500)] if all((folder/f'{i:04d}.json').exists() for i in range(500)) else None

def mean_dict(records,stage,key):
    names=records[0][stage][key]
    return {k:float(np.mean([r[stage][key][k] for r in records])) for k,v in names.items() if isinstance(v,(int,float))}

def summarize(records,terminal_records):
    summary=dict(count=len(records),primary_calls=1,test_read=False,
        placer=dict(loss=mean_dict(records,'placer','loss'),diagnostics=mean_dict(records,'placer','diagnostics')),
        corrected=dict(loss=mean_dict(records,'corrected','loss'),diagnostics=mean_dict(records,'corrected','diagnostics')),
        pairwise=dict(loss_improved=sum(r['corrected']['loss']['total']<r['placer']['loss']['total'] for r in records),
            overlap_improved=sum(r['corrected']['diagnostics']['overlap_area_fraction']<r['placer']['diagnostics']['overlap_area_fraction'] for r in records),
            overlap_worsened=sum(r['corrected']['diagnostics']['overlap_area_fraction']>r['placer']['diagnostics']['overlap_area_fraction'] for r in records),
            raw_hard_legal=sum(r['corrected']['raw_hard_legal'] for r in records)))
    aux=[r for r in records if 'auxiliary' in r];summary['auxiliary']={}
    for call in (2,4,8):
        losses=[r['auxiliary'][str(call)]['loss']['total'] for r in aux]
        summary['auxiliary'][str(call)]=dict(count=len(aux),mean_total=float(np.mean(losses)),
            worse_than_one=sum(v>r['corrected']['loss']['total'] for v,r in zip(losses,aux)),
            mean_N5=float(np.mean([r['auxiliary'][str(call)]['diagnostics']['N5'] for r in aux])),
            checkpoint_selection=False,postprocessed=False)
    if terminal_records:
        q=[r['metrics']['cost'] for r in terminal_records];n=[r['n'] for r in records if r['monitor']]
        # Use member keyed counts, never rely on full-dev/monitor order matching.
        nmap={tuple(r['member']):r['n'] for r in records};n=[nmap[tuple(r['member'])] for r in terminal_records]
        summary['terminal']=dict(count=500,legal=sum(r['metrics']['is_feasible'] for r in terminal_records),
            weighted_Q=official_evaluator().compute_total_score(q,n),mean_Q=float(np.mean(q)),max_Q=float(max(q)),
            time_scope='GPU batched inference amortized plus CPU postprocess; not directly comparable end-to-end latency',
            total_postprocess_worker_seconds=sum(r['postprocess_seconds'] for r in terminal_records))
        summary['terminal']['historical_E90_Q_reused']=all(r.get('terminal_result_reused',False) for r in terminal_records)
        if summary['terminal']['historical_E90_Q_reused']:
            summary['terminal']['Q_scope']='Historical E90 only; identity F, no new terminal measurement or timing claim'
        old_path=existing_source_terminal()
        old=read(old_path);old_records=old['records'];old_map={tuple(r['member']):r for r in old_records}
        oldq=[old_map[tuple(r['member'])]['metrics']['cost'] for r in terminal_records]
        diff=np.array(q)-oldq
        # Source shard is the resampling unit. This exposed dev panel is adaptive.
        families={}
        for r,d in zip(terminal_records,diff):families.setdefault(str(r['member'][0]).split('/')[0],[]).append(float(d))
        gen=np.random.default_rng(0);keys=list(families);boot=[]
        weights=np.exp((np.asarray(n)-max(n))/12)
        for _ in range(1000):
            picked=gen.choice(keys,len(keys),replace=True);ids=[i for key in picked for i,r in enumerate(terminal_records) if str(r['member'][0]).split('/')[0]==key]
            boot.append(float(np.average(diff[ids],weights=weights[ids])))
        summary['terminal']['vs_E90']=dict(source=str(old_path.relative_to(ROOT)),sha256=sha(old_path),
            mean_delta=float(diff.mean()),weighted_delta=float(np.average(diff,weights=weights)),
            worker_family_bootstrap_weighted_delta_95=np.percentile(boot,[2.5,97.5]).tolist(),
            evidence_scope='adaptive exposed development panel, seed0, different source training histories',old_results_reused=True)
    return summary

def run(output):
    output=output.resolve()
    if not output.is_relative_to(HERE/'runs'):raise ValueError('Outside E95/runs')
    begin=time.perf_counter();prior=read(output/'evaluation_budget.json')['charged_seconds'] if (output/'evaluation_budget.json').exists() else 0.
    halt=lambda:(HERE/'runs/STOP').exists() or (output/'STOP').exists() or prior+time.perf_counter()-begin>=3600
    try:
        with exclusive(HERE/'runs/evaluation.lock'):
            state=torch.load(output/'checkpoints/best.pt',map_location='cpu',weights_only=False)
            if state['identity']['sources']!=sources() or state['identity']['loss']!=configuration():raise ValueError('Frozen source/target drift')
            binding=dict(checkpoint_sha256=sha(output/'checkpoints/best.pt'),sources=sources(),primary_calls=1,
                auxiliary_calls=[2,4,8],auxiliary_count=100,steady=STEADY,workers=4,threads_per_worker=2,test_read=False)
            folder=output/'evaluation';folder.mkdir(exist_ok=True);frozen=folder/'frozen.json'
            if frozen.exists() and read(frozen)!=binding:raise ValueError('Evaluation binding changed')
            save(frozen,binding);seed();model=GeometryModel(**CONFIG).cuda().eval();model.load_state_dict(state['model'])
            identity_corrector=bool((model.corrector.head[-1].weight==0).all() and (model.corrector.head[-1].bias==0).all())
            records=raw(model,output,halt)
            if records is None:return False
            del model;torch.cuda.empty_cache()
            terminal_records=terminal(records,output,halt,identity_corrector)
            summary=summarize(records,terminal_records);save(folder/'summary.json',summary)
            report='''# E95 单次调用修正研究\n\n唯一own-Loss best；完整dev2000；测试封存。多步不参与选模。\n\n'''
            for key in ('placer','corrected'):
                v=summary[key];report+=f"- {key}: Loss {v['loss']['total']:.6f}；交面积 {v['diagnostics']['overlap_area_fraction']:.4%}；平均N10 {v['diagnostics']['N10']:.3f}；最坏模块覆盖均值 {v['diagnostics']['max_covered_fraction']:.4%}。\n"
            if 'terminal' in summary:
                v=summary['terminal'];report+=f"\n单次F的flash500：{v['legal']}/500合法，加权Q {v['weighted_Q']:.6f}。\n"
            report+='\n同100例四阶段图见gallery/index.html；辅助2/4/8次结果及同例退化计数见summary.json。\n\n以上仅为自适应开发证据，单候选和单seed不足以作架构/目标的因果归因。\n'
            (folder/'report.md').write_text(report,encoding='utf-8')
            if terminal_records is None:return False
            from .gallery import render
            render(output,records,terminal_records,halt)
            return not halt() and (folder/'gallery/index.html').exists()
    finally:charge(time.perf_counter()-begin,'evaluation',output)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT)
    print('Evaluation complete:',run(p.parse_args().output))
