"""Four requested stages from saved coordinates; never invokes a solver."""
import argparse,datetime,hashlib,json
from pathlib import Path
import numpy as np
from visualization.layouts import plot_case,geometry_details

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
REVISION='E99.raw_previous_best_current_final.v1'


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def save(p,x):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');temp.replace(p)


def planner_signature(options):
    # Defaults and explicit options are equivalent for this comparison.
    import inspect
    from .planner.solver import optimize
    sig={k:v.default for k,v in inspect.signature(optimize).parameters.items() if k not in ('draft','inputs')}
    sig.update(options);return sig


def run_time(out):
    path=out/'registration.json' if (out/'registration.json').exists() else out/'config.json'
    raw=read(path)
    return datetime.datetime.fromisoformat(raw['created_utc'])


def select_previous(out,settings):
    current=read(out/'results.json');members=[r['member'] for r in current['records']]
    source=[r['source_sha256'] for r in current['records']]
    binding=read(out/'binding.json')['identity'];options=planner_signature(binding['planner'])
    cutoff=run_time(out);candidates=[]
    requested=settings.get('comparison',{}).get('previous_best_run','auto')
    known=[HERE/'runs/20261102_area_mass_v2_dev100',HERE/'runs/20261102_deform_v3_dev100']
    known+=sorted((HERE/'runs/tuning').glob('*'))
    if requested!='auto':
        chosen=Path(requested)
        if not chosen.is_absolute():chosen=ROOT/chosen
        known=[chosen.resolve()]
    prefix='experiments/E99_attraction_repulsion/planner/'
    current_planner={k:v for k,v in binding['source_sha256'].items() if k.startswith(prefix) and k.endswith('.py')}
    for candidate in known:
        if candidate.resolve()==out.resolve() or not (candidate/'results.json').is_file():continue
        if not candidate.resolve().is_relative_to(HERE/'runs'):raise ValueError('Previous reference must be inside this experiment/runs')
        r=read(candidate/'results.json')
        if not r.get('finished') or r.get('count')!=100:continue
        ended=datetime.datetime.fromtimestamp((candidate/'results.json').stat().st_mtime,datetime.timezone.utc)
        if requested=='auto' and ended>=cutoff:continue
        if [x['member'] for x in r['records']]!=members or [x['source_sha256'] for x in r['records']]!=source:continue
        if (candidate/'binding.json').exists():
            b=read(candidate/'binding.json')['identity'];old_sources=b['source_sha256']
            if planner_signature(b['planner'])!=options:continue
            score=r['hybrid']['weighted_Q'];legal=r['hybrid']['hard_legal'];key='relaxed'
            label=read(candidate/'settings.json')['label']
        else:
            b=read(candidate/'config.json');old_sources=b['source_sha256']
            config_path='experiments/E99_attraction_repulsion/planner_config.json'
            if old_sources.get(config_path)!=sha(HERE/'planner_config.json') or planner_signature(read(HERE/'planner_config.json'))!=options:continue
            if 'arms' in r:
                score=r['arms']['deformation']['weighted_Q'];legal=r['arms']['deformation']['hard_legal'];key='deformation_draft';label='V3 / 10 steps'
            else:
                score=r['mass_weighted_Q'];legal=r['final_hard_legal'];key='relaxed';label='V2 / 4 steps'
        previous_planner={k:v for k,v in old_sources.items() if k.startswith(prefix) and k.endswith('.py')}
        if previous_planner!=current_planner or legal!=100:continue
        candidates.append(dict(run=candidate.relative_to(ROOT).as_posix(),coordinate_key=key,
            weighted_Q=float(score),label=label,result_sha256=sha(candidate/'results.json')))
    if not candidates:
        raise ValueError('No completed prior AR run matches this panel/draft/planner. Set comparison.previous_best_run to an eligible saved run.')
    best=min(candidates,key=lambda x:(x['weighted_Q'],x['run']))
    return best,candidates


def render_four(out,settings,records,halt=lambda:False):
    out=Path(out).resolve();settings=dict(settings)
    previous,candidates=select_previous(out,settings);old=ROOT/previous['run']
    pairs=[]
    for r in records:
        index=r['index'];current=out/'coordinates'/f'{index:04d}.npz';prior=old/'coordinates'/f'{index:04d}.npz'
        if sha(current)!=r['coordinates_sha256']:raise ValueError('Current coordinate binding changed')
        with np.load(current,allow_pickle=False) as a,np.load(prior,allow_pickle=False) as b:
            np.testing.assert_array_equal(a['positions'],b['positions'])
            np.testing.assert_array_equal(a['constraints'],b['constraints'])
            if b[previous['coordinate_key']].shape!=a['positions'].shape:raise ValueError('Previous module IDs/count mismatch')
        pairs.append(dict(index=index,current_sha256=sha(current),previous_sha256=sha(prior)))
    binding=dict(revision=REVISION,previous=previous,candidates=candidates,pairs=pairs,
        renderer_sha256=sha(Path(__file__)),shared_renderer_sha256=sha(ROOT/'visualization/layouts.py'),
        stages=['raw_draft','previous_best_AR_draft','current_AR_draft','current_full_final'],
        best_rule='Lowest final weighted neutral-runtime Q on the same fixed100 among completed prior compatible AR runs; whole version, no per-case mixing',
        no_solve=True,no_training=True,common_IDs_and_scale=True,score_controls=False)
    key=digest(binding)[:12];folder=out/'gallery'/('four_stage_'+key);folder.mkdir(parents=True,exist_ok=True)
    config_path=folder/'binding.json'
    if config_path.exists():
        if read(config_path)!=binding:raise ValueError('View identity mismatch')
    else:save(config_path,binding)
    settings_snapshot=dict(label=settings['label'],relaxation=settings['relaxation'],previous_reference=previous)
    snapshot=folder/'view_settings.json'
    if not snapshot.exists():save(snapshot,settings_snapshot)
    print('PREVIOUS BEST',previous['label'],previous['weighted_Q'],flush=True)
    assets=[]
    stages=(('raw','Raw draft'),('previous','Previous best AR / '+previous['label']),
            ('current',f"Current AR / {settings['relaxation']['iterations']} steps"),('final','Current AR + final planner'))
    for r,pair in zip(records,pairs):
        if halt():return False
        index=r['index'];path=folder/f'{index:04d}.png';sidecar=path.with_suffix('.json')
        if sidecar.exists():
            item=read(sidecar)
            if path.exists() and sha(path)==item['sha256']:assets.append(item);continue
            if path.exists():raise ValueError('Existing four-stage image changed')
        with np.load(out/'coordinates'/f'{index:04d}.npz',allow_pickle=False) as a,np.load(old/'coordinates'/f'{index:04d}.npz',allow_pickle=False) as b:
            c=a['constraints'].tolist()
            positions=dict(raw=a['positions'],previous=b[previous['coordinate_key']],current=a['relaxed'],final=a['hybrid_final'])
            variants={k:dict(positions=v.tolist(),soft_details=geometry_details(v,c)) for k,v in positions.items()}
        plot_case(dict(case_id=f'dev500:{index}',panel='E99 raw / previous best AR / current AR / current final',constraints=c,variants=variants),path,stages=stages,show_metrics=False)
        item=dict(pair,image=path.name,sha256=sha(path));save(sidecar,item);assets.append(item)
        if index%25==0:print('FOUR STAGES',index+1,'/100',flush=True)
    save(folder/'manifest.json',dict(count=100,assets=assets,**{k:binding[k] for k in ('revision','previous','stages','no_solve','common_IDs_and_scale','score_controls')}))
    html='<!doctype html><meta charset="utf-8"><title>原稿／此前最优／当前／最终</title><style>body{font:16px system-ui}img{width:100%}</style><h1>原始草稿 → 此前最优吸引排斥草稿 → 当前吸引排斥草稿 → 当前完整流程最终布局</h1>'
    html+=f'<p>此前最优：{previous["label"]}；同100例最终加权Q={previous["weighted_Q"]:.9f}。当前：{settings["label"]}。第二列全100例均来自同一旧版本。</p>'
    html+=''.join(f'<p>dev500:{i}</p><img loading="lazy" src="{i:04d}.png">' for i in range(100))
    (folder/'index.html').write_text(html,encoding='utf-8')
    pointer=dict(revision=REVISION,gallery=(folder/'index.html').relative_to(ROOT).as_posix(),
        manifest=(folder/'manifest.json').relative_to(ROOT).as_posix(),previous=previous,count=100,no_solve=True)
    save(out/'gallery/PRIMARY.json',pointer)
    root_index=out/'gallery/index.html';archive=out/'gallery/index_before_four_stage.html'
    if root_index.exists() and not archive.exists():archive.write_bytes(root_index.read_bytes())
    landing='<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0; url='+folder.name+'/index.html"><a href="'+folder.name+'/index.html">原稿／此前最优／当前／最终：100例</a>'
    root_index.write_text(landing,encoding='utf-8')
    latest=HERE/'runs/tuning/LATEST.json'
    if latest.exists():
        current=read(latest)
        if (ROOT/current['run']).resolve()==out:
            current['gallery']=pointer['gallery'];current['display_revision']=REVISION;save(latest,current)
    print('FOUR-STAGE GALLERY',folder/'index.html',flush=True)
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path)
    args=parser.parse_args();out=args.run
    if out is None:out=ROOT/read(HERE/'runs/tuning/LATEST.json')['run']
    elif not out.is_absolute():out=ROOT/out
    result=read(out/'results.json')
    if not result.get('finished') or result.get('count')!=100:raise ValueError('A completed fixed100 numerical run is required')
    settings=read(out/'settings.json')
    # Current display preference can override only the visual prior reference.
    live=read(HERE/'tuning.json')
    settings['comparison']={**settings.get('comparison',{}),'previous_best_run':live.get('comparison',{}).get('previous_best_run','auto')}
    render_four(out,settings,result['records'],lambda:(out/'STOP').exists())


if __name__=='__main__':main()
