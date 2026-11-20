"""Frozen E139 versus E134 dev500 comparison; never optimize or repair layouts."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT))
from helpers import center_setup, center_metrics, intent, blockers, boundary, pair_motion, largest_empty, relation_metrics
RUN = HERE.parent
BASE = ROOT/'experiments/E134_supervision_interval/runs/e119_stride3_seed0'
PRIOR = BASE/'layer_analysis_v1'
STATISTICS = ROOT/'experiments/E136_frontend_F_linear/runs/statistics_dev500_v1'
MODELS = ('E134', 'E139')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)+'\n', encoding='utf-8')


def finite(value):
    return float(value) if np.isfinite(value) else None


def verify():
    plan = read(HERE/'plan.json')
    for name, digest in plan['protected_input_sha256'].items():
        assert sha(ROOT/name) == digest, name
    for owner in (BASE.parents[1], RUN.parents[1]):
        for name, digest in read(owner/'preflight.json')['sources'].items():
            assert sha(ROOT/name) == digest, name
    return plan


def capture():
    import torch
    from datasets.random_reference import RandomReference
    from experiments.E139_delayed_overlap.model import GeometryModel
    from experiments.E139_delayed_overlap.evaluation.run import public_graph
    from experiments.E139_delayed_overlap.batches import decode
    from experiments.E139_delayed_overlap.data import load_rows
    from experiments.E139_delayed_overlap.resident import ResidentRows
    from experiments.E139_delayed_overlap.loss import base_per_case
    verify()
    assert not (HERE/'capture_manifest.json').exists(), 'Preserve completed capture'
    start = time.perf_counter()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ds = RandomReference('dev_monitor500')
    manifest = read(PRIOR/'manifest.json')
    assert [ds.records[i]['member'] for i in ds.indices] == manifest['members']
    checkpoint = torch.load(RUN/'checkpoints/best.pt', map_location='cpu', weights_only=False)
    model = GeometryModel(**checkpoint['model_config']).cuda().eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    g = {k: v.cuda() for k,v in public_graph(ds[0]['solver_inputs']).items()}
    heads = []
    hook = model.head.register_forward_hook(lambda module, args, value: heads.append(value))
    def forward():
        heads.clear()
        output = model(g)
        assert len(heads) == 13
        return decode(output[-1], g)
    states = np.zeros((500,13,120,3), np.float32)
    positions = np.zeros((500,13,120,4), np.float64)
    ref = np.load(PRIOR/'reference.npy', mmap_mode='r')
    areas = np.load(PRIOR/'areas.npy', mmap_mode='r')
    flags = np.load(PRIOR/'flags.npy', mmap_mode='r')
    with torch.inference_mode():
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                forward()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            final = forward()
        for i in range(500):
            item = ds[i]
            old = read(RUN/f'evaluation_no_mib_dev500/cases/{i:04d}.json')
            assert old['member'] == manifest['members'][i] == item['member']
            n = int(item['solver_inputs'][0])
            np.testing.assert_array_equal(ref[i,:n], item['supervision']['reference_xywh'])
            np.testing.assert_array_equal(areas[i,:n], item['solver_inputs'][1])
            np.testing.assert_array_equal(flags[i,:n], item['solver_inputs'][5])
            for key,value in public_graph(item['solver_inputs']).items():
                g[key].copy_(value)
            graph.replay()
            actual = (final[0,:n]*g['scale'][0]).cpu().numpy().astype(float)
            np.testing.assert_array_equal(actual, np.asarray(old['draft_positions']))
            states[i] = torch.stack(heads)[:,0].cpu().numpy()
            for layer, state in enumerate(heads):
                positions[i,layer,:n] = (decode(state,g)[0,:n]*g['scale'][0]).cpu().numpy().astype(float)
            np.testing.assert_array_equal(positions[i,12,:n], actual)
            if (i+1)%50 == 0:
                print('Verified actual S0..S12',i+1,'/500',flush=True)
                save(HERE/'status.json',dict(stage='capture',completed=i+1))
        np.save(HERE/'states.npy', states, allow_pickle=False)
        np.save(HERE/'positions.npy', positions, allow_pickle=False)
        hook.remove()
        assert all(torch.equal(value.cpu(),checkpoint['model'][name]) for name,value in model.state_dict().items())
        del model,graph,g,heads,final
        torch.cuda.empty_cache()
        pack = ResidentRows(load_rows('dev_monitor500'),'cuda',max_nodes=120)
        names = manifest['loss_names']
        losses = np.zeros((500,13,len(names)), np.float64)
        for begin in range(0,500,32):
            ids = list(range(begin,min(begin+32,500)))
            batch = pack.batch(ids)
            raw = torch.as_tensor(states[ids],device='cuda')
            for layer in range(13):
                values = base_per_case(raw[:,layer],batch)
                losses[ids,layer] = torch.stack([values[name] for name in names],-1).cpu().double().numpy()
        np.save(HERE/'losses.npy',losses,allow_pickle=False)
    saved = read(RUN/'evaluation_no_mib_dev500/results.json')
    assert abs(losses[:,12,0].mean()-saved['common_final_new_dev_loss'])<1e-6
    verify()
    save(HERE/'capture_manifest.json',dict(members=manifest['members'],loss_names=names,
        checkpoint_sha256=sha(RUN/'checkpoints/best.pt'),baseline_manifest_sha256=sha(PRIOR/'manifest.json'),
        terminal_bitwise_equal=True,weights_unchanged=True,training=False,optimizer_updates=0,
        postprocessing_calls=0,test_read=False,seconds=time.perf_counter()-start))


def fast_geometry(p, inputs, reference, setup):
    from scoring.draft_geometry import contact_distances, minimum_tree, wirelength, pair_geometry
    from scoring.layout_geometry import measures
    n,areas,_,_,_,flags,_ = inputs
    lo,hi,signed,inter = pair_geometry(p)
    total = p[:,2:].prod(-1).sum()
    geo = measures(p)
    geo['pair_intersection'] = float(inter.sum()/2/total)
    geo['max_module_load'] = float((inter.sum(1)/p[:,2:].prod(-1)).max())
    ri,re = wirelength(reference,inputs[2],inputs[3],inputs[4])
    wi,we = wirelength(p,inputs[2],inputs[3],inputs[4])
    geo['hpwl_reference'] = (wi+we)/(ri+re)
    distances = contact_distances(p)[0]
    groups = []
    group_minimax = {}
    for group in sorted(set(flags[:,3])-{0}):
        members = np.flatnonzero(flags[:,3] == group)
        if len(members)<2:
            continue
        foreign = np.flatnonzero(flags[:,3] != group)
        sub = distances[np.ix_(members,members)]
        edges = minimum_tree(sub)
        costs = np.array([edge[2] for edge in edges])
        block = np.array([blockers(p,int(members[u]),int(members[v]),foreign,areas).any() for u,v,_ in edges])
        costs2 = sub.copy()
        np.fill_diagonal(costs2,0.)
        for k in range(len(members)):
            costs2 = np.minimum(costs2,np.maximum(costs2[:,k,None],costs2[None,k,:]))
        group_minimax[int(group)] = (members,costs2)
        groups.append(dict(group=int(group),members=members.tolist(),
            contact_bottleneck=float(costs.max()),contact_mean=float(costs.mean()),
            edges=[dict(i=int(members[u]),j=int(members[v]),contact=float(cost),blocked=bool(block[z])) for z,(u,v,cost) in enumerate(edges)],
            centerline_blocked_fraction=float(block.mean()),
            connected01=bool(costs.max()<=.01),connected02=bool(costs.max()<=.02),connected05=bool(costs.max()<=.05)))
    for metric in ('contact_bottleneck','contact_mean','centerline_blocked_fraction','connected01','connected02','connected05'):
        geo['group_'+metric] = float(np.mean([row[metric] for row in groups])) if groups else None
    edges = [edge for group in groups for edge in group['edges']]
    geo['group_edge_blocked_fraction'] = float(np.mean([edge['blocked'] for edge in edges])) if edges else None
    bounds = boundary(p, flags, areas)
    geo['boundary_gap_local'] = float(np.mean([row['gap'] for row in bounds])) if bounds else None
    geo['boundary_shadow_fraction'] = float(np.mean([row['shadow']>0 for row in bounds])) if bounds else None
    return geo,groups,distances,group_minimax,bounds


def cohort_metrics(trajectory, inputs, edges, distances, minimax):
    areas = inputs[1]
    flags = inputs[5]
    rows = []
    for edge in edges:
        i,j,group = edge['i'],edge['j'],edge['group']
        foreign = np.flatnonzero(flags[:,3] != group)
        blocked = np.array([blockers(p,i,j,foreign,areas).any() for p in trajectory])
        cost = np.array([d[i,j] for d in distances])
        path = []
        for maps in minimax:
            members,values = maps[group]
            path.append(values[int(np.flatnonzero(members==i)[0]),int(np.flatnonzero(members==j)[0])])
        threshold = edge['contact']/2
        good = ~blocked
        strict = good&(cost<=threshold)
        connected = good&(np.asarray(path)<=threshold)
        rows.append(dict(group=group,i=i,j=j,initial_contact=edge['contact'],
            unblocked=good.tolist(),strict=strict.tolist(),path_strict=connected.tolist(),
            contact=cost.tolist(),minimax_contact=[float(v) for v in path]))
    return rows


def initialize_worker():
    global DS,POS,REF,AREAS,FLAGS
    from datasets.random_reference import RandomReference
    DS = RandomReference('dev_monitor500')
    POS = {'E134':np.load(PRIOR/'positions.npy',mmap_mode='r'),'E139':np.load(HERE/'positions.npy',mmap_mode='r')}
    REF = np.load(PRIOR/'reference.npy',mmap_mode='r')
    AREAS = np.load(PRIOR/'areas.npy',mmap_mode='r')
    FLAGS = np.load(PRIOR/'flags.npy',mmap_mode='r')


def observe_case(index):
    from scoring.draft_geometry import evaluate_geometry
    from shapely.geometry import box
    from shapely.ops import unary_union
    path = HERE/f'cases/{index:04d}.json'
    if path.exists():
        assert read(path)['member'] == DS[index]['member']
        return index
    item = DS[index]
    inputs = tuple(np.asarray(v) if i else int(v) for i,v in enumerate(item['solver_inputs']))
    inputs = (*inputs[:5], inputs[5].astype(np.int64), inputs[6])
    n = inputs[0]
    areas = AREAS[index,:n]
    flags = FLAGS[index,:n]
    reference = REF[index,:n]
    setup = center_setup(reference[:,:2]+reference[:,2:]/2, areas, flags[:,1]!=0)
    refgap,_,_ = intent(reference,setup)
    allowed = refgap>=-.01
    result = dict(index=index,member=item['member'],n=n,models={},group_cohorts={})
    structures = {}
    for model in MODELS:
        trajectory = np.asarray(POS[model][index,:,:n])
        current = []
        distances = []
        minimax = []
        for layer,p in enumerate(trajectory):
            center = p[:,:2]+p[:,2:]/2
            top,_,_ = center_metrics(center,setup)
            _,choice,known = intent(p,setup)
            compatible = np.take_along_axis(allowed,choice[...,None],-1)[...,0]
            eligible = setup['local']&allowed.any(-1)
            top['intent_reference_correct_clear'] = float((compatible&known&eligible).sum()/max(eligible.sum(),1))
            geo,groups,d,m,bounds = fast_geometry(p,inputs,reference,setup)
            current.append(dict(layer=layer,topology={key:finite(v) for key,v in top.items()},geometry=geo,groups=groups,boundary=bounds))
            distances.append(d)
            minimax.append(m)
        if model=='E134':
            prior = read(PRIOR/f'cases/{index:04d}.json')
            for layer in range(13):
                for key in ('local_axis_accuracy','knn6_recall'):
                    assert abs(current[layer]['topology'][key]-prior['layers'][layer]['topology'][key])<1e-12
                assert abs(current[layer]['geometry']['group_contact_bottleneck']-prior['layers'][layer]['geometry']['group_contact_bottleneck'])<1e-10
        evaluation = read((BASE if model=='E134' else RUN)/f'evaluation_no_mib_dev500/cases/{index:04d}.json')
        np.testing.assert_array_equal(trajectory[12],np.asarray(evaluation['draft_positions']))
        motion,exchanges,cumulative = pair_motion(trajectory[3:],areas,flags)
        rel = relation_metrics(trajectory,reference,np.asarray(POS['E134'][index,3,:n]),areas,flags)
        voids = {str(layer):largest_empty(trajectory[layer]) for layer in (3,12)}
        for layer,void in voids.items():
            x,y,w,h = void['rectangle']
            p = trajectory[int(layer)]
            overlap = np.maximum(np.minimum(p[:,:2]+p[:,2:],[x+w,y+h])-np.maximum(p[:,:2],[x,y]),0).prod(-1).sum()
            assert overlap < 1e-7*max(void['area'],1.)
        final_groups = []
        for group in sorted(set(flags[:,3])-{0}):
            ids = np.flatnonzero(flags[:,3]==group)
            p = np.asarray(evaluation['positions'])[ids]
            union = unary_union([box(x,y,x+w,y+h) for x,y,w,h in p])
            count = len(union.geoms) if union.geom_type=='MultiPolygon' else 1
            final_groups.append(dict(group=int(group),size=len(ids),violations=count-1))
        assert sum(g['violations'] for g in final_groups) == evaluation['metrics']['grouping_violations']
        if model=='E134':
            detailed = read(STATISTICS/f'geometry/{index:04d}.json')['draft']['baseline']['geometry']
        else:
            detailed,_,_ = evaluate_geometry(trajectory[12],inputs,reference,details=True)
        result['models'][model] = dict(layers=current,motion=motion,exchanges=exchanges,cumulative=cumulative,
            relation=rel,voids=voids,detailed_S12=detailed,final_groups=final_groups,
            raw_metrics=evaluation['raw_metrics'],final_metrics=evaluation['metrics'],
            complete_seconds=evaluation['complete_seconds'],draft_seconds=evaluation['draft_seconds'],
            repair_seconds=evaluation['repair_seconds'])
        structures[model] = (trajectory,distances,minimax)
    for cohort_name in MODELS:
        groups = result['models'][cohort_name]['layers'][3]['groups']
        edges = [dict(edge,group=g['group']) for g in groups for edge in g['edges'] if edge['blocked'] and edge['contact']>.05]
        result['group_cohorts'][cohort_name+'_S3'] = dict(count=len(edges),edges=edges,
            models={model:cohort_metrics(structures[model][0],inputs,edges,structures[model][1],structures[model][2]) for model in MODELS})
    fixed_bounds = [r for r in result['models']['E134']['layers'][3]['boundary'] if r['gap']>.1 and r['shadow']>0]
    bound_rows = []
    for row in fixed_bounds:
        track = {}
        for model in MODELS:
            rows = [next(b for b in layer['boundary'] if (b['module'],b['bit'])==(row['module'],row['bit'])) for layer in result['models'][model]['layers']]
            track[model] = dict(gaps=[b['gap'] for b in rows],shadow=[b['shadow']>0 for b in rows])
        bound_rows.append(dict(module=row['module'],bit=row['bit'],models=track))
    result['fixed_boundary_cohort'] = bound_rows
    save(path,result)
    return index


def observe(workers):
    from concurrent.futures import ProcessPoolExecutor,as_completed
    verify()
    assert read(HERE/'capture_manifest.json')['terminal_bitwise_equal']
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize_worker) as pool:
        futures = [pool.submit(observe_case,i) for i in range(500)]
        for count,future in enumerate(as_completed(futures),1):
            future.result()
            if count%25==0:
                print('Paired structural audit',count,'/500',flush=True)
                save(HERE/'status.json',dict(stage='structure',completed=count))
    verify()
    save(HERE/'observations_complete.json',dict(count=500,paired=True,all_saved_final_group_counts_reproduced=True))


def analyze():
    from scipy.stats import spearmanr
    verify()
    rows = [read(HERE/f'cases/{i:04d}.json') for i in range(500)]
    members = read(HERE/'capture_manifest.json')['members']
    assert [r['member'] for r in rows] == members
    shards = sorted({m[0] for m in members})
    cluster = np.array([shards.index(m[0]) for m in members])
    rng = np.random.default_rng(20261115)
    counts = rng.multinomial(len(shards),np.full(len(shards),1/len(shards)),size=2000)
    bw = counts[:,cluster].astype(float)
    bw /= bw.sum(1,keepdims=True)
    n = np.array([r['n'] for r in rows])
    qw = np.exp((n-n.max())/12)
    qw /= qw.sum()
    mixedw = .5*(qw+1/500)
    boot_qw = counts[:,cluster]*np.exp((n-n.max())/12)
    boot_qw /= boot_qw.sum(1,keepdims=True)
    boot_mix = .5*(boot_qw+bw)

    def paired(a,b,aggregation='mean'):
        a,b = np.asarray(a,float),np.asarray(b,float)
        mask = np.isfinite(a)&np.isfinite(b)
        if not mask.any():
            raise ValueError('No paired finite observations; check metric schema')
        a,b = a[mask],b[mask]
        weights = mixedw[mask] if aggregation=='mixed_Q' else np.full(len(a),1/len(a))
        weights /= weights.sum()
        boot = boot_mix[:,mask] if aggregation=='mixed_Q' else bw[:,mask]
        boot = boot/boot.sum(1,keepdims=True)
        delta = b-a
        return dict(E134=float(weights@a),E139=float(weights@b),delta=float(weights@delta),
            delta95=np.quantile(boot@delta,[.025,.975]).tolist(),paired_cases=int(mask.sum()),
            lower=int((delta < -1e-9).sum()),higher=int((delta > 1e-9).sum()),tie=int((np.abs(delta)<=1e-9).sum()))

    def get(model, fn):
        return np.array([fn(row['models'][model]) for row in rows],float)

    result = dict(cases=500,source_shards=len(shards),score={},raw_metrics={},final_metrics={},
        terminal_geometry={},topology={},group_cohorts={},strong_relations={},boundary_cohort={},motion={})
    final_names = ('hpwl_gap','area_gap','hpwl_total','bbox_area','grouping_violations','boundary_violations',
        'violations_relative','total_soft_violations')
    for stage in ('raw_metrics','final_metrics'):
        for key in final_names:
            a = get('E134',lambda m:m[stage][key])
            b = get('E139',lambda m:m[stage][key])
            result[stage][key] = paired(a,b)
        for key in ('hpwl_gap','area_gap'):
            result[stage]['positive_'+key] = paired(
                np.maximum(0,get('E134',lambda m:m[stage][key])),np.maximum(0,get('E139',lambda m:m[stage][key])))
    q = {m:get(m,lambda x:x['final_metrics']['cost']) for m in MODELS}
    result['score']['mean_Q'] = paired(q['E134'],q['E139'])
    result['score']['mixed_Q'] = paired(q['E134'],q['E139'],'mixed_Q')
    for model in MODELS:
        saved = read((BASE if model=='E134' else RUN)/'evaluation_no_mib_dev500/results.json')
        assert abs(mixedw@q[model]-saved['mixed_Q'])<1e-12
    # Exact arithmetic attribution of saved scores; no evaluator/solver invocation.
    fields = ('hpwl','area','grouping','boundary')
    def vector(model):
        output = []
        for row in rows:
            m = row['models'][model]['final_metrics']
            den = max(m['max_possible_violations'],1)
            output.append([max(0,m['hpwl_gap']),max(0,m['area_gap']),m['grouping_violations']/den,m['boundary_violations']/den])
        return np.array(output)
    old,new = vector('E134'),vector('E139')
    def formula(x):
        return np.minimum((1+.5*(x[:,0]+x[:,1]))*np.exp(2*(x[:,2]+x[:,3])),9.999999)
    np.testing.assert_allclose(formula(old),q['E134'],rtol=0,atol=1e-12)
    np.testing.assert_allclose(formula(new),q['E139'],rtol=0,atol=1e-12)
    attribution = np.zeros_like(old)
    for permutation in itertools.permutations(range(4)):
        value = old.copy()
        before = formula(value)
        for j in permutation:
            value[:,j] = new[:,j]
            after = formula(value)
            attribution[:,j] += (after-before)/24
            before = after
    np.testing.assert_allclose(attribution.sum(1),q['E139']-q['E134'],atol=1e-12,rtol=0)
    result['score']['attribution'] = {name:dict(mean_Q=float(attribution[:,j].mean()),
        mixed_Q=float(mixedw@attribution[:,j]),mixed_Q95=np.quantile(boot_mix@attribution[:,j],[.025,.975]).tolist()) for j,name in enumerate(fields)}
    for key in ('complete_seconds','draft_seconds','repair_seconds'):
        result[key] = paired(get('E134',lambda m:m[key]),get('E139',lambda m:m[key]))
    from scoring.draft_geometry import repair_movement
    saved_evaluations = {}
    saved_movements = {}
    for model,run in (('E134',BASE),('E139',RUN)):
        saved_evaluations[model] = [read(run/f'evaluation_no_mib_dev500/cases/{i:04d}.json') for i in range(500)]
        saved_movements[model] = [repair_movement(row['draft_positions'],row['positions'],
            np.load(PRIOR/'areas.npy',mmap_mode='r')[i,:row['n']]) for i,row in enumerate(saved_evaluations[model])]
    result['postprocessor_work'] = {key:paired([r['work'][key] for r in saved_evaluations['E134']],
        [r['work'][key] for r in saved_evaluations['E139']]) for key in ('shape_calls','projection_calls','rescue_seeds','shape_native_iterations')}
    result['postprocessor_movement'] = {key:paired([r[key] for r in saved_movements['E134']],
        [r[key] for r in saved_movements['E139']]) for key in saved_movements['E134'][0]}
    for key in ('group_contact_bottleneck','group_contact_mean','group_group_bbox_over_area','group_foreign_bbox_fraction',
        'group_foreign_empty_bbox_fraction','group_tree_swept_foreign_load','group_tree_blocked_fraction',
        'group_connected_0.01','group_connected_0.02','group_connected_0.05','boundary_gap_local',
        'covered_module_max','covered_module_over10','whitespace_bbox','fixed_dimension_relative_mean','preplaced_position_local_mean'):
        result['terminal_geometry'][key] = paired(
            get('E134',lambda m:m['detailed_S12'].get(key,np.nan)),get('E139',lambda m:m['detailed_S12'].get(key,np.nan)))
    layer_rows = []
    for layer in range(13):
        item = dict(layer=layer,topology={},geometry={})
        for section,keys in [('topology',('local_axis_accuracy','global_axis_accuracy','knn4_recall','knn6_recall','knn8_recall','intent_reference_correct_clear')),
            ('geometry',('group_contact_bottleneck','group_contact_mean','group_edge_blocked_fraction','group_centerline_blocked_fraction',
                'group_connected01','group_connected02','group_connected05','boundary_gap_local','boundary_shadow_fraction',
                'repeated_coverage_fraction','bbox_over_area','hpwl_reference','pair_intersection','max_module_load'))]:
            for key in keys:
                item[section][key] = paired(get('E134',lambda m:m['layers'][layer][section][key]),get('E139',lambda m:m['layers'][layer][section][key]))
        layer_rows.append(item)
    result['layers'] = layer_rows
    names = read(HERE/'capture_manifest.json')['loss_names']
    losses = {'E134':np.load(PRIOR/'losses.npy'),'E139':np.load(HERE/'losses.npy')}
    result['common_loss'] = {name:paired(losses['E134'][:,12,j],losses['E139'][:,12,j]) for j,name in enumerate(names)}
    result['loss_trajectories'] = {m:losses[m].mean(0).tolist() for m in MODELS}
    result['loss_names'] = names
    for cohort_name in ('E134_S3','E139_S3'):
        counts = np.array([r['group_cohorts'][cohort_name]['count'] for r in rows])
        total = counts.sum()
        cohort = dict(edges=int(total),cases=int((counts>0).sum()),models={})
        for model in MODELS:
            records = [edge for row in rows for edge in row['group_cohorts'][cohort_name]['models'][model]]
            cohort['models'][model] = {}
            for key in ('unblocked','strict','path_strict'):
                flags = np.asarray([r[key] for r in records],bool)
                cohort['models'][model][key] = dict(S3=int(flags[:,3].sum()),S6=int(flags[:,6].sum()),S12=int(flags[:,12].sum()),
                    S12_fraction=float(flags[:,12].mean()),new_S3_to_S12=int((~flags[:,3]&flags[:,12]).sum()),lost_S3_to_S12=int((flags[:,3]&~flags[:,12]).sum()),
                    persistent_repair_counts_by_layer=np.logical_and.accumulate(flags[:,3:][:,::-1],axis=1)[:,::-1].sum(0).tolist())
        result['group_cohorts'][cohort_name] = cohort
    assert result['group_cohorts']['E134_S3']['edges']==1170, 'Baseline audit definition drift'
    for model in MODELS:
        records = [row['models'][model]['relation'] for row in rows]
        result['strong_relations'][model] = {}
        for cohort_name in ('fixed_baseline_S3_wrong','own_S3_wrong'):
            den = sum(r[cohort_name]['count'] for r in records)
            result['strong_relations'][model][cohort_name] = dict(count=den,cases=sum(r[cohort_name]['count']>0 for r in records),
                strong_repaired=[sum(r[cohort_name]['strong_repaired'][k] for r in records) for k in range(13)],
                direction_repaired=[sum(r[cohort_name]['direction_repaired'][k] for r in records) for k in range(13)])
        exchanges = [r['models'][model]['exchanges'] for r in rows]
        result['motion'][model] = dict(full_side_exchanges=sum(len(e) for e in exchanges),cases_with_full_exchange=sum(bool(e) for e in exchanges),
            arrivals={str(k):sum(e['layer']==k for row in exchanges for e in row) for k in range(4,13)},
            local_order_reversals_by_layer=[sum(r['models'][model]['motion'][k]['local_clear_order_reversals'] for r in rows) for k in range(9)],
            modules_with_order_change_by_layer=[float(np.mean([r['models'][model]['motion'][k]['modules_with_order_change'] for r in rows])) for k in range(9)])
    assert result['motion']['E134']['full_side_exchanges']==2, 'Baseline side-exchange drift'
    bounds = [row for r in rows for row in r['fixed_boundary_cohort']]
    assert len(bounds)==636
    result['boundary_cohort']['edges'] = len(bounds)
    for model in MODELS:
        result['boundary_cohort'][model] = dict(still_shadowed_S12=sum(r['models'][model]['shadow'][12] for r in bounds),
            mean_gap_S12=float(np.mean([r['models'][model]['gaps'][12] for r in bounds])),
            clear_near={str(eps):sum(not r['models'][model]['shadow'][12] and r['models'][model]['gaps'][12]<=eps for r in bounds) for eps in (.01,.02,.05)})
    result['largest_empty'] = {}
    for layer in (3,12):
        result['largest_empty'][f'S{layer}'] = paired(get('E134',lambda m:m['voids'][str(layer)]['bbox_fraction']),get('E139',lambda m:m['voids'][str(layer)]['bbox_fraction']))
    # Group improvements are paired by public group ID, with official connected-component counts reproduced.
    changes = []
    for row in rows:
        a = {g['group']:g for g in row['models']['E134']['final_groups']}
        b = {g['group']:g for g in row['models']['E139']['final_groups']}
        assert a.keys()==b.keys()
        for key in a:
            changes.append(dict(index=row['index'],group=key,size=a[key]['size'],E134=a[key]['violations'],E139=b[key]['violations']))
    result['final_group_changes'] = dict(groups=len(changes),improved=sum(g['E139']<g['E134'] for g in changes),worsened=sum(g['E139']>g['E134'] for g in changes),
        unchanged=sum(g['E139']==g['E134'] for g in changes),E134_total=sum(g['E134'] for g in changes),E139_total=sum(g['E139'] for g in changes))
    save(HERE/'per_group_changes.json',changes)
    dq = q['E139']-q['E134']
    delta_group = get('E139',lambda m:m['final_metrics']['grouping_violations'])-get('E134',lambda m:m['final_metrics']['grouping_violations'])
    result['Q_by_grouping_change'] = {}
    for label,mask in [('better',delta_group<0),('same',delta_group==0),('worse',delta_group>0)]:
        result['Q_by_grouping_change'][label] = dict(cases=int(mask.sum()),mean_Q_delta=float(dq[mask].mean()),
            sum_mixed_Q_delta=float(mixedw[mask]@dq[mask]),mixed_grouping_attribution=float(mixedw[mask]@attribution[mask,2]))
    examples = []
    for index in np.r_[np.argsort(mixedw*dq)[:10],np.argsort(mixedw*dq)[-10:][::-1]]:
        row = rows[int(index)]
        examples.append(dict(index=int(index),member=row['member'],n=row['n'],mean_Q_delta=float(dq[index]),mixed_Q_contribution=float(mixedw[index]*dq[index]),
            group_delta=int(delta_group[index]),components={name:float(attribution[index,j]) for j,name in enumerate(fields)},
            E134_Q=float(q['E134'][index]),E139_Q=float(q['E139'][index])))
    result['largest_mixed_Q_contributors'] = examples
    verification = dict(status='passed',cases=500,source_shards=len(shards),weights_unchanged=True,frozen_sources_unchanged=True,
        actual_E139_S12_bitwise_equal=True,baseline_topology_and_group_bottlenecks_reproduced=True,
        original_baseline_1170_group_barriers_636_boundary_barriers_2_exchanges_reproduced=True,
        final_group_counts_reproduced=True,score_attribution_sum_exact=True,empty_rectangles_unoccupied=True,
        training=False,optimizer_updates=0,postprocessing_calls=0,new_layout_score_calls=0,test_read=False)
    result['verification'] = verification
    save(HERE/'summary.json',result)
    save(HERE/'verification.json',verification)
    save(HERE/'artifact_manifest.json',dict(plan_sha256=sha(HERE/'plan.json'),
        analysis_code_sha256=sha(Path(__file__)),helpers_sha256=sha(HERE/'helpers.py'),
        inputs=read(HERE/'plan.json')['protected_input_sha256'],
        evaluation_case_ledgers={model:hashlib.sha256(''.join(sha((BASE if model=='E134' else RUN)/f'evaluation_no_mib_dev500/cases/{i:04d}.json') for i in range(500)).encode()).hexdigest() for model in MODELS},
        artifacts={name:sha(HERE/name) for name in ('states.npy','positions.npy','losses.npy','summary.json','verification.json')}))
    with (HERE/'paired_metrics.csv').open('w',newline='',encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(['dev_index','source','member_index','model','n','Q','grouping_violations','boundary_violations','hpwl_gap','area_gap','complete_ms','S12_group_contact_bottleneck','S12_group_tree_blocked_fraction','S12_local_axis_accuracy','S12_knn6_recall'])
        for row in rows:
            for model in MODELS:
                r = row['models'][model]
                f = r['final_metrics']
                writer.writerow([row['index'],*row['member'],model,row['n'],f['cost'],f['grouping_violations'],f['boundary_violations'],f['hpwl_gap'],f['area_gap'],r['complete_seconds']*1000,r['detailed_S12']['group_contact_bottleneck'],r['detailed_S12']['group_tree_blocked_fraction'],r['layers'][12]['topology']['local_axis_accuracy'],r['layers'][12]['topology']['knn6_recall']])
    save(HERE/'status.json',dict(status='complete',stage='analysis',cases=500))
    print(json.dumps(dict(score=result['score'],final_group_changes=result['final_group_changes'],group_cohorts=result['group_cohorts'],motion=result['motion']),ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase',choices=('capture','observe','analyze'))
    parser.add_argument('--workers',type=int,default=4)
    args = parser.parse_args()
    if args.phase=='capture':
        capture()
    elif args.phase=='observe':
        observe(args.workers)
    else:
        analyze()
