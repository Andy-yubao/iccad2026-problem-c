"""Quantify reviewed drafts, with conservative low-cost whitespace witnesses.

Diagnostics only. No training, learned-model inference, production postprocessing,
reference layouts, or sealed members. Algorithms stay within the E42 experiment.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['MPLBACKEND'] = 'Agg'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REVIEW = HERE/'runs/20261024_visual_review100_seed20261024'
OUTPUT = HERE/'runs/20261024_feedback_geometry23'
BUDGETS = (0., .02, .05, .10, .20)
THRESHOLDS = (.02, .05, .10, .20)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    os.replace(tmp, path)


def geometry(p):
    from shapely.geometry import box
    from shapely.ops import unary_union
    p = np.asarray(p, dtype=float)
    lo, hi = p[:, :2], p[:, :2]+p[:, 2:]
    intersection = np.maximum(0., np.minimum(hi[:, None], hi[None])-np.maximum(lo[:, None], lo[None])).prod(-1)
    np.fill_diagonal(intersection, 0.)
    area = p[:, 2:].prod(1)
    bbox = float(np.prod(hi.max(0)-lo.min(0)))
    union = float(unary_union([box(x,y,x+w,y+h) for x,y,w,h in p]).area)
    return {'area': area, 'intersection': intersection, 'bbox': bbox, 'union': union,
            'empty': bbox-union, 'empty_fraction': 1.-union/bbox}


def components(graph):
    seen = set(); result = []
    for i in range(len(graph)):
        if i in seen:
            continue
        todo = [i]; seen.add(i); group = []
        while todo:
            j = todo.pop(); group.append(j)
            for k in np.flatnonzero(graph[j]):
                k = int(k)
                if k not in seen:
                    seen.add(k); todo.append(k)
        result.append(group)
    return result


def pileup(p):
    g = geometry(p); n = len(p); a = g['area']; inter = g['intersection']
    load = inter/a[:, None]
    # Both endpoints must lose at least threshold of their own area.
    mutual = inter/np.maximum(a[:, None], a[None])
    curves = []
    for threshold in THRESHOLDS:
        graph = mutual >= threshold
        clusters = [ids for ids in components(graph) if len(ids) >= 3]
        records = []
        for ids in clusters:
            ids = np.asarray(ids)
            severity = load[np.ix_(ids, ids)].sum(1)
            records.append({'ids': ids.tolist(), 'blocks': len(ids), 'block_fraction': len(ids)/n,
                            'area_fraction': float(a[ids].sum()/a.sum()),
                            'mean_coverage_load': float(severity.mean()),
                            'max_coverage_load': float(severity.max()),
                            'energy': float(np.square(severity).sum()/n),
                            'strong_edges': int(graph[np.ix_(ids,ids)].sum()//2)})
        worst = max(records, key=lambda r:r['energy']) if records else None
        curves.append({'threshold': threshold, 'worst': worst,
                       'complex_energy': worst['energy'] if worst else 0.,
                       'largest_strong_component': max([r['blocks'] for r in records], default=0),
                       'all_complex_energy': sum(r['energy'] for r in records)})
    weak = inter > max(1e-10, a.sum()*1e-12)
    plain = components(weak)
    sorted_load = np.sort(load.sum(1))
    top = max(1, int(np.ceil(.1*n)))
    return {'n': n, 'empty_fraction': g['empty_fraction'],
            'pair_overlap_fraction': float(inter.sum()/2/a.sum()),
            'overlap_excess_fraction': float((a.sum()-g['union'])/a.sum()),
            'overlap_pair_count': int(weak.sum()//2),
            'largest_any_component_fraction': max(map(len,plain))/n if weak.any() else 0.,
            'coverage_load_mean': float(load.sum(1).mean()),
            'coverage_load_p90': float(np.quantile(load.sum(1), .90)),
            'coverage_load_top10_mean': float(sorted_load[-top:].mean()),
            'coverage_energy': float(np.square(load.sum(1)).mean()),
            'complex_energy_10pct': curves[2]['complex_energy'],
            'strong_component_fraction_10pct': curves[2]['largest_strong_component']/n,
            'threshold_curve': curves}


def official_metrics(p, inputs):
    from scoring.official import official_evaluator
    from datasets.reference_layouts import public_hard_reference
    n, areas, b2b, p2b, pins, c, specs = inputs
    # Public-hard reference only; arbitrary neutral denominators. Q is discarded.
    m = asdict(official_evaluator().evaluate_solution(
        {'positions': p.tolist(), 'runtime': 1.}, {}, c, b2b, p2b, pins, areas,
        public_hard_reference(areas,c,specs), median_runtime=1.))
    keys = ('hpwl_total','boundary_violations','grouping_violations','mib_violations',
            'area_violations','dimension_violations')
    return {key:m[key] for key in keys}


class ConeBuilder:
    def __init__(self):
        self.cost=[]; self.le_rows=[]; self.le_rhs=[]; self.eq_rows=[]; self.eq_rhs=[]; self.exp=[]
    def var(self):
        self.cost.append(0.); return len(self.cost)-1
    def le(self, row, rhs):
        self.le_rows.append(row); self.le_rhs.append(float(rhs))
    def eq(self, row, rhs):
        self.eq_rows.append(row); self.eq_rhs.append(float(rhs))
    def solve(self):
        import clarabel
        from scipy.sparse import coo_matrix, csc_matrix
        rows = self.eq_rows+self.le_rows; rhs = self.eq_rhs+self.le_rhs
        cones = [clarabel.ZeroConeT(len(self.eq_rows)), clarabel.NonnegativeConeT(len(self.le_rows))]
        for z, sign, width, original in self.exp:
            # Slack (sign*z,1,width/original) lies in the exponential cone.
            rows.extend([{z:-sign},{},{width:-1./original}]); rhs.extend([0.,1.,0.])
            cones.append(clarabel.ExponentialConeT())
        ii=[];jj=[];vv=[]
        for i,row in enumerate(rows):
            for j,value in row.items():
                if value:
                    ii.append(i);jj.append(j);vv.append(value)
        matrix=coo_matrix((vv,(ii,jj)),shape=(len(rows),len(self.cost))).tocsc()
        settings=clarabel.DefaultSettings(); settings.verbose=False; settings.max_threads=1
        settings.max_iter=120; settings.time_limit=2.
        settings.tol_feas=1e-9;settings.tol_gap_abs=1e-8;settings.tol_gap_rel=1e-8
        solver=clarabel.DefaultSolver(csc_matrix((len(self.cost),len(self.cost))),np.asarray(self.cost),matrix,np.asarray(rhs),cones,settings)
        r=solver.solve()
        info={'status':str(r.status),'seconds':r.solve_time,'primal_residual':r.r_prim,'threads':1}
        if str(r.status) not in ('Solved','AlmostSolved','MaxTime') or r.r_prim>1e-6:
            return None,info
        return np.asarray(r.x),info


def candidate(p, inputs, max_move, max_log_shape, direction, wire_allowance=.001):
    """Conservative convex candidate. Exact geometry is checked after recovery."""
    n=len(p); area=p[:,2:].prod(1); scale=np.sqrt(area.sum()); q=p/scale
    centers=q[:,:2]+q[:,2:]/2; widths=q[:,2:]; local=np.sqrt(area)/scale
    lo=q[:,:2].min(0);hi=(q[:,:2]+q[:,2:]).max(0); spans=hi-lo
    b=ConeBuilder()
    xy=np.array([[b.var(),b.var()] for _ in range(n)])
    wh=np.array([[b.var(),b.var()] for _ in range(n)])
    z=np.array([b.var() for _ in range(n)])
    moves=np.array([[b.var(),b.var()] for _ in range(n)])
    lower=np.array([b.var(),b.var()]);upper=np.array([b.var(),b.var()])
    constraints=np.asarray(inputs[5]); locked=constraints[:,:2].any(1); pre=constraints[:,1]!=0
    protected_boundary=[]; shape_locked=locked.copy()
    sides=((0,False,1),(0,True,2),(1,True,4),(1,False,8))
    for i in range(n):
        code=int(constraints[i,4]); needed=[]
        for axis,is_upper,bit in sides:
            if code & bit:
                edge=p[i,axis]+(p[i,axis+2] if is_upper else 0.)
                bound=(hi[axis] if is_upper else lo[axis])*scale
                needed.append((axis,is_upper,abs(edge-bound)<1e-6))
        if needed and all(v[2] for v in needed):
            shape_locked[i]=True
            protected_boundary.extend((i,axis,is_upper) for axis,is_upper,_ in needed)
    for axis in (0,1):
        weight=(1./spans[axis]) if direction=='balanced' else (1./spans[axis] if direction==('x' if axis==0 else 'y') else 0.)
        b.cost[upper[axis]]=weight;b.cost[lower[axis]]=-weight
        b.le({lower[axis]:-1},-lo[axis]);b.le({upper[axis]:1},hi[axis])
    for i in range(n):
        for axis in (0,1):
            b.le({lower[axis]:1,xy[i,axis]:-1,wh[i,axis]:.5},0.)
            b.le({xy[i,axis]:1,wh[i,axis]:.5,upper[axis]:-1},0.)
            b.le({xy[i,axis]:-1,wh[i,axis]:.5},max(0.,-q[i,axis]))
            b.le({xy[i,axis]:1,moves[i,axis]:-local[i]},centers[i,axis])
            b.le({xy[i,axis]:-1,moves[i,axis]:-local[i]},-centers[i,axis])
            b.le({moves[i,axis]:-1},0.)
            b.cost[moves[i,axis]]=1e-7/n
            if pre[i]:b.eq({xy[i,axis]:1},centers[i,axis])
        b.le({moves[i,0]:1,moves[i,1]:1},max_move)
        if shape_locked[i] or not max_log_shape:
            b.eq({z[i]:1},0.)
            for axis in (0,1):b.eq({wh[i,axis]:1},widths[i,axis])
        else:
            b.le({z[i]:1},max_log_shape);b.le({z[i]:-1},max_log_shape)
            b.exp.extend([(z[i],1.,wh[i,0],widths[i,0]),(z[i],-1.,wh[i,1],widths[i,1])])
    b.le({k:1. for k in moves.ravel()},n*max_move/4)
    for i,axis,is_upper in protected_boundary:
        b.eq({xy[i,axis]:1,wh[i,axis]:(.5 if is_upper else -.5),
              (upper[axis] if is_upper else lower[axis]):-1},0.)
    # Existing exactly matching MIB members receive identical shape changes.
    for group in set(constraints[:,2])-{0}:
        ids=np.flatnonzero(constraints[:,2]==group)
        for offset,i in enumerate(ids):
            for j in ids[:offset]:
                if np.allclose(p[i,2:],p[j,2:],rtol=0,atol=1e-10):
                    b.eq({z[i]:1,z[j]:-1},0.);break
    oldlo=q[:,:2];oldhi=oldlo+widths
    for i in range(n):
        for j in range(i+1,n):
            inter=np.minimum(oldhi[i],oldhi[j])-np.maximum(oldlo[i],oldlo[j])
            axes=(int(np.argmin(inter)),) if (inter<=0).any() else (0,1)
            for axis in axes:
                # Chosen right endpoint minus chosen left endpoint bounds actual
                # intersection from above, including interval containment.
                r=i if oldhi[i,axis]<=oldhi[j,axis] else j
                l=i if oldlo[i,axis]>=oldlo[j,axis] else j
                row={}
                for k,v in ((xy[r,axis],1.),(xy[l,axis],-1.),(wh[r,axis],.5),(wh[l,axis],.5)):
                    row[k]=row.get(k,0.)+v
                b.le(row,max(0.,inter[axis]))
    # HPWL epigraph uses the same center-to-center/pin Manhattan definition as
    # the unmodified official evaluator; only the public netlist is used.
    from scoring.official import official_evaluator
    official=official_evaluator()
    oldwire=official.calculate_hpwl_b2b(p.tolist(),inputs[2])+official.calculate_hpwl_p2b(p.tolist(),inputs[3],inputs[4])
    wire={}
    for edges,is_pin in ((inputs[2],False),(inputs[3],True)):
        for e in np.asarray(edges):
            i,j,weight=int(e[0]),int(e[1]),float(e[2])
            if i==-1:continue
            if is_pin:
                if not 0<=j<n or not 0<=i<len(inputs[4]):continue
            elif not (0<=i<n and 0<=j<n):continue
            if weight<0:raise ValueError('Negative net weight')
            for axis in (0,1):
                t=b.var();wire[t]=weight
                if is_pin:
                    pin=float(inputs[4][i][axis])/scale
                    b.le({xy[j,axis]:1,t:-1},pin);b.le({xy[j,axis]:-1,t:-1},-pin)
                elif i==j:
                    b.le({t:-1},0.)
                else:
                    b.le({xy[i,axis]:1,xy[j,axis]:-1,t:-1},0.)
                    b.le({xy[i,axis]:-1,xy[j,axis]:1,t:-1},0.)
    b.le(wire,oldwire*(1+wire_allowance)/scale)
    result,info=b.solve()
    if result is None:return None,info
    shape=np.clip(result[z],-max_log_shape,max_log_shape)
    shape[shape_locked]=0.
    dims=p[:,2:]*np.exp(np.column_stack((shape,-shape)))
    if not max_log_shape:dims=p[:,2:].copy()
    c=result[xy]*scale
    out=np.column_stack((c-dims/2,dims));out[pre]=p[pre]
    return out,info


def witness(p, out, inputs, old_geometry, old_metrics, move_budget, shape_budget):
    g=geometry(out); a=old_geometry['area']; center0=p[:,:2]+p[:,2:]/2;center=out[:,:2]+out[:,2:]/2
    movement=np.abs(center-center0).sum(1)/np.sqrt(a)
    shape=np.abs(np.log(out[:,2]/p[:,2]))
    m=official_metrics(out,inputs)
    reasons=[]
    if np.max(np.abs(out[:,2:].prod(1)/a-1))>1e-8:reasons.append('area_changed')
    if np.max(g['intersection']-old_geometry['intersection'])>max(1e-7,a.max()*1e-7):reasons.append('overlap_increased')
    if g['union']<old_geometry['union']-max(1e-6,old_geometry['bbox']*1e-8):reasons.append('occupied_union_decreased')
    if movement.max()>move_budget+1e-5 or movement.mean()>move_budget/4+1e-5:reasons.append('movement_budget')
    if shape.max()>shape_budget+1e-6:reasons.append('shape_budget')
    if m['hpwl_total']>old_metrics['hpwl_total']*1.001+1e-7:reasons.append('wire_cost')
    if (out[:,:2]<np.minimum(p[:,:2],0)-1e-6).any():reasons.append('negative_coordinates')
    c=np.asarray(inputs[5]);locked=c[:,:2].any(1);pre=c[:,1]!=0
    if not np.allclose(out[locked,2:],p[locked,2:],rtol=0,atol=1e-7) or not np.allclose(out[pre],p[pre],rtol=0,atol=1e-7):reasons.append('immutable_changed')
    soft=[]
    for key in ('boundary_violations','grouping_violations','mib_violations','area_violations','dimension_violations'):
        if m[key]>old_metrics[key]:soft.append(key)
    absolute=old_geometry['empty']-g['empty']
    return {'accepted':not reasons and not soft, 'geometric_accepted':not reasons,
            'reasons':reasons,'soft_regressions':soft,
            'empty_reduction_over_bbox':float(absolute/old_geometry['bbox']),
            'empty_reduction_over_old_empty':float(absolute/old_geometry['empty']) if old_geometry['empty']>0 else 0.,
            'bbox_reduction_fraction':float(1-g['bbox']/old_geometry['bbox']),
            'removable_whitespace_over_old_empty':float((old_geometry['bbox']-g['bbox'])/old_geometry['empty']) if old_geometry['empty']>0 else 0.,
            'union_change_over_bbox':float((g['union']-old_geometry['union'])/old_geometry['bbox']),
            'max_center_move_local':float(movement.max()),'mean_center_move_local':float(movement.mean()),
            'max_center_move_chip_diagonal':float(np.linalg.norm(center-center0,axis=1).max()/np.linalg.norm((p[:,:2]+p[:,2:]).max(0)-p[:,:2].min(0))),
            'max_abs_log_width_change':float(shape.max()),
            'HPWL_relative_change':float(m['hpwl_total']/max(old_metrics['hpwl_total'],1e-12)-1),
            'metrics':m}


def initialize_worker():
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)


def analyze_case(task):
    row,record,annotation,output_string=task
    from datasets.public_member_inputs import public_member_inputs
    started=time.perf_counter();output=Path(output_string);dest=output/'cases_local'/f"{row['id']}.json"
    inputs=public_member_inputs(record,'local_test',authorize_local_test=True)
    if sha(ROOT/row['draft_source'])!=row['draft_sha256']:raise ValueError('Draft changed')
    p=np.asarray(read(ROOT/row['draft_source'])['raw_positions'],float)
    old_geometry=geometry(p); old_metrics=official_metrics(p,inputs)
    probes=[];best_positions={};best_geometric={}
    zero=witness(p,p,inputs,old_geometry,old_metrics,0.,0.)
    for shape_budget in (0.,.05):
        previous=(0.,p.copy(),zero)
        previous_geo=(0.,p.copy(),zero)
        for move_budget in BUDGETS:
            candidates=[(p.copy(),zero,'original')]
            if previous[0]>0:candidates.append((previous[1],previous[2],'smaller_budget_witness'))
            if previous_geo[0]>0:candidates.append((previous_geo[1],previous_geo[2],'smaller_budget_geometric'))
            if shape_budget:
                fixed=best_positions[move_budget]
                candidates.append((fixed[1],fixed[2],'fixed_shape_witness'))
                geo=best_geometric[move_budget]
                candidates.append((geo[1],geo[2],'fixed_shape_geometric'))
            solves=[]
            if move_budget or shape_budget:
                for direction in ('balanced','x','y'):
                    out,info=candidate(p,inputs,move_budget,shape_budget,direction)
                    if out is not None:
                        w=witness(p,out,inputs,old_geometry,old_metrics,move_budget,shape_budget)
                        candidates.append((out,w,direction));info['guard_reasons']=w['reasons']+w['soft_regressions']
                    solves.append(info)
            safe=[r for r in candidates if r[1]['accepted']]
            geometric=[r for r in candidates if r[1]['geometric_accepted']]
            chosen=max(safe,key=lambda r:r[1]['bbox_reduction_fraction'])
            geo=max(geometric,key=lambda r:r[1]['bbox_reduction_fraction'])
            gain=max(0.,chosen[1]['bbox_reduction_fraction']);ggain=max(0.,geo[1]['bbox_reduction_fraction'])
            previous=(gain,chosen[0],chosen[1]);previous_geo=(ggain,geo[0],geo[1])
            if not shape_budget:
                best_positions[move_budget]=previous;best_geometric[move_budget]=previous_geo
            probes.append({'max_move_local':move_budget,'mean_move_cap_local':move_budget/4,
                           'max_log_shape':shape_budget,'strict':chosen[1],
                           'geometric_only':geo[1],'direction':chosen[2],'solves':solves,
                           'positions':chosen[0].tolist()})
    result={'id':row['id'],'member':row['member'],'n':len(p),'annotation':annotation,
            'pileup':pileup(p),'raw_metrics':old_metrics,'probe_curve':probes,
            'source_draft':row['draft_source'],'source_draft_sha256':row['draft_sha256'],
            'diagnostic_source_sha256':sha(Path(__file__)),
            'seconds':time.perf_counter()-started,'constraints_for_plot':np.asarray(inputs[5]).tolist()}
    save(dest,result)
    return {k:v for k,v in result.items() if k not in ('probe_curve','constraints_for_plot')}


def synthetic_checks():
    # 1. One large connected shallow chain versus a dense strong cluster.
    shallow=np.array([[i*.99,0,1,1] for i in range(5)],float)
    dense=np.array([[i*.05,0,1,1] for i in range(5)],float)
    a,b=pileup(shallow),pileup(dense)
    assert a['largest_any_component_fraction']==b['largest_any_component_fraction']==1.
    assert a['complex_energy_10pct']==0 and b['complex_energy_10pct']>5.
    # 2. High pairwise depth alone does not imply a >=3-block complex cluster.
    pair=np.array([[0,0,1,1],[0,0,1,1],[5,0,1,1]],float)
    assert pileup(pair)['complex_energy_10pct']==0 and pileup(pair)['coverage_energy']>0
    # 3. Scale and translation invariance of dimensionless pileup scores.
    transformed=dense*7.;transformed[:,:2]+=100
    assert abs(pileup(transformed)['complex_energy_10pct']-b['complex_energy_10pct'])<1e-10
    return {'passed':True,'checks':['shallow_connected_chain_vs_deep_cluster','isolated_pair_vs_complex_cluster','scale_and_translation_invariance']}


def run(output, limit=None):
    output=output.resolve();output.mkdir(parents=True,exist_ok=True)
    (output/'cases_local').mkdir(exist_ok=True)
    if not output.is_relative_to(HERE/'runs'):raise ValueError('Output outside E42 runs')
    snapshot=output/'feedback_snapshot.json'
    if snapshot.exists():feedback=read(snapshot)
    else:
        export=json.loads(urllib.request.urlopen('http://127.0.0.1:8765/api/export.json').read())
        feedback={'state':export['state'],'events':export['events'],'manifest_sha256':export['manifest_sha256'],
                  'user_evaluation_verbatim':'可以了，就这样吧。\n我写的主要是两个问题。局部复杂重叠情况和留白过多。前者则指的是一大片模块形成重叠连通块，而且两两之间重叠的面积与模块自生面积的比值还不小。后者我觉得更多是由于模块的形状不佳和布局不佳共同导致的。\n\n请你想办法定量衡量这种复杂堆积的严重程度，以及无效留白的严重程度。后者不能简单地认定为空白区域的面积，而是如果我能只付出极小的代价和移动，就可以大幅度缩小留白面积的情况。',
                  'user_ended_rating_round':True,
                  'captured_utc':datetime.now(timezone.utc).isoformat()}
        save(snapshot,feedback)
    if sha(REVIEW/'manifest.json')!=feedback['manifest_sha256']:raise ValueError('Review manifest identity changed')
    manifest=read(REVIEW/'manifest.json');ratings=feedback['state']['ratings']
    rows=[r for r in manifest['cases'] if r['id'] in ratings and (ratings[r['id']].get('score') or ratings[r['id']].get('note') or ratings[r['id']].get('tags'))]
    if limit is not None:rows=rows[:limit]
    source=read(HERE/'runs/20261024_final_test1000_seed20261024/config.json')
    config={'metric_version':'E42.feedback_geometry.v1','feedback_revision':feedback['state']['revision'],
            'count':len(rows),'rated_count':sum(bool(ratings[r['id']].get('score')) for r in rows),
            'cases':[r['id'] for r in rows],'budgets_max_center_L1_over_sqrt_area':list(BUDGETS),
            'mean_move_cap':'one quarter of maximum per-block cap; averages include all blocks',
            'shape_log_width_caps':[0.,.05],'shape_rule':'width=w0*exp(z); height=h0*exp(-z); exact unchanged module area',
            'HPWL_increase_cap_relative':.001,'strong_graph_thresholds':list(THRESHOLDS),
            'primary_strong_threshold':.10,'threads':'4 processes, each solver and Torch/BLAS use one thread',
            'search':'three convex directions per budget: normalized width+height, width, height; conservative overlap endpoint constraints; protect initially satisfied boundary blocks; exact final guards',
            'primary_whitespace':'bbox contraction while occupied union cannot shrink; empty reduction from de-overlap alone is a separate quantity',
            'status':'exploratory metrics defined after visual feedback; no held-out validation or threshold fitting',
            'source_sha256':sha(Path(__file__)),'budget_seconds':1100,
            'test_scope':'Only the 23 previously reviewed/annotated members inside exposed test1000; no remaining sealed cases',
            'implementation_checks':synthetic_checks()}
    save(output/'config.json',config)
    tasks=[]
    for row in rows:
        file=output/'cases_local'/f"{row['id']}.json"
        if file.exists():continue
        tasks.append((row,source['records'][row['source_index']],ratings[row['id']],str(output)))
    begin=time.perf_counter()
    with ProcessPoolExecutor(max_workers=4, initializer=initialize_worker) as pool:
        futures=[pool.submit(analyze_case,t) for t in tasks]
        for i,future in enumerate(as_completed(futures),1):
            r=future.result();print(f"{i}/{len(tasks)} {r['id']} {r['seconds']:.2f}s",flush=True)
            if time.perf_counter()-begin>config['budget_seconds']:raise TimeoutError('Diagnostic budget exhausted; saved per-case evidence retained')
    save(output/'summary.json',{'count':len(rows),'rated_count':config['rated_count'],
          'seconds_this_invocation':time.perf_counter()-begin,'model_inference_calls':0,'training_calls':0,
          'production_postprocessor_calls':0,'status':'diagnostic_geometry_complete','config_sha256':sha(output/'config.json')})


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=OUTPUT)
    ap.add_argument('--limit',type=int)
    args=ap.parse_args();run(args.output,args.limit)
