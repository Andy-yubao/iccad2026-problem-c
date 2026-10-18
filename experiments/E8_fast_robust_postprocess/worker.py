"""Reference-free subprocess boundary; only public solver inputs are accepted."""
import argparse
import json
import signal
from pathlib import Path
import time
import numpy as np


def execute(payload):
    if set(payload) != {'inputs','draft','variant','seconds'}:
        raise ValueError('Only public inputs allowed across worker boundary')
    inputs=(int(payload['inputs'][0]),*[np.asarray(x) for x in payload['inputs'][1:]])
    raw=np.asarray(payload['draft'],dtype=float)
    from .frozen.solver import optimize as control
    from .observe import Observe
    from .fast_solver import optimize as fast
    from .shape_first import optimize as shape_first
    from .certificate import infeasibility_certificate
    from .topology_repair import optimize as topology_repair
    from .frozen import global_refine,solver
    from .guard import guarded_refine
    from .beam_topology import optimize as beam_topology
    from .joint_geometry import optimize as joint_geometry
    from .conic_geometry import optimize as conic_geometry
    from .optimizer import optimize as selected
    start=time.perf_counter()
    try:
        if payload['variant']=='safe_plain120':
            original=solver.refine
            solver.refine=guarded_refine(original)
            try:
                final,stats=control(raw,inputs,seconds=payload['seconds'],shape_steps=24,
                                    reweighted=True,mip_seconds=0.)
            finally:solver.refine=original
        elif payload['variant'] in ('control120','control_short','certified120','certified_short','safe120'):
            with Observe() as observation:
                if payload['variant']=='safe120':
                    observation.patch(solver,'refine',guarded_refine(solver.refine))
                if payload['variant'].startswith('certified'):
                    original=global_refine.solve_lp
                    def screened(problem,seed,edges,**kwargs):
                        certificate=infeasibility_certificate(problem,seed,edges)
                        if certificate is not None:
                            return None,{'status':'certified_infeasible_without_LP','certificate':certificate}
                        return original(problem,seed,edges,**kwargs)
                    observation.patch(global_refine,'solve_lp',screened)
                final,stats=control(raw,inputs,seconds=payload['seconds'],shape_steps=24,
                                    reweighted=True,mip_seconds=0.)
            observed=observation.result()
            stats['observation']=observed
            stats['improvements']=observed.pop('improvements')
        elif payload['variant']=='conic_robust':
            final,stats=selected(raw,inputs,seconds=payload['seconds'])
        elif payload['variant'].startswith('conic_'):
            final,stats=conic_geometry(raw,inputs,seconds=payload['seconds'],
                                      conservative='relaxed' not in payload['variant'],
                                      reduced='dense' not in payload['variant'],
                                      quadratic_wire=('quadratic' in payload['variant'] or 'diagonal' in payload['variant']),
                                      diagonal_wire='diagonal' in payload['variant'])
        elif payload['variant'].startswith('joint_'):
            final,stats=joint_geometry(raw,inputs,seconds=payload['seconds'],
                                      shared=payload['variant'].startswith('joint_shared'),
                                      fixed_shapes=payload['variant']=='joint_fixed')
        elif payload['variant'].startswith('beam_'):
            final,stats=beam_topology(raw,inputs,seconds=payload['seconds'])
        elif payload['variant'].startswith('topology_'):
            final,stats=topology_repair(raw,inputs,seconds=payload['seconds'])
        elif payload['variant'].startswith(('shape_first','linear_area')):
            final,stats=shape_first(raw,inputs,seconds=payload['seconds'],linear_area=payload['variant'].startswith('linear_area'))
        else:
            final,stats=fast(raw,inputs,seconds=payload['seconds'],reduced=payload['variant']=='sparse_short')
        result={'positions':final.tolist(),'stats':stats,'solve_seconds':time.perf_counter()-start}
    except Exception as exc:
        result={'error':{'type':type(exc).__name__,'message':str(exc)},
                'solve_seconds':time.perf_counter()-start}
    return result


def main():
    signal.signal(signal.SIGINT,signal.SIG_IGN)
    parser=argparse.ArgumentParser()
    parser.add_argument('request',type=Path,nargs='?')
    parser.add_argument('response',type=Path,nargs='?')
    parser.add_argument('--serve',type=Path)
    args=parser.parse_args()
    if args.serve:
        from .persistent import retry_file
        directory=args.serve;request=directory/'request.json';response=directory/'response.json'
        idle=time.monotonic()
        while not (directory/'SHUTDOWN').exists():
            if not request.exists():
                if time.monotonic()-idle>90:break
                time.sleep(.01);continue
            payload=json.loads(retry_file(lambda:request.read_text(encoding='utf-8')))
            retry_file(request.unlink)
            result=execute(payload)
            temp=response.with_suffix('.tmp')
            temp.write_text(json.dumps(result,ensure_ascii=False,allow_nan=False),encoding='utf-8')
            retry_file(lambda:temp.replace(response));idle=time.monotonic()
    else:
        if args.request is None or args.response is None:parser.error('request and response required')
        result=execute(json.loads(args.request.read_text(encoding='utf-8')))
        args.response.write_text(json.dumps(result,ensure_ascii=False,allow_nan=False),encoding='utf-8')


if __name__=='__main__':
    main()
