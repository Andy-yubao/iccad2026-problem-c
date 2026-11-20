"""Train-partition-only frozen inference and exact depth-budget Q labels."""
import os
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from datasets.reference_layouts import public_hard_reference
from scoring.evaluation import evaluate_case
from .common import ROOT, OWNER, RUN, read, save, sha, setup, checkpoint, check_stop
from .batches import collate, inputs_only, decode
from .sources.model import GeometryModel
from .perturb import generate
from .postprocessor.solver import optimize
from .postprocessor.repair.canonical_native import warmup

MODELS = []
IDENTITY = None


def initialize(identity):
    global MODELS, IDENTITY
    setup()
    IDENTITY, MODELS = identity, []
    for spec in read(OWNER/'model_manifest.json')['models']:
        path = ROOT/spec['checkpoint']
        assert sha(path) == spec['sha256']
        saved = torch.load(path, map_location='cpu', weights_only=False)
        model = GeometryModel(**saved['model_config']).eval().requires_grad_(False)
        model.load_state_dict(saved['model'], strict=True)
        MODELS.append((spec['id'], model))
    with threadpool_limits(limits=1):
        warmup()
    save(RUN/f'workers/{os.getpid()}.json', dict(status='ready', models=3, threads=1, training=False))


def evaluate(index, supplied_row=None):
    check_stop()
    spec = read(OWNER/'panel.json')['cases'][index]
    output = RUN/f'data/{index:05d}.pt'
    if output.exists():
        cached = torch.load(output, map_location='cpu', weights_only=False)
        assert cached['identity'] == IDENTITY and cached['member'] == spec['member']
        if cached['complete']:
            assert len(cached['metrics']) == 5
            return dict(index=index, cached=True)
    else:
        cached = None
    if cached is None:
        row = supplied_row if supplied_row is not None else torch.load(RUN/f'inputs/{index:05d}.pt', map_location='cpu', weights_only=False)
        assert list(row['member']) == spec['member']
        graph = inputs_only(collate([row['graph']], [0], 'cpu'))
        assert not {'target', 'reference', 'denominator'}.intersection(graph)
        with torch.inference_mode(), threadpool_limits(limits=1):
            centers = [model(graph)[0].detach().clone() for _, model in MODELS]
            states, pairs, directions = generate(centers, graph, spec)
        case = row['case']
        inputs = case.solver_inputs
        case.reference_positions = public_hard_reference(inputs[1], inputs[5], inputs[6])
        cached = dict(identity=IDENTITY, index=index, member=spec['member'], n=spec['n'], split=spec['split'],
                      graph=graph, states=states, pairs=pairs, directions=directions,
                      source_models=[name for name, _ in MODELS], metrics=[], finals=[], repair_seconds=[],
                      case=case, complete=False, test_read=False)
    graph = cached['graph']
    case = cached['case']
    with torch.inference_mode(), threadpool_limits(limits=1):
        for j in range(len(cached['metrics']), 5):
            check_stop()
            draft = (decode(cached['states'][j:j+1], graph)[0]*graph['scale'][0]).numpy().astype(float)
            started = time.perf_counter()
            final, info = optimize(draft, case.solver_inputs)
            seconds = time.perf_counter()-started
            scored = evaluate_case(case, final, seconds)
            assert scored['metrics']['is_feasible'] and scored['independent_hard_checks']['is_feasible']
            cached['metrics'].append(scored['metrics'])
            cached['finals'].append(torch.from_numpy(np.asarray(final).copy()))
            cached['repair_seconds'].append(seconds)
            cached.setdefault('work', []).append(info['work'])
            cached.setdefault('fallback', []).append(info['row_retained'])
            cached['complete'] = j == 4
            if cached['complete']:
                cached['Q'] = torch.tensor([v['cost'] for v in cached['metrics']], dtype=torch.float64)
                cached['auxiliary'] = torch.tensor([[v['hpwl_gap'], v['area_gap'], v['violations_relative']]
                                                     for v in cached['metrics']], dtype=torch.float64)
                del cached['case']
            checkpoint(output, cached)
            save(RUN/f'workers/{os.getpid()}.json', dict(status='labeling', index=index, drafts_complete=j+1,
                                                        training=False, optimizer_updates=0))
    return dict(index=index, cached=False)
