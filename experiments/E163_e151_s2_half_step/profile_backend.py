"""Capture the fixed-shaped E151 backend and leave variable ports in eager L0."""
import json
from time import perf_counter

import torch

from datasets.random_reference import RandomReference
from experiments.E151_frozen_role_initial.model import GeometryModel
from .evaluation.run import public_graph
from .io import HERE, save
from .profile_draft import ReplayedModel, describe, gpu_graph, load_model, timed
from .public import BASE


def backend(model, inputs):
    state = inputs["initial"]
    public = {key: inputs[key] for key in BASE}
    condition = model.problem_encoder(public)
    states = [state]
    for index in model.unit_order:
        output = model.units[index](public, state, condition)
        states.extend(output.unbind(0))
        state = output[-1]
    return torch.stack(states)


def backend_inputs(model, graph):
    return {key: graph[key] for key in BASE} | {"initial": model.initial_generator(graph)[0]}


@torch.inference_mode()
def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, digest = load_model(GeometryModel, HERE / "evaluation/frozen.json")
    indices = (0, 1, 44, 115, 227, 383, 512, 718, 943, 1204, 1561, 1999)
    dataset = RandomReference("dev")
    warm = gpu_graph(public_graph(dataset[indices[0]]["solver_inputs"]))
    initial = backend_inputs(model, warm)
    torch.cuda.synchronize()
    started = perf_counter()
    replay = ReplayedModel(lambda data: backend(model, data), initial)
    replay(initial)
    torch.cuda.synchronize()
    capture_ms = (perf_counter() - started) * 1000
    rows = []
    for index in indices:
        case = dataset[index]["solver_inputs"]
        graph = gpu_graph(public_graph(case))
        torch.cuda.synchronize()
        original, eager_ms = timed(lambda: model(graph))
        accelerated, replay_ms = timed(lambda: replay(backend_inputs(model, graph)))
        rows.append(dict(
            index=index, n=int(case[0]), ports=graph["port_features"].shape[1],
            eager_forward_ms=eager_ms, eager_l0_and_backend_replay_ms=replay_ms,
            all_states_equal=torch.equal(original, accelerated),
            max_abs_normalized_state_error=float((original - accelerated).abs().max()),
        ))
    result = dict(
        device=torch.cuda.get_device_name(), checkpoint_sha256=digest,
        capture_and_warmup_ms=capture_ms, cases=rows, stats=describe(rows),
        exact_cases=sum(row["all_states_equal"] for row in rows),
    )
    save(HERE / "profile/backend.json", result)
    print(json.dumps(result["stats"], indent=2), flush=True)
