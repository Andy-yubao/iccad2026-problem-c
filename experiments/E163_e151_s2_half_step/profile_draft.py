"""Measure E151 eager and graph replay on the same public inputs as E161."""
import os

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[key] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import json
import argparse
from time import perf_counter

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from datasets.random_reference import RandomReference
from .batches import decode
from .evaluation.run import public_graph
from .geometry_feedback import module_features, pair_features
from .shape_feedback import shape_features
from .io import HERE, ROOT, save, sha
from .public import BASE


def timed(action):
    start = perf_counter()
    result = action()
    torch.cuda.synchronize()
    return result, (perf_counter() - start) * 1000


def pad_ports(graph, port_count):
    padding = port_count - graph["port_features"].shape[1]
    data = dict(graph)
    for key in ("port_features", "port_valid", "role_edges", "role_group", "role_hops"):
        axes = (0, 0, 0, padding) if key == "port_features" else (0, padding)
        data[key] = torch.nn.functional.pad(graph[key], axes, value=-1 if key == "role_hops" else 0)
    return data


def gpu_graph(cpu_graph):
    return {key: value.cuda() for key, value in cpu_graph.items()}


def decoded(states, graph, n):
    return (decode(states[-1], graph)[0, :n] * graph["scale"][0]).cpu().numpy()


def geometry_only(states, graph):
    public = {key: graph[key] for key in BASE}
    for state in states[:-1]:
        _, positions = pair_features(state, public)
        module_features(positions, public)
        shape_features(positions, public)


class ReplayedModel:
    def __init__(self, model, inputs):
        self.inputs = {key: value.clone() for key, value in inputs.items()}
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            model(self.inputs)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.states = model(self.inputs)

    def __call__(self, inputs):
        for key, value in inputs.items():
            self.inputs[key].copy_(value)
        self.graph.replay()
        return self.states


def load_model(model_type, frozen_path):
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    checkpoint = ROOT / frozen["checkpoint"]
    if sha(checkpoint) != frozen["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint changed: {checkpoint}")
    weights = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = model_type(**weights["model_config"]).cuda().eval()
    model.load_state_dict(weights["model"], strict=True)
    return model, frozen["checkpoint_sha256"]


def describe(rows):
    stats = {}
    for key in rows[0]:
        if key.endswith("_ms") and rows[0][key] is not None:
            values = [row[key] for row in rows]
            stats[key] = dict(mean=float(np.mean(values)), median=float(np.median(values)),
                              p95=float(np.percentile(values, 95)))
    return stats


@torch.inference_mode()
def main():
    # Read the frozen model entry points only as timing controls.
    from experiments.E151_frozen_role_initial.model import GeometryModel as E151
    from experiments.E161_final_no_mib.model import GeometryModel as E161

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    plan = json.loads((HERE / "evaluation/plan.json").read_text(encoding="utf-8"))
    baseline_paths = sorted((ROOT / plan["baseline_output"] / "cases").glob("*.json"))
    indices = (0, 1, 44, 115, 227, 383, 512, 718, 943, 1204, 1561, 1999)
    dataset = RandomReference("dev")
    inputs = [dataset[index]["solver_inputs"] for index in indices]
    cpu_graphs = [public_graph(case) for case in inputs]
    port_count = max(graph["port_features"].shape[1] for graph in cpu_graphs)
    e151, e151_sha = load_model(E151, HERE / "evaluation/frozen.json")
    e161, e161_sha = load_model(E161, ROOT / "experiments/E161_kaggle_transfer/downloaded/20261123T084205Z/local_final_frozen.json")
    warm = gpu_graph(cpu_graphs[0])
    for model in (e151, e161):
        decoded(model(warm), warm, int(inputs[0][0]))
    started = perf_counter()
    padded_warm = gpu_graph(pad_ports(cpu_graphs[0], port_count))
    replay = None
    capture_error = None
    try:
        captured = ReplayedModel(e151, padded_warm)
        captured(padded_warm)
        torch.cuda.synchronize()
        replay = captured
    except RuntimeError as error:
        capture_error = str(error)
    capture_ms = (perf_counter() - started) * 1000
    rows = []
    for index, case in zip(indices, inputs):
        n = int(case[0])
        cpu, cpu_ms = timed(lambda: public_graph(case))
        graph, transfer_ms = timed(lambda: gpu_graph(cpu))
        padded, padded_transfer_ms = timed(lambda: gpu_graph(pad_ports(cpu, port_count)))
        state151, eager151_ms = timed(lambda: e151(graph))
        _, eager161_ms = timed(lambda: e161(graph))
        _, geometry_ms = timed(lambda: geometry_only(state151, graph))
        _, padded151_ms = timed(lambda: e151(padded))
        original = decoded(state151, graph, n)
        if replay is None:
            replay_ms = equal = max_error = None
        else:
            state_replayed, replay_ms = timed(lambda: replay(padded))
            candidate = decoded(state_replayed, padded, n)
            equal = bool(np.array_equal(original, candidate))
            max_error = float(np.max(np.abs(original - candidate)))
        baseline = json.loads(baseline_paths[index].read_text(encoding="utf-8"))["draft_positions"]
        rows.append(dict(
            index=index, n=n, ports=cpu["port_features"].shape[1],
            cpu_public_graph_ms=cpu_ms, transfer_ms=transfer_ms,
            padded_transfer_ms=padded_transfer_ms, e151_eager_forward_ms=eager151_ms,
            e161_eager_forward_ms=eager161_ms, e151_geometry_only_ms=geometry_ms,
            e151_padded_eager_forward_ms=padded151_ms, e151_graph_replay_ms=replay_ms,
            replay_equal=equal, replay_max_abs_coordinate_error=max_error,
            saved_e151_equal=bool(np.array_equal(original, baseline)),
        ))
    result = dict(device=torch.cuda.get_device_name(), torch_version=torch.__version__,
                  checkpoint_sha256=dict(e151=e151_sha, e161=e161_sha),
                  fixed_port_slots=port_count, capture_and_warmup_ms=capture_ms,
                  capture_error=capture_error,
                  cases=rows, stats=describe(rows))
    path = HERE / "profile/draft.json"
    save(path, result)
    print(json.dumps(result["stats"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.backend_only:
        from .profile_backend import main
    with threadpool_limits(limits=1):
        main()
