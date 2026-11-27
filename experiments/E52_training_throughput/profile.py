"""Bounded five-case throughput probe, no usable model training or checkpoint."""
import json
import statistics
import time
import torch
import numpy as np
from datasets.reference_layouts import ReferenceLayouts
from .io import HERE, ROOT, save, sha
from .train import seed_all
from .data import load_rows, collate
from .batches import inputs_only
from .model import GeometryModel
from .loss import losses
from .reference_loss import losses as reference_losses
from .resident import ResidentRows
from .cuda_step import CapturedStep


def measure(fn, count=12, warm=3):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); times = []
    for _ in range(count):
        start = time.perf_counter(); fn(); torch.cuda.synchronize(); times.append(time.perf_counter()-start)
    return {'median_seconds': statistics.median(times), 'min_seconds': min(times), 'max_seconds': max(times)}


def run():
    started = time.perf_counter(); seed_all(0)
    meta = ReferenceLayouts('train').records
    indices = [min(range(len(meta)), key=lambda i: abs(meta[i]['n']-n)) for n in (25,45,70,95,115)]
    rows = load_rows('train', indices, use_cache=False)
    repeated = [rows[i%5] for i in range(32)]
    checkpoint = ROOT/'experiments/E51_module_overlap/runs/20261025_rectangle_w96d12_unified_v2_main_seed0/checkpoints/best.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = GeometryModel(**state['model_config']).cuda().train(); model.load_state_dict(state['model'])
    # LR=0 measures real AdamW bookkeeping without changing learned weights.
    optimizer = torch.optim.AdamW(model.parameters(), lr=0., weight_decay=3e-4)
    graph, _ = collate(repeated, 'cuda')
    resident = ResidentRows(rows, 'cuda', max_nodes=graph['valid'].shape[1])
    ids = torch.tensor([i%5 for i in range(32)], device='cuda')
    resident_batch = resident.batch(ids)
    assert all(torch.equal(graph[k], resident_batch[k]) for k in graph)
    def eager(batch):
        optimizer.zero_grad(set_to_none=True)
        values = reference_losses(model(inputs_only(batch)), batch, None, 'main', 0.)
        if not torch.isfinite(torch.stack(list(values.values()))).all(): raise RuntimeError('nonfinite')
        values['total'].backward()
        float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
        optimizer.step()
    timings = {}
    timings['collate_host_to_gpu'] = measure(lambda: collate(repeated, 'cuda'))
    timings['resident_batch'] = measure(lambda: resident.batch(ids))
    timings['eager_gpu_batch'] = measure(lambda: eager(graph))
    timings['original_collate_plus_step'] = measure(lambda: eager(collate(repeated, 'cuda')[0]))
    model.eval()
    def forward():
        with torch.no_grad(): model(inputs_only(graph))
    timings['forward_only'] = measure(forward)
    model.train()
    save(HERE/'runs/20261025_profile5/baseline_partial.json', {'timings':timings, 'seconds':time.perf_counter()-started})
    capture_start = time.perf_counter()
    captured = CapturedStep(model, optimizer, resident_batch)
    capture_seconds = time.perf_counter()-capture_start
    def replay():
        resident.batch(ids, out=captured.inputs); captured.replay()
    timings['cuda_graph_resident_step'] = measure(replay)
    # Numerical equivalence with dropout disabled; unchanged weights and FP32.
    model.eval(); eager_values = reference_losses(model(inputs_only(graph)), graph, None, 'main', 0.)
    eager_values['total'].backward()
    eager_total = float(eager_values['total'].detach())
    # Rebuild an eval-mode graph to compare loss and clipped gradients exactly.
    optimizer.zero_grad(set_to_none=True)
    values = reference_losses(model(inputs_only(graph)), graph, None, 'main', 0.)
    values['total'].backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.,foreach=True)
    reference_grads = [p.grad.detach().clone() for p in model.parameters()]
    exact = CapturedStep(model, optimizer, graph); exact.replay()
    max_grad = max(float((p.grad-g).abs().max()) for p,g in zip(model.parameters(),reference_grads))
    assert abs(float(exact.values['total'])-eager_total) < 1e-6
    assert max_grad < 1e-6, max_grad
    # One lightweight CPU profiler pass identifies dispatch-heavy operators.
    model.train()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        eager(graph)
    top = sorted(prof.key_averages(), key=lambda e: e.self_cpu_time_total, reverse=True)[:15]
    out = HERE/'runs/20261025_profile5'
    result = {'status': 'passed', 'seconds': time.perf_counter()-started, 'gpu': torch.cuda.get_device_name(0),
        'torch': str(torch.__version__), 'parameters': sum(p.numel() for p in model.parameters()),
        'unique_cases': 5, 'role': 'train', 'indices': indices, 'module_counts': [r['case'].block_count for r in rows],
        'batch_size': 32, 'repeated_five_cases_for_microbenchmark': True, 'timings': timings,
        'cuda_graph_capture_seconds': capture_seconds, 'eval_loss_abs_delta': abs(float(exact.values['total'])-eager_total),
        'max_clipped_parameter_gradient_abs_delta': max_grad,
        'cpu_top': [{'name':e.key,'self_cpu_ms':e.self_cpu_time_total/1000,'calls':e.count} for e in top],
        'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(), 'dataset_packed_bytes':resident.bytes,
        'checkpoint_sha256':sha(checkpoint), 'usable_training_started':False, 'optimizer_lr':0.,
        'checkpoint_written':False, 'test_read':False}
    save(out/'results.json', result); print(json.dumps(result, indent=2), flush=True)


if __name__=='__main__': run()
