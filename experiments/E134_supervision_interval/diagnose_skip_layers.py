"""Frozen stride3 best63: execute geometry blocks 1,2,3,6,9,12 only."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '1'
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from .model import GeometryModel
from .evaluation.run import public_graph
from .loss import base_per_case
from .data import load_rows
from .resident import ResidentRows

OWNER = Path(__file__).resolve().parent
ROOT = OWNER.parents[1]
RUN = OWNER/'runs/e119_stride3_seed0'
PRIOR = RUN/'layer_analysis_v1'
OUT = RUN/'skip_layers_01236912_v1'
IDS = [0, 1, 2, 3, 6, 9, 12]

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def main():
    from datasets.random_reference import RandomReference
    started = time.perf_counter()
    OUT.mkdir(exist_ok=False)
    prior = read(PRIOR/'manifest.json')
    checkpoint = RUN/'checkpoints/best.pt'
    expected = prior['checkpoint_sha256']
    assert sha(checkpoint) == expected
    gate = read(OWNER/'preflight.json')
    assert all(sha(ROOT/name) == digest for name, digest in gate['sources'].items())
    save(OUT/'plan.json', dict(checkpoint_sha256=expected, kept_readouts=IDS,
        skipped_blocks=[4,5,7,8,10,11], panel='same dev_monitor500',
        semantics='Keep input/global frontend and shared heads. Execute original blocks 1,2,3,6,9,12 sequentially; recompute state embedding and geometry feedback from the previous retained output. No original hidden/state bypass.',
        loss='Original complete public-priority-v1, per-case equal means; original supervision weights on S3/S6/S9/S12 also reported.',
        training=False, optimizer=False, postprocessing_calls=0, Q_calls=0, test_read=False,
        source_sha256=sha(Path(__file__))))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = GeometryModel(**saved['model_config']).cuda().eval()
    model.load_state_dict(saved['model'], strict=True)
    dataset = RandomReference('dev_monitor500')
    assert [dataset.records[i]['member'] for i in dataset.indices] == prior['members']
    g = {key:value.cuda() for key,value in public_graph(dataset[0]['solver_inputs']).items()}

    def forward():
        h = model.input(g['features'])*g['valid'][...,None]
        for block in model.initial_blocks:
            h = block(h, g)
        state = model.head(h)
        outputs = [state]
        for index in IDS[1:]:
            h = model.layers[index-1](h+model.state_embed(state*g['mask']), g, state)
            state = model.head(h)
            outputs.append(state)
        return torch.stack(outputs)

    states = np.zeros((500, len(IDS), 120, 3), np.float32)
    old_states = np.load(PRIOR/'states.npy', mmap_mode='r')
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
            outputs = forward()
        for i in range(500):
            for key,value in public_graph(dataset[i]['solver_inputs']).items():
                g[key].copy_(value)
            graph.replay()
            states[i] = outputs[:,0].cpu().numpy()
            np.testing.assert_array_equal(states[i,:4], old_states[i,:4])
            if (i+1)%50 == 0:
                print('Captured', i+1, '/500; prefix S0-S3 bitwise unchanged', flush=True)
                save(OUT/'status.json', dict(stage='capture', completed=i+1, total=500))
        np.save(OUT/'states.npy', states, allow_pickle=False)
        rows = load_rows('dev_monitor500')
        pack = ResidentRows(rows, 'cuda', max_nodes=120)
        names = prior['loss_names']
        losses = np.zeros((500, len(IDS), len(names)), np.float64)
        for start in range(0, 500, 32):
            indices = list(range(start, min(start+32, 500)))
            batch = pack.batch(indices)
            raw = torch.as_tensor(states[indices], device='cuda')
            for j in range(len(IDS)):
                values = base_per_case(raw[:,j], batch)
                losses[indices,j] = torch.stack([values[name] for name in names], -1).cpu().double().numpy()
        np.save(OUT/'losses.npy', losses, allow_pickle=False)
    original = np.load(PRIOR/'losses.npy')[:,IDS]
    np.testing.assert_array_equal(losses[:,:4], original[:,:4])
    assert np.isfinite(losses).all()
    assert all(torch.equal(value.cpu(), saved['model'][name]) for name,value in model.state_dict().items())
    assert sha(checkpoint) == expected
    assert all(sha(ROOT/name) == digest for name,digest in gate['sources'].items())
    members = prior['members']
    sources = sorted({m[0] for m in members})
    source_index = np.array([sources.index(m[0]) for m in members])
    counts = np.bincount(source_index)
    rng = np.random.default_rng(20261114)
    draws = rng.integers(0, len(sources), (2000, len(sources)))

    def paired(before, after):
        delta = after-before
        sums = np.bincount(source_index, weights=delta, minlength=len(sources))
        means = sums[draws].sum(1)/counts[draws].sum(1)
        return dict(original=float(before.mean()), skipped=float(after.mean()),
            delta=float(delta.mean()), relative_percent=float(100*delta.mean()/before.mean()),
            delta_paired_source_bootstrap95=np.quantile(means,[.025,.975]).tolist(),
            improved=int((delta<0).sum()), worsened=int((delta>0).sum()), unchanged=int((delta==0).sum()))

    comparison = {f'S{layer}':paired(original[:,j,0], losses[:,j,0]) for j,layer in enumerate(IDS)}
    weights = np.array([0,0,0,.2/3,.2/3,.2/3,.8])
    comparison['original_weighted_objective'] = paired(original[:,:,0]@weights, losses[:,:,0]@weights)
    final_components = {name:dict(original=float(original[:,-1,j].mean()),
        skipped=float(losses[:,-1,j].mean()), delta=float((losses[:,-1,j]-original[:,-1,j]).mean()))
        for j,name in enumerate(names)}
    verification = dict(status='passed', cases=500, source_shards=len(sources),
        checkpoint_sha256=expected, weights_unchanged=True, frozen_sources_unchanged=True,
        S0_to_S3_states_and_losses_bitwise_unchanged=True, members_identical=True,
        finite=True, training=False, postprocessing_calls=0, Q_calls=0, test_read=False)
    save(OUT/'results.json', dict(layers=IDS, loss_names=names, comparison=comparison,
        terminal_components=final_components, seconds=time.perf_counter()-started, verification=verification))
    import csv
    with (OUT/'per_case.csv').open('w',newline='',encoding='utf-8-sig') as handle:
        writer=csv.writer(handle)
        writer.writerow(['dev_index','source','member_index','layer','original_loss','skipped_loss','delta'])
        for i,member in enumerate(members):
            for j,layer in enumerate(IDS):
                writer.writerow([i,*member,layer,original[i,j,0],losses[i,j,0],losses[i,j,0]-original[i,j,0]])
    text = ['# Frozen layer skipping: E134 stride3 best63', '',
        'Keep S0 frontend and original geometry blocks 1,2,3,6,9,12. All weights unchanged. Same dev500, no training/postprocessing.', '',
        '|Output|Original mean Loss|Skipped mean Loss|Delta|Relative change|Improved/worsened/unchanged|',
        '|---|---:|---:|---:|---:|---:|']
    for name,value in comparison.items():
        text.append(f"|{name}|{value['original']:.6f}|{value['skipped']:.6f}|{value['delta']:+.6f}|{value['relative_percent']:+.2f}%|{value['improved']}/{value['worsened']}/{value['unchanged']}|")
    text += ['', 'Final component values and source-shard paired confidence intervals: `results.json`. Per-case comparisons: `per_case.csv`.', '',
        'S0-S3 states and Loss are bitwise identical to the original capture; checkpoint and original sources verified unchanged.']
    (OUT/'report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    save(OUT/'status.json', dict(status='complete', seconds=time.perf_counter()-started))
    print(json.dumps(dict(comparison=comparison, terminal_components=final_components, verification=verification),ensure_ascii=False,indent=2),flush=True)

if __name__ == '__main__':
    main()
