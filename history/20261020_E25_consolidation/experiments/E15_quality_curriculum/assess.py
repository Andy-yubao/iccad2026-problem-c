"""Finish full-dev diagnostics for frozen best weights; tests are unavailable."""
import argparse
import json
from pathlib import Path
import sys
import time
import torch
from datasets.quality_layouts import REGISTRY
from .io import HERE, ROOT, sha, save, sources, snapshot
from .data import load_rows
from .model import GeometryModel
from .train import monitor_losses, seed_all


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    begin = time.perf_counter()
    if not args.output.resolve().is_relative_to(HERE/'runs'):
        raise ValueError('Assessment outputs must stay in E15/runs')
    state = torch.load(args.run/'checkpoints/best.pt', map_location='cpu', weights_only=False)
    if state['identity']['implementation_check']:
        raise ValueError('Implementation checkpoint is not a trained research model')
    if state['identity']['registry_sha256'] != sha(REGISTRY) or state['identity']['sources'] != sources():
        raise ValueError('Frozen data/source identity mismatch')
    seed_all(state['identity']['seed'])
    model = GeometryModel(**state['model_config']).to(args.device)
    model.load_state_dict(state['model'])
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output/'config.json', {'command': [sys.executable, *sys.argv],
        'checkpoint_sha256': sha(args.run/'checkpoints/best.pt'), 'identity': state['identity'],
        'scope': 'Full dev2000 geometry/proxy diagnostics only; no training or test access'})
    snapshot(args.output/'source_snapshot.zip', state['identity'])
    rows = load_rows('dev')
    values = monitor_losses(model, rows, args.device, state['identity']['arm'], state['progress']['epoch'], 16)
    save(args.output/'results.json', {'count': len(rows), 'losses': values,
        'elapsed_seconds': time.perf_counter()-begin, 'epoch': state['progress']['epoch'],
        'test_read': False, 'tail_test_read': False, 'status': 'complete'})
    print(json.dumps(values, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
