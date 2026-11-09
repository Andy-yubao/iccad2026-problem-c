"""Explicitly freeze the finished baseline's own validation-Loss best; never launch training."""
import argparse
from ..io import ROOT,HERE,save,sha
from .common import OWNER,read

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--confirm-baseline',action='store_true',required=True)
    parser.parse_args()
    run=HERE/'runs/20261104_generalization96_seed0'
    result=read(run/'results.json')
    assert result['status'] in ('early_stop','max_epochs','budget')
    assert not (OWNER/'baseline_frozen.json').exists()
    checkpoint=run/'checkpoints/best.pt'
    save(OWNER/'baseline_frozen.json',dict(checkpoint=checkpoint.relative_to(ROOT).as_posix(),
        checkpoint_sha256=sha(checkpoint),source_results_sha256=sha(run/'results.json'),
        selection='finished baseline own validation-Loss best, never Q',
        stop_reason=result['status'],best_epoch=result['best_epoch'],test_read=False))
    print('Frozen baseline only; no candidate training started.',flush=True)

if __name__=='__main__':main()
