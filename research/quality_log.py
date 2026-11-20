"""Print saved validation Q summaries; optionally follow one existing queue.

Read-only with respect to experiments/results. Does not import training code,
load checkpoints, evaluate layouts, or launch/restart any experiment.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import time


def summary(result, source):
    if result.get('status') != 'complete':
        return None
    count = int(result['count'])
    return (
        f"[Q] arm={result['arm']} count={count} "
        f"mixed_Q={result['mixed_Q']:.9f} "
        f"weighted_Q={result['weighted_Q']:.9f} mean_Q={result['mean_Q']:.9f} "
        f"hard_legal={result['hard_legal']}/{count} "
        f"best_epoch={result['best_epoch']} "
        f"mean_ms={1000*result['mean_complete_seconds']:.3f} "
        f"p95_ms={1000*result['p95_complete_seconds']:.3f} "
        f"source={source}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--follow', action='store_true')
    args = parser.parse_args()
    root = args.experiment.resolve()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    receipt = args.log.with_suffix(args.log.suffix + '.receipts.json')
    seen = json.loads(receipt.read_text(encoding='utf-8')) if receipt.exists() else {}

    def emit(line):
        stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with args.log.open('a', encoding='utf-8') as stream:
            stream.write(stamp + ' ' + line + '\n')
        print(line, flush=True)

    while True:
        for path in sorted((root/'runs').glob('*/evaluation_no_mib_dev500/results.json')):
            blob = path.read_bytes()
            digest = hashlib.sha256(blob).hexdigest()
            key = path.relative_to(root).as_posix()
            if seen.get(key) == digest:
                continue
            line = summary(json.loads(blob), key)
            if line is not None:
                emit(line)
                seen[key] = digest
                temp = receipt.with_suffix(receipt.suffix + '.tmp')
                temp.write_text(json.dumps(seen, indent=2)+'\n', encoding='utf-8')
                temp.replace(receipt)
        if not args.follow:
            break
        queue = json.loads((root/'queue_state.json').read_text(encoding='utf-8'))
        if queue['status'] in ('complete', 'paused', 'needs_attention', 'cancelled_by_user', 'error'):
            emit('[Q_LOG] queue_status=' + queue['status'])
            break
        time.sleep(5)


if __name__ == '__main__':
    main()
