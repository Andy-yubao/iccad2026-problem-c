"""Apply the preprocessor to a local NPZ; never starts training or evaluation."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from .relaxation import Config, relax


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='NPZ with positions and preplaced arrays')
    parser.add_argument('--output', required=True, type=Path, help='New NPZ; existing files are protected')
    parser.add_argument('--iterations', type=int, default=4)
    args = parser.parse_args()
    config = Config(iterations=args.iterations)
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.input, allow_pickle=False) as saved:
        p, mask = saved['positions'], saved['preplaced']
    out, diagnostics = relax(p, mask, config=config, return_history=True)
    history = diagnostics.pop('history')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as stream:
        np.savez_compressed(stream, positions=out, preplaced=mask, history=history,
                            diagnostics_json=json.dumps(diagnostics), config_json=json.dumps(asdict(config)))
    print(json.dumps({k: v for k, v in diagnostics.items() if k != 'trace'}, indent=2))


if __name__ == '__main__':
    main()
