"""Run the final solver on a public FloorSet case in JSON format."""
import argparse
import json
from pathlib import Path

import numpy as np

from .service import Solver


def public_input(data):
    fields = ('n', 'areas', 'b2b', 'p2b', 'pins', 'constraints', 'specs')
    if data.keys() != set(fields):
        raise ValueError('Expected only n, areas, b2b, p2b, pins, constraints and specs')
    n = data['n']
    if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= 120:
        raise ValueError('Expected an integer n between 1 and 120')
    arrays = [np.asarray(data[key], dtype=np.float32) for key in fields[1:]]
    for i, shape in ((1, 3), (2, 3), (3, 2)):
        if arrays[i].shape == (0,):
            arrays[i] = arrays[i].reshape(0, shape)
    return n, *arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path(__file__).with_name('example.json'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding='utf-8-sig'))
    positions = Solver(args.device).solve(public_input(data))
    result = json.dumps({'positions': positions.tolist()}, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result, encoding='utf-8')
    else:
        print(result, end='')


if __name__ == '__main__':
    main()
