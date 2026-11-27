"""Exercise the actual train entry, architecture identity and exact two-step resume."""
import argparse
import time
import torch
from .io import HERE, save
from .data import load_rows
from .model import GeometryModel
from .arms import model_config
from .train import parser, run

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    folder = HERE/'runs'/args.tag
    if folder.exists():
        raise ValueError('Use a fresh integration-check tag')
    train = load_rows('train', [0, 1], use_cache=False)
    dev = load_rows('dev_monitor500', [0], use_cache=False)
    started = time.perf_counter()
    def settings(name, resume=False):
        command = ['--arm', 'rectangle', '--output', str(folder/name), '--device', args.device,
                   '--max-hours', '.1', '--max-epochs', '5']
        if resume:
            command.append('--resume')
        return parser().parse_args(command)
    for name, steps, resume in [('continuous', 2, False), ('interrupted', 1, False), ('interrupted', 2, True)]:
        run(settings(name, resume), implementation_check=True, stop_after_steps=steps, check_rows=(train*16, dev))
    first = torch.load(folder/'continuous/checkpoints/latest.pt', map_location='cpu', weights_only=False)
    second = torch.load(folder/'interrupted/checkpoints/latest.pt', map_location='cpu', weights_only=False)
    expected = GeometryModel(**model_config('rectangle'))
    expected.load_state_dict(second['model'], strict=True)
    actual = sum(p.numel() for p in expected.parameters())
    assert actual == second['identity']['parameters']
    assert first['model_config'] == second['model_config'] == model_config('rectangle')
    delta = max(float((value-second['model'][key]).abs().max()) for key, value in first['model'].items())
    assert delta == 0
    assert second['progress']['step'] == 2 and second['identity']['implementation_check']
    assert any(block.bias[0].weight[:, 7:].abs().max() > 0 for block in expected.layers)
    report = {'device': args.device, 'arm': 'rectangle', 'model_config': second['model_config'],
              'actual_parameters': actual, 'identity_parameters': second['identity']['parameters'],
              'strict_checkpoint_load': True, 'two_step_resume_max_parameter_delta': delta,
              'continued_elapsed_seconds': second['elapsed_seconds'],
              'new_architecture_parameters_updated': True, 'gradient_batch_size': 32,
              'gradient_members': [list(r['member']) for r in train],
              'validation_members': [list(r['member']) for r in dev],
              'implementation_check': True, 'test_read': False, 'seconds': time.perf_counter()-started}
    save(folder/'results.json', report)
    print(report, flush=True)

if __name__ == '__main__':
    main()
