"""Targeted Windows sharing-lock and synthetic trainer regression; no data tensors."""
import argparse
import ctypes
import json
from pathlib import Path
import sys
import threading
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.graph_proposal import train
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'dataset_tensor_reads': 0}
    target = args.output / 'replace_probe.json'
    train.save_json(target, {'version': 0})
    if sys.platform == 'win32':
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                      ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        kernel.CreateFileW.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        # Allow reads/writes, but deny rename/delete just like an external file reader.
        handle = kernel.CreateFileW(str(target.resolve()), 0x80000000, 3, None, 3, 0, None)
        assert handle != wintypes.HANDLE(-1).value
        blocked = args.output / 'blocked.tmp'
        blocked.write_text('{}', encoding='utf-8')
        try:
            train.os.replace(blocked, target)
        except PermissionError as exc:
            report['reproduced_windows_error'] = exc.winerror
        else:
            raise AssertionError('Expected Windows sharing violation')
        finally:
            blocked.unlink(missing_ok=True)
        timer = threading.Timer(.2, lambda: kernel.CloseHandle(handle))
        timer.start()
        train.save_json(target, {'version': 1})
        timer.join()
        report['real_transient_sharing_lock_retry'] = json.loads(target.read_text()) == {'version': 1}
    for step in range(100):
        train.save_json(target, {'version': step})
        train.append_budget(args.output, {'accumulated_seconds': step / 10, 'completed_steps': step})
    assert train.recorded_seconds(args.output) == 9.9
    report['repeated_atomic_and_append_writes'] = 100
    # Force all trainer batches to use one tiny synthetic layout, without torch.load.
    original_members = train.TrainMembers
    class SyntheticMembers(original_members):
        def __getitem__(self, index):
            c = torch.zeros((3, 5))
            c[0, 0] = 1; c[1, 1] = 1
            ref = torch.tensor([[0., 0., 3., 2.], [4., 0., 2., 2.], [0., 3., 1., 1.]])
            specs = torch.tensor([[-1., -1., 3., 2.], [4., 0., 2., 2.], [-1., -1., -1., -1.]])
            safe = (3, torch.tensor([6., 4., 1.]), torch.tensor([[0., 2., 1.]]),
                    torch.empty((0, 3)), torch.empty((0, 2)), c, specs)
            return safe, ref
    run = args.output / 'synthetic_training'
    command = ['synthetic_io_check', '--legacy-protocol', '--kind', 'deterministic', '--run-dir', str(run),
               '--device', 'cpu', '--batch-size', '1', '--budget-seconds', '60', '--max-batches', '30']
    saved_steps = []
    real_torch_save = train.torch.save
    def observe_checkpoint(payload, path):
        saved_steps.append(payload['total_steps'])
        return real_torch_save(payload, path)
    with patch.object(train, 'TrainMembers', SyntheticMembers), patch.object(sys, 'argv', command), \
            patch.object(train.torch, 'save', observe_checkpoint):
        train.main()
    assert 0 in saved_steps and 1 in saved_steps and 30 in saved_steps, saved_steps
    state = torch.load(run / 'checkpoints/latest.pt', map_location='cpu', weights_only=False)
    assert state['total_steps'] == 30
    command[-1] = '1'
    with patch.object(train, 'TrainMembers', SyntheticMembers), patch.object(sys, 'argv', command + ['--resume']):
        train.main()
    resumed = torch.load(run / 'checkpoints/latest.pt', map_location='cpu', weights_only=False)
    assert resumed['total_steps'] == 31 and resumed['accumulated_seconds'] >= state['accumulated_seconds']
    report.update(synthetic_batches=31, initial_first_batch_and_final_checkpoint=True,
                  strict_resume=True, cumulative_budget_preserved=True)
    train.save_json(args.output / 'verification.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
