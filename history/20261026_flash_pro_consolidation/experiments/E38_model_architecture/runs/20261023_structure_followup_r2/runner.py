"""Sequential authorized structure study with read-only E38 control reuse."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = ROOT/'experiments/E38_model_architecture'
CONTROL = BASE/'runs/20261022_capacity_deep96x12_seed0'
EXPERIMENTS = ('E39_rectangular_attention', 'E40_endpoint_constraint_tokens')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    os.replace(temporary, path)


def certificate():
    frozen = read(CONTROL/'frozen.json')
    assert all(sha(ROOT/name) == digest for name, digest in frozen['sources'].items())
    assert sha(ROOT/frozen['checkpoint']) == frozen['checkpoint_sha256']
    identity = read(CONTROL/'config.json')['identity']
    assert (identity['width'], identity['layers'], identity['seed'], identity['batch_size']) == (96, 12, 0, 32)
    details = {}
    for experiment in EXPERIMENTS:
        path = ROOT/'experiments'/experiment
        prefix = experiment[:3]
        common = ('loss.py', 'losses.py', 'features.py', 'batches.py', 'diagnostics.py', 'cache.py')
        equal = {}
        for name in common:
            original = (BASE/name).read_text(encoding='utf-8').replace('E38', 'EXP')
            copied = (path/name).read_text(encoding='utf-8').replace(prefix, 'EXP')
            equal[name] = original == copied
        assert all(equal.values()), equal
        original = (BASE/'train.py').read_text(encoding='utf-8').replace('E38', 'EXP')
        copied = (path/'train.py').read_text(encoding='utf-8').replace(prefix, 'EXP')
        copied = copied.replace('EXP.rectangular_attention.v2', 'EXP.capacity.v1')
        copied = copied.replace('EXP.endpoint_constraint_tokens.v2', 'EXP.capacity.v1')
        copied = copied.replace("default='geometry_control'", "default='base96x4'")
        copied = copied.replace("default='full_tokens'", "default='base96x4'")
        copied = copied.replace('from .public import inputs_only', 'from .batches import inputs_only')
        copied = copied.replace('model = GeometryModel(**model_config(args.arm)).to(args.device)',
                                "model = GeometryModel(identity['width'], identity['layers']).to(args.device)")
        copied = copied.replace("    if sum(p.numel() for p in model.parameters()) != identity['parameters']:\n        raise ValueError('Actual model parameter count differs from architecture identity')\n", '')
        assert copied == original, experiment+' training protocol changed'
        entry = path/'runs/20261023_training_entry_cuda'
        entry_report = read(entry/'results.json')
        entry_identity = read(entry/'interrupted/config.json')['identity']
        assert entry_report['strict_checkpoint_load'] and entry_report['two_step_resume_max_parameter_delta'] == 0
        assert entry_report['actual_parameters'] == entry_report['identity_parameters']
        assert all(sha(ROOT/name) == digest for name, digest in entry_identity['sources'].items())

        assert sha(path/'reference_model.py') == sha(BASE/'model.py')
        if prefix == 'E39':
            assert (path/'data.py').read_bytes() == (BASE/'data.py').read_bytes()
        sources = {p.relative_to(ROOT).as_posix(): sha(p) for p in path.rglob('*.py') if 'runs' not in p.relative_to(path).parts}
        sources.update({p.relative_to(ROOT).as_posix(): sha(p) for p in path.glob('*.ps1')})
        for name in ('provenance.json', 'requirements-native.txt'):
            sources[(path/name).relative_to(ROOT).as_posix()] = sha(path/name)
        details[prefix] = {'identical_common_files_after_namespace_normalization': equal,
                           'training_difference_allowlist': ['revision', 'default arm', 'E40 public input-key extension', 'v2 full constructor configuration and parameter-count guard'],
                           'training_protocol_equal_after_allowlist': True,
                           'plain_reference_model_sha256': sha(path/'reference_model.py'), 'sources': sources,
                           'actual_training_entry_check': entry_report, 'training_entry_report_sha256': sha(entry/'results.json'),
                           'data_note': 'E40 adds public-only endpoint/group tensors; original graph/cached targets unchanged.' if prefix == 'E40' else 'Same data collation.'}
    return {'control': CONTROL.relative_to(ROOT).as_posix(), 'control_checkpoint_sha256': frozen['checkpoint_sha256'],
            'control_training_identity': {k:v for k,v in identity.items() if k != 'sources'},
            'original_E25_reference': 'experiments/E38_model_architecture/runs/20261022_capacity_e25_reference',
            'retrained_base_reference': 'experiments/E38_model_architecture/runs/20261022_capacity_base96x4_seed0',
            'selected_trunk': {'width': 96, 'layers': 12}, 'seed': 0, 'experiments': details,
            'decision': 'Rank by fixed monitor/full-dev proxies and draft severity; terminal Q did not select trunk.',
            'single_GPU_sequential': True, 'test_read': False, 'loss_change': False}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--prepare-only', action='store_true')
    args = ap.parse_args()
    HERE.mkdir(parents=True, exist_ok=True)
    import msvcrt
    with (HERE/'runner.lock').open('a+b') as lock:
        if lock.tell() == 0:
            lock.write(b'0'); lock.flush()
        lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        plan = certificate()
        plan_path = HERE/'control_reuse.json'
        if plan_path.exists():
            assert read(plan_path) == plan, 'Control/source identity changed; preserve prior evidence'
        else:
            save(plan_path, plan)
        if args.prepare_only:
            print('Read-only control compatibility verified; GPU check and training not launched.', flush=True)
            return
        capacity = BASE/'runs/20261022_capacity_pipeline_seed0'
        assert read(capacity/'results.json')['status'] == 'complete', 'Finish core four-arm pipeline first'
        assert not (capacity/'active.json').exists(), 'Core stage remains active'
        state_path = HERE/'progress.json'
        if state_path.exists() and not args.resume:
            raise ValueError('Existing follow-up requires --resume')
        state = read(state_path) if state_path.exists() else {'stages': [], 'test_read': False}
        env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                   MKL_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4', E39_WIDTH='96', E39_DEPTH='12', E40_WIDTH='96', E40_DEPTH='12')
        stages = []
        for experiment, arm, tag in [('E39_rectangular_attention', 'rectangle', '20261023_rectangle_w96d12_r2'),
                                      ('E40_endpoint_constraint_tokens', 'full_tokens', '20261023_tokens_w96d12_r2')]:
            parent = ROOT/'experiments'/experiment/'runs'
            output = parent/f'{tag}_{arm}_seed0'
            pipeline = parent/f'{tag}_pipeline_seed0'
            stages.append((experiment[:3]+' train', ['-m', f'experiments.{experiment}.pipeline', '--arm', arm, '--stage', 'train',
                           '--tag', tag, '--seed', '0', '--max-hours', '4', '--max-epochs', '200', '--resume'], pipeline, output/'results.json'))
            stages.append((experiment[:3]+' final', ['-m', f'experiments.{experiment}.final', '--output', str(output)], output, output/'terminal/results.json'))
        for kind, arguments, stop_folder, expected in stages:
            if (HERE/'STOP').exists():
                save(HERE/'results.json', {'status': 'stopped', 'next_stage': kind, 'test_read': False})
                return
            if any(stage['kind'] == kind and stage['returncode'] == 0 for stage in state['stages']):
                assert expected.exists(), 'Recorded result missing: '+str(expected)
                continue
            command = [sys.executable, '-u', '-B', *arguments]
            started = time.perf_counter()
            print('Starting '+kind, flush=True)
            with (HERE/'console.log').open('a', encoding='utf-8') as log:
                log.write('\n'+json.dumps({'kind': kind, 'command': command})+'\n'); log.flush()
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                           creationflags=subprocess.CREATE_NO_WINDOW)
                save(HERE/'active.json', {'kind': kind, 'pid': process.pid, 'command': command,
                                         'output': str(stop_folder.relative_to(ROOT)), 'expected': str(expected.relative_to(ROOT))})
                while process.poll() is None:
                    if (HERE/'STOP').exists():
                        stop_folder.mkdir(parents=True, exist_ok=True); (stop_folder/'STOP').touch()
                    try:
                        process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        pass
            stage = {'kind': kind, 'command': command, 'returncode': process.returncode, 'seconds': time.perf_counter()-started}
            (HERE/'active.json').unlink(missing_ok=True)
            if not process.returncode and expected.exists() and kind.endswith('train'):
                if read(expected)['status'] not in ('early_stop', 'budget', 'max_epochs'):
                    save(HERE/'results.json', {'status': 'stopped', 'stage': stage, 'test_read': False})
                    return
            state['stages'].append(stage); save(state_path, state)
            if process.returncode or not expected.exists():
                save(HERE/'results.json', {'status': 'failed', 'stage': stage, 'test_read': False})
                raise RuntimeError(kind+' failed; inspect console.log before recovery')
        save(HERE/'results.json', {'status': 'complete', 'stages': state['stages'], 'test_read': False,
                                 'control_reused_read_only': plan['control']})


if __name__ == '__main__':
    main()
