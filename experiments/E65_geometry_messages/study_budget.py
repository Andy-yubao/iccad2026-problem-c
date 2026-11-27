"""Run ledger is data, not a shared algorithm dependency; serialize its writes."""
from datetime import datetime, timezone
import json
import time
from .io import HERE, ROOT, save, exclusive

PATH = HERE/'long_task.json'


def register(output, max_hours, implementation_check=False):
    if implementation_check:
        return None
    name = str(output.resolve().relative_to(ROOT)).replace('\\', '/')
    with exclusive(PATH.with_suffix('.lock')):
        data = json.loads(PATH.read_text(encoding='utf-8'))
        deadline = datetime.fromisoformat(data['deadline_utc']).timestamp()
        if time.time() >= deadline:
            raise RuntimeError('Long-task calendar budget exhausted')
        ledger = data['run_ledger']
        previous = next((row for row in ledger if row['output'] == name), None)
        if previous is None:
            if len(ledger) >= data['max_formal_runs']:
                raise RuntimeError('Long-task formal-run count exhausted')
            previous = {'output': name, 'elapsed_seconds': 0., 'status': 'registered',
                        'registered_utc': datetime.now(timezone.utc).isoformat()}
            ledger.append(previous)
        other = sum(row['elapsed_seconds'] for row in ledger if row['output'] != name)
        remaining = data['max_cumulative_training_seconds']-other
        if previous['elapsed_seconds'] >= remaining:
            raise RuntimeError('Long-task cumulative training budget exhausted')
        data['status'] = 'running'
        save(PATH, data)
    return {'remaining_including_this': remaining, 'deadline': deadline}


def exhausted(guard, elapsed):
    return guard is not None and (elapsed >= guard['remaining_including_this'] or time.time() >= guard['deadline'])


def charge(output, elapsed, status='running', *, implementation_check=False):
    if implementation_check:
        return
    name = str(output.resolve().relative_to(ROOT)).replace('\\', '/')
    with exclusive(PATH.with_suffix('.lock')):
        data = json.loads(PATH.read_text(encoding='utf-8'))
        row = next(row for row in data['run_ledger'] if row['output'] == name)
        row.update(elapsed_seconds=max(elapsed, row['elapsed_seconds']), status=status,
                   updated_utc=datetime.now(timezone.utc).isoformat())
        data['cumulative_training_seconds'] = sum(row['elapsed_seconds'] for row in data['run_ledger'])
        save(PATH, data)
