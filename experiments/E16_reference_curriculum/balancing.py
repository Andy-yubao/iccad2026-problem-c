"""Deterministic max-min module-count quotas and fresh member ranking."""
import hashlib
from collections import Counter


def rank(seed, tag, value):
    return hashlib.sha256(f'E16.v5:{seed}:{tag}:{value}'.encode()).hexdigest()


def common_quotas(available, *, total=2000, seed=20261019):
    """8*q train and q dev/test give identical normalized histograms.

    Increment each module-count stratum in rounds, limited by its smallest
    partition capacity. Deterministic random tie order spreads remainder.
    """
    keys = sorted(set().union(*(set(v) for v in available.values())))
    caps = {n: min(available['train'].get(n, 0)//8,
                   available['dev'].get(n, 0), available['local_test'].get(n, 0)) for n in keys}
    if sum(caps.values()) < total:
        raise ValueError(f'Insufficient common 8:1:1 capacity: {sum(caps.values())}/{total}; do not silently change the target or partition')
    quotas = {n: 0 for n in keys}
    order = sorted(keys, key=lambda n: (rank(seed, 'quota_remainder', n), n))
    remaining = total
    while remaining:
        for n in order:
            if quotas[n] < caps[n]:
                quotas[n] += 1; remaining -= 1
                if not remaining:
                    break
    return quotas, caps


def select_records(candidates, quotas, parent, *, seed=20261019, factor=1):
    groups = {}
    for record in candidates:
        groups.setdefault(record['n'], []).append(record)
    chosen = []
    for n, quota in sorted(quotas.items()):
        rows = groups.get(n, [])
        rows.sort(key=lambda r: (rank(seed, parent, f'{r["member"][0]}:{r["member"][1]}'), r['member']))
        if len(rows) < factor*quota:
            raise ValueError('Quota exceeds candidate capacity')
        chosen.extend(rows[:factor*quota])
    chosen.sort(key=lambda r: (rank(seed, parent+':order', f'{r["member"][0]}:{r["member"][1]}'), r['member']))
    return chosen


def monitor_records(records, *, seed=20261019, tag, count=500):
    """Proportional fixed monitor; largest remainders with deterministic ties."""
    available = Counter(r['n'] for r in records)
    quotas = {n: count*c//len(records) for n, c in available.items()}
    left = count-sum(quotas.values())
    order = sorted(available, key=lambda n: (-(count*available[n] % len(records)), rank(seed, tag, n)))
    for n in order[:left]:
        quotas[n] += 1
    return select_records(records, quotas, tag, seed=seed)
