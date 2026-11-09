"""Freeze nested 16k/32k/64k train members using existing public-only scan metadata."""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import time
from datasets.paths import dataset_paths
from datasets.reference_layouts import digest
from datasets.expanded_reference import registry
from .io import ROOT,HERE,RUN,REGISTRY,read,save,sha,verify_frozen


def rank(tag,member):
    return hashlib.sha256(f'E58.v6:20261026:{tag}:{member[0]}:{member[1]}'.encode()).hexdigest()


def main():
    started=time.perf_counter();provenance=read(HERE/'provenance.json');verify_frozen()
    if REGISTRY.exists():
        value=registry();print('Existing v6 frozen; reusing',value['manifest_sha256'],flush=True);return
    base=read(ROOT/'protocol/reference_workset_v5.json');split=read(ROOT/'protocol/split_v1.json')
    scan_root=ROOT/provenance['scan_metadata_root'];scan_identity=read(scan_root/'identity.json')
    if scan_identity['preparation_sources']['datasets/reference_layouts.py']!=sha(ROOT/'datasets/reference_layouts.py'):
        raise ValueError('Public MIB candidate-scan semantics changed')
    data_root=dataset_paths()['training_root'];pools=defaultdict(list);stats={};rejections=Counter()
    for shard in split['shards']['train']:
        scan=read(scan_root/'shards'/(shard.replace('/','__')+'.json'))
        st=(data_root/shard).stat()
        if scan['shard']!=shard or scan['parent']!='train' or (st.st_size,st.st_mtime_ns)!=(scan['size'],scan['mtime_ns']):
            raise ValueError('Frozen public candidate metadata changed: '+shard)
        stats[shard]={'size':st.st_size,'mtime_ns':st.st_mtime_ns}
        rejections.update(scan['rejections'])
        for row,n in scan['accepted']:pools[n].append((shard,row))
    old=base['sets']['train']['records'];existing={tuple(x['member']) for x in old}
    old_counts=Counter(x['n'] for x in old);capacity={n:len(rows) for n,rows in pools.items()}
    add32=[];add64=[]
    for n,count in sorted(old_counts.items()):
        eligible=[x for x in pools[n] if x not in existing]
        eligible.sort(key=lambda x:(rank('candidate',x),x))
        if len(eligible)<3*count:
            raise ValueError(f'Insufficient n={n} to preserve v5 distribution: {len(eligible)} extra available, {3*count} required; do not silently resample or alter strata')
        for member in eligible[:count]:add32.append({'member':list(member),'n':n,'case_id':f'train:{member[0]}:{member[1]}'})
        for member in eligible[count:3*count]:add64.append({'member':list(member),'n':n,'case_id':f'train:{member[0]}:{member[1]}'})
    add32.sort(key=lambda x:(rank('order32',x['member']),x['member']))
    add64.sort(key=lambda x:(rank('order64',x['member']),x['member']))
    records=old+add32+add64
    if len(records)!=64000 or len({tuple(x['member']) for x in records})!=64000:
        raise ValueError('Expanded members not unique/exact')
    sets=dict(base['sets']);sets['train']={'count':64000,'n_counts':dict(Counter(x['n'] for x in records)),'records':records}
    source_shards=dict(base['source_shards']);source_shards.update({x['member'][0]:stats[x['member'][0]] for x in records})
    value={'dataset_version':'E58.reference_expansion.v6','task_version':base['task_version'],'seed':20261026,
        'targets':{'train':64000,'dev':2000,'local_test':2000},
        'parent_registry_sha256':sha(ROOT/'protocol/reference_workset_v5.json'),
        'frozen_existing_protocol':provenance['frozen_existing_protocol'],
        'nested_train_views':[16000,32000,64000],
        'partition_rule':'Original split_v1 train shards only for expansion; dev/test/monitor memberships copied exactly from v5',
        'balance_rule':'At 16k/32k/64k, n-count histogram is exactly 1x/2x/4x original v5 train; held-out roles are unchanged, not rebalanced',
        'input_transform':provenance['MIB_rule'],
        'label_source':'Original reference centers and log-aspect; original reference need not be feasible after the mean-area transform',
        'selection_uses_Q':False,'selection_uses_free_labels':False,'new_test_payloads_read':False,
        'candidate_train_n_counts':capacity,'candidate_pool_rejections':dict(rejections),
        'public_scan_metadata_source':provenance['scan_metadata_root'],
        'sets':sets,'source_shards':source_shards}
    value['manifest_sha256']=digest(value)
    save(REGISTRY,value);registry()
    save(RUN/'membership_results.json',{'status':'complete','registry':REGISTRY.relative_to(ROOT).as_posix(),
        'registry_sha256':sha(REGISTRY),'counts':{k:x['count'] for k,x in sets.items()},
        'nested_train_views':[16000,32000,64000],'new_train_cases':48000,
        'same_v5_dev_test_and_monitors':True,'candidate_train_count':sum(capacity.values()),
        'metadata_seconds':time.perf_counter()-started,'test_payloads_read':False})
    verify_frozen();print('v6 membership frozen: train64000/dev2000/test2000; nested16k/32k/64k',flush=True)


if __name__=='__main__':main()
