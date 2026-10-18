"""Metadata-only panel freeze. No tensors are opened by --write."""
import argparse
import hashlib
import json
from .records import HERE,save
from datasets.worksets import workset_manifest,member_pairs


def expected():
    m=workset_manifest()
    def rank(value):return hashlib.sha256(('E7:20261017:'+value).encode()).hexdigest()
    # Entire workers0..39 are conservatively excluded because of prior reads
    # and potentially in-flight shards, rather than only completed case IDs.
    workers=sorted(range(40,100),key=lambda w:rank('worker:'+str(w)))[:50]
    pool=member_pairs(m,'local_test');test=[]
    for w in sorted(workers):
        shard,row=min((p for p in pool if p[0].startswith(f'worker_{w}/')),
                      key=lambda p:rank(f'{p[0]}:{p[1]}'))
        test.append({'shard':shard,'sample_index':row})
    dev=member_pairs(m,'dev_quality')
    smoke=[{'shard':s,'sample_index':r} for s,r in dev if int(s.split('/')[0][7:]) in (0,24,49,74,99)]
    train_shards={p[0] for p in member_pairs(m,'train')}
    assert len(test)==50 and len(smoke)==5 and not(train_shards&{p['shard'] for p in test})
    return {'version':1,'seed':20261017,'workset_sha256':m['manifest_sha256'],
            'test50':test,'dev5':smoke,'previous_exposure_worker_exclusion':list(range(40)),
            'selection':'50 metadata-ranked workers40..99; one metadata-ranked v2 local_test member each',
            'feedback_policy':'User permits diagnostic feedback; repeated selection is adaptive test feedback and must be disclosed. No test gradient or test training targets.'}


def read():
    value=json.loads((HERE/'panels.json').read_text(encoding='utf-8'))
    if value!=expected():raise ValueError('Panel metadata changed')
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--write',action='store_true')
    args=parser.parse_args();value=expected();path=HERE/'panels.json'
    if args.write:
        if path.exists():read()
        else:save(path,value)
    print(json.dumps({'test50':len(value['test50']),'dev5':len(value['dev5']),'tensor_reads':0}))


if __name__=='__main__':main()
