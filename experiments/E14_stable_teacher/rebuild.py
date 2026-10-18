"""Rebuild numeric caches from the exact registered v3 indices, without rescanning."""
import argparse
import json
from pathlib import Path
import time
from .data import HERE,ROOT,VERSION,reader,mean_area_inputs,save,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--split',choices=['train','dev'],required=True)
    ap.add_argument('--output',type=Path)
    ap.add_argument('--smoke-count',type=int,choices=range(1,6))
    args=ap.parse_args()
    if args.smoke_count and args.output is None:
        raise ValueError('Smoke cache requires an explicit separate output directory')
    contract_path=ROOT/'protocol/derived_workset_v3.json'
    contract=json.loads(contract_path.read_text(encoding='utf-8'))
    members=contract['sets'][args.split]['members']
    if args.smoke_count:members=members[:args.smoke_count]
    path=(args.output or HERE/'runs/data_v1')/f'{args.split}.json'
    if path.exists():
        existing=json.loads(path.read_text(encoding='utf-8'))
        if existing['version']!=VERSION or [e['member'] for e in existing['members']]!=members:
            raise ValueError('Existing cache does not match frozen members; refusing overwrite')
        print('EXISTS, VERIFIED',path);return
    begin=time.perf_counter();source=reader(members,args.split);entries=[]
    for i,pair in enumerate(members):
        inputs,changes=mean_area_inputs(source.case(i).solver_inputs)
        if inputs is None:raise ValueError(f'Frozen task became incompatible: {pair}: {changes}')
        entries.append({'member':pair,'changes':changes})
        if len(entries)%1000==0:print('REBUILT',len(entries),flush=True)
    save(path,{'version':VERSION,'split':args.split,'count':len(entries),'members':entries,
               'mode':'smoke' if args.smoke_count else 'full',
               'selection':'exact registered v3 membership, without new filtering or replacement',
               'contract_sha256':sha(contract_path),'seconds':time.perf_counter()-begin,'local_test_read':False})
    print('SAVED',path,len(entries),flush=True)


if __name__=='__main__':main()
