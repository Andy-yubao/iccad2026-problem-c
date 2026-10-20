"""Five mixed-format train/dev cases; no training or postprocessing."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from datasets.quality_layouts import QualityLayouts
from scoring.evaluation import evaluate_case
from .data import prepare
from .cache import build_role, PackedRows
from .io import save


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    report={'cases':[],'postprocessing_calls':0,'formal_updates':0,'test_read':False,'tail_test_read':False}
    for role in ('train','dev'):
        dataset=QualityLayouts(role)
        imported=[i for i,r in enumerate(dataset.records) if '_import.npz' in r['archive_file']]
        fresh=next(i for i,r in enumerate(dataset.records) if '_import.npz' not in r['archive_file'])
        ids=[fresh,*imported[:2]] if role=='train' else [fresh,imported[0]]
        expected=[]
        for index in ids:
            old=dataset[index]
            new=dataset.archive_item(index)
            assert np.array_equal(old['positions'],new['positions'])
            assert old['case'].baseline_metrics==new['case'].baseline_metrics
            for a,b in zip(old['case'].solver_inputs[1:],new['case'].solver_inputs[1:]):
                assert torch.equal(torch.as_tensor(a),torch.as_tensor(b))
            a,b=prepare(old),prepare(new)
            for key in a['graph']:assert torch.equal(a['graph'][key],b['graph'][key]),key
            for key in ('geometry','b2b','p2b','pins'):assert torch.equal(a['target'][key],b['target'][key]),key
            for key in ('wire_anchor','area_anchor'):assert a['target'][key]==b['target'][key]
            metrics=[evaluate_case(row['case'],old['positions'].tolist(),0)['metrics'] for row in (old,new)]
            assert metrics[0]==metrics[1]
            expected.append(a)
            report['cases'].append({'role':role,'index':index,'member':list(new['member']),
                'imported':'_import.npz' in dataset.records[index]['archive_file'],
                'public_inputs_targets_features_anchors_scores_equal':True})
        build_role(role,args.output/'cache'/role,indices=ids,workers=4,chunk_size=2)
        mapped=PackedRows(role,indices=ids,cache_root=args.output/'cache',allow_pilot=True)
        for i,row in enumerate(expected):
            cached=mapped[i]
            for key in row['graph']:assert torch.equal(row['graph'][key],cached['graph'][key]),key
            for key in ('geometry','b2b','p2b','pins'):assert torch.equal(row['target'][key],cached['target'][key]),key
            for key in ('wire_anchor','area_anchor'):assert row['target'][key]==cached['target'][key]
    for role in ('test','tail_test'):
        try:QualityLayouts(role)
        except PermissionError:pass
        else:raise AssertionError('Sealed role opened')
    report.update(passed=True,unique_cases=5,legacy_cases=3,full_cases=2)
    save(args.output/'results.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
