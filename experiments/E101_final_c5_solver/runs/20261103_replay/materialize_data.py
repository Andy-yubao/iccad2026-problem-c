"""Metadata-only raw-data mapping fix; no data payload or test loading."""
from prepare import ROOT,BUNDLE,read,save,junction

if __name__=='__main__':
    assert not (BUNDLE/'state.json').exists(),'Do not alter an executing runtime'
    for sub in ('s42','s74','sf'):
        for name in read(ROOT/'protocol/paths.json').values():
            junction(ROOT/name,BUNDLE/sub/name)
        print(sub,'raw file hardlinks ready; original dataset path guard unchanged',flush=True)
    save(BUNDLE/'data_mapping.json',dict(mapping='Real local directories + read-only file hardlinks; original datasets.paths unchanged',reason='Historical dataset_paths rejects junctions resolving outside isolated runtime root',data_payloads_read=False,test_payloads_read=False,source_bytes_duplicated=0))
