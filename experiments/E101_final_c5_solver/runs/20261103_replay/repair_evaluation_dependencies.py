"""Add missing unmodified upstream imports; preserve frozen training evidence."""
from pathlib import Path
import ast,json,shutil,zipfile,datetime
from prepare import ROOT,BUNDLE,read,save,sha

NAMES=('litetestLoader.py','liteLoader.py','lite_dataset.py','lite_dataset_test.py','visualize.py','prime_dataset.py')

def main():
    runtime=read(BUNDLE/'F_runtime.json');root=Path(runtime['root']);here=Path(runtime['here']);run=Path(runtime['run'])
    fix=BUNDLE/'evaluation_dependency_fix_20261104';fix.mkdir(exist_ok=False)
    for p in (BUNDLE/'controller_sources.json',BUNDLE/'state.json',run/'sources.json',run/'frozen_selection.json',run/'evaluation/dev_monitor500/error.json',BUNDLE/'dev500.log',BUNDLE/'pipeline_stderr.log'):
        shutil.copy2(p,fix/p.name)
    before=read(fix/'controller_sources.json')
    # Only the unused preparation script has been edited; running model/loss/
    # scoring code, training identities and selected model remain unchanged.
    assert all(sha(BUNDLE/n)==h for n,h in before.items() if n!='build_f.py')
    frozen=read(run/'sources.json');assert all(sha(root/n)==h for n,h in frozen.items())
    selection=read(run/'frozen_selection.json');assert sha(here/'delivery/model.pt')==selection['checkpoint_sha256']
    assert not list((run/'evaluation/dev_monitor500/cases').glob('*.json'))
    assert not (run/'evaluation/local_test/frozen.json').exists()
    additions={}
    for name in NAMES:
        source=ROOT/'third_party/FloorSet'/name;target=root/'third_party/FloorSet'/name
        assert not target.exists(),'Refuse to overwrite an existing dependency'
        ast.parse(source.read_text(encoding='utf-8'))
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);assert sha(source)==sha(target)
        additions[target.relative_to(BUNDLE).as_posix()]=sha(target)
    with zipfile.ZipFile(fix/'added_dependencies.zip','w',zipfile.ZIP_DEFLATED) as z:
        for n in additions:z.write(BUNDLE/n,n)
    after=dict(before);after['build_f.py']=sha(BUNDLE/'build_f.py');after['repair_evaluation_dependencies.py']=sha(Path(__file__));after.update(additions)
    save(BUNDLE/'controller_sources.json',after)
    save(fix/'migration.json',dict(registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),reason='dev500 import failed: upstream evaluator transitive loader dependencies omitted by isolated runtime preparation',scope='Add six unmodified upstream Python dependencies; correct unused preparation script; no runtime algorithm edits',added_dependencies=additions,old_controller=before,new_controller=after,all_original_runtime_files_unchanged=True,training_sources_json_sha256=sha(run/'sources.json'),training_source_snapshot_sha256=sha(run/'source_snapshot.zip'),frozen_selection_sha256=sha(run/'frozen_selection.json'),checkpoint_sha256=sha(here/'delivery/model.pt'),dev_results_saved_before_fix=0,test_payloads_read_before_fix=False,retrain=False,reselection=False))
    print('Six missing upstream imports restored; original runtime files and model hashes unchanged',flush=True)

if __name__=='__main__':main()
