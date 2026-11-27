"""Finalize reproducibility records from existing artifacts; no inference or training."""
from datetime import datetime, timezone
from pathlib import Path
import zipfile
from .io import ROOT, HERE, read, save, sha
from .collect import RUNS
from .report import OUT, OLD_DRAFTS, make_report


def main():
    manifest = {'scope': 'Existing development evidence only; original files unchanged',
                'inputs': [], 'checkpoint_checks': {}, 'test_read': False}
    def entry(path, role):
        manifest['inputs'].append({'path': path.relative_to(ROOT).as_posix(),
                                  'role': role, 'sha256': sha(path)})
    forward = read(OUT/'forwards.json')
    for name, folder in RUNS.items():
        full = read(folder/'full_dev.json')
        terminal = read(folder/'terminal/results.json')
        summary = read(folder/'terminal/summary.json')
        checkpoint = folder/'checkpoints/best.pt'
        digest = sha(checkpoint)
        assert full['count'] == 2000 and len(terminal['records']) == 500
        assert full['checkpoint_sha256'] == summary['checkpoint_sha256'] == digest
        embedded_digest = terminal['summary'].get('checkpoint_sha256')
        if embedded_digest is not None:
            assert embedded_digest == digest
        if name in forward['identities']:
            assert forward['identities'][name]['checkpoint_sha256'] == digest
        manifest['checkpoint_checks'][name] = {'path': checkpoint.relative_to(ROOT).as_posix(),
              'sha256': digest, 'full_dev_terminal_summary_and_actual_best_agree': True,
              'terminal_results_embedded_checkpoint_sha256': embedded_digest,
              'terminal_results_header_note': 'Separate terminal/summary.json identity checked; old embedded summary may omit hash'}
        for relative, role in (('full_dev.json','frozen raw dev2000'),
                               ('terminal/results.json','saved flash500 records'),
                               ('terminal/summary.json','saved flash500 aggregate')):
            entry(folder/relative, name+': '+role)
    for relative in ('protocol/reference_workset_v5.json', 'protocol/paths.json'):
        entry(ROOT/relative,'frozen registry / data path resolution')
    examples = read(OUT/'examples.json')['cases']
    for row in examples:
        entry(OLD_DRAFTS/f"case_{row['index']:04d}.json",'previously saved E42 example coordinates')
    save(OUT/'input_manifest.json',manifest)
    make_report(read(OUT/'analysis.json'),read(OUT/'features.json')['records'],forward,
                read(OUT/'followup.json'),examples)
    source_files = sorted([*HERE.glob('*.py'), HERE/'plan.json', HERE/'README.md',
                           *ROOT.glob('datasets/*.py'), *ROOT.glob('visualization/*.py'),
                           ROOT/'protocol/reference_workset_v5.json',
                           ROOT/'docs/protocol.md', ROOT/'docs/workspace_structure.md'])
    with zipfile.ZipFile(OUT/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in source_files:
            archive.write(path,path.relative_to(ROOT).as_posix())
    provenance = read(OUT/'report_provenance.json')
    provenance.update({'source_files': {p.relative_to(ROOT).as_posix(): sha(p) for p in source_files},
          'source_snapshot_sha256': sha(OUT/'source_snapshot.zip'),
          'input_manifest_sha256': sha(OUT/'input_manifest.json'),
          'visual_QA': 'Reviewed all three chart panels and case0029; readable axes, legends and comparison stages. AI inspection is not user acceptance.'})
    save(OUT/'report_provenance.json',provenance)
    result = read(OUT/'results.json')
    result.update({'report_complete': True,'supplementary_analysis_complete': True,
          'report_completed_utc': datetime.now(timezone.utc).isoformat(),
          'supplementary_scope': 'Saved E65 dev500 drafts only; no additional forward or solver',
          'chart_panel_count': 3, 'diagnostic_example_count': len(examples),
          'input_manifest': 'input_manifest.json', 'source_snapshot': 'source_snapshot.zip',
          'elapsed_seconds_scope': 'Original main numerical run only; excludes followup and report assembly'})
    save(OUT/'results.json',result)
    save(OUT/'progress.json',{'status':'complete','stage':'report_and_provenance_complete',
                            'training_started':False,'test_read':False})
    page = (OUT/'index.html').read_text(encoding='utf-8')
    assert page.count('<table>') == page.count('</table>') == 1
    assert page.count('<tr>') == 3 and '|---' not in page
    assert all((OUT/p).exists() for p in ('report.md','case_table.csv','input_manifest.json','source_snapshot.zip'))
    print('Final report complete; all seven checkpoint identities verified; no training/inference/solver run.')


if __name__ == '__main__':
    main()
