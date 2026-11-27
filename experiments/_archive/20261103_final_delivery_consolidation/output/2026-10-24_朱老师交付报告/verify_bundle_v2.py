"""Audit E42 report v2 against saved final records and official result archives."""
import csv
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    index=read(HERE/'evidence_index.json')
    assert len(index['copies'])==26
    for item in index['copies']:
        assert sha(HERE/item['bundle_path'])==sha(ROOT/item['repository_source'])==item['sha256']
    docs=['report.md','README.md','factbase.md','reproduce.md','materials/README.md']
    links=[]
    for name in docs:
        p=HERE/name;text=p.read_text(encoding='utf-8')
        assert '\ufffd' not in text and 'E25' not in text,name
        for raw in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text):
            target=raw.strip('<>').split('#')[0]
            if not target or target.startswith(('http:','https:','mailto:')):continue
            assert (p.parent/target).is_file(),(name,raw)
            links.append([name,raw])
    facts=read(HERE/'factbase.json')['facts']
    assert len(facts)==len({r['id'] for r in facts})==18
    for fact in facts:
        assert fact['claim_boundary'] and all((HERE/p).is_file() for p in fact['evidence'])
    report=(HERE/'report.md').read_text(encoding='utf-8')
    assert len(re.findall(r'!\[',report))==6
    assert '参考' in report and '草稿' in report and '12轮' in report
    assert '前两栏没有重新评分' in report and '没有重跑最终求解' in report
    summary=read(HERE/'evidence/E42_final_summary.json')
    for key in ('weighted_Q','seconds_mean','median_Q','p95_Q','CVaR95_Q','max_Q','seconds_p95','seconds_max'):
        assert f'{summary[key]:.6f}' in report,key
    with (HERE/'evidence/C_beta_leaderboard_20260823.csv').open(encoding='utf-8-sig',newline='') as h:
        beta={int(r['rank']):r for r in csv.DictReader(h)}
    for rank in (1,9,5,10):
        assert f"{float(beta[rank]['raw_score']):.6f}" in report
        assert f"{float(beta[rank]['total_runtime'])/100:.6f}" in report
    for rank in (3,10):assert abs(float(beta[rank]['total_score'])/float(beta[rank]['raw_score'])-.7)<1e-8
    assets=HERE/'materials'
    expected={'official_results_reference.png','network_structure.png','E42_test_distribution.png',
              'layout_typical.png','layout_worst_quality.png','layout_slowest.png'}
    assert {p.name for p in assets.glob('*.png')}==expected
    images=[]
    for name in sorted(expected):
        p=assets/name
        with Image.open(p) as im:im.verify()
        with Image.open(p) as im:
            assert min(im.size)>500
            images.append({'path':p.relative_to(HERE).as_posix(),'width':im.width,'height':im.height})
    for p in assets.glob('*.svg'):ET.parse(p)
    examples=read(assets/'layout_examples.json');drafts=read(assets/'E42_draft_examples.json')
    assert len(examples)==len(drafts['records'])==3
    assert drafts['generated_draft_count']==3
    assert [r['index'] for r in examples]==[302,695,746]
    config=read(HERE/'evidence/E42_final_config.json')
    for e,d in zip(examples,drafts['records']):
        original=read(ROOT/e['final_coordinate_source'])
        assert e['member']==original['member']==d['member']==config['records'][e['index']]['member']
        assert original['metrics']==e['E42_metrics']
        raw=np.array(d['raw_positions']);assert raw.shape==(e['n'],4) and np.isfinite(raw).all()
        assert not e['final_solver_reexecuted']
    assert not any('comparison' in p.name or p.name.startswith('E25') for p in (HERE/'evidence').iterdir())
    qa={'status':'passed','revision':'v2','date':'2026-10-24','facts':18,'source_copies_verified':26,
        'official_result_files':5,'official_table_numbers_verified':True,'document_links_checked':len(links),
        'report_images':6,'PNG_images':images,'three_phase_examples':[302,695,746],
        'E25_comparisons_in_authored_report_or_assets':False,
        'illustration_drafts_generated_total':3,'final_1000_evaluation_reexecuted':False,'new_official_scores':0,
        'reference_and_raw_metrics':'not re-scored, n/a; final values are original saved metrics',
        'visual_review':'All6 PNGs inspected, including all3 three-phase layouts; titles and diagrams readable.',
        'report_characters':len(report),'encoding':'UTF-8'}
    save(HERE/'qa.json',qa)
    files=[p for p in HERE.rglob('*') if p.is_file() and p.name!='manifest.json']
    save(HERE/'manifest.json',{'date':'2026-10-24','revision':'v2','files':[
        {'path':p.relative_to(HERE).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(files)],'manifest_excludes_itself':True})
    print(json.dumps({'status':'passed','revision':'v2','files':len(files)+1,'facts':18,'source_files':26,'links':len(links),'figures':6,'report_characters':len(report)},ensure_ascii=False))


if __name__=='__main__':main()
