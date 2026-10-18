"""Render already-saved reference/draft/final coordinates; never reads datasets."""
import argparse
import html
import json
from pathlib import Path
from visualization.layouts import plot_evaluation_case


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    args=p.parse_args();folder=args.run_dir.resolve();rows=[json.loads(s) for s in (folder/'cases.jsonl').read_text(encoding='utf-8').splitlines()]
    images=[]
    for i,row in enumerate(rows):
        for name,value in row['variants'].items():
            if 'error' in value:continue
            filename=f'{i:03d}_{name}.png'
            plot_evaluation_case(row,name,folder/filename)
            images.append(f'<h3>{html.escape(row["case_id"]+" / "+name)}</h3><img loading="lazy" src="{filename}" style="max-width:100%">')
    (folder/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>E7 layouts</title>'+''.join(images),encoding='utf-8')
    print(json.dumps({'case_count':len(rows),'figures':len(images),'index':str(folder/'index.html')}))


if __name__=='__main__':main()
