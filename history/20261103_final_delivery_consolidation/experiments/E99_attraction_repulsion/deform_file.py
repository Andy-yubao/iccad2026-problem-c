"""V3 file entry: bounded translation and area-preserving deformation."""
import argparse,json
from pathlib import Path
import numpy as np
from .deformation import Config,relax


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    with np.load(args.input,allow_pickle=False) as saved:
        p=saved['positions'];pre=saved['preplaced']
        if 'fixed_shape' in saved:fixed=saved['fixed_shape']
        elif 'constraints' in saved:fixed=saved['constraints'][:,0]!=0
        else:raise ValueError('fixed_shape mask or public constraints are required')
    out,d=relax(p,pre,fixed_shape=fixed,config=Config(),return_history=True)
    history=d.pop('history');args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as stream:
        np.savez_compressed(stream,positions=out,preplaced=pre,fixed_shape=fixed,history=history,diagnostics_json=json.dumps(d))
    print(json.dumps({k:v for k,v in d.items() if k!='trace'},indent=2))


if __name__=='__main__':main()
