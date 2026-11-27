"""Check exact exchangeable public-input twins on the same five train cases."""
import json
import torch
from .io import ROOT,OUT,save
from .cache import load_rows

def main():
    torch.set_num_threads(1)
    ids=json.loads((OUT/'five_case_probe.json').read_text())['train_indices']
    rows=load_rows('train',ids);result=[]
    for row in rows:
        g=row['graph'];n=g['n'];twins=[];feature_matches=0
        for i in range(n):
            for j in range(i+1,n):
                if not g['mask'][i,0] or not g['mask'][j,0]:continue
                if not all(torch.equal(g[k][i],g[k][j]) for k in ('features','area','mask','known','specs','constraints')):continue
                feature_matches+=1
                order=torch.arange(n);order[i]=j;order[j]=i
                if all(torch.equal(g[k],g[k][order][:,order]) for k in ('adjacency','mib','group')):
                    twins.append([i,j])
        result.append({'member':list(row['member']),'n':n,'equal_public_node_candidates':feature_matches,'exact_transposition_twins':twins})
    save(OUT/'structural_probe.json',{'unique_train_cases':5,'test_read':False,'network_inference':False,
        'records':result,'limit':'Sufficient exact transposition test only; does not rule out larger automorphisms, near-symmetry or non-identifiability of the reference.'})
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
