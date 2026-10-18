"""Five-case oracle probe: does shared group movement survive frozen E8?"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from datasets.worksets import LayoutReader,workset_manifest
from .probe import WORKERS,DRAFTS,BASELINE,build_case,worker_call,evaluate_output,summaries
from .records import save,append,read_rows,run_path,provenance,snapshot,sha

VARIANTS=('raw','oracle_global_translation','oracle_group_translation','oracle_center_quarter','oracle_center_half')


def translate(draft,reference,constraints,by_group):
    out=draft.copy();mutable=constraints[:,1]==0
    centers=draft[:,:2]+draft[:,2:]/2;target=reference[:,:2]+reference[:,2:]/2
    partitions=[np.flatnonzero(mutable)]
    if by_group:
        partitions=[np.flatnonzero((constraints[:,3]==group)&mutable)
                    for group in sorted(set(constraints[:,3].astype(int))-{0})]
    changes=[]
    for ids in partitions:
        if not len(ids):continue
        wanted=(target[ids]-centers[ids]).mean(0)
        # One common offset per group, not independent nonnegative clipping.
        # This preserves relative geometry among its movable members exactly.
        shift=np.maximum(wanted,-draft[ids,:2].min(0))
        out[ids,:2]+=shift
        changes.append({'blocks':ids.tolist(),'wanted':wanted.tolist(),'applied':shift.tolist()})
    assert np.array_equal(out[:,2:],draft[:,2:])
    assert np.array_equal(out[~mutable],draft[~mutable])
    assert (out[:,:2]>=0).all()
    return out,changes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--budget-seconds',type=float,default=600.)
    parser.add_argument('--variants',choices=VARIANTS,nargs='+',default=list(VARIANTS[:3]))
    args=parser.parse_args()
    if not 0<args.budget_seconds<1200:parser.error('Five-case probe must remain under 20 minutes')
    output=run_path(args.output);output.mkdir(parents=True,exist_ok=False);begin=time.monotonic()
    manifest=workset_manifest();reader=LayoutReader(manifest,'dev_quality')
    indices=[i for i,pair in enumerate(reader.members) if int(pair[0].split('/')[0].split('_')[1]) in WORKERS]
    assert len(indices)==5
    variants=tuple(args.variants)
    config={**provenance(),'members':[reader.members[i] for i in indices],'variants':variants,
            'seed':0,'solver_seconds':5.,'budget_seconds':args.budget_seconds,'test_payload_reads':0,
            'draft_source':str(DRAFTS),'draft_sha256':sha(DRAFTS),'baseline_source':str(BASELINE),
            'scope':'Oracle diagnostic using dev reference centers; no training or deployable claim. Group0 stays unchanged in group-only variant.'}
    save(output/'config.json',config);snapshot(output,config['source_sha256'])
    drafts={row['case_id']:row for row in read_rows(DRAFTS)};rows=[]
    for order,index in enumerate(indices):
        if time.monotonic()-begin+50>args.budget_seconds or (output/'STOP').exists():break
        raw=reader.case(index);case=build_case(raw)
        draft=np.array(drafts[case.case_id]['draft']);ref=np.array(case.reference_positions);c=np.array(case.solver_inputs[5])
        global_draft,global_changes=translate(draft,ref,c,False)
        group_draft,group_changes=translate(draft,ref,c,True)
        proposals={'raw':draft,'oracle_global_translation':global_draft,'oracle_group_translation':group_draft}
        centers=draft[:,:2]+draft[:,2:]/2;target=ref[:,:2]+ref[:,2:]/2
        for name,fraction in [('oracle_center_quarter',.25),('oracle_center_half',.5)]:
            proposed=draft.copy();mutable=c[:,1]==0
            proposed[mutable,:2]=(draft[mutable,:2]+fraction*(target[mutable]-centers[mutable])).clip(0)
            proposals[name]=proposed
        row={'case_id':case.case_id,'source':case.source,'block_count':case.block_count,'variants':{},
             'changes':{'oracle_global_translation':global_changes,'oracle_group_translation':group_changes}}
        for name in variants[order%len(variants):]+variants[:order%len(variants)]:
            if (output/'STOP').exists() or time.monotonic()-begin+17>args.budget_seconds:break
            row['variants'][name]=evaluate_output(case,worker_call(output,case.solver_inputs,proposals[name]))
            print(json.dumps({'case':case.case_id,'variant':name,'Q':row['variants'][name]['final']['metrics']['cost']}),flush=True)
        append(output/'cases.jsonl',row);rows.append(row)
    summary=summaries(rows,variants)
    result={'status':'completed' if len(rows)==5 and all(len(row['variants'])==len(variants) for row in rows) else 'stopped',
            'summary':summary,'test_payload_reads':0,'wall_seconds':time.monotonic()-begin,
            'scope':'Five fixed development cases; oracle center information only for mechanism diagnosis'}
    save(output/'results.json',result)
    description='只改变草稿位置，保留全部原形状与预放置位置；按组平移仅处理有正分组编号的可移动模块，group0不动。'
    if any(name.startswith('oracle_center_') for name in variants):
        description='逐模块纠正全部非预放置模块（含group0）的参考中心误差的25%/50%，保持原宽高与预放置位置；负坐标单独裁至0，因此边界附近实际中心位移可能不同于名义比例。'
    lines=['# 位置纠正机制的同实例诊断','',
           description+'参考中心仅作为诊断oracle，不能进入部署求解器；所有配置使用同一E8和预算。', '',
           '| 输入 | n | 平均Q | 最大Q | 硬合法 | 平均边界违反 | 平均分组违反 |','|---|---:|---:|---:|---:|---:|---:|']
    for name,v in summary.items():
        lines.append(f'| {name} | {v["n"]} | {v["Q_mean"]:.6f} | {v["Q_max"]:.6f} | {v["feasible"]}/{v["n"]} | {v["boundary_violations_mean"]} | {v["grouping_violations_mean"]} |')
    lines+=['','五例不能当作正式模型选择或泛化证据；oracle平移改善也不代表模型已经学会，失败则不能仅靠增加层数继续投入该机制。','']
    (output/'analysis.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
