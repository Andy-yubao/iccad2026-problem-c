"""Summarize the separately authorized full-repair evaluation from saved evidence."""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import json,math,hashlib,datetime
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    plan=read(HERE/'fullrepair_singlethread_plan.json');folder=HERE/plan['output'];result=read(folder/'results.json');assert result['finished']
    comparisons={};details={};baseline=result['arms']['E85']['weighted_Q']
    for arm in plan['arms']:
        name=arm['name'];records=[read(p) for p in sorted((folder/name/'terminal').glob('*.json'))];assert len(records)==500
        manifest=read(folder/name/'gallery/manifest.json');assert len(manifest['cases'])==100
        for r in manifest['cases']:assert hashlib.sha256((folder/name/'gallery'/r['image']).read_bytes()).hexdigest()==r['sha256']
        weights=np.exp((np.array([r['n'] for r in records])-120)/12);q=np.array([r['metrics']['cost'] for r in records]);fallback=np.array([r['fallback'] for r in records]);w=weights/weights.sum()
        assert all(r['metrics']['is_feasible'] and r['independent_hard_checks']['is_feasible'] for r in records)
        score=float(q@w);assert abs(score-result['arms'][name]['weighted_Q'])<1e-12
        prior=[read(HERE/arm['previous_terminal']/f'{i:04d}.json') for i in range(500)]
        assert [r['member'] for r in prior]==[r['member'] for r in records]
        gaps=[]
        for r,p in zip(records,prior):
            a=np.array(r['draft_positions']);b=np.array(p['draft_positions']);scale=np.sqrt((a[:,2]*a[:,3]).sum());gaps.append(float(np.abs(a-b).max()/max(scale,1e-12)))
        oldq=float(np.array([r['metrics']['cost'] for r in prior])@w)
        comparisons[name]=dict(full_weighted_Q=score,previous_slim3_weighted_Q=oldq,Q_change=score-oldq,delta_to_new_E85=score-baseline,old_fallbacks=sum(r['fallback'] for r in prior),new_fallbacks=int(fallback.sum()),saved_draft_normalized_max_error=max(gaps),draft_cases_over_2e_minus4=int(sum(x>2e-4 for x in gaps)))
        details[name]=dict(fallback_indices=[r['index'] for r in records if r['fallback']],fallback_weight_mass=float(w[fallback].sum()),fallback_excess_Q=float(((q-1)*w)[fallback].sum()),gallery_hashes_verified=100,all_hard_legal=True)
    report=['# 旧E85完整后处理：现有候选的单线程补测','',
        '本次仅按用户新授权评价现有唯一own-Loss best，未训练、未重新选checkpoint，测试封存。固定同dev500，所有臂串行、每次一个请求，CPU数学库/求解器全部1线程，GPU推理保持FP32和原模型。原8候选训练账本与关闭时间不改写，自动唤醒保持关闭。','',
        '| 方案 | 完整加权Q | 旧三步Q（历史） | 行回退 | 完整均时/秒 | P95/秒 | 超过.25秒 |','|---|---:|---:|---:|---:|---:|---:|']
    for a in plan['arms']:
        name=a['name'];r=result['arms'][name];old=comparisons[name]['previous_slim3_weighted_Q']
        report.append(f"| [{name}]({name}/gallery/index.html) | {r['weighted_Q']:.9f} | {old:.6f} | {r['fallbacks']} | {r['mean_complete_seconds']:.5f} | {r['p95_complete_seconds']:.5f} | {r['cases_over_025s']} |")
    report += ['',
        '表中七个动态原生关系路径直接使用旧E85完整后处理配置；C4/C5保留学习关系功能的两臂与fixed_native_control使用同完整预算和既有固定关系接口。C4/C5若丢弃关系头、只接原生动态关系，其物理草稿等于C1，直接复用C1结果，不能把复用冒称对关系头有效性的检验。','',
        '旧三步结果采用此前双线程配置，仅作历史质量/失败参照；本次统一单线程且GPU运行状态不同，不能把耗时差或细小分数差全部单因子归于修复器。新E85对照与所有新候选则使用相同单线程完整配置。神经草稿新旧数值差与全部成员匹配检查见saved_comparison.json。','',
        '完整请求包含公开特征/padding、传输、GPU网络及内部固定几何、decode/返回、全部完整修复/扩时/robust seed/最终几何优化及必要硬检查。模型/预热/捕获费用单列，官方评分和绘图在计时外；正式计时全部结束后才绘图，未用批量摊销冒充单例。旧完整自适应预算没有额外增加。','',
        '单线程核验：所有BLAS/OpenMP线程池为1，PyTorch intra/inter-op与Numba为1，每次Clarabel构造断言max_threads=1，HiGHS显式设置1；没有worker池或并发请求。独立求解副本只改线程设置，原源码与权重保护。五例预检5/5合法、0回退，图/普通输出差0。','',
        '全部逐例坐标/trace保存在各臂terminal目录，同100图各自位于gallery/index.html。原rawdev2000与历史草稿指标继续复用，没有重复完整raw推理。各臂100图SHA已全部核验；硬合法逐例核验与加权聚合复核均通过。','',
        '详细数值：[结果](results.json) · [新旧保存结果对照](saved_comparison.json) · [原计划](../../fullrepair_singlethread_plan.json)。本报告是自适应开发面板上的补测，不是独立泛化或正式比赛成绩。']
    data=dict(status='audited',timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),comparisons=comparisons,details=details,no_new_inference_or_solver_calls=True,test_read=False)
    (folder/'saved_comparison.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');(folder/'report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(json.dumps(comparisons,indent=2))

if __name__=='__main__':main()
