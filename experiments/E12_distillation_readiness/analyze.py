"""Analyze saved evidence only; no solver, reference data load or training."""
import argparse,json
from pathlib import Path
from .io import HERE,HISTORY,np,save

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def panel(path):return [read(p) for p in sorted((path/'cases').glob('*.json'))]
def gate(path):
    rows=panel(path);teachers=np.array([r['teacher_q'] for r in rows]);result=read(path/'results.json');out={}
    for name,summary in result['summary'].items():
        pairs=[(r,t) for r in rows for t in r['trials'] if t['arm']==name]
        deltas=np.array([t['metrics']['cost']-r['teacher_q'] for r,t in pairs]);ideal=name.endswith('geo0')
        checks={'full_panel':len(pairs)==100,'hard':summary['hard_legal_count']==100,
          'mean_q':summary['Q_mean']<=teachers.mean()+(.002 if ideal else .005),
          'max_q':summary['Q_max']<=teachers.max()+(.05 if ideal else .15),
          'mean_repair_time':summary['worker_seconds_mean']<=.35}
        if ideal:checks['p95_q']=summary['Q_p95']<=np.quantile(teachers,.95)+.01
        else:checks['fallback']=summary['row_retained_count']<=1
        checks={k:bool(v) for k,v in checks.items()}
        out[name]={'pass':all(checks.values()),'checks':checks,'summary':summary,
          'paired_delta_max':float(deltas.max()),'paired_delta_p95':float(np.quantile(deltas,.95)),
          'paired_delta_min':float(deltas.min()),
          'official_and_independent_all_legal':all(t['independent_hard_checks']['is_feasible'] and t['metrics']['official_is_feasible_before_nonnegative_check'] for r,t in pairs)}
    return {'path':str(path.relative_to(HERE)),'teacher_mean':float(teachers.mean()),
      'teacher_p95':float(np.quantile(teachers,.95)),'teacher_max':float(teachers.max()),'arms':out}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--confirmation',default='20261018_polished_confirmation100_seed0_r2');args=ap.parse_args()
    dev=gate(HERE/'runs/20261018_polished_dev100_seed0');confirm=gate(HERE/'runs'/args.confirmation)
    draftpath=HISTORY/'20261018_stageA_dev100_seed0/drafts.jsonl';teacherpath=HISTORY/'20261018_robust_dev100_seed0/cases.jsonl'
    old={r['case_id']:r for r in map(json.loads,draftpath.read_text(encoding='utf-8').splitlines())}
    errors=[]
    for r in map(json.loads,teacherpath.read_text(encoding='utf-8').splitlines()):
        x=np.asarray(r['variants']['conic_robust']['final']['positions']);d=np.asarray(old[r['case_id']]['draft'])
        e=((d[:,:2]+d[:,2:]/2)-(x[:,:2]+x[:,2:]/2))/np.sqrt(x[:,2:].prod(1))[:,None]
        logerror=np.log(d[:,2]/d[:,3])-np.log(x[:,2]/x[:,3])
        changed=(np.linalg.norm(e,axis=1)>1e-6)|(np.abs(logerror)>1e-6)
        errors.append({'case_id':r['case_id'],'center_local_rmse_all':float(np.sqrt((e**2).mean())),
          'logratio_rmse_all':float(np.sqrt((logerror**2).mean())),
          'center_local_rmse_changed':float(np.sqrt((e[changed]**2).mean())) if changed.any() else 0.})
    ledger=[]
    for p in sorted((HERE/'runs').glob('*/results.json')):
        r=read(p)
        if 'elapsed_seconds' in r:ledger.append({'run':p.parent.name,'seconds':r['elapsed_seconds']})
    state={'dev':dev,'confirmation':confirm,'old_model_teacher_errors':errors,
      'ledger':ledger,'total_compute_seconds':20.+sum(r['seconds'] for r in ledger),
      'legacy_single_case_reserve_seconds':20.,'noise_is_trained_model':False,'training_started':False}
    out=HERE/'runs/20261018_readiness_report';out.mkdir(exist_ok=True);save(out/'analysis.json',state)
    lines=['# E12 蒸馏准备证据','',
      '这些是教师信息直接输入的接口/误差诊断。未训练学生；两组均是既有开发面板，不是独立测试或正式比赛分数。',
      '', '| 面板 | 输入 | 平均Q | 相对教师ΔQ | 硬合法 | 回退 | 修复+通信秒/例 | 加0.10秒学生预留 | 门槛 |',
      '|---|---|---:|---:|---:|---:|---:|---:|---|']
    for label,pan in [('dev100',dev),('confirmation100',confirm)]:
        for name,arm in pan['arms'].items():
            s=arm['summary'];lines.append(f"| {label} | {name} | {s['Q_mean']:.6f} | {s['Q_delta_mean_vs_teacher']:+.6f} | {s['hard_legal_count']}/{s['count']} | {s['row_retained_count']} | {s['worker_seconds_mean']:.4f} | {s['mean_plus_010_network_reserve']:.4f} | {'通过' if arm['pass'] else '未通过'} |")
    lines+=['','所有组均获得教师实际接触森林、已满足边界和MIB形状类别。geo仅表示全体分离方向由草图几何推导；teacher另提供教师分离。学生将来必须自己预测结构信息。',
      '', '原模型距教师的中心误差（按教师块尺度归一化、含公开已知块）每例RMSE的中位数：'+f"{np.median([r['center_local_rmse_all'] for r in errors]):.4f}"+'；log长宽比RMSE中位数：'+f"{np.median([r['logratio_rmse_all'] for r in errors]):.4f}"+'。2%噪声不等于当前模型已达到的精度。',
      '', '完整逐例尾部、配对差值、全部合法标记及累计计算账本见analysis.json。门槛沿用README预注册值；5%噪声作为破坏性边界诊断，不能把它的失败删掉。',
      '', '## 重叠与留白', '', '重叠率是矩形面积和减去精确并集面积后除以面积和；内部留白是bbox内未被并集覆盖的比例。它们是草图诊断，不替代Q。低重叠仍可能因错误拓扑、接触/MIB失配而失败，旧oracle运行提供了反例。',
      '', '| 输入 | 草图重叠面积比例 | bbox内部留白 | 最终ΔQ |', '|---|---:|---:|---:|']
    for name,arm in dev['arms'].items():
        s=arm['summary'];lines.append(f"| {name} | {s['draft_overlap_excess_mean']:.6f} | {s['draft_whitespace_mean']:.6f} | {s['Q_delta_mean_vs_teacher']:+.6f} |")
    lines+=['', '## 计时边界', '', '修复计时含进程通信、问题准备和候选池处理，不含教师标签提取/官方评分、数据或模型加载、预热；另显式预留学生网络0.10秒。归一化/结构输出解码主要在probe主进程，目前未并入worker计时；需用真实学生端到端计时最终确认。confirmation原缓存仅有9例推理时间，缺失值记null，不补造；目标用预留时长核对。',
      '', '## 计算账本', '', f"已记录计算合计 {state['total_compute_seconds']:.2f} 秒（含20秒早期单例排错保守预留）；上限1100秒。标签schema的只读缓存检查另在工作记录披露。", '', '| 运行 | 秒 |','|---|---:|']
    lines.extend(f"| {r['run']} | {r['seconds']:.2f} |" for r in ledger)
    (out/'analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['lower_geo0','lower_geo005','lower_geo02','lower_geo05'];noise=[0,.5,2,5]
    fig,axs=plt.subplots(1,2,figsize=(10,3.6))
    axs[0].plot(noise,[dev['arms'][n]['summary']['Q_delta_mean_vs_teacher'] for n in names],'o-');axs[0].axhline(.005,color='red',ls='--',lw=1)
    axs[0].set(xlabel='Geometry noise (%)',ylabel='Mean Q minus teacher',title='All 100 cases, including fallbacks')
    axs[1].plot(noise,[dev['arms'][n]['summary']['worker_seconds_mean']+.1 for n in names],'o-');axs[1].axhline(.5,color='red',ls='--',lw=1)
    axs[1].set(xlabel='Geometry noise (%)',ylabel='Repair + 0.10s network reserve (s)',ylim=(0,.55),title='Measured repair; network is a reserve')
    fig.tight_layout();fig.savefig(out/'readiness.png',dpi=180);plt.close(fig)
    print(json.dumps({'dev':{k:v['pass'] for k,v in dev['arms'].items()},'confirmation':{k:v['pass'] for k,v in confirm['arms'].items()},'compute_seconds':state['total_compute_seconds']},indent=2))

if __name__=='__main__':main()
