"""Summarize the two completed frozen evaluations without new inference."""
import json
import numpy as np
from ..io import HERE,ROOT,save,sha

def main():
    plan=json.loads((HERE/'evaluation/plan.json').read_text(encoding='utf-8'))
    models={}
    records={}
    for arm in ('terminal','dense'):
        out=HERE/'runs'/f'{arm}_seed0'/'evaluation_no_mib_dev500'
        models[arm]=json.loads((out/'results.json').read_text(encoding='utf-8'))
        assert models[arm]['status']=='complete'
        assert models[arm]['checkpoint_sha256']==plan['models'][arm]['checkpoint_sha256']
        assert sha(ROOT/plan['models'][arm]['checkpoint'])==models[arm]['checkpoint_sha256']
        records[arm]=[json.loads((out/f'cases/{i:04d}.json').read_text(encoding='utf-8')) for i in range(500)]
    a,b=models['terminal'],models['dense']
    delta=np.array([y['metrics']['cost']-x['metrics']['cost'] for x,y in zip(records['terminal'],records['dense'])])
    assert all(x['member']==y['member'] for x,y in zip(records['terminal'],records['dense']))
    result=dict(status='complete',models=models,
        dense_minus_terminal={k:b[k]-a[k] for k in ('weighted_Q','mean_Q','mixed_Q','common_final_dev_loss',
            'raw_repeated_coverage','raw_bbox_over_area','mean_complete_seconds','p95_complete_seconds')},
        paired_dense_Q_better=int((delta < -1e-12).sum()),paired_equal=int((np.abs(delta)<=1e-12).sum()),
        paired_dense_Q_worse=int((delta>1e-12).sum()),
        parameters_per_model=612307,test_read=False,selection_frozen_before_Q=True,
        training_paused=True,staged_model_updates=0,
        scope='Single seed, fixed repeatedly monitored parent dev500, original areas/no MIB/all-module draft; not historical E74/E105/A panel',
        caveat='Own objectives and training amounts differ; dense vs terminal does not answer whether layer4/12 supervision is better')
    save(HERE/'evaluation/comparison.json',result)
    rows=[('Best epoch','best_epoch'),('Train epochs','training_epochs'),('Train seconds','training_seconds'),
        ('Parameters','parameters'),('Common final dev Loss','common_final_dev_loss'),('Weighted Q','weighted_Q'),
        ('Mean Q','mean_Q'),('Mixed Q','mixed_Q'),('Raw hard legal','raw_hard_legal'),('Final hard legal','hard_legal'),
        ('Fallbacks','fallbacks'),('Raw repeated coverage','raw_repeated_coverage'),('Raw bbox / area','raw_bbox_over_area'),
        ('Mean complete seconds','mean_complete_seconds'),('P95 complete seconds','p95_complete_seconds'),
        ('Max complete seconds','max_complete_seconds'),('Cases over250ms','over250ms')]
    text='# E114：两个已完成模型的固定dev500验证\n\n'
    text+='训练队列已用户暂停，staged仍为0更新。两组唯一own-Loss best先冻结再评价；原始面积/无MIB/全模块预测、同500例和同固定深度后处理，测试未读。\n\n'
    text+='|指标|末层监督|密集监督|\n|---|---:|---:|\n'
    for label,key in rows:
        text+=f'|{label}|{a[key]:.9g}|{b[key]:.9g}|\n'
    text+=f'\n密集监督逐例Q胜/平/负：{result["paired_dense_Q_better"]}/{result["paired_equal"]}/{result["paired_dense_Q_worse"]}。\n\n'
    text+='两组各612307参数；共同末层Loss可比较，各自训练总Loss不可直接比较。Q为不计MIB、中性时间的本地研究分数，越低越好，不是官方比赛成绩。完整时间包含公开输入复制、特征、传输、网络、全部修复/回退及独立硬检查；加载、初始化、评分、诊断几何、磁盘及画图单列/排除。跨运行延迟只作实测描述。\n\n'
    text+='这是单seed开发面板结果，不能作为独立泛化或稳定排序证明；不同own-Loss选模及实际更新量也会影响结果。尚无第4/12层模型，不能据此判断稀疏监督优劣。历史E74/E105/A数据/任务不同，不能直接排名。\n\n'
    text+='[同五例逐层查看](../../../visualization/E114_supervision/20261107_two_models/index.html) · [机器汇总](comparison.json)\n'
    (HERE/'evaluation/report.md').write_text(text,encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='models'},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
