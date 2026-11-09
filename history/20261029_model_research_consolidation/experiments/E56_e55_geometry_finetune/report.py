"""Read-only comparison with pre-training E55 and accepted E42 outputs."""
import json
from .io import ROOT, HERE, save
from .pipeline import OUTPUT
from .initialization import metadata, verify_protected


def main():
    def read(p):
        return json.loads(p.read_text(encoding='utf-8'))
    protected = verify_protected()
    train, progress, final = [read(OUTPUT/name) for name in ('results.json', 'progress.json', 'terminal/results.json')]
    start = progress['history'][0]
    best = next(r for r in progress['history'] if r['epoch'] == train['best_epoch'])
    comparisons = {}
    sources = {'E55': ROOT/'experiments/E55_reference_position_shape/runs/20261026_reference_position_shape_seed0',
               'E42': ROOT/'experiments/E42_joint_regularization/runs/20261023_rectangle_w96d12_ab_seed0'}
    assert len(final['records']) == 500
    for name, source in sources.items():
        old = read(source/'terminal/results.json')
        old_map = {r['index']: r for r in old['records']}
        deltas = []
        for row in final['records']:
            control = old_map[row['index']]
            assert row['member'] == control['member']
            deltas.append(row['metrics']['cost']-control['metrics']['cost'])
        comparisons[name] = {'summary': old['summary'], 'wins': sum(d < -1e-8 for d in deltas),
                             'ties': sum(abs(d) <= 1e-8 for d in deltas), 'losses': sum(d > 1e-8 for d in deltas)}
    result = {'training': train, 'initialization': metadata(), 'epoch0': start,
              'best_monitor': best, 'flash500': final['summary'], 'paired': comparisons,
              'pretraining_plus_posttraining_seconds': metadata()['pretraining_elapsed_seconds']+train['elapsed_seconds'],
              'protected_checkpoints_unchanged': protected, 'test_read': False}
    save(OUTPUT/'analysis.json', result)
    summary = final['summary']
    lines = ['# E56：从E55 best40出发，以E42八项Loss后训练', '',
             f'训练状态：{train["status"]}；本阶段{train["epochs"]}轮，best{train["best_epoch"]}。',
             f'同一固定八项Loss的dev500：起点{start["dev_monitor"]["total"]:.9f} → best {best["dev_monitor"]["total"]:.9f}。',
             f'flash500硬合法 {summary["hard_legal"]}/500，加权Q {summary["weighted_Q"]:.9f}，P95 {summary["p95_Q"]:.9f}，最大Q {summary["max_Q"]:.9f}，完整均时{summary["seconds_mean"]:.6f}秒。']
    for name, row in comparisons.items():
        lines.append(f'相对{name}原同500例：{row["wins"]}胜／{row["ties"]}平／{row["losses"]}负；其原加权Q {row["summary"]["weighted_Q"]:.9f}。')
    lines += ['', '[同100例reference／本轮draft／冻结E42 draft](draft_comparison100/index.html)。草稿原始质量、中心／形状保持与修复负担应优先于辅助Q判断。',
              '', '本轮仅改变训练目标并从E55权重进入新阶段；旧优化器／调度器不继承，E55 best/latest原文件及E42权重哈希复核不变。单seed、已暴露开发集，不是独立盲测。E42历史选模使用G+occupancy，本轮依照当前固定Loss规范使用G，不能称完整重现历史训练协议。']
    (OUTPUT/'analysis.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'training': train, 'flash500': summary}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
