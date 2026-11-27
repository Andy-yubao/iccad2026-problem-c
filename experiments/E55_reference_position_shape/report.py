"""Report saved training/flash500 evidence without additional inference."""
import json
import numpy as np
from .io import ROOT, HERE, save
from .pipeline import OUTPUT


def main():
    def read(p):
        return json.loads(p.read_text(encoding='utf-8'))
    train = read(OUTPUT/'results.json')
    progress = read(OUTPUT/'progress.json')
    final = read(OUTPUT/'terminal/results.json')
    control = read(ROOT/'experiments/E42_joint_regularization/runs/20261023_rectangle_w96d12_ab_seed0/terminal/results.json')
    assert len(final['records']) == len(control['records']) == 500
    baseline = {r['index']: r for r in control['records']}
    delta, raw, repair = [], [], []
    for row in final['records']:
        old = baseline[row['index']]
        assert row['member'] == old['member']
        delta.append(row['metrics']['cost']-old['metrics']['cost'])
        if 'draft_diagnostics' in row:
            raw.append({'member': row['member'], **row['draft_diagnostics']})
        if 'mean_center_move_local' in row:
            repair.append(row['mean_center_move_local'])
    best = next(r for r in progress['history'] if r['epoch'] == train['best_epoch'])
    paired = {'wins': sum(d < -1e-8 for d in delta), 'ties': sum(abs(d) <= 1e-8 for d in delta),
              'losses': sum(d > 1e-8 for d in delta)}
    result = {'training': train, 'best_same_loss_dev500': best['dev_monitor'],
              'best_same_loss_train_monitor500': best['train_monitor'],
              'flash500': final['summary'], 'paired_E42_flash500': paired,
              'draft_diagnostics': raw, 'mean_repair_local': float(np.mean(repair)) if repair else None,
              'raw_hard_legal': sum(r.get('raw_hard_legal', False) for r in final['records']),
              'test_read': False, 'independent_blind_test': False}
    save(OUTPUT/'analysis.json', result)
    summary = final['summary']
    lines = ['# E55：只拟合 reference 位置与形状', '',
             '从头训练 seed0；固定 Loss = center_local + 0.25 × shape。全部额外几何惩罚关闭。', '',
             f'训练状态：{train["status"]}；{train["epochs"]}轮，best{train["best_epoch"]}。',
             f'best同口径 dev500 Loss：{best["dev_monitor"]["total"]:.9f}；train_monitor500：{best["train_monitor"]["total"]:.9f}。',
             f'原始draft硬合法：{result["raw_hard_legal"]}/500。',
             f'flash500最终硬合法：{summary["hard_legal"]}/500，加权Q {summary["weighted_Q"]:.9f}，P95 {summary["p95_Q"]:.9f}，最大 {summary["max_Q"]:.9f}。',
             f'完整均时 {summary["seconds_mean"]:.6f}秒；同E42 flash胜/平/负：{paired["wins"]}/{paired["ties"]}/{paired["losses"]}。', '',
             '草稿主判断应结合[同100例reference／本轮draft／E42 draft](draft_comparison100/index.html)和固定Loss。Q仅作辅助；原始几何及修复负担见analysis.json与terminal/results.json。', '',
             '本次为已暴露开发集的单种子机制实验，不是独立盲测；不得与E42历史总Loss横向比较。既有E42和pro均未修改或重新选择。']
    (OUTPUT/'analysis.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'training': train, 'flash500': summary, 'paired': paired}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
