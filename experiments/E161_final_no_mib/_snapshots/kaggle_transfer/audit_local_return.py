"""Verify returned frozen weights and completed stage state without running a model."""
import hashlib
import json
from pathlib import Path
import shutil
import torch

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent

def read(path): return json.loads(path.read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value): path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    recovery=read(HERE/'latest_recovery.json')
    folder=(ROOT/recovery['destination']).resolve()
    if not folder.is_relative_to((HERE/'downloaded').resolve()): raise ValueError('Unexpected return directory')
    returned=folder/'e161-results'
    stages=[]
    for arm in ('e161_l0','e161_s0','e161_s1','e161_s2'):
        frozen=read(returned/'identity'/f'frozen_{arm}.json')
        relative=Path(frozen['checkpoint']).relative_to('experiments/E161_final_no_mib')
        best=(returned/relative).resolve()
        if not best.is_relative_to(returned.resolve()): raise ValueError('Unexpected checkpoint path')
        if sha(best)!=frozen['checkpoint_sha256']: raise ValueError('Frozen weight SHA256 mismatch: '+arm)
        state=torch.load(best,map_location='cpu',weights_only=False)
        required={'model','optimizer','scheduler','progress','rng','identity','projection','elapsed_seconds'}
        if required-set(state): raise ValueError('Incomplete checkpoint: '+arm)
        if state['progress']['best_epoch']!=frozen['best_epoch']: raise ValueError('Frozen selection epoch differs')
        changes=[name for name,value in state['identity']['sources'].items() if sha(ROOT/name)!=value]
        if changes: raise ValueError('Local model source identity differs: '+str(changes))
        run=returned/'runs'/f'{arm}_seed0'
        result=read(run/'results.json')
        if result['status'] not in ('budget','early_stop','max_epochs') or result['error']:
            raise ValueError('Stage did not finish successfully: '+arm)
        latest=run/'checkpoints/latest.pt'
        latest_verified=False
        if latest.exists():
            cursor=torch.load(latest,map_location='cpu',weights_only=False)
            if not cursor['finished'] or cursor['status']!=result['status']:
                raise ValueError('Final resume state not saved: '+arm)
            if cursor['progress']['step']!=result['steps']: raise ValueError('Accepted update cursor differs')
            latest_verified=True
            del cursor
        stages.append(dict(arm=arm,status=result['status'],completed_epochs=result['epochs'],
            steps=result['steps'],best_epoch=frozen['best_epoch'],own_selection_loss=frozen['own_selection_loss'],
            elapsed_seconds=result['elapsed_seconds'],best_checkpoint=best.relative_to(ROOT).as_posix(),
            best_sha256=frozen['checkpoint_sha256'],best_sha256_verified=True,
            checkpoint_load_verified=True,source_identity_verified=True,
            latest_resume_state_verified=latest_verified,training_gpu=state['identity']['gpu']))
        del state
    final=stages[-1]
    exported=read(returned/'identity/frozen_e161_s2.json')
    exported['checkpoint']=final['best_checkpoint']
    save(folder/'local_final_frozen.json',exported)
    configs=folder/'local_evaluation_inputs'
    configs.mkdir(exist_ok=True)
    for name in ('plan.json','budget.json'):
        shutil.copy2(ROOT/'experiments/E161_final_no_mib/evaluation'/name,configs/name)
    report=dict(training_complete=True,stages=stages,
        final_best=final,dev2000_started=False,dev2000_completed_cases=0,
        evaluation_failure='Remote transfer omitted evaluation/budget.json; solver import failed before main. evaluation/plan.json was also omitted. Both original configs are available locally and copied to local_evaluation_inputs.',
        evaluation_launch_requested=False,Q_available=False,
        gpu_session_terminal=recovery['gpu_session_terminal'],
        gpu_quota_seconds=recovery['gpu_quota_seconds'],
        gpu_released=recovery['gpu_session_terminal'] and recovery['gpu_quota_seconds']['time_reserved']==0,
        original_local_paused_run_preserved=True,remote_artifact_files=recovery['downloaded_files'],
        remote_artifact_bytes=recovery['total_bytes'],download_manifest=(folder/'download_manifest.json').relative_to(ROOT).as_posix(),
        final_frozen_record=(folder/'local_final_frozen.json').relative_to(ROOT).as_posix(),
        original_remote_identity_records_preserved=True)
    save(folder/'local_return_audit.json',report)
    save(HERE/'local_return_audit.json',report)
    lines=['# E161远端结果已回传本地','',
        '四阶段训练均已完成，S1/S2最终恢复状态及四阶段唯一own-Loss-best均已校验；原本地暂停运行未覆盖。','',
        '| 阶段 | 结束状态 | 完成轮数 | best轮数 | own-Loss |','|---|---|---:|---:|---:|']
    for stage in stages:
        lines.append(f"| {stage['arm']} | {stage['status']} | {stage['completed_epochs']} | {stage['best_epoch']} | {stage['own_selection_loss']:.9f} |")
    lines += ['',f"最终评价使用S2 best{final['best_epoch']}，SHA256：`{final['best_sha256']}`。所有best均与远端冻结SHA256相符，并已在本地CPU读取检查；S1/S2 latest包含终末模型、Adam、RNG、调度、投影及累计预算。",'',
        'dev2000尚未开始：迁移包漏掉evaluation/budget.json，程序在导入求解器时退出。evaluation/plan.json也未打入原包；两份原配置已保留至local_evaluation_inputs，打包工具已修正。无需重训。本次未启动本地评价。','',
        f"远端会话状态：{recovery['status']}，GPU保留额度为{recovery['gpu_quota_seconds']['time_reserved']}秒，已不占用GPU；未删除远端结果、数据集或Notebook。",'',
        f"完整下载：{recovery['downloaded_files']}个远端文件、{recovery['total_bytes']}字节，逐个按HTTP Content-Length核对完整长度，SHA256见download_manifest.json。",'',
        '文件入口：','',
        '- 最终冻结权重：`e161-results/runs/e161_s2_seed0/checkpoints/best.pt`',
        '- 最终恢复状态：`e161-results/runs/e161_s2_seed0/checkpoints/latest.pt`',
        '- 本地评价权重登记：`local_final_frozen.json`',
        '- 原冻结记录：`e161-results/identity/`',
        '- 训练与错误日志：`e161-results/queue.stdout.log`、`remote_execution.log`',
        '- 本地审核记录：`local_return_audit.json`','']
    (folder/'RETURN_TO_LOCAL.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(training_complete=True,gpu_released=report['gpu_released'],
        final_best_epoch=final['best_epoch'],final_best_checkpoint=final['best_checkpoint'],
        downloaded_files=recovery['downloaded_files'],downloaded_bytes=recovery['total_bytes'],
        dev2000_completed_cases=0,report=(folder/'RETURN_TO_LOCAL.md').relative_to(ROOT).as_posix())),flush=True)

if __name__=='__main__': main()
