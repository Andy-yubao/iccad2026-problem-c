"""Register the user's new full-repair single-thread evaluation, no training."""
import datetime,hashlib,json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    target=HERE/'repair85_one_thread';assert not target.exists()
    edits=[('max_threads=2','max_threads=1'),("'threads':2","'threads':1"),("setOptionValue('threads',2)","setOptionValue('threads',1)")]
    manifest=[]
    for original in sorted((HERE/'repair85').rglob('*.py')):
        relative=original.relative_to(HERE/'repair85');copy=target/relative;copy.parent.mkdir(parents=True,exist_ok=True)
        text=original.read_text(encoding='utf-8');changes={a:text.count(a) for a,b in edits if a in text}
        for a,b in edits:text=text.replace(a,b)
        copy.write_text(text,encoding='utf-8')
        manifest.append(dict(original=original.relative_to(ROOT).as_posix(),original_sha256=sha(original),copy=copy.relative_to(ROOT).as_posix(),copy_sha256=sha(copy),thread_replacements=changes))
    original=HERE/'solver_relations.py';copy=HERE/'solver_relations_one_thread.py'
    copy.write_text(original.read_text(encoding='utf-8').replace('.repair85.','.repair85_one_thread.'),encoding='utf-8')
    manifest.append(dict(original=original.relative_to(ROOT).as_posix(),original_sha256=sha(original),copy=copy.relative_to(ROOT).as_posix(),copy_sha256=sha(copy),changes='imports only: .repair85. to .repair85_one_thread.'))
    arms=[]
    def arm(name,path,kind,prior,relations=False):
        checkpoint=ROOT/path;assert checkpoint.is_file()
        arms.append(dict(name=name,checkpoint=path,sha256=sha(checkpoint),model_kind=kind,fixed_relations=relations,previous_terminal=prior))
    arm('E85','experiments/E85_cooperative_space_refinement/runs/20261030_P74_F81_cooperative_seed0/checkpoints/best.pt','nn85.model','runs/20261101_slim3_dev500/E85_slim3')
    study=json.loads((HERE/'study.json').read_text(encoding='utf-8'))
    for entry in study['candidate_ledger']:
        i=entry['candidate']
        if i in (4,5):continue
        kind='student' if i<=3 else ('deep_geometry' if i==6 else 'normalized_geometry')
        arm(f'C{i}',f"experiments/E97_gpu_hybrid_solver/{entry['output']}/checkpoints/best.pt",kind,entry['output']+'/evaluation/terminal')
    c4=next(x for x in study['candidate_ledger'] if x['candidate']==4)
    arm('fixed_native_control',f"experiments/E97_gpu_hybrid_solver/{c4['output']}/checkpoints/untrained.pt",'relation_model',c4['output']+'/control/terminal',True)
    for i in (4,5):
        entry=next(x for x in study['candidate_ledger'] if x['candidate']==i)
        arm(f'C{i}_fixed_relations',f"experiments/E97_gpu_hybrid_solver/{entry['output']}/checkpoints/best.pt",'relation_model',entry['output']+'/evaluation/terminal',True)
    plan=dict(revision='E97.fullrepair_singlethread.v1',registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),authorization='2026-11-02用户明确要求用旧后处理器测试现有方案，每次只跑一线程；仅评价，不恢复已结束训练预算或自动唤醒。',
        panel='same frozen dev_monitor500, original order; exactly one complete request per arm/member; all existing own-Loss best fixed, no checkpoint selection or coefficient sweep',arms=arms,
        full_repair=dict(relaxed=False,mode=0,adaptive=True,fast_budget=False,late_extension=True,plateau=True,robust_seeds=True,certified=True,quick_project=True,quadratic_project=True,incumbent_project=True,snap=1e-4),
        relations='C4/C5 retain existing fixed learned relation interface at full budget, with required untrained fixed-native control. Under the original dynamic-native interface their physical drafts equal C1 and C1 result applies; no duplicate geometry solve claimed as a test of the discarded relation head.',
        threading='One sequential process, no pool/workers, torch intra/inter-op1, BLAS/OpenMP/Numba1, Clarabel/HiGHS explicit1, CUDA GPU inference allowed. All arms sequential, no overlapping timed requests.',
        measurement='Resident GPU batch1 complete public feature/padding/H2D/NN/internal geometry/decode/D2H, optional relation transfer/build, all old full CPU repair/rescue/polish and returned hard checks. Official scoring and gallery outside timer. Cold prepare/warmup/capture separately.',
        preflight='Original five train IDs only, E85 full repair with one-thread enforcement and graph/natural consistency; no dev probe or parameter changes',
        reuse='Previous raw2000 and source/reference/L/F geometry reused, no repeat raw2000. Same100 final galleries generated from new saved terminal results. Prior slim3 two-thread results are historical comparisons, not matched one-factor timing control.',
        stopping='STOP files or user stop at case boundaries; numeric/legality error preserves failure and stops, no automatic retry; resume only missing members with identical binding. No new training, no test, no automation.',
        output='runs/20261102_fullrepair_singlethread_dev500',copy_manifest=manifest,test_read=False)
    write(HERE/'fullrepair_singlethread_plan.json',plan)
    print('Registered',len(arms),'sequential arms;',len(manifest),'copied source records')

if __name__=='__main__':main()
