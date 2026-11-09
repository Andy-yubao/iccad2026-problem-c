"""Resume raw dev2000 only; never call Q. Copy monitor with explicit valid-N slicing.
The frozen original passed padded zero rectangles to coverage, producing0/0.
Only coverage-report slicing is fixed; Loss, weights, model and decoding unchanged.
"""
import os
os.environ['E74_THREADS']='4'
import sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from experiments.E90_auxiliary_weight_placer.io import HERE,sha,save,exclusive
from experiments.E90_auxiliary_weight_placer.data import load_rows,collate
from experiments.E90_auxiliary_weight_placer.model import GeometryModel,decode
from experiments.E90_auxiliary_weight_placer.batches import inputs_only
from experiments.E90_auxiliary_weight_placer.loss import per_case
from experiments.E90_auxiliary_weight_placer.diagnostics import geometry_checks,distribution
from experiments.E90_auxiliary_weight_placer.train import seed_all,selection
from experiments.E90_auxiliary_weight_placer.loss_config import load_coefficients
OUTPUT=HERE/'runs/20261031_aux3_seed0'

@torch.no_grad()
def monitor_losses(model, rows, device, arm, epoch, batch_size, *, should_stop=lambda: False, start_epoch=6, return_records=False):
    model.eval(); sums = {}; hard_legal = 0; inference_seconds = 0.; diagnostics = []
    for start in range(0, len(rows), batch_size):
        if should_stop():
            return None
        selected = [rows[i] for i in range(start, min(start+batch_size, len(rows)))]
        if str(device).startswith('cuda'): torch.cuda.synchronize(device)
        started = time.perf_counter()
        graph, targets = collate(selected, device)
        state = model(inputs_only(graph))
        positions = (decode(state, graph)*graph['scale'][:, None, None]).cpu().double().numpy()
        if str(device).startswith('cuda'): torch.cuda.synchronize(device)
        inference_seconds += time.perf_counter()-started
        per_layout = per_case(state, graph, arm)
        values = {key: value.mean() for key, value in per_layout.items()}
        diagnostic_keys = tuple(per_layout)
        loss_records = torch.stack([per_layout[k] for k in diagnostic_keys], -1).cpu().tolist()
        for row, predicted, row_losses in zip(selected, positions, loss_records) if return_records else ():
            _, checks, diagnostic = geometry_checks(predicted, row['case'].solver_inputs)
            from experiments.E90_auxiliary_weight_placer.metrics import module_overlap
            n=int(row['case'].solver_inputs[0])
            assert n==int(row['graph']['n'])
            assert np.all(predicted[n:]==0), 'Unexpected nonzero padding'
            diagnostic.update(module_overlap(predicted[:n]))
            diagnostic.update({'loss_'+k: value for k, value in zip(diagnostic_keys, row_losses)})
            hard_legal += int(checks['is_feasible'])
            diagnostics.append({'member': list(row['member']), **diagnostic})
        keys=list(values)
        scalars=torch.stack([values[k].detach() for k in keys]).cpu().tolist()
        for key, value in zip(keys,scalars): sums[key] = sums.get(key, 0.) + value*len(selected)
    result = {key: value/len(rows) for key, value in sums.items()}
    result.update(count=len(rows), selection_proxy=selection(result), raw_hard_legal=hard_legal if return_records else None,
                  raw_hard_legal_rate=hard_legal/len(rows) if return_records else None,
                  batched_inference_seconds_per_case=inference_seconds/len(rows))
    result['draft_diagnostics'] = distribution(diagnostics,[k for k,v in diagnostics[0].items() if isinstance(v,(int,float,bool))]) if diagnostics else {}
    if return_records:
        result['diagnostic_records'] = diagnostics
    return result


def main():
    with exclusive(OUTPUT.with_suffix('.lock')):
        identity=json.loads((OUTPUT/'frozen.json').read_text(encoding='utf-8'))
        from experiments.E90_auxiliary_weight_placer.io import sources
        assert sources()==identity['sources']
        assert sha(ROOT/identity['checkpoint'])==identity['checkpoint_sha256']
        assert sha(OUTPUT/'checkpoints/latest.pt')==identity['latest_checkpoint_sha256']
        terminal=OUTPUT/'terminal/results.json';terminal_sha=sha(terminal)
        assert len(json.loads(terminal.read_text(encoding='utf-8'))['records'])==500
        target=OUTPUT/'full_dev.json'
        if target.exists():
            assert json.loads(target.read_text(encoding='utf-8'))['checkpoint_sha256']==identity['checkpoint_sha256']
            print('Existing complete raw result retained');return
        def stopped():return (OUTPUT/'STOP').exists() or (HERE/'runs/STOP').exists()
        if stopped():print('STOP present');return
        signal.signal(signal.SIGINT,lambda *_:(OUTPUT/'STOP').touch())
        state=torch.load(ROOT/identity['checkpoint'],map_location='cpu',weights_only=False)
        load_coefficients(state['identity']['calibration']['path'])
        seed_all(state['identity']['seed'],4);device=state['identity']['device']
        model=GeometryModel(**state['model_config']).to(device).eval();model.load_state_dict(state['model'])
        rows=load_rows('dev');assert len(rows)==2000
        partial=OUTPUT/'full_dev_trimmed_partial.json'
        chunks=json.loads(partial.read_text(encoding='utf-8')) if partial.exists() else []
        cursor=sum(c['count'] for c in chunks)
        assert cursor<=2000
        save(OUTPUT/'raw_reporting_recovery.json',dict(status='running',pid=os.getpid(),
            error='original coverage reporting received zero-area padding; nonfinite JSON rejected',
            fix='Slice predictions to public N before exact module overlap coverage. Same original Loss and decoding.',
            original_monitor_sha256=sha(HERE/'train.py'),recovery_source_sha256=sha(__file__),
            checkpoint_sha256=identity['checkpoint_sha256'],terminal_sha256=terminal_sha,
            new_postprocessing_calls=0,test_read=False))
        save(HERE/'state.json',dict(status='validation_running',stage='raw_dev2000_padding_fix',pid=os.getpid(),
            best_epoch=identity['best_epoch'],terminal_complete=True,test_read=False))
        for start in range(cursor,len(rows),32):
            if stopped():print('Stopped; raw partial retained');return
            value=monitor_losses(model,rows[start:start+32],device,'simple',0.,32,return_records=True)
            assert value is not None
            chunks.append(value);save(partial,chunks)
            if len(chunks)%10==0:print('Raw validation:',min(start+32,len(rows)),'/2000',flush=True)
        records=[r for chunk in chunks for r in chunk['diagnostic_records']]
        assert len(records)==2000
        numeric=[k for k,v in chunks[0].items() if isinstance(v,(int,float)) and k not in ('count','raw_hard_legal','raw_hard_legal_rate')]
        values={k:sum(c[k]*c['count'] for c in chunks)/2000 for k in numeric}
        values.update(count=2000,raw_hard_legal=sum(c['raw_hard_legal'] for c in chunks))
        values['raw_hard_legal_rate']=values['raw_hard_legal']/2000
        values['draft_diagnostics']=distribution(records,[k for k,v in records[0].items() if isinstance(v,(int,float,bool))])
        assert sha(terminal)==terminal_sha
        assert sha(ROOT/identity['checkpoint'])==identity['checkpoint_sha256']
        assert sha(OUTPUT/'checkpoints/latest.pt')==identity['latest_checkpoint_sha256']
        save(target,dict(checkpoint_sha256=identity['checkpoint_sha256'],losses=values,records=records,count=2000,
            postprocessing_calls=0,test_read=False,reporting_fix='coverage restricted to each public real module count, excludes padded zeros',
            recovery_source_sha256=sha(__file__)))
        partial.unlink()
        recovery=json.loads((OUTPUT/'raw_reporting_recovery.json').read_text(encoding='utf-8'));recovery['status']='complete'
        save(OUTPUT/'raw_reporting_recovery.json',recovery)
        save(HERE/'state.json',dict(status='validation_complete',best_epoch=identity['best_epoch'],terminal_count=500,
            raw_count=2000,training_checkpoints_unchanged=True,terminal_Q_unchanged=True,test_read=False,
            next_stage='await review; no automatic training or Q rerun'))
        print('Raw dev2000 complete; Q and checkpoints unchanged',flush=True)

if __name__=='__main__':main()
