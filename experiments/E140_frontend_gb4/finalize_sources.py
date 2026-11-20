"""Adapt preparation validation and E139 comparison reporting; never start workers."""
import ast
from pathlib import Path

HERE=Path(__file__).resolve().parent


def replace_function(text,name,replacement):
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)
    lines=text.splitlines(keepends=True)
    return ''.join(lines[:node.lineno-1])+replacement.rstrip()+'\n'+''.join(lines[node.end_lineno:])


def main():
    path=HERE/'preflight.py';text=path.read_text(encoding='utf-8')
    text=text.replace("PARENT = ROOT/'experiments/E134_supervision_interval'","PARENT = ROOT/'experiments/E139_delayed_overlap'")
    start="        previous = next(r for r in parent_gate['arms'] if r['arm'] == 'e119_stride3')\n        if digest != previous['initial_state_sha256']:"
    assert start in text
    text=text.replace(start,"        if digest != parent_gate['initial_state_sha256']:")
    text=replace_function(text,'inherited_engine', '''def inherited_engine():
    provenance=read(HERE/'provenance.json');gate=read(PARENT/'preflight.json')
    if sha(PARENT/'preflight.json')!=provenance['parent_preflight_sha256']:
        raise ValueError('Parent preflight changed')
    if gate['status']!='passed' or not gate['ctrl_c_resume_verified']:
        raise ValueError('Missing parent continuation proof')
    proof=gate['continuation_proof']
    if not all(proof[name] for name in ('training_resume_verified','queue_resume_verified','evaluation_prefix_verified')):
        raise ValueError('Inherited E139 continuation proof missing')
    checked=[]
    for name,record in provenance['copied_originals'].items():
        parent=ROOT/record['parent']
        if sha(parent)!=record['sha256'] or gate['sources'].get(record['parent'])!=record['sha256']:
            raise ValueError('Parent verified source drift: '+name)
        if name=='preflight.py':continue
        expected=parent.read_text(encoding='utf-8')
        for old,new in provenance['identity_mapping'].items():expected=expected.replace(old,new)
        if name=='loss.py':
            expected=expected.replace("    if layer in NO_OVERLAP_LAYERS:\\n        # Omit",
                "    if layer in NO_OVERLAP_LAYERS:\\n        out['grouping_weighted'] = 4.0*out['grouping_weighted']\\n        out['boundary_weighted'] = 4.0*out['boundary_weighted']\\n        # Omit",1)
        actual=(HERE/name).read_text(encoding='utf-8')
        if name=='run_queue.py':
            def control_tree(source):
                tree=ast.parse(source)
                tree.body=[n for n in tree.body if not isinstance(n,ast.FunctionDef) or n.name!='write_comparison']
                return ast.dump(tree,include_attributes=False)
            equal=control_tree(actual)==control_tree(expected)
        else:equal=actual==expected
        if not equal:raise ValueError('Unregistered numerical/controller change: '+name)
        checked.append(name)
    return dict(mode='Inherited E139/E134 continuation proof with source identity/report substitutions and registered early G/B multipliers only',
        parent_preflight_sha256=provenance['parent_preflight_sha256'],checked_files=checked,
        training_resume_verified=True,queue_resume_verified=True,evaluation_prefix_verified=True,new_interruption_simulations=0)
''')
    text=replace_function(text,'objective_checks', '''def objective_checks(output,batch,arm):
    from experiments.E139_delayed_overlap.loss import base_per_case as parent_base,per_case as parent_per_case,supervised_per_case as parent_supervised
    weights=layer_weights(arm);states=output.detach().clone().requires_grad_(True)
    actual=per_case(states,batch,arm);expected=0;extra=0
    for layer,weight in weights.items():
        full=parent_base(states[layer-1],batch)
        common=base_per_case(states[layer-1],batch)
        own=supervised_per_case(states[layer-1],batch,layer)
        old=parent_supervised(states[layer-1],batch,layer)
        for name in full:torch.testing.assert_close(common[name],full[name],rtol=0,atol=0)
        if layer in NO_OVERLAP_LAYERS:
            retained=sum(full[name]*(4 if name in ('grouping_weighted','boundary_weighted') else 1)
                for name in TOTAL_COMPONENTS if name!='overlap_weighted')
            extra=extra+weight*3*(full['grouping_weighted']+full['boundary_weighted'])
            assert torch.count_nonzero(own['overlap_weighted'])==0
        else:
            retained=full['total']
            for name in old:torch.testing.assert_close(own[name],old[name],rtol=0,atol=0)
        torch.testing.assert_close(own['total'],retained,rtol=0,atol=0)
        for name in TOTAL_COMPONENTS:
            if name=='overlap_weighted' and layer in NO_OVERLAP_LAYERS:continue
            factor=4 if layer in NO_OVERLAP_LAYERS and name in ('grouping_weighted','boundary_weighted') else 1
            torch.testing.assert_close(own[name],factor*full[name],rtol=0,atol=0)
        expected=expected+weight*retained
    torch.testing.assert_close(actual['total'],expected,rtol=0,atol=0)
    parent_total=parent_per_case(states,batch,'e119_delayed_overlap')['total']
    torch.testing.assert_close(actual['total']-parent_total,extra,rtol=1e-4,atol=2e-5)
    grad=torch.autograd.grad(actual['total'].sum(),states,retain_graph=True)[0]
    expected_grad=torch.autograd.grad(expected.sum(),states,retain_graph=True)[0]
    torch.testing.assert_close(grad,expected_grad,rtol=1e-6,atol=1e-6)
    parent_grad=torch.autograd.grad(parent_total.sum(),states)[0]
    extra_grad=torch.autograd.grad(extra.sum(),states)[0]
    torch.testing.assert_close(grad-parent_grad,extra_grad,rtol=1e-4,atol=1e-6)
    for layer in (1,3,4,5,7,9,11):assert torch.count_nonzero(grad[layer-1])==0
    torch.testing.assert_close(actual['final_total'],parent_base(states[11],batch)['total'],rtol=0,atol=0)
    collision=torch.zeros_like(states[0]);collision[...,0]=torch.arange(collision.shape[1],device=collision.device)*.001
    full=parent_base(collision,batch)
    assert bool((full['overlap_weighted']>0).all())
    for layer in NO_OVERLAP_LAYERS:
        own=supervised_per_case(collision,batch,layer)
        correct=sum(full[name]*(4 if name in ('grouping_weighted','boundary_weighted') else 1)
            for name in TOTAL_COMPONENTS if name!='overlap_weighted')
        torch.testing.assert_close(own['total'],correct,rtol=0,atol=0)
        assert torch.count_nonzero(own['overlap_weighted'])==0
        for name in ('penetration','module_load_mean','module_max_excess10','overlap_total'):
            torch.testing.assert_close(own[name],full[name],rtol=0,atol=0)
    return dict(common_parent_full_loss_exact=True,early_grouping_boundary_exact_x4=True,
        all_other_early_terms_exact=True,late_S8_S10_S12_all_components_exact=True,
        aggregate_value_and_gradient_verified=True,E139_delta_formula_and_gradient_verified=True,
        unsupervised_readout_direct_gradients_zero=True,synthetic_nonzero_overlap_still_omitted=True)
''')
    path.write_text(text,encoding='utf-8')
    path=HERE/'run_queue.py';text=path.read_text(encoding='utf-8')
    text=replace_function(text,'write_comparison', '''def write_comparison():
    plan=read(OWNER/'plan.json');control=plan['comparison_controls'][0]
    if sha(ROOT/control['result'])!=control['results_sha256'] or sha(ROOT/control['checkpoint'])!=control['checkpoint_sha256']:
        raise ValueError('Frozen E139 control changed')
    rows=[dict(read(ROOT/control['result']),arm='E139_existing_best42')]
    paths=[(ROOT/control['result']).parent]
    for arm in ARMS:
        path=OWNER/'runs'/f'{arm}_seed0/evaluation_no_mib_dev500/results.json'
        result=read(path)
        if result['status']!='complete' or result['count']!=500:raise ValueError('Evaluation incomplete')
        rows.append(dict(result,arm=arm));paths.append(path.parent)
    panel=None
    for row,path in zip(rows,paths):
        records=[read(path/f'cases/{i:04d}.json') for i in range(500)]
        members=[r['member'] for r in records]
        if panel is None:panel=members
        elif panel!=members:raise ValueError('Paired evaluation panel differs')
        row.update(grouping_violations_total=sum(r['metrics']['grouping_violations'] for r in records),
            boundary_violations_total=sum(r['metrics']['boundary_violations'] for r in records),
            all_soft_satisfied_cases=sum(r['metrics']['total_soft_violations']==0 for r in records),
            grouping_satisfied_cases=sum(r['metrics']['grouping_violations']==0 for r in records),
            boundary_satisfied_cases=sum(r['metrics']['boundary_violations']==0 for r in records))
    save(OWNER/'comparison.json',dict(cells=rows,only_intervention='S2/S6 G/B x4; E139 otherwise exact',
        note='Frozen E139 control reused. Single seed and adaptive stopping limit inference. Different own-Loss values not ranked. Boundary contradictions require separate evidence.'))
    text='# E140 early G/B x4 versus E139\\n\\nOne changed factor: S2/S6 grouping and boundary coefficients x4. E139 best42 reused without retraining.\\n\\n'
    text+='|Model|Mixed Q|Hard legal|Grouping violations|Boundary violations|All soft satisfied|Mean/P95 ms|Common final Loss|\\n|---|---:|---:|---:|---:|---:|---:|---:|\\n'
    for r in rows:
        text+=f"|{r['arm']}|{r['mixed_Q']:.6f}|{r['hard_legal']}/500|{r['grouping_violations_total']}|{r['boundary_violations_total']}|{r['all_soft_satisfied_cases']}/500|{r['mean_complete_seconds']*1000:.1f}/{r['p95_complete_seconds']*1000:.1f}|{r['common_final_new_dev_loss']:.6f}|\\n"
    (OWNER/'comparison.md').write_text(text,encoding='utf-8')
''')
    path.write_text(text,encoding='utf-8')


if __name__=='__main__':main()
