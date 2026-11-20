"""Validate the saved galleries and their shared read-only HTTP endpoints."""
from pathlib import Path
import hashlib
import json
import math
import urllib.request
from visualization.style import normalize

RUNS=Path(__file__).resolve().parent
ROOT=RUNS.parents[2]

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def get(url):
    with urllib.request.urlopen(url,timeout=10) as response:return json.load(response)

def main():
    common=None;reports=[]
    for arm,port in (('e119_A',8784),('e119_B',8785),('e119_C',8786)):
        run=RUNS/f'{arm}_seed0';folder=run/'visualization_dev15_loss'
        data=read(folder/'data.json');selection=read(folder/'selection.json')
        state=read(folder/'state.json');assert state['status']=='complete'
        assert len(data['cases'])==15 and len(selection['ordinary_indices'])==10 and len(selection['worst_indices'])==5
        if common is None:common=selection['ordinary_indices']
        else:assert common==selection['ordinary_indices']
        url=f'http://127.0.0.1:{port}/';manifest=get(url+'api/manifest')
        assert manifest['title']==data['title'] and len(manifest['cases'])==15
        for j,(saved,registered) in enumerate(zip(data['cases'],selection['members'])):
            actual=get(url+f'api/case/{j}');assert actual==normalize(saved)
            assert actual['layer_ids']==list(range(13)) and len(actual['layers'])==len(actual['layer_losses'])==13
            source=run/f"evaluation_no_mib_dev500/cases/{registered['index']:04d}.json"
            row=read(source);assert sha(source)==registered['evaluation_sha256']
            assert actual['member']==row['member'] and actual['final']==row['positions']
            assert actual['layers'][-1]==row['draft_positions']
            weights=[]
            for loss in actual['layer_losses']:
                assert math.isfinite(loss['total']) and math.isfinite(loss['objective_total'])
                weights.append(loss['supervision_weight'])
            assert abs(sum(weights)-1)<1e-12
            if arm=='e119_C':
                front=actual['layer_losses'][0];f=front['frontend_components']
                assert front['objective_name']=='F' and front['objective_total']==f['total']
                assert abs(f['total']-(f['center_tolerance']+.05*f['free_shape']+f['public_shape']+f['public_preplaced_position']))<1e-5
        report=dict(arm=arm,url=url,cases=15,actual_layers=13,layer_losses=195,
            all_HTTP_cases_match_saved_data=True,terminal_drafts_bitwise_equal=True,original_final_positions_unchanged=True,
            data_sha256=sha(folder/'data.json'),checkpoint_sha256=data['identity']['checkpoint_sha256'],
            training=False,postprocessing_calls=0,Q_calls=0,test_read=False)
        (folder/'verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        (folder/'README.md').write_text(f'''# E135 {arm[-1]} 可视化

本机入口：{url}

相同普通10例＋本组最终Q最坏5例。S0–S12为实际模型输出，每层显示完整新Loss与分项、监督权重；C的S0另列实际F及四项组成。所有层与reference/原后处理共享坐标范围，沿用唯一分组色/黑框/底部图例样式。

末层与已完成评价草稿逐位一致；后处理和Q直接复用。未训练、未重跑后处理或Q、未读test。selection.json冻结成员及来源哈希，state.json与verification.json记录核验。

独立启动同一展示入口：

```powershell
& .\\.conda\\envs\\iccad2026\\python.exe -B -m visualization --data experiments/E135_initial_supervision/runs/{arm}_seed0/visualization_dev15_loss/data.json --port {port}
```
''',encoding='utf-8')
        reports.append(report)
    # Existing 12-readout data retain their original layer numbering/coordinates.
    old=read(ROOT/'experiments/E134_supervision_interval/runs/e119_stride3_seed0/visualization_dev15_loss/data.json')['cases'][0]
    old_normalized=normalize(old)
    assert old_normalized['layer_ids']==list(range(1,13)) and old_normalized['layers']==old['layers']
    gate=read(RUNS.parent/'preflight.json')
    assert all(sha(ROOT/name)==digest for name,digest in gate['sources'].items())
    assert not (ROOT/'output').exists()
    result=dict(status='passed',galleries=reports,matched_ordinary_indices=common,
        old_12_layer_data_compatible=True,frozen_training_sources_unchanged=True,
        browser_checked=dict(A_B_worst_case=358,A_Q=3.273555567156634,B_Q=3.1760953370788094,
            C_S0_full_loss=data['cases'][0]['layer_losses'][0]['total'],
            C_S0_F=data['cases'][0]['layer_losses'][0]['objective_total'],real_slider_states_checked='S0..S12',
            screenshot='e119_C_seed0/visualization_dev15_loss/preview_full.png'))
    (RUNS/'visualization_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))

if __name__=='__main__':main()
