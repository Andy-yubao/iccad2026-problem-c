"""Read-only decomposition of already completed full-solver dev500 evidence."""
import json,math,re,datetime
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))

def main():
    output=HERE/'runs/20261102_round2_saved_diagnosis';output.mkdir(exist_ok=False)
    source=HERE/'runs/20261102_fullrepair_singlethread_dev500'
    official=(ROOT/'third_party/FloorSet/iccad2026contest/iccad2026_evaluate.py').read_text(encoding='utf-8')
    constants={k:float(re.search(r'^'+k+r'\s*=\s*([0-9.]+)',official,re.M)[1]) for k in ['ALPHA','BETA']}
    arms={};all_rows={}
    for name in ['E85','C1','C2','C3','C6','C7','C8']:
        rows=[read(p) for p in sorted((source/name/'terminal').glob('*.json'))];assert len(rows)==500
        weights=[math.exp((r['n']-120)/12) for r in rows];mass=sum(weights);weights=[w/mass for w in weights]
        terms=[]
        for r,w in zip(rows,weights):
            m=r['metrics'];wire=constants['ALPHA']*max(0,m['hpwl_gap']);area=constants['ALPHA']*max(0,m['area_gap'])
            soft=(1+wire+area)*math.expm1(constants['BETA']*m['violations_relative'])
            assert abs(1+wire+area+soft-m['cost'])<1e-10
            terms.append(dict(index=r['index'],member=r['member'],n=r['n'],cost=m['cost'],weight=w,weighted_excess=w*(m['cost']-1),wire=w*wire,area=w*area,soft=w*soft,boundary=m['boundary_violations'],grouping=m['grouping_violations'],mib=m['mib_violations'],hpwl_gap=m['hpwl_gap'],area_gap=m['area_gap'],violations_relative=m['violations_relative']))
        excess=sum(t['weighted_excess'] for t in terms);ranked=sorted(terms,key=lambda r:r['weighted_excess'],reverse=True)
        arms[name]=dict(weighted_Q=1+excess,excess_decomposition={k:sum(t[k] for t in terms) for k in ['wire','area','soft']},excess_shares={k:sum(t[k] for t in terms)/excess for k in ['wire','area','soft']},weighted_violation_counts={k:sum(t[k]*t['weight'] for t in terms) for k in ['boundary','grouping','mib']},cases_with_soft_violations=sum(r['metrics']['total_soft_violations']>0 for r in rows),top5_share=sum(t['weighted_excess'] for t in ranked[:5])/excess,top10_share=sum(t['weighted_excess'] for t in ranked[:10])/excess,top10=ranked[:10])
        all_rows[name]=terms
    comparisons={}
    for name in ['C1','C2','C3','C6','C7','C8']:
        delta=[dict(index=a['index'],n=a['n'],weighted_Q_delta=a['weight']*(a['cost']-b['cost']),boundary_change=a['boundary']-b['boundary'],grouping_change=a['grouping']-b['grouping'],wire_gap_change=a['hpwl_gap']-b['hpwl_gap'],area_gap_change=a['area_gap']-b['area_gap']) for a,b in zip(all_rows[name],all_rows['E85'])]
        comparisons[name]=dict(weighted_Q_change=sum(r['weighted_Q_delta'] for r in delta),worst10=sorted(delta,key=lambda r:r['weighted_Q_delta'],reverse=True)[:10],best10=sorted(delta,key=lambda r:r['weighted_Q_delta'])[:10])
    result=dict(status='complete',timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Read saved full500 metrics only; zero new inference, optimizer, solve or scorer calls',formula='cost-1=alpha*positive_wire_gap+alpha*positive_area_gap+(1+those)*(exp(beta*V)-1); neutral runtime, uncapped saved scores',constants=constants,arms=arms,comparisons=comparisons,test_read=False)
    (output/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:{x:v[x] for x in ['weighted_Q','excess_decomposition','excess_shares','weighted_violation_counts','cases_with_soft_violations','top5_share','top10_share']} for k,v in arms.items()},indent=2))

if __name__=='__main__':main()
