"""Public-only floor-aware selector, calibrated globally on development evidence."""
import math

def choose(rows,base,area_scale,draft_wire,nsoft):
    # Dense boxes offer little useful area headroom; prioritize wire there.
    # No references, metrics dictionary, case ID or true Q may enter.
    dense=base is not None and base['area']/area_scale<=1.02
    area_anchor=area_scale*(1.02 if dense else 1.)
    wire_anchor=max(draft_wire,1e-12)*(.98 if dense else 1.)
    distance_weight=.003 if dense else 0.
    def score(row):
        quality=1+.5*max(0.,row['area']/area_anchor-1)+.5*max(0.,row['wire']/wire_anchor-1)
        return quality*math.exp(2*sum(row['soft_violations'])/max(1,nsoft))+distance_weight*row['distance']['D']
    chosen=min(rows,key=score)
    return chosen,{'regime':'dense' if dense else 'loose','area_anchor':area_anchor,'wire_anchor':wire_anchor,
        'distance_weight':distance_weight,'selected_public_score':score(chosen),'mathematical_Q_guarantee':False}
