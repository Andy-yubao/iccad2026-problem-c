"""Frozen public-only selection rule; no reference fields or true Q may enter."""
import math

def estimate(row,area_scale,draft_wire,nsoft):
    quality=1+.5*max(0.,row['area']/area_scale-1)+.5*max(0.,row['wire']/max(draft_wire,1e-12)-1)
    return quality*math.exp(2*sum(row['soft_violations'])/max(1,nsoft))

def select(rows,area_scale,draft_wire,nsoft):
    return min(rows,key=lambda r:estimate(r,area_scale,draft_wire,nsoft)+.01*r['distance']['D'])
