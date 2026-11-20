"""Four-process same-case gallery, after all sequential timed solves."""
from pathlib import Path
import numpy as np

def render(args):
    from visualization.layouts import plot_case,geometry_details
    index,reference,constraints,parent,draft,final,destination=args
    variants={key:dict(positions=p,soft_details=geometry_details(p,constraints)) for key,p in (
        ('reference',reference),('parent',parent),('candidate',draft),('final',final))}
    plot_case(dict(case_id=f'dev500:{index}',panel='E98 reference075 posttrain / E98 depth repair',
        constraints=constraints,variants=variants),Path(destination),
        stages=(('reference','Reference'),('parent','Enhanced baseline L'),('candidate','Learned Loss L'),('final','Learned Loss final')),
        show_metrics=False)
    return index
