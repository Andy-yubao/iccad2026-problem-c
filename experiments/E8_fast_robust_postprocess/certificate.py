"""Input-only necessary feasibility checks for fixed-size separation graphs."""
from collections import deque
import numpy as np


def infeasibility_certificate(problem, seed, edges):
    """Certify cycles or a required path that exceeds a public fixed anchor.

    Return None when undecided. All separation lengths are positive; propagating
    lower bounds from nonnegative coordinates and public preplaced coordinates
    is a necessary condition of the fixed-size LP, including contact variants.
    """
    n=problem.n
    tolerance=1e-7*max(1.,problem.scale)
    for axis in (0,1):
        outgoing=[[] for _ in range(n)]
        degree=[0]*n
        for i,j,a in edges:
            if a==axis:
                outgoing[i].append(j);degree[j]+=1
        ready=deque(i for i in range(n) if degree[i]==0)
        lower=np.zeros(n)
        lower[problem.preplaced]=problem.specs[problem.preplaced,axis]
        path=[[i] for i in range(n)]
        count=0
        while ready:
            i=ready.popleft();count+=1
            if problem.preplaced[i] and lower[i] > problem.specs[i,axis]+tolerance:
                return {'kind':'preplaced_path_exceeds_anchor','axis':axis,'path':path[i],
                        'required_coordinate':float(lower[i]),'public_coordinate':float(problem.specs[i,axis])}
            for j in outgoing[i]:
                bound=lower[i]+seed[i,axis+2]
                if bound>lower[j]:lower[j]=bound;path[j]=path[i]+[j]
                degree[j]-=1
                if degree[j]==0:ready.append(j)
        if count!=n:
            return {'kind':'positive_separation_cycle','axis':axis}
    return None
