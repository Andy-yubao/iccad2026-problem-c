"""Joint center/log-aspect search with exact-area representation and short LP polish."""
import time
import numpy as np
from scipy.optimize import minimize
from .frozen.packing import prepare_problem
from .frozen.feasible import Objective,row_control
from .frozen.global_refine import topology,contact_tree,solve_lp
from .frozen.canonical_geometry import canonicalize
from .frozen.solver import hard_valid
from .guard import repair_tiny_negative
from .reduction import reduce_edges
from .certificate import infeasibility_certificate


class Parameterization:
    def __init__(self,p,raw,shared=False,fixed_shapes=False):
        self.p=p;self.scale=p.scale;self.locked=p.fixed|p.preplaced
        self.area=p.areas.copy();self.area[self.locked]=p.specs[self.locked,2:].prod(1)
        self.known=np.log(raw[:,2]/raw[:,3]);self.index=np.full(p.n,-1,dtype=int)
        self.classes=[];unassigned=set(np.flatnonzero(~self.locked).tolist())
        if shared and not fixed_shapes:
            for group in np.unique(p.constraints[:,2]):
                if group<=0:continue
                ids=np.flatnonzero(p.constraints[:,2]==group)
                rigid=[int(i) for i in ids if self.locked[i]]
                for i in ids:
                    i=int(i)
                    if i not in unassigned:continue
                    compatible=[j for j in rigid if .990001*p.areas[i]<=p.specs[j,2:].prod()<=1.009999*p.areas[i]]
                    if compatible:
                        j=min(compatible,key=lambda j:abs(self.known[i]-self.known[j]))
                        self.area[i]=p.specs[j,2:].prod();self.known[i]=self.known[j];unassigned.remove(i)
                pending=[int(i) for i in ids if i in unassigned]
                while pending:
                    point=min(1.009999*p.areas[i] for i in pending)
                    members=[i for i in pending if .990001*p.areas[i]<=point<=1.009999*p.areas[i]]
                    lo=max(.990001*p.areas[i] for i in members);hi=min(1.009999*p.areas[i] for i in members)
                    common=float(np.clip(np.exp(np.log(p.areas[members]).mean()),lo,hi))
                    self.area[members]=common;self.classes.append(members)
                    for i in members:pending.remove(i);unassigned.remove(i)
        if not fixed_shapes:
            self.classes.extend([[i] for i in sorted(unassigned)])
        for k,ids in enumerate(self.classes):self.index[ids]=k
        self.area/=self.scale**2
        self.initial_centers=(raw[:,:2]+raw[:,2:]/2)/self.scale
        ratios=[float(self.known[ids].mean()) for ids in self.classes]
        self.initial=np.r_[self.initial_centers.ravel(),ratios]
        self.bounds=[]
        upper=max(3.,float(np.max(self.initial_centers))+1.)
        for i in range(p.n):
            for a in (0,1):
                coordinate=(p.specs[i,a]+p.specs[i,a+2]/2)/self.scale
                self.bounds.append((coordinate,coordinate) if p.preplaced[i] else (0.,upper))
        self.bounds.extend([(-np.log(25.),np.log(25.))]*len(self.classes))

    def decode(self,variables):
        p=self.p
        centers=variables[:2*p.n].reshape(p.n,2).copy()
        centers[p.preplaced]=(p.specs[p.preplaced,:2]+p.specs[p.preplaced,2:]/2)/self.scale
        ratios=self.known.copy();free=self.index>=0
        ratios[free]=variables[2*p.n+self.index[free]]
        wh=np.sqrt(self.area)[:,None]*np.exp(ratios[:,None]*np.array([.5,-.5]))
        wh[self.locked]=p.specs[self.locked,2:]/self.scale
        return centers,wh

    def positions(self,variables):
        c,wh=self.decode(variables)
        out=np.column_stack((c-wh/2,wh))*self.scale
        out[self.locked,2:]=self.p.specs[self.locked,2:]
        out[self.p.preplaced]=self.p.specs[self.p.preplaced]
        return out

    def pullback(self,center_gradient,dimension_gradient,wh):
        center_gradient=center_gradient.copy();center_gradient[self.p.preplaced]=0.
        per_block=.5*(wh[:,0]*dimension_gradient[:,0]-wh[:,1]*dimension_gradient[:,1])
        free=self.index>=0
        ratios=np.bincount(self.index[free],weights=per_block[free],minlength=len(self.classes))
        return np.r_[center_gradient.ravel(),ratios]


class GeometryObjective:
    def __init__(self,p,parameterization):
        self.p=p;self.rep=parameterization;self.i,self.j=np.triu_indices(p.n,1)
        self.sigma2=(np.sqrt(self.rep.area[self.i])+np.sqrt(self.rep.area[self.j]))**2
        seed=self.rep.positions(self.rep.initial)
        self.tree=contact_tree(seed,p,topology(seed,p.constraints))
        self.nsoft=Objective(p).nsoft
        self.bi=p.b2b[:,:2].astype(int);self.pi=p.p2b[:,:2].astype(int)
        self.norm=max(p.total_wire_weight,1e-12)
        self.eps=1e-4

    def value_gradient(self,variables,overlap_weight=100.):
        p=self.p;n=p.n;c,wh=self.rep.decode(variables)
        gc=np.zeros_like(c);gd=np.zeros_like(wh);loss=0.
        left=c-wh/2;right=c+wh/2
        # Overlap can change its separating direction at each evaluation.
        delta=c[self.i]-c[self.j];penetration=(wh[self.i]+wh[self.j])/2-np.abs(delta)
        axis=np.argmin(penetration,axis=1);amount=penetration[np.arange(len(axis)),axis]
        active=(penetration>0).all(1)
        amount=np.where(active,amount,0.)
        weight=overlap_weight/max(n,1)/self.sigma2
        loss+=float(np.sum(weight*amount**2));derivative=2*weight*amount
        for a in (0,1):
            k=np.where(axis==a,derivative,0.);sign=np.where(delta[:,a]>=0,1.,-1.)
            gc[:,a]+=np.bincount(self.i,weights=-k*sign,minlength=n)+np.bincount(self.j,weights=k*sign,minlength=n)
            gd[:,a]+=.5*(np.bincount(self.i,weights=k,minlength=n)+np.bincount(self.j,weights=k,minlength=n))
        # Nonnegative lower edges; preplaced coordinates remain exact by construction.
        negative=np.minimum(left,0.)
        penalty=1000/max(n,1)/self.rep.area[:,None]
        loss+=float(np.sum(penalty*negative**2));edge_gradient=2*penalty*negative
        gc+=edge_gradient;gd-=edge_gradient/2
        # Public weighted Manhattan wire objective.
        if len(self.bi):
            i,j=self.bi.T;diff=c[i]-c[j];weights=p.b2b[:,2]/self.norm
            loss+=float(np.sum(weights[:,None]*np.abs(diff)))
            for a in (0,1):
                signed=weights*np.sign(diff[:,a]);gc[:,a]+=np.bincount(i,weights=signed,minlength=n)-np.bincount(j,weights=signed,minlength=n)
        if len(self.pi):
            pin,block=self.pi.T;diff=c[block]-p.pins[pin]/p.scale;weights=p.p2b[:,2]/self.norm
            loss+=float(np.sum(weights[:,None]*np.abs(diff)))
            for a in (0,1):gc[:,a]+=np.bincount(block,weights=weights*np.sign(diff[:,a]),minlength=n)
        # Bbox area and boundary distance to the actual moving bbox.
        low_index=left.argmin(0);high_index=right.argmax(0)
        lower=left[low_index,np.arange(2)];upper=right[high_index,np.arange(2)];span=upper-lower
        loss+=float(span.prod());gl=np.zeros_like(c);gr=np.zeros_like(c)
        for a in (0,1):gl[low_index[a],a]-=span[1-a];gr[high_index[a],a]+=span[1-a]
        for bit,a,at_upper in ((1,0,False),(2,0,True),(4,1,True),(8,1,False)):
            ids=np.flatnonzero(p.constraints[:,4]&bit)
            if not len(ids):continue
            gap=upper[a]-right[ids,a] if at_upper else left[ids,a]-lower[a]
            smooth=np.sqrt(gap**2+self.eps**2);weight=12./self.nsoft
            loss+=float(weight*np.sum(smooth-self.eps));g=weight*gap/smooth
            if at_upper:gr[ids,a]-=g;gr[high_index[a],a]+=g.sum()
            else:gl[ids,a]+=g;gl[low_index[a],a]-=g.sum()
        gc+=gl+gr;gd+=(gr-gl)/2
        # Soft geometric contact preferences inferred only from the draft/public groups.
        for i,j,a in self.tree:
            weight=8./self.nsoft;t=1-a
            gap=c[j,a]-c[i,a]-(wh[i,a]+wh[j,a])/2
            smooth=np.sqrt(gap**2+self.eps**2);g=weight*gap/smooth
            loss+=weight*(smooth-self.eps);gc[i,a]-=g;gc[j,a]+=g;gd[i,a]-=g/2;gd[j,a]-=g/2
            diff=c[j,t]-c[i,t];small=i if wh[i,t]<=wh[j,t] else j
            gap=abs(diff)-(wh[i,t]+wh[j,t])/2+.02*wh[small,t]
            if gap>0:
                smooth=np.sqrt(gap**2+self.eps**2);g=weight*gap/smooth
                loss+=weight*(smooth-self.eps);sgn=1. if diff>=0 else -1.
                gc[i,t]-=g*sgn;gc[j,t]+=g*sgn;gd[i,t]-=g/2;gd[j,t]-=g/2;gd[small,t]+=.02*g
        movement=c-self.rep.initial_centers
        loss+=.02/max(n,1)*float(np.sum(movement**2));gc+=.04/max(n,1)*movement
        return float(loss),self.rep.pullback(gc,gd,wh)


class BudgetReached(Exception):pass


def optimize(raw,inputs,seconds=1.3,shared=True,fixed_shapes=False):
    begin=time.perf_counter();deadline=begin+seconds;p=prepare_problem(*inputs)
    rep=Parameterization(p,np.asarray(raw),shared=shared,fixed_shapes=fixed_shapes)
    objective=GeometryObjective(p,rep);quality=Objective(p)
    best=row_control(p);best_value=quality(best);events=[];trace=[]
    if not hard_valid(best,p):raise RuntimeError('Invalid public fallback')
    def accept(candidate,label):
        nonlocal best,best_value
        if candidate is None:return False
        candidate=canonicalize(candidate,p);candidate,_=repair_tiny_negative(candidate,p)
        if not hard_valid(candidate,p):return False
        value=quality(candidate)
        if value<best_value-1e-10:
            best,best_value=candidate.copy(),value
            events.append({'elapsed':time.perf_counter()-begin,'proxy':float(value),'phase':label,'positions':best.tolist()})
        return True
    accept(np.asarray(raw).copy(),'input')
    state=rep.initial.copy();gradient_stop=begin+seconds*.55
    def diagnostics(candidate):
        i,j=np.triu_indices(p.n,1)
        penetration=np.minimum(candidate[i,:2]+candidate[i,2:],candidate[j,:2]+candidate[j,2:])-np.maximum(candidate[i,:2],candidate[j,:2])
        active=(penetration>1e-6).all(1)
        return {'overlap_pairs':int(active.sum()),'max_penetration':float(np.minimum(penetration[active,0],penetration[active,1]).max()) if active.any() else 0.,
                'min_coordinate':float(candidate[:,:2].min()),'bbox_area':float(np.prod((candidate[:,:2]+candidate[:,2:]).max(0)-candidate[:,:2].min(0)))}
    initial_diagnostics=diagnostics(rep.positions(state))
    for phase,weight in enumerate((10.,1000.,100000.)):
        phase_stop=begin+seconds*.55*(phase+1)/3
        if time.perf_counter()>=gradient_stop:break
        calls=0;phase_best=[float('inf'),state.copy()]
        def fun(vector):
            nonlocal calls
            if time.perf_counter()>=phase_stop:raise BudgetReached()
            value,grad=objective.value_gradient(vector,weight);calls+=1
            if value<phase_best[0]:phase_best[:]=[value,vector.copy()]
            return value,grad
        before=time.perf_counter();status='iterations'
        try:
            solution=minimize(fun,state,jac=True,bounds=rep.bounds,method='L-BFGS-B',
                              options={'maxiter':10000,'maxls':25,'ftol':1e-12,'gtol':1e-7,'maxcor':15})
            state=solution.x;status=str(solution.message)
        except BudgetReached:
            state=phase_best[1];status='phase_budget'
        candidate=rep.positions(state);legal=accept(candidate,'joint_continuous')
        trace.append({'phase':'joint_continuous','weight':weight,'calls':calls,'seconds':time.perf_counter()-before,'legal':legal,'status':status,**diagnostics(candidate)})
    candidate=rep.positions(state)
    for mode in (0,1):
        remaining=deadline-time.perf_counter()-.03
        if remaining<.04:break
        full=topology(candidate,p.constraints,mode);edges=reduce_edges(full,p.n)
        tree=contact_tree(candidate,p,full)
        for contact in (0.,8.):
            remaining=deadline-time.perf_counter()-.025
            if remaining<.04:break
            before=time.perf_counter()
            projected,stats=solve_lp(p,candidate,edges,contacts=tree if contact else (),contact=contact,limit=min(.35,remaining))
            legal=accept(projected,'terminal_position_lp')
            trace.append({'phase':'terminal_position_lp','mode':mode,'contact':contact,'seconds':time.perf_counter()-before,'legal':legal,'edges':len(edges),
                          'certificate':infeasibility_certificate(p,candidate,full) if projected is None else None,**stats})
    if not hard_valid(best,p):raise AssertionError('Invalid final incumbent')
    return best,{'seconds':time.perf_counter()-begin,'proxy':float(best_value),'trace':trace,'guard_fallback':False,
                 'ratio_variables':len(rep.classes),'shared':shared,'fixed_shapes':fixed_shapes,'improvements':events,
                 'initial_diagnostics':initial_diagnostics,'final_diagnostics':diagnostics(candidate),
                 'illegal_continuous_proposal':candidate.tolist() if not hard_valid(candidate,p) else None}
