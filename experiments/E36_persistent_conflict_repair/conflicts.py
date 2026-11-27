"""Public-input conflict history and bounded, fixed-shape coordinate sweeps."""
import time
import numpy as np


def pair_overlaps(x, units, tolerance=1e-6):
    i,j=np.triu_indices(len(x),1)
    overlap=np.minimum(x[i,:2]+x[i,2:],x[j,:2]+x[j,2:])-np.maximum(x[i,:2],x[j,:2])
    hit=(overlap>tolerance).all(1)
    depth=np.minimum(overlap[hit,0],overlap[hit,1])/np.minimum(units[i[hit]],units[j[hit]])
    return {(int(a),int(b)):float(d) for a,b,d in zip(i[hit],j[hit],depth)}


class ConflictHistory:
    def __init__(self, p):
        self.units=np.sqrt(p.areas.copy());locked=p.fixed|p.preplaced
        self.units[locked]=np.sqrt(p.specs[locked,2:].prod(1));self.counts={}

    def observe(self,x):
        depths=pair_overlaps(x,self.units)
        self.counts={pair:self.counts.get(pair,0)+1 for pair,depth in depths.items() if depth>1e-3}
        persistent=sorted((pair for pair,count in self.counts.items() if count>=3),key=lambda pair:(-self.counts[pair],-depths[pair],pair))
        return {'depths':depths,'persistent':persistent,'longest':max(self.counts.values(),default=0),
                'count':len(depths),'max_depth':max(depths.values(),default=0.)}


def nearest_clear_coordinates(x,block,axis,max_shift):
    """Two one-dimensional interval sweeps, no planning solver or resizing.

    Return at most one nearest clear coordinate for this axis. Other rectangles
    stay fixed; preplaced eligibility is checked by the caller.
    """
    tangent=1-axis;others=np.arange(len(x))!=block
    tangent_overlap=np.minimum(x[block,tangent]+x[block,tangent+2],x[:,tangent]+x[:,tangent+2])-np.maximum(x[block,tangent],x[:,tangent])
    obstacles=x[others&(tangent_overlap>1e-6)]
    candidates=np.r_[0.,obstacles[:,axis]-x[block,axis+2],obstacles[:,axis]+obstacles[:,axis+2]]
    candidates=np.unique(candidates[(candidates>=0)&(np.abs(candidates-x[block,axis])<=max_shift)])
    if not len(candidates):return []
    if len(obstacles):
        overlap=np.minimum(candidates[:,None]+x[block,axis+2],obstacles[None,:,axis]+obstacles[None,:,axis+2])-np.maximum(candidates[:,None],obstacles[None,:,axis])
        candidates=candidates[~(overlap>1e-6).any(1)]
    if not len(candidates):return []
    return [float(candidates[np.argmin(np.abs(candidates-x[block,axis]))])]


def local_repair(x,p,history,diagnostic,objective,*,max_moves=4,max_block_shift=.5,seconds=.008):
    started=time.perf_counter();state=x.copy();before=pair_overlaps(state,history.units)
    moves=[];evaluated=0;persistent=set(diagnostic['persistent']);blocked=set()
    initial_centers=x[:,:2]+x[:,2:]/2
    for _ in range(max_moves):
        if time.perf_counter()-started>=seconds:break
        current=pair_overlaps(state,history.units)
        if not current:break
        critical=sorted(current,key=lambda pair:(pair not in persistent,-current[pair],pair))
        candidates=list(dict.fromkeys(i for pair in critical[:4] for i in pair if not p.preplaced[i] and i not in blocked))
        best=None;best_key=None
        for block in candidates:
            for axis in (0,1):
                if time.perf_counter()-started>=seconds:break
                for coordinate in nearest_clear_coordinates(state,block,axis,max_block_shift*history.units[block]):
                    trial=state.copy();trial[block,axis]=coordinate
                    movement=np.linalg.norm((trial[block,:2]+trial[block,2:]/2)-initial_centers[block])/history.units[block]
                    if movement>max_block_shift+1e-12:continue
                    remaining=pair_overlaps(trial,history.units);evaluated+=1
                    if set(remaining)-set(current) or len(remaining)>=len(current):continue
                    # Resolve collisions first. Input-only quality and displacement
                    # choose among repairs with equal collision burden.
                    key=(len(remaining),sum(remaining.values()),float(objective(trial)),movement,block,axis)
                    if best_key is None or key<best_key:best=trial;best_key=key
            if time.perf_counter()-started>=seconds:break
        if best is None:break
        changed=np.flatnonzero(np.any(best[:,:2]!=state[:,:2],axis=1));assert len(changed)==1
        block=int(changed[0]);delta=best[block,:2]-state[block,:2]
        moves.append({'block':block,'delta':delta.tolist(),'block_units':float(np.linalg.norm(delta)/history.units[block])});blocked.add(block);state=best
    after=pair_overlaps(state,history.units)
    assert np.array_equal(state[:,2:],x[:,2:])
    assert np.array_equal(state[p.preplaced],x[p.preplaced])
    assert not (set(after)-set(before))
    return (state if moves else None),{'seconds':time.perf_counter()-started,'evaluated':evaluated,'moves':moves,
        'before_overlap_pairs':len(before),'after_overlap_pairs':len(after),
        'before_max_depth':max(before.values(),default=0.),'after_max_depth':max(after.values(),default=0.),
        'persistent_pairs':[list(pair) for pair in diagnostic['persistent']],
        'geometry_legal_by_pair_test':not bool(after)}


def chain_repair(x,p,history,diagnostic,objective,*,max_blocks=8,max_block_shift=.5,seconds=.012):
    """Monotone single-axis pushes; failed plans never modify the input."""
    started=time.perf_counter();before=pair_overlaps(x,history.units);plans=[];attempts=0
    for pair in diagnostic['persistent'][:2]:
        for block,obstacle in (pair,pair[::-1]):
            if p.preplaced[block]:continue
            for axis in (0,1):
                for sign in (-1,1):
                    if time.perf_counter()-started>=seconds:break
                    trial=x.copy();moved=set();pending=[];failed=False;attempts+=1
                    def shift(i,coordinate):
                        nonlocal failed
                        if p.preplaced[i] or coordinate<0 or abs(coordinate-x[i,axis])>max_block_shift*history.units[i]+1e-10:
                            failed=True;return
                        if sign*(coordinate-trial[i,axis])<=1e-8:return
                        trial[i,axis]=coordinate;moved.add(i);pending.append(i)
                        if len(moved)>max_blocks:failed=True
                    target=trial[obstacle,axis]+trial[obstacle,axis+2] if sign>0 else trial[obstacle,axis]-trial[block,axis+2]
                    shift(block,target);steps=0
                    while pending and not failed and steps<64:
                        steps+=1;i=pending.pop(0)
                        overlap=np.minimum(trial[i,:2]+trial[i,2:],trial[:,:2]+trial[:,2:])-np.maximum(trial[i,:2],trial[:,:2])
                        ids=np.flatnonzero((overlap>1e-6).all(1));ids=ids[ids!=i]
                        # Resolve immutable obstacles first by moving this same block.
                        anchors=ids[p.preplaced[ids]]
                        if len(anchors):
                            target=max(trial[j,axis]+trial[j,axis+2] for j in anchors) if sign>0 else min(trial[j,axis]-trial[i,axis+2] for j in anchors)
                            shift(i,target);continue
                        for j in ids:
                            target=trial[i,axis]+trial[i,axis+2] if sign>0 else trial[i,axis]-trial[j,axis+2]
                            shift(int(j),target)
                            if failed:break
                    if failed or pending or not moved:continue
                    after=pair_overlaps(trial,history.units)
                    if pair in after or set(after)-set(before) or len(after)>=len(before):continue
                    distance=np.abs(trial[:,axis]-x[:,axis])/history.units
                    key=(len(after),sum(after.values()),float(objective(trial)),float(distance.sum()),len(moved))
                    plans.append((key,trial,axis,distance))
    best=min(plans,key=lambda row:row[0]) if plans else None
    out=best[1] if best else None;moves=[]
    if best:
        for i in np.flatnonzero(best[3]>1e-10):moves.append({'block':int(i),'delta':(out[i,:2]-x[i,:2]).tolist(),'block_units':float(best[3][i])})
        assert np.array_equal(out[:,2:],x[:,2:]) and np.array_equal(out[p.preplaced],x[p.preplaced])
        assert len(moves)<=max_blocks and max(m['block_units'] for m in moves)<=max_block_shift+1e-8
    return out,{'seconds':time.perf_counter()-started,'attempts':attempts,'successful_plans':len(plans),'moves':moves,
        'before_overlap_pairs':len(before),'after_overlap_pairs':best[0][0] if best else len(before),
        'persistent_pairs':[list(pair) for pair in diagnostic['persistent']]}


def acyclic(edges,n):
    for axis in (0,1):
        part=edges[edges[:,2]==axis];counts=np.bincount(part[:,1],minlength=n)
        adjacency=[[] for _ in range(n)]
        for a,b in part[:,:2]:adjacency[a].append(b)
        queue=np.flatnonzero(counts==0).tolist();visited=0
        while queue:
            a=queue.pop();visited+=1
            for b in adjacency[a]:
                counts[b]-=1
                if counts[b]==0:queue.append(int(b))
        if visited!=n:return False
    return True


def choose_pivot(x,p,full,history,diagnostic,tried):
    """Change one disjunction; use only public geometry and DAG consistency."""
    choices=[]
    for pair in diagnostic['persistent'][:2]:
        i,j=pair;at=j*(j-1)//2+i;old=tuple(int(k) for k in full[at]);tabu=tried.setdefault(pair,set());tabu.add(old)
        for axis in (0,1):
            for a,b in ((i,j),(j,i)):
                edge=(a,b,axis)
                if edge in tabu or (p.preplaced[a] and p.preplaced[b]):continue
                gap=max(0.,x[a,axis]+x[a,axis+2]-x[b,axis])
                costs=[]
                if not p.preplaced[b]:costs.append(gap/history.units[b])
                if not p.preplaced[a] and x[a,axis]>=gap:costs.append(gap/history.units[a])
                if not costs:continue
                trial=full.copy();trial[at]=edge
                if not acyclic(trial,p.n):continue
                choices.append((min(costs),pair,edge,old))
    if not choices:return None
    cost,pair,edge,old=min(choices);tried[pair].add(edge)
    return {'pair':pair,'edge':edge,'old':old,'minimum_move_bound':cost}
