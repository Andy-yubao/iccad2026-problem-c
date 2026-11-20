"""Transparent secondary controls after primary structural audit."""
import itertools
import numpy as np
from .audit_unsupervised import OUT,PRIOR,LAYERS,UNSUP,read,save,largest_empty,blockers,check

def main():
    from scoring.draft_geometry import contact_distances,minimum_tree
    from shapely.geometry import box
    from shapely.ops import unary_union
    _,manifest=check()
    positions=np.load(PRIOR/'positions.npy',mmap_mode='r');flags=np.load(PRIOR/'flags.npy');areas=np.load(PRIOR/'areas.npy')
    refs=np.load(PRIOR/'reference.npy',mmap_mode='r')
    sensitivity={str(t):dict(events=[],reference_free=True) for t in (.005,.02,.05)}
    center_crossings={str(t):[] for t in (.02,.05,.1)}
    path_events=[];reference_edges=[];occupancy_checks=0
    for index in range(500):
        row=read(OUT/f'cases/{index:04d}.json');n=row['n'];p=np.asarray(positions[index,3:,:n]);a=areas[index,:n];f=flags[index,:n]
        center=p[:,:,:2]+p[:,:,2:]/2;d=center[:,:,None,:]-center[:,None,:,:]
        half=(p[:,:,None,2:]+p[:,None,:,2:])/2;scale=(np.sqrt(a)[:,None]+np.sqrt(a)[None,:])/2
        pairs=np.triu(np.ones((n,n),bool),1)&~((f[:,1]!=0)[:,None]&(f[:,1]!=0)[None,:])
        for margin in (.005,.02,.05):
            clear=np.where(np.abs(d)-half>margin*scale[None,:,:,None],np.sign(d),0)
            last=clear[0].copy();cross_layer=np.zeros((n,n,2),int)
            for k in range(1,10):
                old=d[k-1];new=d[k]
                for axis in range(2):
                    cross=old[...,axis]*new[...,axis]<0
                    t=np.divide(old[...,axis],old[...,axis]-new[...,axis],out=np.zeros_like(scale),where=cross)
                    orth=1-axis;delta=old[...,orth]+t*(new[...,orth]-old[...,orth])
                    wh=half[k-1,...,orth]+t*(half[k,...,orth]-half[k-1,...,orth])
                    through=cross&(wh-np.abs(delta)>margin*scale)&pairs
                    cross_layer[...,axis]=np.where(through,k+3,cross_layer[...,axis])
                exchanged=(last*clear[k]<0)&(cross_layer>0)&pairs[...,None]
                for i,j,axis in np.argwhere(exchanged):
                    sensitivity[str(margin)]['events'].append(dict(case=index,i=int(i),j=int(j),axis=int(axis),
                        crossing_layer=int(cross_layer[i,j,axis]),completion_layer=k+3))
                resolved=clear[k]!=0
                last=np.where(resolved,clear[k],last);cross_layer=np.where(resolved,0,cross_layer)
        for margin in (.02,.05,.1):
            counts=[]
            for k in range(1,10):
                old=d[k-1];new=d[k];total=0;modules=np.zeros(n,bool)
                for axis in range(2):
                    cross=(old[...,axis]*new[...,axis]<0)&(np.abs(old[...,axis])>margin*scale)&(np.abs(new[...,axis])>margin*scale)
                    t=np.divide(old[...,axis],old[...,axis]-new[...,axis],out=np.zeros_like(scale),where=cross)
                    orth=1-axis;delta=old[...,orth]+t*(new[...,orth]-old[...,orth])
                    wh=half[k-1,...,orth]+t*(half[k,...,orth]-half[k-1,...,orth])
                    through=cross&(wh-np.abs(delta)>.02*scale)&pairs
                    total+=int(through.sum());modules|=through.any(0)|through.any(1)
                counts.append(dict(layer=k+3,events=total,module_fraction=float(modules.mean())))
            center_crossings[str(margin)].append(counts)
        # Alternative same-group chains can replace a fixed initial MST edge.
        for group in sorted(set(f[:,3])-{0}):
            members=np.flatnonzero(f[:,3]==group)
            if len(members)<2:continue
            edge_rows=[e for e in row['group_edges'] if e['group']==group and e['initial_bad']]
            if not edge_rows:continue
            path=[]
            for value in p:
                distances=contact_distances(value)[0][np.ix_(members,members)]
                np.fill_diagonal(distances,0.)
                for k in range(len(members)):
                    distances=np.minimum(distances,np.maximum(distances[:,k,None],distances[None,k,:]))
                path.append(distances)
            for e in edge_rows:
                u=int(np.flatnonzero(members==e['i'])[0]);v=int(np.flatnonzero(members==e['j'])[0])
                costs=np.array([value[u,v] for value in path]);clear=np.array(e['blocker_counts'])==0
                good=clear&(costs<=costs[0]/2)
                persistent=np.logical_and.accumulate(good[::-1])[::-1]
                path_events.append(dict(case=index,group=int(group),i=e['i'],j=e['j'],costs=costs.tolist(),
                    clear_and_path_halved_layer=int(np.flatnonzero(persistent)[0])+3 if persistent.any() else None))
        ref=refs[index,:n];rd=contact_distances(ref)[0]
        for group in sorted(set(f[:,3])-{0}):
            members=np.flatnonzero(f[:,3]==group);foreign=np.flatnonzero(f[:,3]!=group)
            if len(members)<2:continue
            for u,v,cost in minimum_tree(rd[np.ix_(members,members)]):
                i,j=int(members[u]),int(members[v])
                blocked=bool(blockers(ref,i,j,foreign,a).any())
                reference_edges.append(dict(blocked=blocked,cost=cost,bad=bool(blocked and cost>.05)))
        if index%25==0:
            for k in (0,3,6,9):
                rect=row['void'][k]['rectangle'];empty=box(rect[0],rect[1],rect[0]+rect[2],rect[1]+rect[3])
                union=unary_union([box(x,y,x+w,y+h) for x,y,w,h in p[k]])
                assert empty.intersection(union).area<=1e-8*a.sum()
                occupancy_checks+=1
        if (index+1)%100==0:print('Secondary controls',index+1,'/500',flush=True)
    for margin,value in sensitivity.items():
        value['count']=len(value['events']);value['cases']=len({e['case'] for e in value['events']})
        value['crossing_unsupervised']=sum(e['crossing_layer'] in UNSUP for e in value['events'])
        value['completion_unsupervised']=sum(e['completion_layer'] in UNSUP for e in value['events'])
    crossing_summary={}
    for margin,values in center_crossings.items():
        crossing_summary[margin]=[dict(layer=k+4,total=sum(v[k]['events'] for v in values),
            cases=sum(v[k]['events']>0 for v in values),mean_module_fraction=float(np.mean([v[k]['module_fraction'] for v in values]))) for k in range(9)]
    # Independent exhaustive search on small, nonuniform-coordinate layouts.
    synthetic=[np.array([[0,0,1,1],[2,0,1,2],[0,2,2,1]],float),
        np.array([[0,0,1,4],[3,0,1,4]],float),
        np.array([[0,0,2,2],[1,1,2,2],[3,0,1,3]],float)]
    for p in synthetic:
        x=np.unique(np.r_[p[:,0],p[:,0]+p[:,2]]);y=np.unique(np.r_[p[:,1],p[:,1]+p[:,3]])
        brute=0.
        for l,r in itertools.combinations(x,2):
            for b,t in itertools.combinations(y,2):
                overlap=(np.minimum(r,p[:,0]+p[:,2])-np.maximum(l,p[:,0])>1e-10)&(np.minimum(t,p[:,1]+p[:,3])-np.maximum(b,p[:,1])>1e-10)
                if not overlap.any():brute=max(brute,(r-l)*(t-b))
        np.testing.assert_allclose(largest_empty(p)['area'],brute,rtol=0,atol=1e-10)
    save(OUT/'followup.json',dict(passage_sensitivity=sensitivity,overlap_order_crossings=crossing_summary,
        fixed_edge_alternative_group_paths=path_events,
        clear_and_group_path_halved_count=sum(e['clear_and_path_halved_layer'] is not None for e in path_events),
        reference_center_line_controls=dict(edges=len(reference_edges),blocked=sum(e['blocked'] for e in reference_edges),bad=sum(e['bad'] for e in reference_edges)),
        maximum_empty_rectangle_checks=dict(bruteforce_layouts=len(synthetic),actual_no_occupancy_checks=occupancy_checks,status='passed')))
    print('Followup controls passed',flush=True)

if __name__=='__main__':main()
