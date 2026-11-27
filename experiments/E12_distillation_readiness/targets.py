"""Versioned teacher targets. No original free-block reference is a target."""
import numpy as np
from .certificate import extract

SCHEMA='E12.teacher.v1'

def decode_prediction(geometry,p,mib_labels):
    """Student output + predicted classes + public inputs; no teacher geometry."""
    from .certificate import coordinate_shapes
    g=np.asarray(geometry,dtype=np.float64);areas=p.areas*.9900002;logs=np.clip(g[:,2],-5,5)
    ratio=np.exp(logs);wh=np.column_stack((np.sqrt(areas*ratio),np.sqrt(areas/ratio)))
    locked=p.fixed|p.preplaced;wh[locked]=p.specs[locked,2:]
    wh=coordinate_shapes(wh,p,mib_labels,'lower',areas,logs)
    x=np.column_stack((np.maximum(0.,g[:,:2]*p.scale-wh/2),wh));x[p.preplaced]=p.specs[p.preplaced]
    return x

def encode(positions,p):
    x=np.asarray(positions,dtype=np.float64);cert=extract(x,p);n=p.n
    geometry=np.column_stack(((x[:,:2]+x[:,2:]/2)/p.scale,np.log(x[:,2]/x[:,3]))).astype(np.float32)
    separation=np.full((n,n),-1,dtype=np.int8)
    contacts=np.full((n,n),-1,dtype=np.int8)
    for i,j,a in cert['edges']:separation[i,j]=a;separation[j,i]=a+2
    for i,j,a in cert['contacts']:contacts[i,j]=a;contacts[j,i]=a+2
    labels=np.asarray(cert['mib_labels'],dtype=np.int32)
    components=np.asarray(cert['group_components'],dtype=np.int32)
    overlap=np.minimum(x[:,None,:2]+x[:,None,2:],x[None,:,:2]+x[None,:,2:])-np.maximum(x[:,None,:2],x[None,:,:2])
    touching=((overlap[:,:,0]>=0)&(overlap[:,:,1]>0))|((overlap[:,:,0]>0)&(overlap[:,:,1]>=0))
    samegroup=(p.constraints[:,3,None]==p.constraints[None,:,3])&(p.constraints[:,3,None]>0)
    all_contacts=touching&samegroup;np.fill_diagonal(all_contacts,False)
    return {'positions':x,'geometry':geometry,'separation':separation,'contact_forest':contacts,
        'group_contact_positive':all_contacts,
        'geometry_learned_mask':np.column_stack((np.repeat((~p.preplaced)[:,None],2,axis=1),~(p.fixed|p.preplaced))),
        'boundary_satisfied':np.asarray(cert['boundary'],dtype=np.int8),
        'mib_same':labels[:,None]==labels[None,:], 'mib_labels':labels,
        'group_same_component':components[:,None]==components[None,:],
        'group_components':components,'teacher_area_ratio':(x[:,2:].prod(1)/p.areas).astype(np.float32)}

def validate(target,p):
    n=p.n
    assert target['geometry'].shape==(n,3) and np.isfinite(target['geometry']).all()
    assert np.all(np.diag(target['separation'])==-1)
    assert np.count_nonzero(target['separation']>=0)==n*(n-1)
    for field in ('mib_same','group_same_component'):
        assert np.array_equal(target[field],target[field].T) and np.diag(target[field]).all()
    c=target['contact_forest'];a,b=np.where((c>=0)&(c<2))
    assert np.all(p.constraints[a,3]==p.constraints[b,3]) and np.all(p.constraints[a,3]>0)
    assert np.all(c[b,a]==c[a,b]+2)
    return True
