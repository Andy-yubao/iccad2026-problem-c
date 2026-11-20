"""Exact FP32 disk-mapped batches; never resident-copy the 64k set to VRAM."""
import gc
import numpy as np
import torch
from .common import ROOT,HERE,RUN,ARCHIVE,read,save,sha,check_stop
from datasets.expanded_reference import registry,REGISTRY
PUBLIC=('features','adjacency','mib','group','mask','area','known','specs','constraints','scale','valid')
NODE=('features','mask','area','known','specs','constraints','target','reference')
PAIR=('adjacency','mib','group')
CACHE=RUN/'cache'
OWNER=ROOT/'history/20261029_model_research_consolidation/experiments/E58_training_data_v6'

def resolve(name):
    name=name.replace('\\','/')
    if name.startswith('experiments/E58_training_data_v6/'):
        p=OWNER/name.split('experiments/E58_training_data_v6/',1)[1]
    else:
        p=ROOT/name
        if not p.exists():p=ARCHIVE/name
    p=p.resolve()
    if not p.is_relative_to(ROOT):raise ValueError('Cache path escapes workspace')
    return p

def manifest(role):
    if role not in ('train','dev'):raise PermissionError('No training test access')
    base=OWNER/'runs/20261026_reference_v6_64k';m=read(base/'cache'/role/'manifest.json');ident=m['identity']
    for name in ('preparation_status.json','results.json','cache/results.json'):
        assert read(base/name)['status']=='complete'
    assert read(base/'verification.json')['status']=='passed'
    assert ident['registry_sha256']==sha(REGISTRY) and ident['paths_sha256']==sha(ROOT/'protocol/paths.json')
    for n,h in ident['feature_sources'].items():assert sha(OWNER/n)==h,n
    for n,h in ident['dataset_sources'].items():assert sha(ROOT/n)==h,n
    for n in ('features.py','batches.py'):assert sha(HERE/'placer'/n)==ident['feature_sources'][n]
    assert m['members']==[v['member'] for v in registry()['sets'][role]['records']]
    for entry in m['parts']:
        s=resolve(entry['path']).stat();assert (s.st_size,s.st_mtime_ns)==(entry['size'],entry['mtime_ns'])
    return m

def create(role):
    folder=CACHE/role
    if (folder/'complete.json').exists():return
    check_stop();m=manifest(role);count=m['count'];folder.mkdir(parents=True,exist_ok=True)
    arrays=None;max_group=0;done=0
    for entry in m['parts']:
        check_stop();part=torch.load(resolve(entry['path']),map_location='cpu',weights_only=False,mmap=True)
        assert part['identity']==m['part_identities'][entry['identity_key']] and part['indices']==entry['indices']
        if arrays is None:
            first=part['rows'][0]['graph'];arrays={}
            for k in (*NODE,*PAIR,'scale','n'):
                if k in PAIR:shape=(count,120,120);dtype=np.float32
                elif k=='n':shape=(count,);dtype=np.int16
                elif k=='scale':shape=(count,);dtype=np.float32
                else:shape=(count,120,*first[k].shape[1:]);dtype=first[k].numpy().dtype
                arrays[k]=np.lib.format.open_memmap(folder/(k+'.npy'),mode='w+',dtype=dtype,shape=shape)
        for index,row in zip(part['indices'],part['rows']):
            assert list(row['member'])==m['members'][index];g=row['graph'];n=g['n'];assert 0<n<=120
            arrays['n'][index]=n;arrays['scale'][index]=float(g['scale'])
            for k in (*NODE,*PAIR):
                arrays[k][index]=0
                if k in PAIR:arrays[k][index,:n,:n]=g[k].numpy()
                else:arrays[k][index,:n]=g[k].numpy()
            groups=g['constraints'][:,3]
            max_group=max(max_group,max([int((groups==v).sum()) for v in groups.unique() if v>0] or [0]));done+=1
        if done%1024==0 or done==count:
            for a in arrays.values():a.flush()
            save(CACHE/'progress.json',dict(role=role,completed=done,total=count,max_public_group=max_group,test_read=False))
            print('Exact disk cache',role,done,'/',count,flush=True)
        del part;gc.collect()
    assert done==count
    for a in arrays.values():a.flush()
    save(folder/'complete.json',dict(count=count,max_public_group=max_group,registry_sha256=sha(REGISTRY),manifest_sha256=sha(OWNER/'runs/20261026_reference_v6_64k/cache'/role/'manifest.json'),format='E101.exact_FP32_memmap.v1',test_read=False))

class Data:
    def __init__(self,role):
        if role not in ('train','dev_monitor500','train_monitor500'):raise PermissionError(role)
        parent='train' if role.startswith('train') else 'dev';self.parent=parent
        meta=registry()['sets'];full=meta[parent]['records'];lookup={tuple(r['member']):i for i,r in enumerate(full)}
        rows=meta[role]['records'];self.members=[r['member'] for r in rows];self.indices=np.array([lookup[tuple(r['member'])] for r in rows])
        self.meta=read(CACHE/parent/'complete.json');self.arrays={k:np.load(CACHE/parent/(k+'.npy'),mmap_mode='r') for k in (*NODE,*PAIR,'scale','n')}
    def __len__(self):return len(self.indices)
    def row(self,i):
        j=self.indices[i];n=int(self.arrays['n'][j]);g={k:torch.from_numpy(self.arrays[k][j,:n,:n].copy() if k in PAIR else self.arrays[k][j,:n].copy()) for k in (*NODE,*PAIR)}
        g.update(n=n,scale=torch.tensor(float(self.arrays['scale'][j])))
        return dict(member=self.members[i],graph=g)

class Packer:
    def __init__(self,data,supervision=True):
        self.data=data;self.keys=(*PUBLIC,'target','reference') if supervision else PUBLIC;self.host={}
        for k in self.keys:
            if k=='valid':shape=(32,120);dtype=torch.bool
            elif k=='scale':shape=(32,);dtype=torch.float32
            else:
                a=data.arrays[k];shape=(32,*a.shape[1:]);dtype=torch.from_numpy(np.empty((),dtype=a.dtype)).dtype
            self.host[k]=torch.zeros(shape,dtype=dtype,pin_memory=True)
    def pack(self,ids):
        count=len(ids);assert 0<count<=32
        ids=list(ids)+[ids[-1]]*(32-count);actual=self.data.indices[ids]
        for k,v in self.host.items():
            if k=='valid':np.less(np.arange(120)[None],self.data.arrays['n'][actual,None],out=v.numpy())
            else:np.take(self.data.arrays[k],actual,axis=0,out=v.numpy())
        return self.host,count
    def gpu(self,ids):
        values,count=self.pack(ids);return {k:v.to('cuda',non_blocking=True) for k,v in values.items()},count

def public(g):return {k:g[k] for k in PUBLIC}
