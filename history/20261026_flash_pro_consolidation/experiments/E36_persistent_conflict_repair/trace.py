"""Input-only event recorder. Scoring is deferred outside all solver clocks."""
import contextvars
from contextlib import contextmanager
import functools
import hashlib
import time
import numpy as np

CURRENT=contextvars.ContextVar('E35_observer',default=None)
ARM=contextvars.ContextVar('E35_ablation',default='full')


def clock():
    obs=CURRENT.get()
    return time.perf_counter()-(obs.overhead if obs else 0.)


def emit(kind,**values):
    obs=CURRENT.get()
    if obs:obs.record(kind,**values)


class Observer:
    def __init__(self):
        self.events=[];self.layouts={};self.overhead=0.;self.stack=[]
    def layout(self,x):
        if x is None:return None
        a=np.asarray(x,dtype=float)
        if a.ndim!=2 or a.shape[1]!=4:return None
        key=hashlib.sha256(a.tobytes()).hexdigest()[:24]
        if key not in self.layouts:self.layouts[key]=a.copy()
        return key
    def record(self,kind,**values):
        start=time.perf_counter()
        try:
            for k,v in list(values.items()):
                if isinstance(v,np.ndarray):values[k]=self.layout(v)
                elif isinstance(v,np.generic):values[k]=v.item()
            self.events.append({'id':len(self.events),'kind':kind,'parent':self.stack[-1] if self.stack else None,
                'depth':sum(self.events[i].get('module')=='optimize' for i in self.stack),**values})
            return len(self.events)-1
        finally:self.overhead+=time.perf_counter()-start
    def wrap(self,module,function):
        @functools.wraps(function)
        def call(*args,**kwargs):
            if CURRENT.get() is not self:return function(*args,**kwargs)
            incoming=next((v for v in args if isinstance(v,np.ndarray) and v.ndim==2 and v.shape[1]==4),None)
            identifier=self.record('call',module=module,input=incoming);self.stack.append(identifier);started=clock()
            try:
                result=function(*args,**kwargs)
            except BaseException as error:
                self.events[identifier].update(error=repr(error));raise
            else:
                elapsed=clock()-started;start=time.perf_counter();event=self.events[identifier]
                candidate=result[0] if isinstance(result,tuple) else result
                if isinstance(candidate,np.ndarray):event['output']=self.layout(candidate)
                elif isinstance(candidate,(bool,np.bool_)):event['returned_bool']=bool(candidate)
                if isinstance(result,tuple) and len(result)>1 and isinstance(result[1],dict):
                    event['stats']={k:v for k,v in result[1].items() if k!='trace'}
                if module in ('topology','edge_reduction','contact_tree'):
                    event['output_count']=len(result)
                    if module=='edge_reduction':event['input_count']=len(args[0])
                event['seconds_excluding_observer']=elapsed;self.overhead+=time.perf_counter()-start
                return result
            finally:self.stack.pop()
        return call
    @contextmanager
    def activate(self):
        from . import solver,fast_geometry,certify_polish
        from .backend import rescue
        token=CURRENT.set(self);changes=[]
        specs=[(solver,'optimize','optimize'),(solver,'row_control','row_fallback'),
            (solver,'step','shape_solve'),(solver,'project','fixed_shape_projection'),
            (solver,'topology','topology'),(solver,'reduce_edges','edge_reduction'),
            (solver,'contact_tree','contact_tree'),(solver,'canonicalize','canonicalize'),
            (solver,'repair_tiny_negative','negative_roundoff_guard'),(solver,'hard_valid','hard_guard'),
            (fast_geometry,'contact_tree','contact_tree'),(rescue,'seed_pack','packed_seed'),
            (certify_polish,'polish','certified_polish')]
        try:
            for owner,name,label in specs:
                original=getattr(owner,name);changes.append((owner,name,original));setattr(owner,name,self.wrap(label,original))
            yield self
        finally:
            for owner,name,original in reversed(changes):setattr(owner,name,original)
            CURRENT.reset(token)


def accept_event(label,raw,clean,before,after,before_value,after_value,score=None,reason=None):
    emit('acceptance',module=label,raw=raw,clean=clean,before=before,after=after,
        before_proxy=float(before_value),after_proxy=float(after_value),candidate_proxy=score,
        hard_legal=clean is not None,updated=after_value<before_value,reason=reason)


def gate(module,triggered,reason,**values):emit('gate',module=module,triggered=bool(triggered),reason=reason,**values)
