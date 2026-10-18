"""Reusable public-only child, using files to avoid Windows named-pipe errors."""
import atexit
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def retry_file(operation,seconds=2.):
    """Windows scanners/sandbox handles may briefly deny an atomic file handoff."""
    deadline=time.monotonic()+seconds
    while True:
        try:return operation()
        except PermissionError:
            if time.monotonic()>=deadline:raise
            time.sleep(.02)


class PersistentWorker:
    def __init__(self,output,root,package):
        self.output=output;self.root=root;self.package=package
        self.process=None;self.directory=None;self.log=None;self.temp=None
        self.startup_calls=0;atexit.register(self.close)

    def call(self,payload):
        begin=time.perf_counter()
        if self.process is None:
            self.temp=tempfile.TemporaryDirectory(prefix='public_worker_',dir=self.output)
            self.directory=Path(self.temp.name)
            self.log=(self.output/'persistent_worker.log').open('a',encoding='utf-8')
            self.process=subprocess.Popen([sys.executable,'-B','-m',self.package+'.worker',
                                           '--serve',str(self.directory)],cwd=self.root,
                                          stdout=self.log,stderr=self.log,
                                          creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            self.startup_calls+=1
        try:
            request=self.directory/'request.json';response=self.directory/'response.json'
            if response.exists():raise RuntimeError('Unconsumed public worker response')
            temp=request.with_suffix('.tmp')
            temp.write_text(json.dumps(payload,allow_nan=False),encoding='utf-8')
            retry_file(lambda:temp.replace(request))
            deadline=begin+payload['seconds']+10.
            while not response.exists():
                if self.process.poll() is not None:raise RuntimeError('Persistent worker exited; see its log')
                if time.perf_counter()>deadline:raise TimeoutError('Persistent worker exceeded per-case guard')
                time.sleep(.01)
            value=json.loads(retry_file(lambda:response.read_text(encoding='utf-8')))
            retry_file(response.unlink)
        except Exception as exc:
            value={'error':{'type':type(exc).__name__,'message':str(exc)},'solve_seconds':time.perf_counter()-begin}
            self.close()
        value['worker_wall_seconds']=time.perf_counter()-begin
        return value

    def close(self):
        if self.process is not None:
            (self.directory/'SHUTDOWN').touch()
            try:self.process.wait(timeout=2.)
            except subprocess.TimeoutExpired:
                self.process.terminate();self.process.wait(timeout=5.)
            self.process=None
        if self.log is not None:self.log.close();self.log=None
        if self.temp is not None:self.temp.cleanup();self.temp=None
