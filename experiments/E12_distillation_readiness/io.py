"""Experiment-local process/record helpers; algorithms stay in this experiment."""
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import uuid

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'

import numpy as np
from datasets.paths import ROOT

HERE=Path(__file__).resolve().parent
HISTORY=ROOT/'history/20261018_E10_consolidation/experiments/E8_fast_robust_postprocess/runs'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        for attempt in range(5):
            try:os.replace(temporary,path);break
            except OSError:
                if attempt==4:raise
                time.sleep(.05*(attempt+1))
    finally:
        if temporary.exists():temporary.unlink()


class Worker:
    def __init__(self,log):
        self.messages=queue.Queue();self.log=Path(log).open('w',encoding='utf-8')
        self.process=subprocess.Popen([sys.executable,'-B','-m','experiments.E12_distillation_readiness.worker'],
             cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,encoding='utf-8')
        def read():
            for line in self.process.stdout:self.messages.put(line)
            self.messages.put(None)
        threading.Thread(target=read,daemon=True).start()
        assert self.receive(30).get('ready')
    def receive(self,timeout):
        line=self.messages.get(timeout=timeout)
        if line is None:raise RuntimeError('Worker exited; inspect log')
        return json.loads(line)
    def call(self,request):
        begin=time.perf_counter();self.process.stdin.write(json.dumps(request,allow_nan=False)+'\n');self.process.stdin.flush()
        response=self.receive(15);response['worker_wall_seconds']=time.perf_counter()-begin;return response
    def close(self):
        if self.process.poll() is None:
            try:self.process.stdin.write('{"quit":true}\n');self.process.stdin.flush()
            except BrokenPipeError:pass
            try:self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
        self.log.close()
