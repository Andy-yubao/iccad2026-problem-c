"""One active numerical process, at most two CPU execution slots."""
import os
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS',
              'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[_name]='1'
os.environ['OMP_MAX_ACTIVE_LEVELS']='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'

def configure():
    import ctypes
    import torch
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetCurrentProcess.restype=ctypes.c_void_p
    handle=kernel.GetCurrentProcess()
    current=ctypes.c_size_t();system=ctypes.c_size_t()
    kernel.GetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)]
    if not kernel.GetProcessAffinityMask(handle,ctypes.byref(current),ctypes.byref(system)):raise ctypes.WinError(ctypes.get_last_error())
    allowed=[i for i in range(64) if current.value&(1<<i)]
    # Separate physical cores on the current x86 host, with a safe fallback.
    requested=os.environ.get('E36_CPU_SLOT')
    chosen=[int(requested)] if requested is not None else allowed[:1]
    if chosen[0] not in allowed:raise RuntimeError('Requested CPU is outside inherited process affinity')
    kernel.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
    if not kernel.SetProcessAffinityMask(handle,sum(1<<i for i in chosen)):raise ctypes.WinError(ctypes.get_last_error())
    torch.set_num_threads(1)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    return {'pid':os.getpid(),'affinity':chosen,
            'torch_intraop':torch.get_num_threads(),'torch_interop':torch.get_num_interop_threads(),
            'workers':1,'max_cpu_execution_slots':1,
            'environment':{name:os.environ[name] for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS',
                'OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS')}}
