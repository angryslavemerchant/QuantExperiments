"""Programmatic dependent launch (PDL, sm_90+) for CuPy RawKernels via the driver's cuLaunchKernelEx.

A kernel launched with PDL may start while the previous kernel in the stream is still running,
once every block of that kernel has executed `griddepcontrol.launch_dependents` (PDL_TRIGGER) or
exited. It must execute `griddepcontrol.wait` (PDL_WAIT) before it reads anything the previous
kernels write, or writes anything they read; that wait returns when the previous grid has
completed and its memory is visible. Work before the wait (copying a codebook into shared memory,
loading weights) overlaps the previous kernel's tail and the launch gap. Captured into CUDA graphs
as programmatic edges.

Kernels include PDL_DEFS: the macros compile to nothing below sm_90 and are no-ops when the kernel
was launched normally, so one binary serves both launch modes.
"""

from __future__ import annotations

import ctypes
import sys

import numpy as np
import torch

PDL_DEFS = r"""
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define PDL_WAIT() asm volatile("griddepcontrol.wait;" ::: "memory")
#define PDL_TRIGGER() asm volatile("griddepcontrol.launch_dependents;" ::: "memory")
#else
#define PDL_WAIT()
#define PDL_TRIGGER()
#endif
"""

_ATTR_PROGRAMMATIC_STREAM_SERIALIZATION = 6


class _Value(ctypes.Union):           # CUlaunchAttributeValue (64 bytes)
    _fields_ = [("i", ctypes.c_int), ("pad", ctypes.c_ubyte * 64)]


class _Attr(ctypes.Structure):
    _fields_ = [("id", ctypes.c_int), ("pad", ctypes.c_ubyte * 4), ("value", _Value)]


class _Config(ctypes.Structure):
    _fields_ = [("gridDimX", ctypes.c_uint), ("gridDimY", ctypes.c_uint), ("gridDimZ", ctypes.c_uint),
                ("blockDimX", ctypes.c_uint), ("blockDimY", ctypes.c_uint), ("blockDimZ", ctypes.c_uint),
                ("sharedMemBytes", ctypes.c_uint), ("hStream", ctypes.c_void_p),
                ("attrs", ctypes.POINTER(_Attr)), ("numAttrs", ctypes.c_uint)]


_lib = None


def _driver():
    global _lib
    if _lib is None:
        _lib = ctypes.WinDLL("nvcuda.dll") if sys.platform == "win32" else ctypes.CDLL("libcuda.so.1")
        _lib.cuLaunchKernelEx.restype = ctypes.c_int
        _lib.cuLaunchKernelEx.argtypes = [ctypes.POINTER(_Config), ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
                                          ctypes.c_void_p]
    return _lib


def supported(device=None) -> bool:
    return torch.cuda.get_device_capability(device)[0] >= 9


_CT = {np.uint64: ctypes.c_uint64, np.int64: ctypes.c_int64, np.int32: ctypes.c_int32, np.uint32: ctypes.c_uint32,
       np.float32: ctypes.c_float}


def launch(kernel, grid: int, block: int, args, shared_mem: int = 0, stream=None):
    """Launch a cupy RawKernel with programmatic stream serialization on the current torch stream.
    args: numpy/cupy scalars (cp.uint64 pointers, cp.int32, cp.float32), as for RawKernel calls."""
    vals = [_CT[type(a)](a) for a in args]
    ptrs = (ctypes.c_void_p * len(vals))(*[ctypes.cast(ctypes.byref(v), ctypes.c_void_p) for v in vals])
    attr = _Attr()
    attr.id = _ATTR_PROGRAMMATIC_STREAM_SERIALIZATION
    attr.value.i = 1          # programmaticStreamSerializationAllowed
    cfg = _Config(grid, 1, 1, block, 1, 1, shared_mem,
                  ctypes.c_void_p(stream if stream is not None else torch.cuda.current_stream().cuda_stream),
                  ctypes.pointer(attr), 1)
    rc = _driver().cuLaunchKernelEx(ctypes.byref(cfg), ctypes.c_void_p(kernel.kernel.ptr), ptrs, None)
    if rc != 0:
        raise RuntimeError(f"cuLaunchKernelEx failed: CUresult {rc}")
