"""GPU memory trace for finding memory vLLM does not budget (diagnostics only).

SHAPEQUANT_MEMTRACE=<path.csv> starts a daemon thread in every vLLM process that has CUDA initialized (the
engine core) and appends one row per second:

  t, pid, used_mib (total - free as CUDA reports it, includes other programs), torch_alloc_mib,
  torch_reserved_mib, torch_reserved_peak_mib, torch_alloc_peak_mib, inactive_split_mib, alloc_retries,
  phase

`non-torch` memory of this process = used - reserved - (other programs). `phase` is the most recent marker
set with mark(); the plugin marks model load / profiling / KV allocation / CUDA graphs via wrappers.
"""

from __future__ import annotations

import os
import threading
import time

_PHASE = ["start"]


def mark(phase: str) -> None:
    _PHASE[0] = phase


def _loop(path: str) -> None:
    import torch
    while not torch.cuda.is_initialized():
        time.sleep(0.5)
    pid = os.getpid()
    new = not os.path.exists(path)
    with open(path, "a", buffering=1) as f:
        if new:
            f.write("t,pid,used_mib,torch_alloc_mib,torch_reserved_mib,torch_reserved_peak_mib,"
                    "torch_alloc_peak_mib,inactive_split_mib,alloc_retries,phase\n")
        while True:
            try:
                free, total = torch.cuda.mem_get_info()
                s = torch.cuda.memory_stats()
                f.write(f"{time.time():.1f},{pid},{(total - free) >> 20},{s.get('allocated_bytes.all.current', 0) >> 20},"
                        f"{s.get('reserved_bytes.all.current', 0) >> 20},{s.get('reserved_bytes.all.peak', 0) >> 20},"
                        f"{s.get('allocated_bytes.all.peak', 0) >> 20},{s.get('inactive_split_bytes.all.current', 0) >> 20},"
                        f"{s.get('num_alloc_retries', 0)},{_PHASE[0]}\n")
            except Exception:  # noqa: BLE001
                pass
            time.sleep(1.0)


def _wrap(cls, name: str, phase: str) -> None:
    fn = getattr(cls, name, None)
    if fn is None or getattr(fn, "_sq_mt", False):
        return

    def w(*a, **k):
        mark(phase)
        try:
            return fn(*a, **k)
        finally:
            mark(phase + "_done")

    w._sq_mt = True
    setattr(cls, name, w)


def install() -> bool:
    path = os.environ.get("SHAPEQUANT_MEMTRACE")
    if not path:
        return False
    try:
        from vllm.v1.worker import gpu_worker as gw
        W = gw.Worker
        _wrap(W, "load_model", "load_model")
        _wrap(W, "determine_available_memory", "profile")
        _wrap(W, "initialize_from_config", "kv_alloc")
        _wrap(W, "compile_or_warm_up_model", "warmup_graphs")
        _wrap(W, "execute_model", "serving")
    except Exception:  # noqa: BLE001
        pass
    # record who initializes CUDA in each process (the API front-end should not)
    import torch
    real = torch.cuda._lazy_init
    if not getattr(real, "_sq_mt", False):
        def lazy_init(*a, **k):
            if not torch.cuda.is_initialized():
                import traceback
                with open(path + f".cudainit.{os.getpid()}.txt", "w") as f:
                    f.write("".join(traceback.format_stack()))
            return real(*a, **k)
        lazy_init._sq_mt = True
        torch.cuda._lazy_init = lazy_init
    threading.Thread(target=_loop, args=(path,), daemon=True, name="sq-memtrace").start()
    return True
