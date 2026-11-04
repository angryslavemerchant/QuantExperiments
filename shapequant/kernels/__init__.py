"""GPU kernels for running packed stack quants (shapequant.packed).

gemv.py        PackedLinear + Triton batch-1 GEMV (v1 generic, v2 d=4 8-byte loads)
cuda_gemv.py   CUDA (NVRTC via CuPy) warp-per-row GEMV, shared-memory table: the
               fastest so far on Turing (10-bit tags ~25% of DRAM bandwidth)
cuda_slab.py   v4: slab-loaded tag words (no gain: not memory-parallelism bound)
cuda_tiles.py  v3: mask-sorted tiles with per-tile shared input (setup cost too high)
cuda_l1.py     v5: no shared memory, everything through L1 (random L1 gathers slower)

The CUDA variants need CuPy (cupy-cuda13x); they compile at runtime with NVRTC,
so no host C++ compiler is required. Profiling notes are in the module docstrings.
"""
