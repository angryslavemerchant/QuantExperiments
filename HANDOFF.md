# Handoff (2026-11-08)

State of the project for a fresh session (for example on the work 3090 machine). The previous
sessions' notes lived outside the repo; this file carries what matters.

## What this is

**shapequant**: a learned weight-shape codebook quantizer. The shipped artifact is
**Qwen3.8-27B at ~2.5 bpw** (Fisher-allocated non-uniform recipe, fine-tuned):

- Model: private HF repo `luckymushy/Qwen3.8-27B-shapequant-2.5bpw`, file `s16_25_ft.safetensors`
  (9.75 GB; 2.51 bpw in the quantized blocks, 2.90 bpw whole file incl. q8 lm_head + 4-bit embedding).
- Quality: KL 0.1198 vs bf16 (UD-IQ2_S: 0.1408). GSM8K 93.3 / MMLU-Pro 71.5 / IFEval 88.0
  (UD-IQ2_S: 91.1 / 65.9 / 89.1). wikitext NLL in vLLM 2.113 (bf16 2.121).
- Runtime: a **vLLM quantization plugin** (`shapequant/vllm_ext/`, registered through the
  `vllm.general_plugins` entry point; `core.py` is vLLM-free, `plugin.py` hooks vLLM). vLLM 0.33,
  CUDA 13 wheels: NVIDIA driver >= 580 required.
- Deploy package: `deploy/` (`setup_server.sh`, `serve.sh`, `RUNBOOK.md`). Start with the RUNBOOK.

## Measured speeds (Qwen3.8-27B, tok/s)

Single request, decode:

| GPU | ours in vLLM | ours standalone | UD-IQ2_S (llama.cpp) | UD-Q4_K_M | bf16 vLLM |
|---|---|---|---|---|---|
| RTX 3090 | **50.3** (commit 9665543; 43.3 before) | 50.7 | 45.3 | 40.6 | does not fit |
| RTX PRO 6000 | 68 (before 9665543) | 100 | 104 | 77 | 30 |

3090 vLLM batched (vllm_smoke, 256-token prompts, aggregate tok/s): conc 1/4/8/32 = 46.8/134/222/343.
3090 serve.sh defaults: ~130k KV tokens. Long context: 128k fits with bf16 KV (~159k KV tokens,
GPU_UTIL 0.94, MAX_SEQS 2), 262k with fp8 KV (~316k tokens).
Speculative decoding (3090, before 9665543): DSpark 79 tok/s single (base 38), 123 at 4 requests;
MTP (K=3) 58.5 single, 164 at 4 requests. Vision (bf16 tower, `VISION=1` export) works.

## Where single-decode time goes (3090, after 9665543: ~19.9 ms/token)

From a torch-profiler trace of vLLM decode: our `gemv_runs_b` matmuls dominate; the q8 lm_head was
3.7 ms and is now ~1.5 ms (tile-major layout, ~880 GB/s); DeltaNet / attention / norms ~1.5 ms;
vLLM CPU overhead is negligible (async scheduling and full CUDA graphs changed nothing).
At B=1 the lane-major batch-1 kernel (`proto_runs.gemv`) is still 10-25% faster than the tile
kernel on several shapes (down_proj 48.8 vs ~58 us), but needs a second copy of the weights that
does not fit in 24 GB. Next lever: a B=1 kernel that reads the tile layout.

## Open items

1. **Long-context retrieval accuracy** at 128k (bf16 KV) / 262k (fp8 KV): `scripts/long_ctx.py`
   (needle test; dataset id fixed, never completed a run).
2. **Quality side-by-side vs UD-Q4_K_M**: `scripts/quality_prompts.py` (ours done: 20/20 sensible;
   the llama.cpp download failed with a 504).
3. **RUNBOOK numbers** predate 9665543 (it says ~36 tok/s single; now ~47 through the server).
   DSpark / MTP speeds should be re-measured on the new kernels.
4. **MTP in serve.sh**: needs the MTP export (`scripts/spec_stage.sh` shows the steps:
   bf16 embed/head + `mtp.safetensors`); only `SPEC=dspark` is wired up.
5. **serve.sh with SPEC=dspark / VISION=1** verified only through test scripts plus a dry run of
   the argument assembly, not launched end to end through serve.sh.
6. **5090 / PRO 6000 no-regression check** of the 2026-11-08 kernel changes (never ran: rental
   hosts failed).
7. **Thinking-mode evals** (GPQA diamond, MATH-500) of ours vs the GGUFs never completed;
   lighteval needs a fix for huggingface_hub 2.4 (`list_models(model_name=)` was removed).
8. Large-batch vLLM throughput (>= 64 concurrent) is below bf16 on the PRO 6000; irrelevant for a
   personal server.
9. **Rebuild the 27B with mixed calibration + fine-tune data** (needs a 96 GB card, like stage 16).
   Tested 2026-11-12 on Qwen3.5-4B (d4 K1024 G4 LDLQ + 256-step codebook fine-tune, 2.52 bpw,
   `configs/calmix/`; the fine-tune draws fresh sequences from the same mixture). Post-fine-tune KL
   by eval domain, FineWeb-only / public mix / own answers / combined:
   wikitext .204/.211/.222/.214, code .208/.133/.130/.131, chat .164/.131/.163/.135,
   math .140/.069/.085/.087, de .318/.228/.385/.235, zh .362/.296/.412/.301,
   the model's own thinking text .162/.125/.102/.108.
   Public mix = FineWeb-edu 40, codeparrot 20, UltraChat 20, MATH algebra 10, de/zh Wikipedia 5+5.
   Combined (best overall) = own answers 35 (`scripts/selfgen_calib.py`: the model's thinking-on
   answers to chat / MATH / Magicoder prompts), FineWeb 30, code 15, chat 10, de 5, zh 5.
   Own answers alone hurt de/zh. Caveats: code/math evals share their source datasets with the
   mix; one seed (seed swing ~4-5% KL). Only ~300k tokens of own answers are needed (35% of
   calibration + fine-tune data). The e-means codebook fit (`kmeans.fit: ema`) is within seed noise
   at equal compute; 4x compute or 100 Lloyd iterations gave 5-7% (one seed).

## Key scripts

- `scripts/export_vllm_checkpoint.py`: packed file -> vLLM checkpoint (`--vision`, `--embed-head`).
- `scripts/vllm_smoke.py`: wikitext NLL + greedy chats + throughput through vLLM.
- `scripts/diag_vllm_paths.py`: per-component correctness on the real file.
- `scripts/bench_small_b.py`: per-matrix kernel timings at small batch.
- `scripts/dev3090_stage.sh`, `overnight_chain.sh`, `spec_stage.sh`, `wishlist_stage.sh`,
  `vision_stage.sh`, `vision_check.py`, `long_ctx.py`, `quality_prompts.py`: the rental-box stages
  (paths assume `/workspace/vllm_env` and `/workspace/vllm_ckpt`; adapt for a local machine).

## Working preferences (from the user)

- Commit locally with clear messages; push only when asked.
- Every change passes tests and a correctness check (wikitext NLL ~2.11, greedy outputs) before
  speed counts.
- Use they/them for the user.
- Vast.ai rentals (`vast/ops.py`, see CLAUDE.md and `.claude/skills/vast-ops`) need an explicit
  dollar cap from the user each time; never commit `vast/secrets.env`, weights or datasets.
  For interactive debug boxes: setup job, then live SSH work, with `stall_minutes` >= the TTL.
- On the home PC GPU runs went to rentals; on the work 3090, running locally is the point.
- Time matters more than small cost differences; prefer fast, reliable hardware.
- Default formats are locked: q8 lm_head + 4-bit embedding; masks (M>1) dropped; 2.5 bpw
  non-uniform is the focus.

## Known pitfalls

- vLLM pickles the quantization config into the EngineCore process (`__init__` does not rerun):
  the Qwen3.5 embedding patch is applied in `ShapeQuantConfig.__setstate__`; the head refuses to
  build if the embedding was not ours (that bug gave NLL 12.4 = log vocab).
- One CuPy build only (`cupy-cuda13x`); two installed gives `CUDA_ERROR_INVALID_IMAGE`.
- vLLM `max_num_seqs` must stay within the DeltaNet state slots; the 1024 default fails.
- Scripts that start vLLM's offline `LLM` need an `if __name__ == "__main__":` guard (spawn).
- GPU-marked tests fail with "Invalid device id" when CUDA is hidden; run them on a GPU.
