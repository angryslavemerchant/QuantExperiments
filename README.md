# QuantExperiments: learned weight-shape codebook

Post-training, simulated-quantization experiment: compress every linear layer
inside a Qwen2.5 model's transformer blocks, decode back to bf16, and measure
how far the model's behaviour moves. The full plan is in
[`Spec v2 Learned Weight-Shape Codebook (lean build).md`](Spec%20v2%20Learned%20Weight-Shape%20Codebook%20(lean%20build).md).

## Layout

| Path | What |
| --- | --- |
| `shapequant/rtn.py` | Round-to-nearest group baselines (Q4/Q3/Q2, ternary) |
| `shapequant/codebook.py` | Chunking, k-means codebook fit, snap (`assign`), repaint (`decode`) |
| `shapequant/memory.py` | Memory accounting by formula |
| `shapequant/evaluate.py` | WikiText-2 KL / top-1 / perplexity, GSM8K generation check |
| `shapequant/gguf_ref.py` | Published llama.cpp GGUF quants as reference points (block linears only) |
| `shapequant/pipeline.py` | Applies a method to every block linear, collects stats |
| `scripts/run_experiment.py` | One entry point, driven by a YAML config |
| `scripts/run_parallel.py` | Runs a config as N parallel shards on one GPU, then merges |
| `configs/` | One config per stage |
| `vast/` | Guarded Vast.ai tooling for rented-GPU runs |

## Reproduce

```bash
pip install torch transformers datasets pyyaml matplotlib   # see requirements-remote.txt for pinned versions
python scripts/run_experiment.py --config configs/stage0_debug.yaml      # sanity, Qwen2.5-0.5B
python scripts/run_experiment.py --config configs/stage1_baselines.yaml  # baselines, Qwen2.5-1.5B-Instruct
python scripts/run_experiment.py --config configs/stage1_baselines_7b.yaml  # baselines, Qwen2.5-7B-Instruct
python scripts/run_experiment.py --config configs/stage2_codebook_7b.yaml   # codebook cells, Qwen2.5-7B-Instruct
python scripts/run_parallel.py --config configs/stage2b_sweep_7b.yaml --shards 3  # bits-matched d sweep + GGUF references
python scripts/run_parallel.py --config configs/stage2c_weighted_7b.yaml --shards 2  # importance-weighted k-means
python -m pytest tests -q
```

Each run writes `results.csv`, per-run JSON, GSM8K generations and the two
plots to the config's `output_dir` under `artifacts/qwen_s01/` (one folder for all stages, so a remote run declares a single artifact path that always exists). Rerunning the same
config skips runs already present in `results.csv`.

## Implementation notes

- **Reference cache.** Full float32 reference distributions for ~100k tokens
  would be ~60 GB, so the cache holds the original model's final hidden
  states instead. The lm_head is never compressed, so reference
  log-probabilities rebuilt from the cache are identical. Log-probabilities
  are computed in float32 (float32 lm_head matmul on the bf16 hidden states).
- **Evaluation positions.** 49 non-overlapping windows of 2,048 tokens; every
  position with a next-token target (2,047 per window, 100,303 total) is
  scored.
- **RTN scale.** Symmetric, one fp16 scale per group of 32, integer levels
  -2^(b-1)..2^(b-1)-1. The signed max-magnitude weight maps exactly to the
  lowest level (llama.cpp Q4_0 style), so all 2^b levels are usable.
- **16-bit storage.** Group scales, row scales and codebook entries are
  rounded to fp16 before decoding, matching the memory accounting.
- **`start` column.** `fp16` means the method was applied to the original
  bf16 weights; `q4` means to the Q4 RTN dequantized weights (codebook only).
- **Memory.** MB = 10^6 bytes. Embedding/lm_head (tied in Qwen2.5-0.5B/1.5B,
  counted once), norms and biases count at 16 bits.
- **Stage 0 on 0.5B.** Qwen2.5-0.5B passed every sanity check except Q4
  near-lossless (mean KL 0.126, perplexity +12.8%); its Q4 weight error
  (0.8% relative) is normal for 4-bit, so this reads as small-model
  fragility. Stage 0 is repeated on Qwen2.5-7B (`configs/stage0_7b.yaml`).
- **Spec deviation.** The d = 8, K = 65,536 Stage 2 cell is deliberately
  dropped: its per-matrix codebook alone adds ~1.25 bits/weight, and the K/V
  projections have fewer chunks than codewords.
- **Main model for Stage 2 onward.** Stage 2 runs on Qwen2.5-7B-Instruct
  instead of 1.5B (user decision, 2026-11-02), with the Stage 1 baselines
  repeated on 7B (`configs/stage1_baselines_7b.yaml`) so every comparison is
  within one model. The 1.5B Stage 1 results are kept for reference. On 7B
  the embedding and lm_head are untied and both count at 16 bits.
- **GGUF reference points.** `method: gguf` takes only the 196 block linears
  from a published llama.cpp GGUF (bartowski's imatrix quants of
  Qwen2.5-7B-Instruct), dequantized with gguf-py; embedding, lm_head, norms
  and biases stay 16-bit like every other method. Bits per weight are the
  GGUF tensors' real storage over the block linears, so mixed-precision files
  report their true average. A per-matrix relative-error guard (> 0.5 aborts)
  catches any tensor-mapping mistake.
- **Revised plan after Stage 2 (2026-11-02).** Next steps, in order:
  GGUF references and a bits-matched d sweep (Stage 2b); activation-weighted
  k-means; then shuffle and masks (spec Stages 3-4) under the weighted
  objective; then codebook fine-tuning (Stage 5) on the best configuration.
- **Importance weighting (Stage 2c).** `weighting: act_row` fits k-means
  on per-element weighted squared error with weight s_r² · h_c, where s_r is
  the row scale and h_c = E[x_c²] is the input second moment per column
  (`shapequant/calibrate.py`). That is the diagonal approximation of each
  layer's output error, the same statistic as llama.cpp's imatrix. h comes from
  128 × 2,048 tokens of FineWeb-Edu, never the WikiText-2 evaluation text.
  Ablations: `row` (s_r² only, plain weight MSE) and `act` (h_c only).
