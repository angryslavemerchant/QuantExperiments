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
| `shapequant/pipeline.py` | Applies a method to every block linear, collects stats |
| `scripts/run_experiment.py` | One entry point, driven by a YAML config |
| `configs/` | One config per stage |
| `vast/` | Guarded Vast.ai tooling for rented-GPU runs |

## Reproduce

```bash
pip install torch transformers datasets pyyaml matplotlib   # see requirements-remote.txt for pinned versions
python scripts/run_experiment.py --config configs/stage0_debug.yaml      # sanity, Qwen2.5-0.5B
python scripts/run_experiment.py --config configs/stage1_baselines.yaml  # baselines, Qwen2.5-1.5B-Instruct
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
- **Spec deviation.** The d = 8, K = 65,536 Stage 2 cell is deliberately
  dropped: its per-matrix codebook alone adds ~1.25 bits/weight, and the K/V
  projections have fewer chunks than codewords.
