# Spec v2: Learned Weight-Shape Codebook (lean build)

Nov 1, 2026 · @Dr. Sorenzo

## The question

Does a learned codebook of weight shapes, at about 1.5 bits per weight, keep a small Qwen model usable, and do two cheap tricks (a column shuffle and a small codebook of masks) make it better?

Primary target: d = 8 weights per shape, K = 4,096 shapes in the codebook. That is 12 bits per tag covering 8 weights, so 1.5 bits per weight before overhead. It must be judged against plain rounding at 4, 3 and 2 bits, and against snapped ternary as the floor.

This is a post-training, simulated-quantization experiment. No model training in the core work. Compress, decode back to 16-bit, run the model, measure how much it changed. Memory is computed by formula, never measured from files. Fast kernels and real packed storage are out of scope.

The long original spec (Nov 1, 2026) is a menu; this document is the order. Everything here is required unless marked optional.

## Vocabulary (use these words in code and logs)

The method is paint-by-numbers for weight matrices. Every weight keeps its slot; what sits in the slot gets cheaper.

| Term | Meaning | Code name |
| --- | --- | --- |
| Shape / chunk | d consecutive weights in one row, taken along the input dimension | `chunk` |
| d | Weights per shape | `d` |
| Codebook / legend | The list of K learned shapes, each d numbers long, stored once per matrix at 16-bit | `codebook` |
| K | Number of shapes in the codebook. This counts whole shapes, not per-weight values. K = 4,096 means 4,096 shapes, not 4,096 to the power of anything | `K` |
| Tag / index | The integer stored in place of each chunk, pointing at its codebook entry. Costs log2(K) bits, rounded up to whole bits | `idx` |
| Row scale | One 16-bit number per output row; the row is divided by it before chunking and multiplied back on decode | `row_scale` |
| Snap | Replace a chunk with its nearest codebook entry (squared Euclidean distance) | `assign` |
| Repaint / decode | Rebuild the full matrix from tags, codebook and row scales, as ordinary 16-bit weights | `decode` |
| Ternary-equivalent | K = 3 to the power of d. The codebook size at which the method pays the same tag cost as ternary weights chunked the same way | — |

Why d > 1 matters: with one tag per weight (d = 1) every combination of neighbouring values is automatically allowed, so the legend cannot drop combinations that never occur. Once the legend entry is a whole shape, it can. That is the only thing being bought, and it costs codebook storage. If d = 8 does not beat d = 1 at equal bits, the shapes are not helping.

## Setup

All of these are config values with the defaults below, never hard-coded.

| Setting | Default | Note |
| --- | --- | --- |
| Debug model | Qwen2.5-0.5B (base) | Get the pipeline end to end here first |
| Main model | Qwen2.5-1.5B-Instruct | Instruct so the generation check means something |
| Reference precision | bfloat16 | The original model; all KL and agreement numbers compare against it |
| Layers compressed | Every linear layer inside transformer blocks: attention Q, K, V, O and MLP gate, up, down |  |
| Layers left alone | Token embedding, lm\_head, all norms and biases | Counted at 16 bits in memory accounting |
| Chunk direction | Along the input dimension, consecutive weights within one output row |  |
| Codebook scope | One codebook per weight matrix |  |
| k-means | k-means++ init, 25 iterations, fit on a random subsample of at most 1,000,000 chunks, then assign all | Re-seed empty clusters from the chunks with the largest error |
| Seed | Fixed, logged in every result row | k-means init, subsampling and calibration text must all be reproducible |
| Hardware | Any GPU that holds the 1.5B model plus one matrix's chunks for k-means | Batch assignment on GPU for K = 4,096 and up |

## Core pipeline: snap, repaint, run

Every configuration produces an ordinary 16-bit model holding decoded weights, then evaluates it. Nothing is packed into small bits.

1. Load the original model in bfloat16. Run it once on the evaluation text and cache its next-token log-probabilities so every configuration compares against the same reference.
2. Optional pre-step, on by default: 4-bit round-to-nearest group quantization (group size 32, one 16-bit scale per group, integer levels -8 to 7), dequantized back to real values. This is the Q4 baseline and the default input to the codebook. A `start_from_fp16` flag skips it; run that variant for the primary target to measure what the Q4 start costs.
3. Optional shuffle (see Runs): reorder the input columns of the matrix by a fixed permutation, store the permutation, undo it on decode.
4. Row normalization: for each output row compute its root-mean-square, store it as the row scale, divide the row by it. Any earlier group scales are discarded.
5. Chunking: reshape each normalized row into chunks of d. If the row length is not divisible by d, zero-pad and record the padding.
6. Codebook fitting: k-means on the chunks to get K real-valued codewords. Fit on the subsample, assign everything.
7. Snap: replace every chunk with the index of its nearest codeword. Log the fraction of codewords used and the mean squared error between the chunks and their codewords.
8. Repaint: look up every tag, reshape, strip padding, multiply row scales back in, undo the shuffle if any, write the result into the model as bfloat16 weights.
9. Evaluate with the protocol below.

Step 8 must be exactly reproducible: decoding the same tags, codebook and scales twice gives identical tensors. Test this on one matrix before any sweep.

## Runs, in order

Do each stage fully before starting the next. Stop and report if a sanity check fails.

**Stage 0: sanity (debug model).**

- [ ] Original model scored against its own cache gives KL = 0 and 100% top-1 agreement.
- [ ] Q4 round-to-nearest gives mean KL well under 0.1 and perplexity close to the original. If not, the quantizer is broken.
- [ ] Decode is exact (identical tensors on repeat) for one matrix.
- [ ] Memory formula checked by hand for one matrix at d = 8, K = 4,096.

**Stage 1: baselines (main model, run once each).**

| Baseline | How | Role |
| --- | --- | --- |
| Original bfloat16 | No change | Reference; KL = 0 |
| Q4 RTN | Group 32, 16 levels | Should be near-lossless |
| Q3 RTN | Group 32, 8 levels | Scalar comparison at 3 bits |
| Q2 RTN | Group 32, 4 levels | Scalar comparison at 2 bits |
| Ternary RTN | Per group of 32: scale = mean absolute weight, round to -1, 0, +1 | The floor; expected to break the model |

**Stage 2: the target and its neighbours.** Learned codebook, no shuffle, from Q4.

| Cell | d | K | Bits per weight (tags) | Why |
| --- | --- | --- | --- | --- |
| Target | 8 | 4,096 | 1.5 | The idea at the memory level we care about |
| Neighbour: shorter shapes | 4 | 4,096 | 3.0 | Same codebook, half the coverage; shows what d costs |
| Neighbour: bigger codebook | 8 | 65,536 | 2.0 | Same shapes, more entries; shows what K buys |
| Ternary-equivalent | 8 | 6,561 | 1.58 | Same tag cost as ternary; learned shapes vs the fixed grid |
| Scalar control | 1 | 3 | 1.58 (2 rounded) | Learned per-weight values; if the target does not beat this, shapes are not helping |

Also run the target cell with `start_from_fp16` to measure the Q4-start cost.

**Stage 3: column shuffle.** Before row normalization, reorder the matrix's input columns so columns with similar statistics sit next to each other. Default rule: sort columns by their mean absolute value over all rows; config flag to use the calibration-input scale instead (same calibration as Stage 5). One permutation per matrix, stored at 16 bits per column index. Rerun the Stage 2 target and ternary-equivalent cells with the shuffle on. Report the delta.

**Stage 4: mask codebook (the main new idea).** Keep a small set of M column permutations (default M = 16) and let each row pick the one under which its chunks snap with the least error. Mask tag cost is log2(M) bits per row, negligible.

- Fitting, greedy: start from M random permutations plus the Stage 3 sort; alternate (a) assign each row its best mask by total snap error, (b) refit the codebook on the re-ordered chunks; 5 rounds.
- Rerun the target cell with masks on, with and without the Stage 3 sort as the initial mask. Report the delta and the mask usage histogram.
- If Stage 3 and Stage 4 both show no gain within noise, stop here and write that up as the finding.

**Stage 5 (optional): codebook fine-tune.** Freeze tags and masks, train only codebook entries and row scales to minimize KL against the original on about 128 sequences of calibration text (never the evaluation text), a few hundred steps. Run on the best Stage 2 to 4 configuration only. This is the cheapest cousin of what trained-ternary methods do and the most likely thing to turn almost-works into works.

## Measurement

Mean KL divergence against the original model is the headline number. The rest catch what the mean hides.

**Text metrics.** WikiText-2 test split, about 100,000 tokens, context 2,048, non-overlapping windows. Compare full next-token distributions at every position, in float32 from log-probabilities.

| Metric | What it catches |
| --- | --- |
| Mean KL(original ‖ compressed) | Overall behaviour change |
| 99th-percentile KL | Rare large failures from badly matched chunks in important spots |
| Top-1 agreement (%) | How often greedy output would diverge |
| Perplexity | Comparison with published numbers |

**Generation check.** First 200 GSM8K test problems, greedy, up to 512 new tokens, exact match on the final number, one fixed prompt template for every configuration. Catches compounding errors the text metrics cannot see.

**Codebook health.** Fraction of codewords used, mean squared chunk reconstruction error against the original 16-bit weights, and for Stage 4 the mask usage histogram. Low usage means k-means collapsed.

**Memory, by formula.** For one compressed matrix with R output rows and C input columns:

```latex
\text{bits} = R \cdot \lceil C/d \rceil \cdot \lceil \log_2 K \rceil + K \cdot d \cdot 16 + R \cdot 16 + \text{shuffle} + \text{masks}
```

The terms are tags, codebook, row scales, then C × 16 for a Stage 3 permutation and M × C × 16 + R × log2(M) for Stage 4 masks (zero when off). Report both the rounded-up tag width and the ideal log2(K). RTN baselines: R × C × bits + (R × C / 32) × 16 for group scales; ternary at 2 bits per weight, also reporting the ideal 1.58. Per configuration report effective bits per weight over compressed layers, total MB including the 16-bit layers, and percent of the original size.

## Deliverables and acceptance

One script reruns everything from a config file. Done means every row below exists and the summary answers its four questions.

1. `results.csv`, one row per run: stage, method, d, K, shuffle on/off, masks on/off (and M), fine-tune on/off, start (Q4 or fp16), seed, effective bits per weight (rounded and ideal), total MB, percent of original, mean KL, 99th-percentile KL, top-1 agreement, perplexity, GSM8K accuracy, codebook usage, reconstruction error, compression wall-clock.
2. Two plots: mean KL vs effective bits per weight with every run as a labelled point and baselines as distinct markers, log scale on KL; the same for GSM8K accuracy.
3. A one-page summary answering, in this order:
   1. Where does the d = 8, K = 4,096 target land relative to Q4, Q3, Q2 and ternary RTN on KL and GSM8K?
   2. Does the target beat the d = 1, K = 3 scalar control at the same tag cost? (If not, shapes are not helping.)
   3. What did the shuffle and the mask codebook each add, in KL and GSM8K, with the memory they cost?
   4. What did the Q4 start cost versus fp16, and (if run) what did fine-tuning add?
4. Code, config and a README with the exact reproduce command.

Read the result honestly. A null result on shuffle or masks is a finding, not a failure, and gets written up the same way.

## Out of scope and pitfalls

Not in this build:

- The full d × bits grid from the original spec. Only the cells in Stage 2.
- Fast inference kernels, lookup-table execution, packed storage, GGUF export.
- Random-rotation preprocessing (QuIP#, QuaRot, TurboQuant style). It erases the structure a learned codebook depends on; note it as a comparison for later, do not implement it.
- Shared codebooks across matrices, importance-weighted k-means. Phase 2 candidates if Stage 4 shows promise.
- Training the model itself. Stage 5 trains only the codebook and row scales.

Do not:

- Evaluate on calibration or k-means-subsample text. Keep evaluation text separate.
- Compress the embedding or lm\_head.
- Change the GSM8K prompt template between runs.
- Let a mean-KL number stand alone; the 99th percentile and GSM8K must be in the same row.
- Interpret K as a per-weight vocabulary anywhere in code, comments or logs. It is the count of whole shapes.
- Spend time on Stage 4 before Stage 2 and 3 have complete rows. The baseline comes first.
