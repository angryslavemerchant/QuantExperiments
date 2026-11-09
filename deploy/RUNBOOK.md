# Running the shapequant 27B as a server (RTX 3090 / any 24 GB+ NVIDIA GPU)

What you get: an OpenAI-compatible endpoint (`/v1/chat/completions`, `/v1/models`) serving
Qwen3.8-27B at 2.5 bpw through vLLM + the shapequant plugin. Thinking mode, 32k context per
request, up to 16 requests at once by default, protected by an API key.

**Single user on a 24 GB card (the work 3090), everything on** — images, DSpark speculative
decoding, RecoverSSM, fp8 KV cache (measured 2026-11-09, WSL2):

```bash
MODEL_DIR=~/models/q27-sq-vl SPEC=dspark RSSM=1 KV_DTYPE=fp8 MAX_LEN=81920 MAX_SEQS=2 TOOL_PARSER=qwen3_coder bash deploy/serve.sh
python scripts/smoke_server.py --api-key "$SQ_API_KEY"      # in a second shell: 5 checks, "ALL OK"
```

| | |
|---|---|
| Weights in VRAM | 14.0 GiB: text model 9.85, DSpark drafter ~2.5, vision tower 0.92, FlashInfer workspace 0.75 |
| KV cache | ~87k tokens (fp8, default 2.5 GB headroom) shared by the running requests; `MAX_LEN` up to ~82k for one request. `HEADROOM_MB=1536` gives ~117k / 98k but spills (prefill ~5x slower) once the desktop holds ~0.6 GB (remote session + browser) |
| Decode, one request | ~122 tok/s on a short context; 90 at 8k, 68 at 32-64k, 61 at 90k already in the prompt (without `SPEC`: 48 / 47 / 44 / 42 / 40) |
| Prompt processing | 8k in ~4 s, 32k in ~17 s, 64k in ~43 s, 90k in ~67 s (~2000 tok/s falling to ~1350; INT8 prefill GEMMs) |
| Same card, llama.cpp Q4_K_M | prefill 1303 / 1184 / 1050 tok/s at 8k / 32k / 64k; decode 41 / 38 / 33 at 8k / 32k / 64k |
| Checks | needle retrieval 15/15 at 8k-128k (fp8 KV), GSM8K 5-shot identical with/without INT8 prefill |

Needs the patched RecoverSSM venv (`deploy/patches/make_recoverssm_venv.sh`, once) and the
checkpoint with the vision tower (section 3, "Images").

## 0. Requirements

- Linux, or Windows with **WSL2** (Ubuntu 22.04/24.04). vLLM does not run on native Windows.
  - WSL2: install the normal Windows NVIDIA driver; inside WSL `nvidia-smi` must show the GPU.
- NVIDIA driver **580 or newer** (`nvidia-smi` top right must show "CUDA Version: 13.0" or higher).
  The vLLM 0.33 packages are built for CUDA 13; on an older driver (e.g. 570 = CUDA 12.8) vLLM fails
  with `libcudart.so.13: cannot open shared object file`. Updating the driver is the fix.
- Python 3.10-3.12, git, ~40 GB free disk (15 GB vLLM environment, 10 GB model, caches).
- A C compiler, Python headers and the CUDA toolkit (`nvcc`): Triton and FlashInfer compile kernels
  at engine start. Rental images ship these; a fresh Ubuntu / WSL2 install does not (errors:
  `Failed to find C compiler`, `Could not find nvcc`). On WSL2 Ubuntu 24.04 (as root):
  ```bash
  apt-get install -y build-essential python3.12-dev
  wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
  dpkg -i cuda-keyring_1.1-1_all.deb && apt-get update && apt-get install -y cuda-toolkit-13-2
  ```
  Then run vLLM with the venv's `bin` (for `ninja`) and CUDA on `PATH`:
  `export PATH="$HOME/sq-venv/bin:/usr/local/cuda/bin:$PATH" CUDA_HOME=/usr/local/cuda`
  (`No such file or directory: 'ninja'` means the venv's `bin` is missing from `PATH`).
  On WSL2 do **not** set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (the rental-box scripts
  do): the plugin's per-shape tuning at startup then crawls (22+ min stuck in `PackedShards.tune`,
  CPU-bound, GPU idle) instead of finishing in ~2 min.
- No `pip` in the system Python (Ubuntu 24.04)? Install uv standalone first:
  `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- A Hugging Face **read** token with access to `luckymushy/Qwen3.8-27B-shapequant-2.5bpw`.

## 1. Get the code onto the machine

```bash
git clone https://github.com/angryslavemerchant/QuantExperiments.git quant-experiments && cd quant-experiments
```

(Without GitHub access: `git bundle create qe.bundle HEAD` on a machine that has the repo, copy the
file over, then `git clone qe.bundle quant-experiments`.)

## 2. One-time setup (~15-25 min, mostly downloads)

```bash
HF_TOKEN=hf_xxx bash deploy/setup_server.sh
```

Creates `~/sq-venv`, installs vLLM + CuPy + this repo, downloads the packed model and exports a
vLLM checkpoint to `~/models/q27-sq`. Safe to re-run.

## 3. Start the server

```bash
export SQ_API_KEY=$(openssl rand -hex 24)      # keep this; clients need it
echo $SQ_API_KEY > ~/.sq_api_key && chmod 600 ~/.sq_api_key
bash deploy/serve.sh
```

The first start compiles the kernels and tunes each matrix shape: wait for
`Application startup complete` (a few minutes; later starts are faster).

Measured on an RTX 3090 (2026-11-08): wikitext NLL 2.113 (= the reference); ~50 tok/s decode for
one request (llama.cpp on the same card: UD-IQ2_S 45, UD-Q4_K_M 41), ~220 tok/s total at 8
requests, ~340 tok/s at 32. Prompts are processed at ~2000-2400 tok/s up to 32k tokens (INT8
prefill GEMMs; llama.cpp Q4_K_M on the same card: ~1200-1300). The KV cache room (shared by all
running requests) is logged at start as `GPU KV cache size: N tokens`.

Keep it running after you log out: run it inside `tmux` (`tmux new -s sq`, start, then `Ctrl-b d`
to detach; `tmux attach -t sq` to come back).

Useful settings (environment variables for `serve.sh`):

| Variable | Default | What it does |
|---|---|---|
| `HOST` | `127.0.0.1` | Address to listen on. Use your Tailscale IP (below) to reach it from home |
| `PORT` | `8000` | Port |
| `MAX_LEN` | `32768` | Max tokens per request (prompt + answer) |
| `MAX_SEQS` | `16` | Max requests processed at once |
| `GPU_UTIL` | computed | Share of the *total* GPU memory vLLM may use (the rest of its share becomes KV cache). Default: (free VRAM at start − `HEADROOM_MB`) / total, so memory other programs hold is left alone |
| `HEADROOM_MB` | `2560` | VRAM kept free beyond what is in use at start. vLLM itself uses ~1.2-1.7 GB more than it budgets (full stack, measured 2026-11-09); the rest covers desktop apps opened later. Too little shows as prefill dropping to ~370 tok/s (VRAM spilled to system RAM) |
| `CHUNK` | `8192` | Prefill chunk (`--max-num-batched-tokens`); 4096 frees ~0.3 GB for KV at ~10% slower prompts |
| `KV_DTYPE` | `auto` | `fp8` halves KV-cache memory: longer contexts / more requests |
| `TOOL_PARSER` | off | Enables tool / function calling (see notes) |
| `SPEC` | off | `dspark`: speculative decoding with the DSpark drafter (FP8 build); `mtp`: Qwen's MTP head |
| `RSSM` | off | `1`: RecoverSSM (with `SPEC`): much more KV room; needs the patched venv |
| `MODEL_DIR` | `~/models/q27-sq` | Checkpoint; `~/models/q27-sq-vl` = the one with the vision tower |

### Long context

The model supports up to 262k tokens. Without a drafter, fp8 KV fits `MAX_LEN=262144` (needle
retrieval 49/49 up to 256k, measured 2026-11-08). With DSpark + RecoverSSM (text-only checkpoint,
computed `GPU_UTIL`): ~142k fp8 KV tokens, `MAX_LEN=131072` fits, needles 15/15 at 8k / 64k / 128k,
a 128k prompt takes ~110 s.

```bash
KV_DTYPE=fp8 MAX_LEN=262144 MAX_SEQS=1 bash deploy/serve.sh                       # longest, no drafter
SPEC=dspark RSSM=1 KV_DTYPE=fp8 MAX_LEN=131072 MAX_SEQS=1 bash deploy/serve.sh    # fast decode + 128k
```

The GPU-memory numbers above are what vLLM's own accounting allows. **On Windows/WSL2, going over
the card's memory does not fail**: the driver moves part of it to system RAM and everything runs
~5x slower (first request fast, later ones crawl). The computed `GPU_UTIL` avoids it; if you set
`GPU_UTIL` by hand keep weights + KV + ~2 GB of Windows/desktop use under 24 GB, or set the NVIDIA
Control Panel's "CUDA - Sysmem Fallback Policy" to "Prefer No Sysmem Fallback" (then it is an
out-of-memory error instead).

### Faster single requests: DSpark speculative decoding

Measured on the 3090 (WSL2, thinking-mode prompts, `scripts/spec_bench.py`): **~105-125 tok/s
for one request instead of 51**, ~205 tok/s total at 4 requests. The default drafter is the FP8
build (`inference-optimization/Qwen3.8-27B-DSpark-FP8-DYNAMIC`, ~2.5 GB of VRAM; the bf16 original
is slower and larger). The draft length (`SPEC_K`, default 7) was swept: 6 and 7 tie, 5 / 8 / 10 are
10-25% slower. Acceptance varies with the text (2.5-3.2 accepted tokens per draft).

```bash
SPEC=dspark RSSM=1 KV_DTYPE=fp8 MAX_LEN=65536 MAX_SEQS=4 bash deploy/serve.sh
```

The first start downloads the drafter and compiles (can take ~30 min); later starts are faster.
Best when you are the only user; for many parallel requests use MTP (below) or leave it off.
`RSSM=1` (RecoverSSM, vLLM PR #61309 patched into a second venv by
`deploy/patches/make_recoverssm_venv.sh`) stops the DeltaNet layers from keeping a full state copy
per draft token: ~3x the KV room with the drafter. fp8 KV next to the drafter works since the
plugin's sliding-window block fix (2026-11-09; before it the drafter's pages were padded ~29x).

### Several requests at once: MTP speculative decoding

Qwen3.8's own multi-token-prediction head: 78 tok/s for one request, **~235 tok/s total at 4
requests** (vs 146 without), KV cache ~42k tokens. Needs a checkpoint with the MTP head and a bf16
embedding / lm_head (+~3 GB; downloads the head from the base model):

```bash
~/sq-venv/bin/python scripts/export_vllm_checkpoint.py --packed-hf luckymushy/Qwen3.8-27B-shapequant-2.5bpw:s16_25_ft.safetensors \
    --out ~/models/q27-sq-mtp --embed-head bf16 --mtp
MODEL_DIR=~/models/q27-sq-mtp SPEC=mtp MAX_LEN=16384 MAX_SEQS=4 bash deploy/serve.sh
```

`SPEC_K` defaults to 2 (2 and 3 tie; 4 and 5 are slower).

For comparison on the same card: llama.cpp UD/lmstudio Q4_K_M 41 tok/s single, 72 with its MTP head.

### Images (vision)

The base model's vision tower can be added (kept in bf16, +1 GB; the text model stays 2.5 bpw).
Export with it at setup time (if you already set up text-only, `rm -rf ~/models/q27-sq` first):

```bash
VISION=1 HF_TOKEN=hf_xxx bash deploy/setup_server.sh
```

Or, keeping a text-only checkpoint next to it, from the local packed file (~5 min, CPU only):

```bash
~/sq-venv/bin/python scripts/export_vllm_checkpoint.py --packed <path to s16_25_ft.safetensors> --out ~/models/q27-sq-vl --vision
```

`serve.sh` then accepts images automatically (`IMAGES=4` per request by default). Send them as
OpenAI `image_url` parts (an `https://` URL or a `data:image/png;base64,...` URL). Measured on the
3090 (2026-11-09, with DSpark + RecoverSSM + fp8 KV): both test images (a synthetic number/shapes
image and a COCO photo of two cats with remotes) described correctly, ~0.9 s per short answer.
The vision tower costs ~1.4 GB in all (0.92 weights + activation room), roughly 40k fp8 KV tokens.

## 4. Check it

One command, from the server or any machine that reaches it (stdlib Python only):

```bash
python3 scripts/smoke_server.py --endpoint http://127.0.0.1:8000 --api-key "$SQ_API_KEY"
```

It checks the model list and that requests without the key are refused, a chat answer, streaming
(time to first token), thinking (reasoning + answer) and an image (skipped on a text-only
checkpoint), and ends with `ALL OK` or the failed checks.

By hand, on the server:

```bash
curl -s localhost:8000/health && echo ok
curl -s localhost:8000/v1/chat/completions -H "Authorization: Bearer $SQ_API_KEY" \
  -H 'Content-Type: application/json' -d '{
    "model": "qwen3.8-27b-sq",
    "messages": [{"role": "user", "content": "What is 7 times 8?"}],
    "max_tokens": 512}' | python3 -m json.tool
```

The reply has `message.content` (the answer) and `message.reasoning_content` (the thinking).
To turn thinking off for a request add `"chat_template_kwargs": {"enable_thinking": false}`.
Recommended sampling (Qwen): temperature 0.6, top_p 0.95, top_k 20 with thinking;
0.7 / 0.8 / 20 without.

## 5. Reach it from home

### Option A: Tailscale (recommended)

Both machines join your private Tailscale network; nothing is exposed to the internet and no
router or firewall changes are needed. **Check your workplace's IT policy first.**

1. Install on the work machine (inside WSL2 if you use it): `curl -fsSL https://tailscale.com/install.sh | sh`
   then `sudo tailscale up` and log in.
2. Install Tailscale on the home PC and log in with the same account.
3. On the work machine: `tailscale ip -4` (e.g. `100.101.102.103`), then start the server bound
   to it: `HOST=100.101.102.103 bash deploy/serve.sh`
4. From home: `http://100.101.102.103:8000/v1` with the API key.

### Option B: SSH tunnel

If you can SSH into the work machine (directly or through a jump host):

```bash
ssh -N -L 8000:127.0.0.1:8000 you@work-machine
```

then use `http://127.0.0.1:8000/v1` on the home PC while the tunnel is open. The server keeps
`HOST=127.0.0.1` (not reachable from the network at all).

## 6. Calling it

Any OpenAI-compatible client works: base URL `http://<host>:8000/v1`, API key = `SQ_API_KEY`,
model `qwen3.8-27b-sq`.

```python
from openai import OpenAI
client = OpenAI(base_url="http://100.101.102.103:8000/v1", api_key=open("key.txt").read().strip())
r = client.chat.completions.create(
    model="qwen3.8-27b-sq", temperature=0.6, top_p=0.95, max_tokens=4096,
    extra_body={"top_k": 20},
    messages=[{"role": "user", "content": "Explain KV caching in two paragraphs."}])
print(r.choices[0].message.content)
```

Chat apps: Open WebUI, LibreChat, Chatbox, Continue (VS Code) etc. take an "OpenAI-compatible"
connection with that URL and key.

## 7. Troubleshooting

- **Out of memory at start**: lower `MAX_LEN` or `MAX_SEQS`, or `KV_DTYPE=fp8`; close other GPU
  programs (`nvidia-smi`).
- **Fast first answer, then everything ~5x slower (Windows / WSL2)**: GPU memory overflowed into
  system RAM. Task Manager > Performance > GPU shows "Shared GPU memory" growing and "Dedicated" at
  the top. Leave `GPU_UTIL` unset (computed) or lower it; see "Long context". Note `nvidia-smi`
  inside WSL2 does not count what Windows programs hold.
- **`nvidia-smi` missing in WSL**: update the Windows NVIDIA driver and WSL (`wsl --update`).
- **CuPy errors (`CUDA_ERROR_INVALID_IMAGE`)**: two CuPy builds installed. Run
  `~/sq-venv/bin/pip uninstall -y cupy-cuda12x cupy-cuda13x`, `rm -rf ~/.cupy`, then re-run
  `deploy/setup_server.sh`.
- **Slow first answer after start**: kernels compile on first use of each batch size; it settles
  after a few requests.
- **Tool calling**: `TOOL_PARSER=qwen3_coder bash deploy/serve.sh` (verified on a 3090: an
  OpenAI `tools` request comes back as a proper `tool_calls` entry with `finish_reason: tool_calls`).

## Security notes

- Always set `SQ_API_KEY`. Without Tailscale/SSH, never bind `HOST=0.0.0.0` on a network you
  don't control.
- The API key is a shared secret: anyone with it and network access can use the GPU.
