# Running the shapequant 27B as a server (RTX 3090 / any 24 GB+ NVIDIA GPU)

What you get: an OpenAI-compatible endpoint (`/v1/chat/completions`, `/v1/models`) serving
Qwen3.8-27B at 2.5 bpw through vLLM + the shapequant plugin. Thinking mode, 32k context per
request, up to 16 requests at once by default, protected by an API key.

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
requests, ~340 tok/s at 32; KV cache room for ~130k tokens in total with the defaults (shared by
all running requests).

Keep it running after you log out: run it inside `tmux` (`tmux new -s sq`, start, then `Ctrl-b d`
to detach; `tmux attach -t sq` to come back).

Useful settings (environment variables for `serve.sh`):

| Variable | Default | What it does |
|---|---|---|
| `HOST` | `127.0.0.1` | Address to listen on. Use your Tailscale IP (below) to reach it from home |
| `PORT` | `8000` | Port |
| `MAX_LEN` | `32768` | Max tokens per request (prompt + answer) |
| `MAX_SEQS` | `16` | Max requests processed at once |
| `GPU_UTIL` | `0.92` | Fraction of GPU memory vLLM may use (the rest of it becomes KV cache) |
| `KV_DTYPE` | `auto` | `fp8` halves KV-cache memory: longer contexts / more requests |
| `TOOL_PARSER` | off | Enables tool / function calling (see notes) |
| `SPEC` | off | `dspark`: speculative decoding with RedHat's DSpark drafter (+4 GB download) |

### Long context

The model supports up to 262k tokens. Measured on the 3090 (`GPU_UTIL=0.94`, `MAX_SEQS=2`): with
the default KV cache, `MAX_LEN=131072` fits (room for ~159k KV tokens); with `KV_DTYPE=fp8`,
`MAX_LEN=262144` fits (~316k KV tokens). Retrieval accuracy at those lengths has not been measured yet.

```bash
KV_DTYPE=fp8 MAX_LEN=262144 MAX_SEQS=2 GPU_UTIL=0.94 bash deploy/serve.sh
```

### Faster single requests: DSpark speculative decoding

Measured on the 3090: 79 tok/s for one request instead of 38 (2.1x), 123 vs 102 tok/s total at 4
requests. The drafter takes ~4 GB of GPU memory, so the KV cache shrinks to ~25k tokens total:

```bash
SPEC=dspark MAX_LEN=16384 MAX_SEQS=4 bash deploy/serve.sh
```

The first start downloads the drafter and compiles (can take ~30 min); later starts are faster.
Best when you are the only user; for many parallel requests leave it off.

### Images (vision)

The base model's vision tower can be added (kept in bf16, +1 GB; the text model stays 2.5 bpw).
Export with it at setup time (if you already set up text-only, `rm -rf ~/models/q27-sq` first):

```bash
VISION=1 HF_TOKEN=hf_xxx bash deploy/setup_server.sh
```

`serve.sh` then accepts images automatically (`IMAGES=4` per request by default). Send them as
OpenAI `image_url` parts (an `https://` URL or a `data:image/png;base64,...` URL). Measured on the
3090 (2026-11-08): both test images (a synthetic number/shapes image and a COCO photo of two cats
with remotes) described correctly, ~1.3 s per short answer; KV cache ~92k tokens.

## 4. Check it

On the server:

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
