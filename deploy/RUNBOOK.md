# Running the shapequant 27B as a server (RTX 3090 / any 24 GB+ NVIDIA GPU)

What you get: an OpenAI-compatible endpoint (`/v1/chat/completions`, `/v1/models`) serving
Qwen3.8-27B at 2.5 bpw through vLLM + the shapequant plugin. Thinking mode, 32k context per
request, up to 16 requests at once by default, protected by an API key.

## 0. Requirements

- Linux, or Windows with **WSL2** (Ubuntu 22.04/24.04). vLLM does not run on native Windows.
  - WSL2: install the normal Windows NVIDIA driver; inside WSL `nvidia-smi` must show the GPU.
- NVIDIA driver supporting CUDA 12.8 or newer (`nvidia-smi` top right shows "CUDA Version").
- Python 3.10-3.12, git, ~40 GB free disk (15 GB vLLM environment, 10 GB model, caches).
- A Hugging Face **read** token with access to `luckymushy/Qwen3.8-27B-shapequant-2.5bpw`.

## 1. Get the code onto the machine

The repo is not on GitHub. On the home PC, from the repo folder:

```bash
git bundle create qe.bundle HEAD          # one file with the full history
```

Copy `qe.bundle` to the work machine (USB, scp, cloud drive), then there:

```bash
git clone qe.bundle quant-experiments && cd quant-experiments
```

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
- **Tool calling**: set `TOOL_PARSER` to the parser matching Qwen3.8's chat template
  (`qwen3_coder` for the XML `<function=...>` format, `hermes` for JSON `<tool_call>`); check the
  template in `~/models/q27-sq/chat_template.jinja`.

## Security notes

- Always set `SQ_API_KEY`. Without Tailscale/SSH, never bind `HOST=0.0.0.0` on a network you
  don't control.
- The API key is a shared secret: anyone with it and network access can use the GPU.
