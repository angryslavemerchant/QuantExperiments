"""Hand-apply the two hunks of vLLM PR #61309 (RecoverSSM for Qwen3.5 GDN, head 3a559cc) that do
not apply to vLLM 0.33.0: the architecture tuples in config/vllm.py and the in-place guard in
qwen_gdn_linear_attn.py. Run with the patched venv's python; VENV (default ~/sq-venv-rssm)."""
import os

sp = os.path.join(os.environ.get("VENV", os.path.expanduser("~/sq-venv-rssm")), "lib/python3.12/site-packages/vllm")

p = f"{sp}/config/vllm.py"
s = open(p).read()
if "_QWEN_GDN_RECOVERSSM_ARCHS = (" not in s:
    anchor = "logger = init_logger(__name__)\n"
    assert s.count(anchor) == 1
    s = s.replace(anchor, anchor + '''
# Architectures whose GDN (and PLE) layers run RecoverSSM under --use-replayssm.
_QWEN4EXP_RECOVERSSM_ARCHS = (
    "Qwen4ExpForCausalLM",
    "Qwen4ExpForConditionalGeneration",
    "Qwen4ExpMTP",
)
# Qwen3-Next / Qwen3.5 share the Qwen GDN layer (and its RecoverSSM path) with Qwen4Exp.
_QWEN_GDN_RECOVERSSM_ARCHS = (
    *_QWEN4EXP_RECOVERSSM_ARCHS,
    "Qwen3NextForCausalLM",
    "Qwen3NextMTP",
    "Qwen3_5ForCausalLM",
    "Qwen3_5MoeForCausalLM",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration",
    "Qwen3_5MTP",
    "Qwen3_5MoeMTP",
)
''', 1)
    open(p, "w").write(s)
    print("config/vllm.py: arch tuples added")

p = f"{sp}/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"
s = open(p).read()
old = ("        elif spec_sequence_masks is not None:\n"
       "            core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)\n")
new = ("        elif spec_sequence_masks is not None:\n"
       "            if core_attn_out_spec.data_ptr() != core_attn_out.data_ptr():\n"
       "                core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)\n")
if new not in s:
    print("gdn: occurrences of old block:", s.count(old))
    assert s.count(old) == 1
    s = s.replace(old, new)
    open(p, "w").write(s)
    print("qwen_gdn_linear_attn.py: in-place guard added")
import py_compile
for f in ("config/vllm.py", "model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
          "model_executor/layers/mamba/gdn/recoverssm_gdn.py", "model_executor/models/qwen3_5.py",
          "v1/attention/backends/gdn_recoverssm.py"):
    py_compile.compile(f"{sp}/{f}", doraise=True)
print("compile ok")
