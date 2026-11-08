"""Run lighteval's CLI unchanged except for vLLM-side compatibility and engine settings in its vLLM wrapper.

lighteval 0.13.0 (src/lighteval/models/vllm/vllm_model.py) imports get_tokenizer from a module newer
vLLM moved (aliased here), generates each task in 4 sequential splits (set to 1: one long tail
instead of four), and builds vllm.LLM with
"enforce_eager": True (no CUDA graphs: decode several times slower for every model, plugin or
bf16) and "swap_space": 4 (an argument newer vLLM releases may reject). This wrapper swaps the
LLM class lighteval imported for one that sets enforce_eager=False (LIGHTEVAL_EAGER=1 keeps it)
and drops keyword arguments the installed vLLM does not accept. Tasks, prompts, answer extraction
and metrics are lighteval's own.

  python scripts/lighteval_run.py vllm "model_name=DIR,dtype=bfloat16,..." "gpqa:diamond@k=1@n=4|0" --output-dir OUT
"""

from __future__ import annotations

import dataclasses
import os
import sys


def _shim_tokenizer_module() -> None:
    """lighteval imports get_tokenizer from vllm.transformers_utils.tokenizer, which newer vLLM
    moved to vllm.tokenizers(.registry): alias the old module path."""
    import importlib
    import types
    try:
        importlib.import_module("vllm.transformers_utils.tokenizer")
        return
    except ImportError:
        pass
    try:
        from vllm.tokenizers import get_tokenizer
    except ImportError:
        from vllm.tokenizers.registry import get_tokenizer
    mod = types.ModuleType("vllm.transformers_utils.tokenizer")
    mod.get_tokenizer = get_tokenizer
    sys.modules["vllm.transformers_utils.tokenizer"] = mod
    print("[lighteval_run] aliased vllm.transformers_utils.tokenizer.get_tokenizer", flush=True)


def _shim_hub_list_models() -> None:
    """lighteval 0.13.0 builds its LLM-judge metrics at import and calls
    HfApi.list_models(model_name=...), a keyword huggingface_hub 1.x/2.x removed (s37: every run
    died at import). Map it to search=."""
    from huggingface_hub import HfApi
    import inspect
    orig = HfApi.list_models
    if "model_name" in inspect.signature(orig).parameters:
        return

    def list_models(self, *args, model_name=None, **kw):
        if model_name is not None:
            kw.setdefault("search", model_name)
        return orig(self, *args, **kw)

    HfApi.list_models = list_models
    print("[lighteval_run] shimmed HfApi.list_models(model_name=)", flush=True)


def _patch_vllm() -> None:
    _shim_tokenizer_module()
    import vllm
    from vllm.engine.arg_utils import EngineArgs
    import lighteval.models.vllm.vllm_model as vm

    known = {f.name for f in dataclasses.fields(EngineArgs)}
    real = vllm.LLM

    def LLM(**kw):
        if os.environ.get("LIGHTEVAL_EAGER") != "1":
            kw["enforce_eager"] = False
        dropped = sorted(k for k in kw if k not in known and k != "model")
        for k in dropped:
            kw.pop(k)
        print(f"[lighteval_run] vllm.LLM({', '.join(f'{k}={v!r}' for k, v in sorted(kw.items()))})"
              + (f"; dropped {dropped}" if dropped else ""), flush=True)
        return real(**kw)

    vm.LLM = LLM
    # lighteval generates a task in DATASET_SPLITS (4) sequential vLLM calls; with 32k-token
    # reasoning each call ends in a long single-sequence tail, so submit everything at once
    vm.VLLMModel.DATASET_SPLITS = int(os.environ.get("LIGHTEVAL_SPLITS", "1"))


def main() -> int:
    _shim_hub_list_models()
    if len(sys.argv) > 1 and sys.argv[1] == "vllm":
        _patch_vllm()
    from lighteval.__main__ import app
    sys.argv = ["lighteval"] + sys.argv[1:]
    app()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
