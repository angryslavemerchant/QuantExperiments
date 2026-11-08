"""Scoring helpers of the thinking-mode lm-eval tasks and the eval-checkpoint name mapping."""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tt = _load(ROOT / "configs" / "lm_eval_tasks" / "think_tasks.py", "think_tasks")
sys.path.insert(0, str(ROOT / "scripts"))
ex = _load(ROOT / "scripts" / "export_eval_checkpoint.py", "export_eval_checkpoint")


def test_last_boxed_nested():
    assert tt.last_boxed(r"so \boxed{1} then \boxed{\frac{1}{2}} done") == r"\frac{1}{2}"
    assert tt.last_boxed("no box") is None
    assert tt.last_boxed(r"\boxed{unclosed") is None


def test_gpqa_results():
    doc = {"gold": "C"}
    assert tt.gpqa_results(doc, [r"... \boxed{C}"]) == {"exact_match": 1, "no_answer": 0}
    assert tt.gpqa_results(doc, [r"\boxed{\text{B}}"]) == {"exact_match": 0, "no_answer": 0}
    assert tt.gpqa_results(doc, ["The answer is (C)."]) == {"exact_match": 1, "no_answer": 0}
    assert tt.gpqa_results(doc, ["I am not sure"]) == {"exact_match": 0, "no_answer": 1}


def test_gpqa_docs_deterministic_and_gold_correct():
    datasets = pytest.importorskip("datasets")
    rows = [{"Question": f"q{i}", "Correct Answer": f"right{i}", "Incorrect Answer 1": "w1",
             "Incorrect Answer 2": "w2", "Incorrect Answer 3": "w3"} for i in range(20)]
    a = tt.gpqa_docs(datasets.Dataset.from_list(rows))
    b = tt.gpqa_docs(datasets.Dataset.from_list(rows))
    assert [d["gold"] for d in a] == [d["gold"] for d in b]
    for i, d in enumerate(a):
        assert d[d["gold"]] == f"right{i}"
    assert len({d["gold"] for d in a}) > 1          # actually shuffled


def test_math_results():
    doc = {"answer": r"\frac{1}{2}"}
    r = tt.math_results(doc, [r"thus \boxed{\frac12}"])     # the string normalisation alone matches this
    assert r["exact"] == 1 and r["math_verify"] == 1
    assert tt.math_results(doc, [r"\boxed{2}"]) == {"math_verify": 0, "exact": 0, "no_answer": 0}
    assert tt.math_results(doc, ["cut off mid"]) == {"math_verify": 0, "exact": 0, "no_answer": 1}


def test_block_names_multimodal_layout():
    names = ["model.language_model.layers.0.mlp.down_proj.weight",
             "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
             "model.language_model.layers.0.linear_attn.in_proj_a.weight",      # DeltaNet gate: not a block linear
             "model.language_model.layers.3.self_attn.q_proj.weight",
             "model.language_model.layers.3.self_attn.q_norm.weight",
             "model.language_model.embed_tokens.weight", "lm_head.weight",
             "mtp.layers.0.mlp.down_proj.weight", "model.visual.blocks.0.attn.qkv.weight"]
    p = ex.text_prefix(names)
    assert p == "model.language_model."
    assert ex.block_names(names, p) == {
        "model.layers.0.mlp.down_proj": "model.language_model.layers.0.mlp.down_proj.weight",
        "model.layers.0.linear_attn.in_proj_qkv": "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
        "model.layers.3.self_attn.q_proj": "model.language_model.layers.3.self_attn.q_proj.weight"}


def test_block_names_text_layout():
    names = ["model.layers.1.mlp.up_proj.weight", "model.embed_tokens.weight"]
    assert ex.block_names(names, ex.text_prefix(names)) == {"model.layers.1.mlp.up_proj": "model.layers.1.mlp.up_proj.weight"}


te = _load(ROOT / "scripts" / "think_eval.py", "think_eval")


def test_think_eval_scores_only_the_final_answer():
    doc = {"gold": "B"}
    assert te.score("gpqa", doc, r"<think>maybe \boxed{A}</think>So \boxed{B}")["exact_match"] == 1
    r = te.score("gpqa", doc, r"<think>still thinking \boxed{B}")      # cut off inside the thinking
    assert r["exact_match"] == 0 and r["no_answer"] == 1 and r["think_closed"] == 0
    m = te.score("math500", {"answer": "3"}, r"<think>x</think>\boxed{3}")
    assert m["math_verify"] == 1 and m["think_closed"] == 1


def test_think_eval_summary():
    rows = [{"scores": {"exact_match": e, "no_answer": 0, "think_closed": 1}, "gen_tokens": 10, "finish": "stop"}
            for e in (1, 0, 1, 1)]
    s = te.summarize(rows, "exact_match")
    assert s["n"] == 4 and s["exact_match"] == 0.75 and s["exact_match_stderr"] > 0 and s["max_gen_tokens_hit"] == 0
