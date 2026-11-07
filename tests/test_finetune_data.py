"""run_experiment.finetune_data: fresh fine-tune sequences come right after (never overlap) the
calibration ones; held-out fineweb after those; held-out wikitext from the TRAIN split."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_experiment as rx  # noqa: E402
import shapequant.calibrate as cal  # noqa: E402


def _stream(tok, ccfg, seed=0):
    n, ctx = ccfg["num_sequences"], ccfg["context"]
    return torch.arange(n * ctx).reshape(n, ctx)          # token i of the stream = i


def test_finetune_data_disjoint(monkeypatch):
    monkeypatch.setattr(cal, "load_calibration_sequences", _stream)
    seen = {}

    def fake_wiki(tok, w):
        seen.update(w)
        return torch.zeros(w["num_windows"], w["context"], dtype=torch.long)

    monkeypatch.setattr(rx, "load_wikitext_windows", fake_wiki)
    cfg = {"calibration": {"num_sequences": 4, "context": 8},
           "eval": {"wikitext": {"dataset": "x", "subset": "y", "split": "test", "context": 8, "num_windows": 2}}}
    calib = _stream(None, cfg["calibration"])
    fcfg = {"seq_len": 4, "sequences": 3, "heldout": {"fineweb": 3, "wikitext_train": 5}}
    ft, held = rx.finetune_data(fcfg, cfg, None, calib, 0)
    assert ft.shape == (3, 8) and int(ft.min()) == 32 and int(ft.max()) == 55      # right after calibration
    assert held["fineweb"].shape == (3, 4) and int(held["fineweb"].min()) == 56     # after the fine-tune data
    assert held["wikitext_train"].shape == (5, 4)
    assert seen["split"] == "train" and seen["context"] == 4
    # default: the calibration sequences themselves, no held-out sets
    ft, held = rx.finetune_data({"seq_len": 4}, cfg, None, calib, 0)
    assert ft is calib and held == {}


def test_upload_packed_fake_hub(monkeypatch, tmp_path):
    """upload_packed: private repo under the token's user, model card + file, receipt with sha256."""
    import hashlib
    import json
    import huggingface_hub
    calls = []

    class FakeApi:
        def whoami(self):
            return {"name": "someone"}

        def create_repo(self, repo, private, exist_ok):
            calls.append(("create", repo, private))

        def upload_file(self, path_or_fileobj, path_in_repo, repo_id):
            calls.append(("upload", path_in_repo, repo_id, path_or_fileobj if path_in_repo == "README.md" else None))

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    f = tmp_path / "m_ft.safetensors"
    f.write_bytes(b"x" * 1000)
    row = {"mean_kl": 0.12, "bits_per_weight_rounded": 2.51, "deploy_h8e4_kl": 0.121}
    r = rx.upload_packed(f, {"repo": "q27-test"}, {"model": "Qwen/Qwen3.8-27B"}, row, {"bpw": 2.5, "bpw_all": 2.9},
                         tmp_path, lambda *_: None)
    assert calls[0] == ("create", "someone/q27-test", True)
    assert [c[1] for c in calls[1:]] == ["README.md", "m_ft.safetensors"]
    assert b"0.12" in calls[1][3] and b"base_model: Qwen/Qwen3.8-27B" in calls[1][3]
    assert r["sha256"] == hashlib.sha256(b"x" * 1000).hexdigest()
    assert json.loads((tmp_path / "upload_receipt.json").read_text())["repo"] == "someone/q27-test"
