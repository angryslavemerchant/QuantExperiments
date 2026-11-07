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
