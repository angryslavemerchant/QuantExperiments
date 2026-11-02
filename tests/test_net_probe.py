from pathlib import Path

from vast import net_probe


def _url(tmp_path: Path, size: int) -> str:
    f = tmp_path / "blob.bin"
    f.write_bytes(b"\0" * size)
    return f.as_uri()


def test_probe_reads_up_to_max_bytes(tmp_path):
    result = net_probe.probe(_url(tmp_path, 3_000_000), max_bytes=2_000_000, max_seconds=10)
    assert result["bytes"] == 2_000_000
    assert result["mb_per_s"] > 0


def test_main_exit_codes(tmp_path, capsys):
    url = _url(tmp_path, 1_000_000)
    assert net_probe.main(["--url", url, "--min-mb-per-s", "0"]) == 0
    assert net_probe.main(["--url", url, "--min-mb-per-s", "1e12"]) == 3
    assert net_probe.main(["--url", (tmp_path / "missing.bin").as_uri()]) == 4
    assert '"ok": false' in capsys.readouterr().out
