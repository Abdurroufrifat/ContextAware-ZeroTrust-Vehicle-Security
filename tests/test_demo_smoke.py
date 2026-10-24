from __future__ import annotations

from coact_cps.demo import run_demo


def test_end_to_end_demo(tmp_path) -> None:
    result = run_demo(tmp_path, epochs=1, n_steps=300, seed=5)
    assert (tmp_path / "detector.pt").exists()
    assert (tmp_path / "adjacency.npy").exists()
    assert (tmp_path / "test_windows.npz").exists()
    assert (tmp_path / "demo_results.json").exists()
    assert result["controlled_attack_check"]["clean_status"] == "VERIFIED"
    assert result["controlled_attack_check"]["attacked_status"] != "VERIFIED"
