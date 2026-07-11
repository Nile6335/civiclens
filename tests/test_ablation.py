"""Unit tests for the retrieval ablation runner and chart writer (no DB, no models)."""

import json
from pathlib import Path

import pytest

import evals.ablation as ablation
from evals.ablation import run_ablation, save_ablation
from evals.dataset import GoldenItem, Span

FIXED_METRICS: dict[str, dict] = {
    "dense": {"hit_rate": 0.4, "mrr": 0.25, "n": 2},
    "hybrid": {"hit_rate": 0.6, "mrr": 0.45, "n": 2},
    "hybrid_rerank": {"hit_rate": 0.8, "mrr": 0.7, "n": 2},
}


def _fake_metrics(items: list[GoldenItem], k: int = 5, mode: str = "hybrid") -> dict:
    return dict(FIXED_METRICS[mode])


def _items() -> list[GoldenItem]:
    span = Span(
        city="oakland",
        source_type="transcript",
        meeting_id="mtg-001",
        chunk_index=3,
        text_snippet="The council approved the FY25 budget amendment by a 6-1 vote.",
    )
    return [
        GoldenItem(
            id="g1",
            question="What did the council approve?",
            answer="The FY25 budget amendment.",
            difficulty="easy",
            source_type="transcript",
            city="oakland",
            spans=[span],
            sample=True,
        ),
        GoldenItem(
            id="g2",
            question="How did the vote on the budget amendment go?",
            answer="It passed 6-1.",
            difficulty="multi-hop",
            source_type="transcript",
            city="oakland",
            spans=[span],
            sample=True,
        ),
    ]


class TestRunAblation:
    def test_shape_and_values(self) -> None:
        results = run_ablation(_items(), k=5, metrics_fn=_fake_metrics)
        assert results["k"] == 5
        assert results["generated_at"] is None
        assert set(results["modes"]) == {"dense", "hybrid", "hybrid_rerank"}
        for mode, expected in FIXED_METRICS.items():
            assert results["modes"][mode] == expected

    def test_metrics_fn_called_once_per_mode_with_kwargs(self) -> None:
        calls: list[tuple[int, str, int]] = []

        def recording_metrics(items: list[GoldenItem], k: int, mode: str) -> dict:
            calls.append((len(items), mode, k))
            return {"hit_rate": 0.5, "mrr": 0.5, "n": len(items)}

        items = _items()
        run_ablation(items, k=3, metrics_fn=recording_metrics)
        assert calls == [(2, "dense", 3), (2, "hybrid", 3), (2, "hybrid_rerank", 3)]

    def test_custom_modes_subset(self) -> None:
        results = run_ablation(_items(), k=5, modes=("dense",), metrics_fn=_fake_metrics)
        assert list(results["modes"]) == ["dense"]
        assert results["modes"]["dense"]["n"] == 2


class TestSaveAblation:
    def test_writes_json_and_png(self, tmp_path: Path) -> None:
        results = run_ablation(_items(), k=5, metrics_fn=_fake_metrics)
        json_path, png_path = save_ablation(results, out_dir=tmp_path)

        assert json_path == tmp_path / "ablation.json"
        assert png_path == tmp_path / "ablation.png"
        assert json_path.is_file()
        assert png_path.is_file()

        round_tripped = json.loads(json_path.read_text(encoding="utf-8"))
        assert round_tripped == results
        assert png_path.stat().st_size > 1024

    def test_creates_missing_out_dir(self, tmp_path: Path) -> None:
        results = run_ablation(_items(), k=5, metrics_fn=_fake_metrics)
        out_dir = tmp_path / "nested" / "results"
        json_path, png_path = save_ablation(results, out_dir=out_dir)
        assert json_path.is_file()
        assert png_path.is_file()


class TestMain:
    def test_missing_dataset_exits_1(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(ablation, "DATASET_PATH", tmp_path / "golden_dataset.json")
        with pytest.raises(SystemExit) as excinfo:
            ablation.main()
        assert excinfo.value.code == 1
        out = capsys.readouterr().out
        assert "not found" in out
        assert "generate" in out.lower()

    def test_happy_path_prints_table(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        dataset_path = tmp_path / "golden_dataset.json"
        dataset_path.write_text("[]", encoding="utf-8")
        monkeypatch.setattr(ablation, "DATASET_PATH", dataset_path)
        monkeypatch.setattr(ablation, "load_dataset", lambda: _items())
        monkeypatch.setattr(
            ablation,
            "run_ablation",
            lambda items: run_ablation(items, metrics_fn=_fake_metrics),
        )
        saved: list[dict] = []

        def fake_save(results: dict) -> tuple[Path, Path]:
            saved.append(results)
            return tmp_path / "ablation.json", tmp_path / "ablation.png"

        monkeypatch.setattr(ablation, "save_ablation", fake_save)

        ablation.main()

        assert len(saved) == 1
        assert saved[0]["generated_at"] is not None
        out = capsys.readouterr().out
        assert "Dense only" in out
        assert "Hybrid (RRF)" in out
        assert "Hybrid + rerank" in out
        assert "0.800" in out
