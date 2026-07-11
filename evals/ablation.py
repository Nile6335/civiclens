"""Auto-run, auto-charted retrieval ablation: dense-only vs hybrid vs hybrid+rerank.

Runs the golden dataset through each retrieval mode via ``evals.metrics.retrieval_metrics``
(hit@k + MRR), writes ``evals/results/ablation.json`` and a grouped bar chart
``evals/results/ablation.png`` for the README. Run with ``python -m evals.ablation``.
"""

import json
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from evals.dataset import DATASET_PATH, RESULTS_DIR, GoldenItem, load_dataset

DEFAULT_MODES: tuple[str, ...] = ("dense", "hybrid", "hybrid_rerank")

MODE_LABELS: dict[str, str] = {
    "dense": "Dense only",
    "hybrid": "Hybrid (RRF)",
    "hybrid_rerank": "Hybrid + rerank",
}

MetricsFn = Callable[..., dict]


def run_ablation(
    items: list[GoldenItem],
    k: int = 5,
    modes: tuple[str, ...] = DEFAULT_MODES,
    metrics_fn: MetricsFn | None = None,
) -> dict:
    """Run retrieval metrics once per mode.

    ``metrics_fn(items, k=..., mode=...)`` must return a dict with ``hit_rate``, ``mrr``
    and ``n``; it defaults to ``evals.metrics.retrieval_metrics``. ``generated_at`` is
    left as None for the caller to stamp.
    """
    if metrics_fn is None:
        from evals.metrics import retrieval_metrics  # lazy: avoids retrieval deps at import

        metrics_fn = retrieval_metrics
    mode_results: dict[str, dict] = {}
    for mode in modes:
        metrics = metrics_fn(items, k=k, mode=mode)
        mode_results[mode] = {
            "hit_rate": float(metrics["hit_rate"]),
            "mrr": float(metrics["mrr"]),
            "n": int(metrics["n"]),
        }
    return {"k": k, "modes": mode_results, "generated_at": None}


def _render_chart(results: dict, png_path: Path) -> None:
    """Grouped bar chart (hit@k + MRR per mode) rendered headlessly to png_path."""
    import matplotlib

    matplotlib.use("Agg")  # must be set before pyplot import: headless render
    import matplotlib.pyplot as plt

    modes = list(results["modes"])
    labels = [MODE_LABELS.get(mode, mode) for mode in modes]
    hit_rates = [results["modes"][mode]["hit_rate"] for mode in modes]
    mrrs = [results["modes"][mode]["mrr"] for mode in modes]
    k = results["k"]
    n = max((results["modes"][mode]["n"] for mode in modes), default=0)

    width = 0.35
    positions = range(len(modes))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    hit_bars = ax.bar([p - width / 2 for p in positions], hit_rates, width, label=f"hit@{k}")
    mrr_bars = ax.bar([p + width / 2 for p in positions], mrrs, width, label="MRR")
    for bars in (hit_bars, mrr_bars):
        ax.bar_label(bars, fmt="%.2f", padding=2)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("score")
    ax.set_title(f"CivicLens retrieval ablation (k={k}, n={n})")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def save_ablation(results: dict, out_dir: Path = RESULTS_DIR) -> tuple[Path, Path]:
    """Write ablation.json and ablation.png into out_dir; return both paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "ablation.json"
    png_path = out_dir / "ablation.png"
    json_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
    _render_chart(results, png_path)
    return json_path, png_path


def main() -> None:
    if not DATASET_PATH.exists():
        print(
            f"Golden dataset not found at {DATASET_PATH}.\n"
            "Run the dataset generator first: uv run python -m evals.generate"
        )
        sys.exit(1)
    items = load_dataset()
    results = run_ablation(items)
    results["generated_at"] = datetime.now(UTC).isoformat()
    json_path, png_path = save_ablation(results)

    k = results["k"]
    print(f"{'mode':<18} {f'hit@{k}':>8} {'mrr':>8} {'n':>5}")
    for mode, metrics in results["modes"].items():
        label = MODE_LABELS.get(mode, mode)
        print(f"{label:<18} {metrics['hit_rate']:>8.3f} {metrics['mrr']:>8.3f} {metrics['n']:>5}")
    print(f"wrote {json_path} and {png_path}")


if __name__ == "__main__":
    main()
