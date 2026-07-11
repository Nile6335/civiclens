"""Eval orchestrator: retrieval ablation + RAGAS metrics + the CI regression gate.

    python -m evals.run                          # full local eval (ablation + RAGAS subset)
    python -m evals.run --sample-only --subset 20 --check-baseline   # the CI gate
    python -m evals.run --write-baseline         # refresh the committed baseline

Baselines are keyed by (llm backend/model | embedding model | scope) so a CI run with the
small-footprint model is never compared against numbers produced by a different judge.
"""

import argparse
import json
import logging
import sys
from datetime import UTC, datetime

from common.llm import llm_description
from common.settings import get_settings
from evals.ablation import run_ablation, save_ablation
from evals.dataset import RESULTS_DIR, GoldenItem, load_dataset

logger = logging.getLogger(__name__)

BASELINE_PATH = RESULTS_DIR / "baseline.json"

# Gate floors are metric-aware: MRR is deterministic (retrieval math), so a strict 5%
# relative floor is safe; faithfulness is scored by a small LLM judge over few samples
# and carries real run-to-run variance, so it gets an additional absolute grace —
# a genuine regression still trips it, judge noise does not.
GATE_RULES: dict[str, dict[str, float]] = {
    "mrr": {"rel": 0.05, "abs_grace": 0.0},
    "faithfulness": {"rel": 0.05, "abs_grace": 0.05},
}


def config_key(sample_only: bool) -> str:
    settings = get_settings()
    scope = "sample" if sample_only else "full"
    return f"{llm_description()}|{settings.embedding_model}|{scope}"


def _stratified_subset(items: list[GoldenItem], limit: int) -> list[GoldenItem]:
    groups: dict[str, list[GoldenItem]] = {}
    for item in items:
        groups.setdefault(item.source_type, []).append(item)
    picked: list[GoldenItem] = []
    while len(picked) < limit and any(groups.values()):
        for key in sorted(groups):
            if groups[key] and len(picked) < limit:
                picked.append(groups[key].pop(0))
    return picked


def run_eval(subset: int, sample_only: bool, skip_ragas: bool, k: int = 5) -> dict:
    items = load_dataset()
    validated = [i for i in items if i.validated]
    if sample_only:
        validated = [i for i in validated if i.sample]
    logger.info("evaluating %d validated items (sample_only=%s)", len(validated), sample_only)
    if not validated:
        raise SystemExit("no validated items to evaluate — run evals.generate + evals.validate")

    ablation = run_ablation(validated, k=k)
    ablation["generated_at"] = datetime.now(UTC).isoformat()
    json_path, png_path = save_ablation(ablation)
    logger.info("ablation written to %s / %s", json_path, png_path)

    ragas: dict = {}
    if not skip_ragas:
        from evals.ragas_runner import collect_answers, ragas_scores

        subset_items = _stratified_subset(validated, subset)
        logger.info("collecting pipeline answers for %d questions...", len(subset_items))
        records = collect_answers(subset_items)
        logger.info("scoring with RAGAS (local LLM judge)...")
        ragas = ragas_scores(records)

    results = {
        "generated_at": datetime.now(UTC).isoformat(),
        "config": {
            "llm": llm_description(),
            "embedding_model": get_settings().embedding_model,
            "reranker_model": get_settings().reranker_model,
            "scope": "sample" if sample_only else "full",
            "k": k,
        },
        "dataset": {
            "total": len(items),
            "evaluated": len(validated),
            "by_type": _counts(validated),
            "multi_hop": sum(i.difficulty == "multi-hop" for i in validated),
        },
        "retrieval": ablation,
        "ragas": ragas,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # scope-suffixed so a CI sample run never clobbers the full-corpus numbers
    path = RESULTS_DIR / ("results-sample.json" if sample_only else "results.json")
    path.write_text(json.dumps(results, indent=1), encoding="utf-8")
    logger.info("results written to %s", path)
    return results


def _counts(items: list[GoldenItem]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item.source_type] = counts.get(item.source_type, 0) + 1
    return counts


def _gate_values(results: dict) -> dict[str, float]:
    values = {"mrr": results["retrieval"]["modes"]["hybrid_rerank"]["mrr"]}
    if results.get("ragas"):
        values["faithfulness"] = results["ragas"].get("faithfulness", 0.0)
    return values


def write_baseline(results: dict, sample_only: bool) -> None:
    baseline = {}
    if BASELINE_PATH.exists():
        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    baseline[config_key(sample_only)] = {
        **_gate_values(results),
        "generated_at": results["generated_at"],
    }
    BASELINE_PATH.write_text(json.dumps(baseline, indent=1), encoding="utf-8")
    print(f"baseline updated for {config_key(sample_only)!r}")


def check_baseline(results: dict, sample_only: bool) -> bool:
    key = config_key(sample_only)
    if not BASELINE_PATH.exists():
        print(f"FAIL: no baseline file at {BASELINE_PATH}")
        return False
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    if key not in baseline:
        print(f"FAIL: no baseline entry for config {key!r} (run --write-baseline)")
        return False
    current = _gate_values(results)
    ok = True
    for metric, rule in GATE_RULES.items():
        base = baseline[key].get(metric)
        cur = current.get(metric)
        if base is None or cur is None:
            continue
        floor = base * (1 - rule["rel"]) - rule["abs_grace"]
        status = "OK" if cur >= floor else "REGRESSION"
        print(f"{metric}: current={cur:.4f} baseline={base:.4f} floor={floor:.4f} → {status}")
        if cur < floor:
            if metric == "faithfulness":
                print(
                    "  note: faithfulness is LLM-judged; if MRR is OK and this dip is "
                    "small, suspect judge variance before suspecting the change."
                )
            ok = False
    return ok


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=24, help="questions for RAGAS scoring")
    parser.add_argument("--sample-only", action="store_true", help="sample-corpus items only")
    parser.add_argument("--skip-ragas", action="store_true")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument("--check-baseline", action="store_true")
    args = parser.parse_args()

    results = run_eval(args.subset, args.sample_only, args.skip_ragas, args.k)

    print("\n=== retrieval (all validated span-backed items) ===")
    for mode, m in results["retrieval"]["modes"].items():
        print(f"  {mode:<15} hit@{args.k}={m['hit_rate']:.3f}  mrr={m['mrr']:.3f}  n={m['n']}")
    if results.get("ragas"):
        print("=== ragas (subset) ===")
        for name in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
            if name in results["ragas"]:
                print(f"  {name:<18} {results['ragas'][name]:.3f}")

    if args.write_baseline:
        write_baseline(results, args.sample_only)
    if args.check_baseline and not check_baseline(results, args.sample_only):
        sys.exit(1)


if __name__ == "__main__":
    main()
