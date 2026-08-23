"""Gates reproductibles avant de promouvoir un checkpoint post-training v4.5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate_gates(baseline_path: Path, candidate_path: Path,
                   corrected_ood: Path | None = None, model_name: str | None = None,
                   ood_floor: int = 7, facts_floor: int = 5) -> dict:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    base_ids = {row["task_id"] for row in baseline["rows"]}
    candidate_ids = {row["task_id"] for row in candidate["rows"]}
    if base_ids != candidate_ids:
        raise ValueError("les profils baseline/candidat n'évaluent pas exactement les mêmes tâches")
    b = baseline["summary"]["overall"]
    c = candidate["summary"]["overall"]
    pass_k_key = next(key for key in b if key.startswith("pass@") and key != "pass@1")
    checks = {
        "dev_pass1_non_regression": c["pass@1"] >= b["pass@1"],
        "dev_passk_non_regression": c[pass_k_key] >= b[pass_k_key],
        "dev_success_rate_improves": c["success_rate"] > b["success_rate"],
        "entropy_not_collapsed": c["mean_entropy"] >= 0.65 * b["mean_entropy"],
    }
    ood_summary = None
    if corrected_ood is not None:
        corrected = json.loads(corrected_ood.read_text(encoding="utf-8"))
        if not corrected.get("fully_adjudicated"):
            raise ValueError("le rapport OOD n'est pas intégralement corrigé à la main")
        totals = corrected["totals"]
        if model_name is None:
            if len(totals) != 1:
                raise ValueError("--model-name requis quand le rapport contient plusieurs modèles")
            model_name = next(iter(totals))
        if model_name not in totals:
            raise ValueError(f"modèle absent du rapport corrigé : {model_name}")
        values = totals[model_name]
        reasoning = values["reasoning"]
        facts = values["facts"]
        checks["ood_manual_floor"] = reasoning[0] >= ood_floor
        checks["facts_manual_floor"] = facts[0] >= facts_floor
        ood_summary = {"model": model_name, "reasoning": reasoning, "facts": facts}
    report = {
        "schema": "frlm-posttrain-gates-v45-1",
        "baseline": str(baseline_path), "candidate": str(candidate_path),
        "checks": checks, "passed": all(checks.values()),
        "dev": {"baseline": b, "candidate": c}, "ood": ood_summary,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--corrected-ood", type=Path)
    parser.add_argument("--model-name")
    parser.add_argument("--ood-floor", type=int, default=7)
    parser.add_argument("--facts-floor", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("posttrain_gates_v45.json"))
    args = parser.parse_args()
    report = evaluate_gates(args.baseline, args.candidate, args.corrected_ood,
                            args.model_name, args.ood_floor, args.facts_floor)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
