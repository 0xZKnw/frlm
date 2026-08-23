"""Prépare puis valide une correction humaine exhaustive du bench OOD v2."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def prepare(raw_path: Path, output: Path) -> dict:
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    rows = []
    for record in raw["records"]:
        rows.append({**record, "manual_correct": None, "manual_note": ""})
    payload = {"schema": "frlm-ood-v2-manual-1", "source": str(raw_path),
               "instructions": "Remplir manual_correct avec true ou false pour chaque ligne.",
               "records": rows}
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def finalize(labels_path: Path, output: Path) -> dict:
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    records = labels.get("records") or []
    if not records:
        raise ValueError("aucune réponse à adjuger")
    missing = [index for index, row in enumerate(records)
               if type(row.get("manual_correct")) is not bool]
    if missing:
        raise ValueError(f"manual_correct manque ou n'est pas booléen aux lignes {missing[:12]}")
    keys = [(row["model"], row["section"], row["item_index"]) for row in records]
    if len(keys) != len(set(keys)):
        raise ValueError("items dupliqués dans l'adjudication")
    totals = defaultdict(lambda: {"reasoning": [0, 0], "facts": [0, 0], "by_category": {}})
    disagreements = []
    for row in records:
        model = row["model"]
        section = row["section"]
        correct = bool(row["manual_correct"])
        totals[model][section][0] += int(correct)
        totals[model][section][1] += 1
        category = row["category"]
        cat = totals[model]["by_category"].setdefault(category, [0, 0])
        cat[0] += int(correct)
        cat[1] += 1
        if correct != bool(row["auto_correct"]):
            disagreements.append({"model": model, "section": section,
                                  "item_index": row["item_index"],
                                  "auto": row["auto_correct"], "manual": correct,
                                  "note": row.get("manual_note", "")})
    report = {"schema": "frlm-ood-v2-corrected-1", "source": str(labels_path),
              "fully_adjudicated": True, "totals": dict(totals),
              "auto_manual_disagreements": disagreements}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path, help="rapport .raw.json ou fichier .manual.json")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.finalize:
        output = args.output or args.raw.with_name(args.raw.stem.replace(".manual", "")
                                                   + ".corrected.json")
        report = finalize(args.raw, output)
        print(json.dumps(report["totals"], ensure_ascii=False, indent=2))
    else:
        output = args.output or args.raw.with_suffix(".manual.json")
        prepare(args.raw, output)
        print(f"Template manuel : {output}")


if __name__ == "__main__":
    main()
