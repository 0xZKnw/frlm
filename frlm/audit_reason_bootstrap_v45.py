"""Audit CPU bloquant du bootstrap raisonnement v4.5 avant Modal."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from frlm import data as D
from frlm.reason_bootstrap_v45 import RECIPE_NAME, evaluate_ast, operator_signature


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _audit_bin(path: Path, seq_len: int) -> dict:
    mask_path = path.with_suffix(".mask")
    if not path.is_file() or not mask_path.is_file():
        raise FileNotFoundError(f"bin ou masque absent : {path}")
    tokens = np.memmap(path, dtype=D.DTYPE, mode="r")
    masks = np.memmap(mask_path, dtype=np.uint8, mode="r")
    if len(tokens) != len(masks):
        raise ValueError(f"tokens/mask incohérents : {path}")
    ends = np.flatnonzero(np.asarray(tokens) == 0).astype(np.int64) + 1
    starts = np.concatenate((np.array([0], dtype=np.int64), ends[:-1]))
    if not len(ends) or int(ends[-1]) != len(tokens):
        raise ValueError(f"conversation finale non terminée par EOT : {path}")
    lengths = ends - starts
    if int(lengths.max()) > seq_len + 1:
        raise ValueError(f"conversation > {seq_len + 1} tokens : {path}")
    supervised = np.add.reduceat(np.asarray(masks, dtype=np.int64), starts)
    if np.any(supervised <= 0):
        raise ValueError(f"conversation sans token assistant : {path}")
    return {"tokens": len(tokens), "conversations": len(starts),
            "supervised": int(masks.sum()), "max_tokens": int(lengths.max())}


def audit(data_dir: Path, require_replay_local: bool = False) -> dict:
    data_dir = Path(data_dir)
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    section = meta.get("reason_bootstrap_v45") or {}
    if section.get("recipe") != RECIPE_NAME:
        raise ValueError(f"recette bootstrap absente ou ancienne : {section.get('recipe')}")
    seq_len = int(section["max_conversation_tokens"])
    bins = {}
    deferred_replay = []
    for name, capability in section["capabilities"].items():
        for split in ("train", "val"):
            path = data_dir / capability[f"{split}_path"]
            key = str(path.resolve())
            if capability.get("replay_source") and not path.is_file():
                if require_replay_local:
                    raise FileNotFoundError(f"bin de rétention absent : {path}")
                deferred_replay.append(str(path))
                continue
            if key not in bins:
                bins[key] = _audit_bin(path, seq_len)
            expected = int(capability[f"{split}_tokens"])
            if bins[key]["tokens"] != expected:
                raise ValueError(f"meta incohérente pour {path}: {bins[key]['tokens']} != {expected}")

    raw_dir = data_dir / "raw"
    train = _read_jsonl(raw_dir / "reason_bootstrap_v45_train.jsonl")
    eval_sets = {split: _read_jsonl(data_dir / relative)
                 for split, relative in section["eval_manifests"].items()}
    if len(train) != int(section["examples"]):
        raise ValueError("nombre d'exemples train incohérent")
    ids = [row["id"] for row in train]
    ids.extend(row["id"] for rows in eval_sets.values() for row in rows)
    if len(ids) != len(set(ids)):
        raise ValueError("fuite de prompts entre train et évaluations")
    train_signatures = {operator_signature(row["program_ast"]) for row in train}
    structure_signatures = {
        operator_signature(row["program_ast"])
        for row in eval_sets["structure_holdout"]
    }
    overlap = train_signatures & structure_signatures
    if overlap:
        raise ValueError(f"fuite de topologies AST : {sorted(overlap)}")
    for rows in (train, *eval_sets.values()):
        for row in rows:
            value = evaluate_ast(row["program_ast"])
            if row["objective"] in ("execute", "number_only") and str(value) != row["target"]:
                raise ValueError(f"cible fausse : {row['id']}")
    return {
        "ok": True, "recipe": RECIPE_NAME, "train_examples": len(train),
        "eval_examples": {key: len(value) for key, value in eval_sets.items()},
        "train_signatures": len(train_signatures),
        "structure_signatures": len(structure_signatures), "overlap": [],
        "unique_bins": len(bins), "bins": bins,
        "replay_bins_deferred_to_modal_preflight": sorted(set(deferred_replay)),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data-v4")
    parser.add_argument("--require-replay-local", action="store_true",
                        help="exige aussi une copie locale des anciens bins v4.5")
    args = parser.parse_args()
    print(json.dumps(audit(Path(args.data_dir), args.require_replay_local),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
