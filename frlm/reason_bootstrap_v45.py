"""Bootstrap supervisé v4.5 basé sur des AST exécutables et des holdouts structurels.

OOD v2 n'est jamais lu par ce module. Les structures arithmétiques de validation
sont séparées du train par leur topologie d'AST, pas seulement par une autre seed.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

from frlm import data as D


RECIPE_NAME = "v4.5-reason-bootstrap-ast-1"
TRAIN_SPLIT = "train"
EVAL_SPLITS = ("iid", "surface_holdout", "structure_holdout")
OPS = ("add", "sub", "mul")
SYMBOLS = {"add": "+", "sub": "−", "mul": "×"}
OBJECTIVE_WEIGHTS = {
    "execute": 0.40,
    "masked_step": 0.10,
    "order_steps": 0.08,
    "find_error": 0.12,
    "number_only": 0.10,
}
REPLAY_WEIGHTS = {
    "general_response": 0.10,
    "verified_short_code": 0.03,
    "uncertainty": 0.03,
    "grounded_transformation": 0.02,
    "style_identity": 0.02,
}


def evaluate_ast(node: dict) -> int:
    if node["op"] == "const":
        return int(node["value"])
    left, right = (evaluate_ast(child) for child in node["args"])
    if node["op"] == "add":
        return left + right
    if node["op"] == "sub":
        return left - right
    if node["op"] == "mul":
        return left * right
    raise ValueError(f"opération AST inconnue : {node['op']}")


def operator_signature(node: dict) -> str:
    if node["op"] == "const":
        return "C"
    return f"{node['op']}({','.join(operator_signature(child) for child in node['args'])})"


def _const(rng: random.Random, small: bool = False) -> dict:
    limit = 9 if small else 35
    value = rng.randint(2, limit)
    if rng.random() < 0.12:
        value *= -1
    return {"op": "const", "value": value}


def _binary(rng: random.Random, left: dict, right: dict) -> dict:
    return {"op": rng.choice(OPS), "args": [left, right]}


def _candidate_ast(rng: random.Random, split: str) -> dict:
    if split == "structure_holdout":
        if rng.random() < 0.55:
            # Topologie volontairement absente du train : branche droite imbriquée.
            return _binary(rng, _const(rng),
                           _binary(rng, _const(rng, True), _const(rng, True)))
        # Deux sous-programmes indépendants ensuite combinés.
        return _binary(rng,
                       _binary(rng, _const(rng, True), _const(rng, True)),
                       _binary(rng, _const(rng, True), _const(rng, True)))
    if rng.random() < 0.35:
        return _binary(rng, _const(rng), _const(rng, True))
    # Topologie train/IID/surface : chaîne strictement imbriquée à gauche.
    return _binary(rng,
                   _binary(rng, _const(rng, True), _const(rng, True)),
                   _const(rng, True))


def make_ast(seed: int, split: str) -> dict:
    rng = random.Random(seed)
    for _ in range(200):
        node = _candidate_ast(rng, split)
        values = []

        def collect(current):
            values.append(evaluate_ast(current))
            for child in current.get("args", []):
                collect(child)

        collect(node)
        if all(abs(value) <= 600 for value in values) and abs(evaluate_ast(node)) >= 2:
            return node
    raise RuntimeError("impossible de générer un AST borné")


def _natural(node: dict, split: str, rng: random.Random) -> str:
    if node["op"] == "const":
        return str(node["value"])
    left = _natural(node["args"][0], split, rng)
    right = _natural(node["args"][1], split, rng)
    if split == "surface_holdout":
        templates = {
            "add": ("{a} augmenté de {b}", "le total obtenu avec {a} puis {b}"),
            "sub": ("{a} auquel on retire {b}", "ce qui reste de {a} après retrait de {b}"),
            "mul": ("{a} multiplié par {b}", "{b} fois la quantité {a}"),
        }
    else:
        templates = {
            "add": ("la somme de {a} et {b}", "{a} plus {b}"),
            "sub": ("la différence entre {a} et {b}", "{a} moins {b}"),
            "mul": ("le produit de {a} par {b}", "{a} fois {b}"),
        }
    return rng.choice(templates[node["op"]]).format(a=left, b=right)


def trace_ast(node: dict) -> tuple[list[dict], str]:
    rows: list[dict] = []

    def walk(current: dict) -> str:
        if current["op"] == "const":
            return str(current["value"])
        left = walk(current["args"][0])
        right = walk(current["args"][1])
        index = len(rows) + 1
        value = evaluate_ast(current)
        rows.append({"step": index, "left": left, "right": right,
                     "op": current["op"], "value": value})
        return f"r{index}"

    output = walk(node)
    return rows, output


def _trace_line(row: dict, value: int | str | None = None) -> str:
    result = row["value"] if value is None else value
    return (f"Étape {row['step']} : r{row['step']} = {row['left']} "
            f"{SYMBOLS[row['op']]} {row['right']} = {result}")


def make_example(seed: int, split: str = "train", objective: str = "execute") -> dict:
    if split not in (TRAIN_SPLIT, *EVAL_SPLITS):
        raise ValueError(f"split inconnu : {split}")
    if objective not in OBJECTIVE_WEIGHTS:
        raise ValueError(f"objectif inconnu : {objective}")
    ast = make_ast(seed, split)
    rng = random.Random(seed ^ 0x45A57)
    result = evaluate_ast(ast)
    trace, _ = trace_ast(ast)
    expression = _natural(ast, split, rng)
    answer: str
    if objective == "execute":
        prompt = f"Calcule {expression}. Donne les étapes utiles puis la réponse finale."
        answer = (f"{D.THINK}\n" + "\n".join(_trace_line(row) for row in trace)
                  + f"\n{D.THINK_END}\nRéponse : {result}")
        target = str(result)
    elif objective == "number_only":
        prompt = f"Calcule {expression}. Réponds uniquement par le nombre final."
        answer = str(result)
        target = str(result)
    elif objective == "masked_step":
        hidden = rng.randrange(len(trace))
        shown = [_trace_line(row, "[MASQUÉ]" if index == hidden else None)
                 for index, row in enumerate(trace)]
        prompt = ("Complète exactement le résultat masqué dans cette exécution :\n"
                  + "\n".join(shown))
        target = str(trace[hidden]["value"])
        answer = f"Réponse : {target}"
    elif objective == "order_steps":
        order = list(range(len(trace)))
        rng.shuffle(order)
        labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        shown = [f"{labels[pos]}. {_trace_line(trace[index])}"
                 for pos, index in enumerate(order)]
        expected = [labels[order.index(index)] for index in range(len(trace))]
        target = ",".join(expected)
        prompt = ("Remets ces étapes dans l'ordre d'exécution. Réponds seulement par "
                  "les lettres séparées par des virgules :\n" + "\n".join(shown))
        answer = target
    else:
        wrong = rng.randrange(len(trace))
        shown = []
        for index, row in enumerate(trace):
            value = row["value"] + rng.choice((-3, -2, -1, 1, 2, 3)) if index == wrong else None
            shown.append(_trace_line(row, value))
        target = str(wrong + 1)
        prompt = ("Une trace contient une erreur de calcul. Donne uniquement le numéro "
                  "de la première étape fausse :\n" + "\n".join(shown))
        answer = target
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return {
        "id": prompt_hash[:20], "seed": seed, "split": split,
        "objective": objective, "program_ast": ast,
        "operator_signature": operator_signature(ast), "surface_family": split,
        "prompt": prompt, "answer": answer, "target": target,
        "messages": [{"role": "user", "text": prompt},
                     {"role": "assistant", "text": answer}],
    }


def _encode_record(tok, record: dict, max_len: int) -> tuple[list[int], list[int]] | None:
    ids: list[int] = []
    mask: list[int] = []
    for text, supervised in D.chat_segments(record["messages"]):
        encoded = tok.encode(text).ids
        ids.extend(encoded)
        mask.extend([int(supervised)] * len(encoded))
    ids.append(tok.token_to_id(D.EOT))
    mask.append(0)
    if len(ids) > max_len + 1 or not any(mask):
        return None
    return ids, mask


def _write_records(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
                    encoding="utf-8")


def _build_bucket(tok, records: list[dict], bin_path: Path, max_len: int) -> dict:
    writer = D.BinWriter(bin_path, with_mask=True)
    supervised = conversations = dropped = 0
    lengths = []
    for record in records:
        encoded = _encode_record(tok, record, max_len)
        if encoded is None:
            dropped += 1
            continue
        ids, mask = encoded
        writer.write(ids, mask)
        supervised += sum(mask)
        conversations += 1
        lengths.append(len(ids))
    tokens = writer.n
    writer.close()
    return {"tokens": tokens, "supervised": supervised, "conversations": conversations,
            "dropped": dropped, "max_tokens": max(lengths, default=0)}


def _unique_examples(count: int, split: str, seed: int, used_ids: set[str],
                     objective_rng: random.Random | None = None) -> list[dict]:
    """Produit exactement ``count`` prompts uniques, de façon déterministe."""
    rows: list[dict] = []
    attempts = 0
    objectives = tuple(OBJECTIVE_WEIGHTS)
    probabilities = tuple(OBJECTIVE_WEIGHTS.values())
    while len(rows) < count:
        if attempts > count * 100:
            raise RuntimeError(f"espace de prompts épuisé pour {split} ({len(rows)}/{count})")
        objective = (objective_rng.choices(objectives, probabilities, k=1)[0]
                     if objective_rng is not None
                     else objectives[len(rows) % len(objectives)])
        row = make_example(seed + attempts * 97, split, objective)
        attempts += 1
        if row["id"] in used_ids:
            continue
        used_ids.add(row["id"])
        rows.append(row)
    return rows


def _append_file(source: Path, destination) -> None:
    with source.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            destination.write(chunk)


def prepare(data_dir: Path, examples: int = 20_000, max_len: int = 256,
            seed: int = 455_500, eval_per_split: int = 120) -> dict:
    """Génère les bins sans téléchargement et publie un manifeste auditable."""
    if examples < 1_000 or eval_per_split < 30:
        raise ValueError("bootstrap trop petit pour mesurer la généralisation")
    data_dir = Path(data_dir)
    tok = D.load_tokenizer(data_dir / "tokenizer.json")
    rng = random.Random(seed)
    used_ids: set[str] = set()
    train = _unique_examples(examples, "train", seed, used_ids, rng)
    eval_sets = {
        split: _unique_examples(eval_per_split, split,
                                seed + 10_000_000 + split_index * 1_000_003,
                                used_ids)
        for split_index, split in enumerate(EVAL_SPLITS)
    }
    train_signatures = {row["operator_signature"] for row in train}
    structure_signatures = {row["operator_signature"]
                            for row in eval_sets["structure_holdout"]}
    overlap = train_signatures & structure_signatures
    if overlap:
        raise RuntimeError(f"contamination structurelle train/dev : {sorted(overlap)[:5]}")
    all_ids = [row["id"] for row in train]
    all_ids.extend(row["id"] for rows in eval_sets.values() for row in rows)
    if len(all_ids) != len(set(all_ids)) or len(all_ids) != len(used_ids):
        raise RuntimeError("prompts dupliqués entre les splits")

    raw_dir = data_dir / "raw"
    _write_records(raw_dir / "reason_bootstrap_v45_train.jsonl", train)
    for split, rows in eval_sets.items():
        _write_records(raw_dir / f"reason_bootstrap_v45_{split}.jsonl", rows)

    capabilities: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix=".reason-v45-", dir=data_dir) as tmp_name:
        tmp = Path(tmp_name)
        aggregate_train = D.BinWriter(tmp / "reason_v45_train.bin", with_mask=True)
        aggregate_val = D.BinWriter(tmp / "reason_v45_val.bin", with_mask=True)
        for objective, weight in OBJECTIVE_WEIGHTS.items():
            rows = [row for row in train if row["objective"] == objective]
            val_rows = [row for row in eval_sets["iid"] if row["objective"] == objective]
            train_path = tmp / f"reason_v45_train_{objective}.bin"
            val_path = tmp / f"reason_v45_val_{objective}.bin"
            train_stats = _build_bucket(tok, rows, train_path, max_len)
            val_stats = _build_bucket(tok, val_rows, val_path, max_len)
            _append_file(train_path, aggregate_train.f)
            _append_file(train_path.with_suffix(".mask"), aggregate_train.fm)
            aggregate_train.n += train_stats["tokens"]
            _append_file(val_path, aggregate_val.f)
            _append_file(val_path.with_suffix(".mask"), aggregate_val.fm)
            aggregate_val.n += val_stats["tokens"]
            capabilities[f"ast_{objective}"] = {
                "sampling_weight": weight, "actual_supervised": train_stats["supervised"],
                "train_path": train_path.name, "train_tokens": train_stats["tokens"],
                "train_conversations": train_stats["conversations"],
                "val_path": val_path.name, "val_tokens": val_stats["tokens"],
                "val_conversations": val_stats["conversations"],
            }
        train_tokens, val_tokens = aggregate_train.n, aggregate_val.n
        aggregate_train.close()
        aggregate_val.close()
        publications = list(tmp.glob("reason_v45_*.bin")) + list(tmp.glob("reason_v45_*.mask"))
        for source in publications:
            os.replace(source, data_dir / source.name)

    old_meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    old_capabilities = (old_meta.get("sft_v45") or {}).get("capabilities") or {}
    for name, weight in REPLAY_WEIGHTS.items():
        if name not in old_capabilities:
            raise ValueError(f"capacité de rétention v4.5 absente : {name}")
        source = dict(old_capabilities[name])
        source["sampling_weight"] = weight
        source["replay_source"] = "sft_v45"
        capabilities[f"replay_{name}"] = source

    report = {
        "recipe": RECIPE_NAME, "seed": seed, "examples": examples,
        "max_conversation_tokens": max_len, "sequence_isolation_required": True,
        "train_path": "reason_v45_train.bin", "train_tokens": train_tokens,
        "val_path": "reason_v45_val.bin", "val_tokens": val_tokens,
        "actual_supervised_tokens": sum(
            int(row["actual_supervised"]) for name, row in capabilities.items()
            if name.startswith("ast_")
        ),
        "objective_counts": dict(Counter(row["objective"] for row in train)),
        "operator_signatures_train": sorted(train_signatures),
        "operator_signatures_structure_holdout": sorted(structure_signatures),
        "structure_overlap": [], "eval_per_split": eval_per_split,
        "eval_manifests": {split: f"raw/reason_bootstrap_v45_{split}.jsonl"
                           for split in EVAL_SPLITS},
        "capabilities": capabilities,
    }
    old_meta["reason_bootstrap_v45"] = report
    tmp_meta = data_dir / "meta.json.reason-v45.tmp"
    tmp_meta.write_text(json.dumps(old_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_meta.replace(data_dir / "meta.json")
    return report
