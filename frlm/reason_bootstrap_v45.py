"""Bootstrap supervisé v4.5 basé sur des AST exécutables et des holdouts structurels.

OOD v2 n'est jamais lu par ce module. Les structures arithmétiques de validation
sont séparées du train par leur topologie d'AST, pas seulement par une autre seed.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import random
import tempfile
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from frlm import data as D


RECIPE_NAME = "v4.5-reason-bootstrap-ast-1"
BALANCED_RECIPE_NAME = "v4.5-reason-bootstrap-balanced-2"
CORRECTED_RECIPE_NAME = "v4.5-reason-bootstrap-corrected-3"
TRAIN_SPLIT = "train"
EVAL_SPLITS = ("iid", "surface_holdout", "structure_holdout")
FINAL_SPLIT = "final_sealed"
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

# Seconde passe courte : environ une exposition au corpus AST au lieu des
# 3,3 expositions de la première recette, avec une majorité de sorties SFT
# diversifiées pour préserver la politique conversationnelle.
BALANCED_AST_WEIGHTS = {
    "execute": 0.18,
    "masked_step": 0.04,
    "order_steps": 0.04,
    "find_error": 0.07,
    "number_only": 0.02,
}
BALANCED_REPLAY_WEIGHTS = {
    "general_response": 0.23,
    "grounded_transformation": 0.07,
    "verified_reasoning": 0.08,
    "constraints_structure": 0.10,
    "multiturn": 0.06,
    "verified_short_code": 0.04,
    "uncertainty": 0.04,
    "style_identity": 0.03,
}


def _check_weights(ast_weights: dict[str, float], replay_weights: dict[str, float]) -> None:
    if abs(sum(ast_weights.values()) + sum(replay_weights.values()) - 1.0) > 1e-9:
        raise ValueError("les poids du mélange SFT doivent totaliser 1")
    if any(weight <= 0 for weight in (*ast_weights.values(), *replay_weights.values())):
        raise ValueError("les poids du mélange SFT doivent être strictement positifs")


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


def _candidate_ast(rng: random.Random, split: str, operations: int) -> dict:
    if split == "structure_holdout":
        if operations == 2:
            # Topologie volontairement absente du train : branche droite imbriquée.
            return _binary(rng, _const(rng),
                           _binary(rng, _const(rng, True), _const(rng, True)))
        # Deux sous-programmes indépendants ensuite combinés.
        return _binary(rng,
                       _binary(rng, _const(rng, True), _const(rng, True)),
                       _binary(rng, _const(rng, True), _const(rng, True)))
    # Topologie train/IID/surface/final : chaîne imbriquée à gauche.
    node = _const(rng)
    for _ in range(operations):
        node = _binary(rng, node, _const(rng, True))
    return node


def program_key(node: dict) -> str:
    """Identité indépendante de la surface et de l'ordre des opérandes commutatifs."""
    def canonical(current):
        if current["op"] == "const":
            return ("const", current["value"])
        children = [canonical(child) for child in current["args"]]
        if current["op"] in ("add", "mul"):
            children.sort(key=repr)
        return (current["op"], *children)
    return hashlib.sha256(repr(canonical(node)).encode()).hexdigest()


def program_split(node: dict) -> str:
    bucket = int(program_key(node)[:8], 16) % 10
    return {6: "iid", 7: "surface_holdout", 8: "structure_holdout",
            9: FINAL_SPLIT}.get(bucket, "train")


def make_ast(seed: int, split: str, operations: int | None = None) -> dict:
    rng = random.Random(seed)
    operations = operations or rng.choice((2, 3) if split == "structure_holdout" else (1, 2, 3))
    if operations not in ((2, 3) if split == "structure_holdout" else (1, 2, 3)):
        raise ValueError("nombre d'opérations incompatible avec le split")
    for _ in range(10_000):
        node = _candidate_ast(rng, split, operations)
        if program_split(node) != split:
            continue
        values = []

        def collect(current):
            values.append(evaluate_ast(current))
            for child in current.get("args", []):
                collect(child)

        collect(node)
        # Pas de sous-calcul annulé (×0), identité (×1) ou résultat trivial.
        if all(2 <= abs(value) <= 600 for value in values):
            return node
    raise RuntimeError("impossible de générer un AST borné")


def _natural(node: dict, split: str, rng: random.Random) -> str:
    if node["op"] == "const":
        return f"({node['value']})" if node["value"] < 0 else str(node["value"])
    left = _natural(node["args"][0], split, rng)
    right = _natural(node["args"][1], split, rng)
    if split == FINAL_SPLIT:
        templates = {
            "add": ("l'addition de {a} avec {b}",),
            "sub": ("la soustraction de {b} à {a}",),
            "mul": ("la multiplication de {a} par {b}",),
        }
    elif split == "surface_holdout":
        templates = {
            "add": ("{a} augmenté de {b}", "le total de {a} et de {b}"),
            "sub": ("{a} auquel on retire {b}", "ce qui reste de {a} après retrait de {b}"),
            "mul": ("{a} multiplié par {b}", "{b} fois la quantité {a}"),
        }
    else:
        templates = {
            "add": ("la somme de {a} et {b}", "{a} plus {b}"),
            "sub": ("la soustraction de {b} à partir de {a}", "{a} moins {b}"),
            "mul": ("le produit de {a} par {b}", "{a} fois {b}"),
        }
    return "(" + rng.choice(templates[node["op"]]).format(a=left, b=right) + ")"


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


def make_example(seed: int, split: str = "train", objective: str = "execute",
                 operations: int | None = None, error_position: int | None = None) -> dict:
    if split not in (TRAIN_SPLIT, *EVAL_SPLITS, FINAL_SPLIT):
        raise ValueError(f"split inconnu : {split}")
    if objective not in OBJECTIVE_WEIGHTS:
        raise ValueError(f"objectif inconnu : {objective}")
    rng = random.Random(seed ^ 0x45A57)
    minimum = 1 if objective in ("execute", "number_only") and split != "structure_holdout" else 2
    operations = operations if operations is not None else rng.randint(minimum, 3)
    if operations < minimum:
        raise ValueError("les objectifs de trace exigent au moins deux opérations")
    ast = make_ast(seed, split, operations)
    result = evaluate_ast(ast)
    trace, _ = trace_ast(ast)
    expression = _natural(ast, split, rng)
    answer: str
    valid_orders = []
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
        instruction = {
            "surface_holdout": "Retrouve la valeur cachée dans la trace ci-dessous :",
            FINAL_SPLIT: "Quel entier doit remplacer [MASQUÉ] dans ces calculs ?",
        }.get(split, "Complète exactement le résultat masqué dans cette exécution :")
        prompt = instruction + "\n" + "\n".join(shown)
        target = str(trace[hidden]["value"])
        answer = f"Réponse : {target}"
    elif objective == "order_steps":
        order = list(range(len(trace)))
        rng.shuffle(order)
        labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        # Les noms n'encodent ni l'ordre d'exécution ni l'ordre d'affichage.
        names = rng.sample(("ambre", "brique", "cendre", "dune", "ecume", "feuille",
                            "galet", "houle", "iris", "jade", "kiwi", "lune"), len(trace))
        variables = {f"r{index + 1}": name for index, name in enumerate(names)}
        shown = []
        for pos, index in enumerate(order):
            row = trace[index]
            left = variables.get(row["left"], row["left"])
            right = variables.get(row["right"], row["right"])
            shown.append(f"{labels[pos]}. {names[index]} = {left} {SYMBOLS[row['op']]} {right}")
        expected = [labels[order.index(index)] for index in range(len(trace))]
        target = ",".join(expected)
        for permutation in itertools.permutations(range(len(trace))):
            ready = set()
            for index in permutation:
                row = trace[index]
                dependencies = {value for value in (row["left"], row["right"])
                                if value.startswith("r")}
                if not dependencies <= ready:
                    break
                ready.add(f"r{index + 1}")
            else:
                valid_orders.append(",".join(labels[order.index(index)] for index in permutation))
        instruction = {
            "surface_holdout": "Classe les affectations selon leurs dépendances. Donne un ordre valide de lettres séparées par des virgules :",
            FINAL_SPLIT: "Chaque variable doit être définie avant son utilisation. Fournis une suite valide des lettres, avec des virgules :",
        }.get(split, "Remets ces étapes dans un ordre d'exécution valide. Réponds seulement par les lettres séparées par des virgules :")
        prompt = instruction + "\n" + "\n".join(shown)
        answer = target
    else:
        wrong = rng.randrange(len(trace)) if error_position is None else error_position
        if not 0 <= wrong < len(trace):
            raise ValueError("position d'erreur hors trace")
        shown = []
        displayed = {}
        for index, row in enumerate(trace):
            operands = [displayed[part] if part in displayed else int(part)
                        for part in (row["left"], row["right"])]
            value = evaluate_ast({"op": row["op"], "args": [
                {"op": "const", "value": part} for part in operands]})
            if index == wrong:
                value += rng.choice((-3, -2, -1, 1, 2, 3))
            displayed[f"r{index + 1}"] = value
            shown.append(_trace_line(row, value))
        target = str(wrong + 1)
        instruction = {
            "surface_holdout": "Indique seulement l'indice de l'unique erreur arithmétique de cette trace :",
            FINAL_SPLIT: "Vérifie ces égalités successives. Réponds par le numéro de la première égalité incorrecte, sans autre texte :",
        }.get(split, "Une trace contient une erreur de calcul. Donne uniquement le numéro de la première étape fausse :")
        prompt = instruction + "\n" + "\n".join(shown)
        answer = target
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return {
        "id": prompt_hash[:20], "seed": seed, "split": split,
        "objective": objective, "program_ast": ast,
        "recipe": CORRECTED_RECIPE_NAME, "program_key": program_key(ast),
        "operations": len(trace), "difficulty": f"ops_{len(trace)}",
        "valid_orders": valid_orders,
        "operator_signature": operator_signature(ast),
        "surface_family": split if split in ("surface_holdout", FINAL_SPLIT) else "base",
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


def _unique_examples(count: int, split: str, seed: int, used_ids: set[str]) -> list[dict]:
    """Produit exactement ``count`` prompts uniques, de façon déterministe."""
    rows: list[dict] = []
    attempts = 0
    objectives = tuple(OBJECTIVE_WEIGHTS)
    counts = Counter()
    error_counts = Counter()
    while len(rows) < count:
        if attempts > count * 100:
            raise RuntimeError(f"espace de prompts épuisé pour {split} ({len(rows)}/{count})")
        # Le stock est équilibré ; le mélange d'entraînement est pondéré séparément
        # en tokens assistant, et non deux fois (stock puis sampler).
        objective = objectives[len(rows) % len(objectives)]
        minimum = 1 if objective in ("execute", "number_only") and split != "structure_holdout" else 2
        operations = minimum + counts[objective] % (4 - minimum)
        position = error_counts[operations] % operations
        row = make_example(seed + attempts * 97, split, objective, operations, position)
        attempts += 1
        if row["id"] in used_ids:
            continue
        used_ids.add(row["id"])
        rows.append(row)
        counts[objective] += 1
        if objective == "find_error":
            error_counts[operations] += 1
    return rows


def _append_file(source: Path, destination) -> None:
    with source.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            destination.write(chunk)


def prepare(data_dir: Path, examples: int = 20_000, max_len: int = 512,
            seed: int = 455_900, eval_per_split: int = 120) -> dict:
    """Publie uniquement reason45c, sans téléchargement ni écrasement historique."""
    if examples < 1_000 or eval_per_split < 30:
        raise ValueError("bootstrap trop petit pour mesurer la généralisation")
    data_dir = Path(data_dir)
    tok = D.load_tokenizer(data_dir / "tokenizer.json")
    used_ids: set[str] = set()
    train = _unique_examples(examples, "train", seed, used_ids)
    eval_sets = {
        split: _unique_examples(eval_per_split, split,
                                seed + 10_000_000 + split_index * 1_000_003,
                                used_ids)
        for split_index, split in enumerate((*EVAL_SPLITS, FINAL_SPLIT))
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

    old_meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    old_capabilities = (old_meta.get("sft_v45") or {}).get("capabilities") or {}
    # Ne pas réutiliser les compartiments qui contiennent les défauts synthétiques
    # corrigés ici (remises et instruction JSON incomplète).
    replay_weights = {name: weight for name, weight in BALANCED_REPLAY_WEIGHTS.items()
                      if name not in ("verified_reasoning", "constraints_structure")}
    total_replay = sum(replay_weights.values())
    replay_weights = {name: weight * 0.65 / total_replay
                      for name, weight in replay_weights.items()}
    for name in replay_weights:
        if name not in old_capabilities:
            raise ValueError(f"capacité de rétention v4.5 absente : {name}")
    raw_dir = data_dir / "raw"
    if (old_meta.get("reason_bootstrap_v45c") or any(data_dir.glob("reason_v45c_*"))
            or any(raw_dir.glob("reason_bootstrap_v45c_*"))):
        raise FileExistsError("reason45c existe déjà : utiliser un autre répertoire de données")
    capabilities: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix=".reason-v45c-", dir=data_dir) as tmp_name, ExitStack() as cleanup:
        tmp = Path(tmp_name)
        _write_records(tmp / "raw" / "reason_bootstrap_v45c_train.jsonl", train)
        for split, rows in eval_sets.items():
            _write_records(tmp / "raw" / f"reason_bootstrap_v45c_{split}.jsonl", rows)
        aggregate_train = D.BinWriter(tmp / "reason_v45c_train.bin", with_mask=True)
        cleanup.callback(aggregate_train.close)
        aggregate_val = D.BinWriter(tmp / "reason_v45c_val.bin", with_mask=True)
        cleanup.callback(aggregate_val.close)
        for objective, weight in BALANCED_AST_WEIGHTS.items():
            rows = [row for row in train if row["objective"] == objective]
            val_rows = [row for row in eval_sets["iid"] if row["objective"] == objective]
            train_path = tmp / f"reason_v45c_train_{objective}.bin"
            val_path = tmp / f"reason_v45c_val_{objective}.bin"
            train_stats = _build_bucket(tok, rows, train_path, max_len)
            val_stats = _build_bucket(tok, val_rows, val_path, max_len)
            if train_stats["dropped"] or val_stats["dropped"]:
                raise ValueError(f"{objective} dépasse seq-len={max_len}; augmenter --seq-len")
            _append_file(train_path, aggregate_train.f)
            _append_file(train_path.with_suffix(".mask"), aggregate_train.fm)
            aggregate_train.n += train_stats["tokens"]
            _append_file(val_path, aggregate_val.f)
            _append_file(val_path.with_suffix(".mask"), aggregate_val.fm)
            aggregate_val.n += val_stats["tokens"]
            capabilities[f"ast_{objective}"] = {
                "target_token_weight": weight, "actual_supervised": train_stats["supervised"],
                "train_path": train_path.name, "train_tokens": train_stats["tokens"],
                "train_conversations": train_stats["conversations"],
                "val_path": val_path.name, "val_tokens": val_stats["tokens"],
                "val_conversations": val_stats["conversations"],
            }
        train_tokens, val_tokens = aggregate_train.n, aggregate_val.n
        aggregate_train.close()
        aggregate_val.close()
        publications = list(tmp.glob("reason_v45c_*.bin")) + list(tmp.glob("reason_v45c_*.mask"))
        raw_dir.mkdir(exist_ok=True)
        for source in (tmp / "raw").glob("*.jsonl"):
            os.replace(source, raw_dir / source.name)
        for source in publications:
            os.replace(source, data_dir / source.name)

    for name, weight in replay_weights.items():
        source = dict(old_capabilities[name])
        source.pop("sampling_weight", None)
        source["target_token_weight"] = weight
        source["replay_source"] = "sft_v45"
        capabilities[f"replay_{name}"] = source

    report = {
        "recipe": CORRECTED_RECIPE_NAME, "seed": seed, "examples": examples,
        "max_conversation_tokens": max_len, "sequence_isolation_required": True,
        "sampling_unit": "supervised_tokens",
        "train_manifest": "raw/reason_bootstrap_v45c_train.jsonl",
        "final_split": FINAL_SPLIT,
        "final_use": "une seule évaluation après gel du checkpoint et du protocole",
        "train_path": "reason_v45c_train.bin", "train_tokens": train_tokens,
        "val_path": "reason_v45c_val.bin", "val_tokens": val_tokens,
        "actual_supervised_tokens": sum(
            int(row["actual_supervised"]) for name, row in capabilities.items()
            if name.startswith("ast_")
        ),
        "objective_counts": dict(Counter(row["objective"] for row in train)),
        "operator_signatures_train": sorted(train_signatures),
        "operator_signatures_structure_holdout": sorted(structure_signatures),
        "structure_overlap": [], "eval_per_split": eval_per_split,
        "eval_manifests": {split: f"raw/reason_bootstrap_v45c_{split}.jsonl"
                           for split in (*EVAL_SPLITS, FINAL_SPLIT)},
        "manifest_sha256": {split: hashlib.sha256(
            (raw_dir / f"reason_bootstrap_v45c_{split}.jsonl").read_bytes()).hexdigest()
            for split in ("train", *EVAL_SPLITS, FINAL_SPLIT)},
        "capabilities": capabilities,
    }
    old_meta["reason_bootstrap_v45c"] = report
    tmp_meta = data_dir / "meta.json.reason-v45c.tmp"
    tmp_meta.write_text(json.dumps(old_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_meta.replace(data_dir / "meta.json")
    return report


def prepare_balanced(data_dir: Path) -> dict:
    """Publie le mélange 35 % AST / 65 % rétention sans régénérer les bins."""
    _check_weights(BALANCED_AST_WEIGHTS, BALANCED_REPLAY_WEIGHTS)
    data_dir = Path(data_dir)
    meta_path = data_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    ast_section = meta.get("reason_bootstrap_v45") or {}
    sft_section = meta.get("sft_v45") or {}
    if ast_section.get("recipe") != RECIPE_NAME:
        raise ValueError("prépare d'abord la recette reason45 et ses bins AST")
    sft_capabilities = sft_section.get("capabilities") or {}

    capabilities: dict[str, dict] = {}
    for objective, weight in BALANCED_AST_WEIGHTS.items():
        name = f"ast_{objective}"
        if name not in ast_section["capabilities"]:
            raise ValueError(f"capacité AST absente : {name}")
        capability = dict(ast_section["capabilities"][name])
        capability["sampling_weight"] = weight
        capabilities[name] = capability
    for name, weight in BALANCED_REPLAY_WEIGHTS.items():
        if name not in sft_capabilities:
            raise ValueError(f"capacité de rétention v4.5 absente : {name}")
        capability = dict(sft_capabilities[name])
        capability["sampling_weight"] = weight
        capability["replay_source"] = "sft_v45"
        capabilities[f"replay_{name}"] = capability

    report = {
        "recipe": BALANCED_RECIPE_NAME,
        "source_recipe": RECIPE_NAME,
        "seed": ast_section["seed"],
        "examples": ast_section["examples"],
        "max_conversation_tokens": ast_section["max_conversation_tokens"],
        "sequence_isolation_required": True,
        "train_path": ast_section["train_path"],
        "train_tokens": ast_section["train_tokens"],
        "val_path": ast_section["val_path"],
        "val_tokens": ast_section["val_tokens"],
        "actual_supervised_tokens": ast_section["actual_supervised_tokens"],
        "ast_fraction": sum(BALANCED_AST_WEIGHTS.values()),
        "retention_fraction": sum(BALANCED_REPLAY_WEIGHTS.values()),
        "eval_per_split": ast_section["eval_per_split"],
        "eval_manifests": ast_section["eval_manifests"],
        "operator_signatures_train": ast_section["operator_signatures_train"],
        "operator_signatures_structure_holdout": (
            ast_section["operator_signatures_structure_holdout"]
        ),
        "structure_overlap": ast_section["structure_overlap"],
        "capabilities": capabilities,
    }
    meta["reason_bootstrap_v45b"] = report
    tmp_meta = data_dir / "meta.json.reason-v45b.tmp"
    tmp_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_meta.replace(meta_path)
    return report
