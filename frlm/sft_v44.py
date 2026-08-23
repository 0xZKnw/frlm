"""Préparation du SFT v4.4 équilibré par capacités.

Contrairement à la v4.2, un quota manquant ne traverse jamais une frontière de
capacité. Les bins restent séparés afin que le trainer contrôle réellement le mix
au lieu de le laisser dépendre du nombre et de la longueur des conversations.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np

from frlm import data as D
from frlm import synth_programs


RECIPE_NAME = "v4.4-balanced-capabilities-18m"
DEFAULT_TARGET_SUPERVISED = 18_000_000
VAL_SUPERVISED_PER_SOURCE = 40_000

SOURCE_RULES = {
    "chat_human":     dict(max_repeat=1.0, max_prompt=1800, max_final=1200, max_think=0),
    "croissant":      dict(max_repeat=1.0, max_prompt=1600, max_final=1200, max_think=0),
    "oasst_fr":       dict(max_repeat=1.0, max_prompt=1800, max_final=1200, max_think=0),
    "distill":        dict(max_repeat=1.0, max_prompt=900, max_final=500, max_think=500),
    "openhermes_fr":  dict(max_repeat=1.0, max_prompt=1400, max_final=900, max_think=0),
    "reasoning_v44":  dict(max_repeat=1.0, max_prompt=1000, max_final=220, max_think=500),
    "maths_sft":      dict(max_repeat=1.0, max_prompt=700, max_final=280, max_think=600),
    "gsm8k":          dict(max_repeat=1.0, max_prompt=1400, max_final=350, max_think=700),
    "grounded_v44":   dict(max_repeat=1.0, max_prompt=1800, max_final=500, max_think=400),
    "qa_human_v44":   dict(max_repeat=1.0, max_prompt=6500, max_final=900, max_think=0),
    "calibration_v44": dict(max_repeat=1.0, max_prompt=1000, max_final=300, max_think=0),
    "multiturn_v44":  dict(max_repeat=1.0, max_prompt=1200, max_final=300, max_think=0),
    "identity":       dict(max_repeat=1.0, max_prompt=500, max_final=500, max_think=0),
}

# Les parts internes servent uniquement à distribuer une capacité entre ses
# sources. Une pénurie peut être reprise dans la même capacité, jamais ailleurs.
CAPABILITIES = {
    "native_chat": {
        "weight": 0.25,
        "sources": {"chat_human": 0.68, "croissant": 0.28, "oasst_fr": 0.04},
    },
    "student_distill": {
        "weight": 0.20,
        "sources": {"distill": 0.25, "openhermes_fr": 0.75},
    },
    "verified_reasoning": {
        "weight": 0.30,
        "sources": {"reasoning_v44": 0.75, "maths_sft": 0.20, "gsm8k": 0.05},
    },
    "grounded_qa": {
        "weight": 0.12,
        "sources": {"grounded_v44": 0.70, "qa_human_v44": 0.30},
    },
    "constraints_calibration": {
        "weight": 0.08,
        "sources": {"calibration_v44": 1.0},
    },
    "multiturn_identity": {
        "weight": 0.05,
        "sources": {"multiturn_v44": 0.98, "identity": 0.02},
    },
}

# Caps globaux issus du rapport. Les autres sources restent sous le cap générique
# de 20 % par construction et par leur quota de capacité.
GLOBAL_SOURCE_CAPS = {"openhermes_fr": 0.15, "gsm8k": 0.03}


def _validate_recipe() -> None:
    if not math.isclose(sum(c["weight"] for c in CAPABILITIES.values()), 1.0,
                        abs_tol=1e-9):
        raise ValueError("les poids des capacités SFT v4.4 ne somment pas à 1")
    for name, cfg in CAPABILITIES.items():
        if not math.isclose(sum(cfg["sources"].values()), 1.0, abs_tol=1e-9):
            raise ValueError(f"les poids internes de {name} ne somment pas à 1")
        unknown = set(cfg["sources"]) - set(SOURCE_RULES)
        if unknown:
            raise ValueError(f"sources sans règles dans {name}: {sorted(unknown)}")


def _convert_qa_mid(raw_dir: Path) -> dict:
    """Convertit le QA causal du mid v4.3 en conversations assistant-only."""
    source = raw_dir / "frenchqa_mid_v43.jsonl"
    target = raw_dir / "qa_human_v44.jsonl"
    if not source.exists():
        return {"missing": True, "path": str(source)}
    tmp = target.with_suffix(".tmp")
    read = kept = 0
    with source.open(encoding="utf-8") as inp, tmp.open("w", encoding="utf-8") as out:
        for line in inp:
            read += 1
            try:
                text = str(json.loads(line).get("t") or "")
            except json.JSONDecodeError:
                continue
            if "\nRéponse : " not in text:
                continue
            prompt, answer = text.rsplit("\nRéponse : ", 1)
            prompt = prompt.removeprefix("Question : ").strip()
            answer = answer.strip()
            if not prompt or not answer:
                continue
            messages = [{"role": "user", "text": prompt},
                        {"role": "assistant", "text": answer}]
            record = {"t": D.render_chat(messages), "m": messages, "k": "grounded_qa",
                      "capability_id": "grounded_qa", "schema_id": "human_grounded_qa",
                      "surface_family_id": "human_source", "license": "source-dependent"}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1
    os.replace(tmp, target)
    return {"path": str(target), "read": read, "kept": kept,
            "bytes": target.stat().st_size}


def _ensure_raw(data_dir: Path, seed: int, skip_download: bool) -> dict:
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {}
    D._write_curated_identity(raw_dir / "identity.jsonl")

    required_inputs = {
        "chat_human": 80_000_000,
        "oasst_fr": 20_000_000,
        "openhermes_fr": 180_000_000,
        "gsm8k": 12_000_000,
        "maths_sft": 35_000_000,
    }
    for index, (source, budget) in enumerate(required_inputs.items()):
        path = raw_dir / f"{source}.jsonl"
        if path.exists() and path.stat().st_size > 0:
            report[f"{source}_input"] = {"reused": True, "path": str(path),
                                         "bytes": path.stat().st_size}
        elif skip_download:
            raise FileNotFoundError(f"--skip-download mais source absente : {path}")
        else:
            report[f"{source}_input"] = D.download_source(
                source, budget, path, seed=seed + 500 + index * 17
            )

    # La distillation student-aware demande un teacher externe. Une absence devient
    # un shortfall mesuré, jamais un remplacement silencieux par OpenHermes.
    distill = raw_dir / "distill.jsonl"
    if not distill.exists():
        distill.touch()
        report["distill_input"] = {
            "missing_teacher_data": True, "path": str(distill), "bytes": 0,
        }
    else:
        report["distill_input"] = {"reused": True, "path": str(distill),
                                   "bytes": distill.stat().st_size}

    # Croissant est fini et peut rester sous le quota : le shortfall sera explicite.
    croissant = raw_dir / "croissant.jsonl"
    if not croissant.exists() and skip_download:
        raise FileNotFoundError(f"--skip-download mais source absente : {croissant}")
    if not croissant.exists():
        report["croissant_download"] = D.download_source(
            "croissant", 35_000_000, croissant, seed=seed + 31
        )
    else:
        report["croissant_input"] = {"reused": True, "path": str(croissant),
                                     "bytes": croissant.stat().st_size}

    qa_mid = raw_dir / "frenchqa_mid_v43.jsonl"
    if not qa_mid.exists() and not skip_download:
        from frlm.mid_v43 import _download_qa_sources

        report["qa_download"] = _download_qa_sources(
            raw_dir, qa_chars=30_000_000, seed=seed + 41, skip_download=False
        )
    report["qa_conversion"] = _convert_qa_mid(raw_dir)

    synth_specs = {
        "reasoning_v44": (85_000_000, seed + 101),
        "grounded_v44": (90_000_000, seed + 211),
        "calibration_v44": (45_000_000, seed + 307),
        "multiturn_v44": (35_000_000, seed + 401),
    }
    for kind, (budget, source_seed) in synth_specs.items():
        path = raw_dir / f"{kind}.jsonl"
        if path.exists() and path.stat().st_size > 0:
            report[kind] = {"reused": True, "path": str(path),
                            "bytes": path.stat().st_size}
        else:
            report[kind] = synth_programs.write_jsonl(
                path, target_chars=budget, seed=source_seed, kind=kind
            )
    return report


def _encode_source(tok, path: Path, source: str, train_w: D.BinWriter,
                   val_w: D.BinWriter, seen_prompts: set[bytes], max_len: int,
                   source_rules: dict | None = None) -> dict:
    stats = {"read": 0, "kept": 0, "duplicates": 0,
             "train_conversations": 0, "val_conversations": 0,
             "train_supervised": 0, "val_supervised": 0}
    rejected: dict[str, int] = {}
    answer_counts: dict[str, int] = {}
    rules = source_rules or SOURCE_RULES
    max_same_answer = int(rules[source].get("max_same_answer", 100))
    eot = tok.token_to_id(D.EOT)
    for record in D.iter_jsonl(path):
        stats["read"] += 1
        messages, fingerprint, note = D._prepare_sft_messages(
            record, source, recipe=rules
        )
        if messages is None or fingerprint is None:
            key = note or "invalide"
            rejected[key] = rejected.get(key, 0) + 1
            continue
        if fingerprint in seen_prompts:
            stats["duplicates"] += 1
            continue
        answer_key = " ".join(
            " ".join(str(message.get("text") or "").split())
            for message in messages if message.get("role") == "assistant"
        ).casefold()
        if answer_counts.get(answer_key, 0) >= max_same_answer:
            rejected["reponse_exacte_surrepresentee"] = (
                rejected.get("reponse_exacte_surrepresentee", 0) + 1
            )
            continue
        answer_counts[answer_key] = answer_counts.get(answer_key, 0) + 1
        seen_prompts.add(fingerprint)
        ids: list[int] = []
        mask: list[int] = []
        for text, learn in D.chat_segments(messages, ensure_think=False):
            encoded = tok.encode(text).ids
            ids.extend(encoded)
            mask.extend([1 if learn else 0] * len(encoded))
        supervised = sum(mask)
        if len(ids) > max_len or supervised == 0:
            rejected["trop_de_tokens"] = rejected.get("trop_de_tokens", 0) + 1
            continue
        ids.append(eot)
        mask.append(0)
        explicit = record.get("split")
        if explicit in ("train", "val"):
            to_val = explicit == "val"
        else:
            to_val = int.from_bytes(fingerprint[:8], "little") % 1000 < 5
        writer = val_w if to_val else train_w
        writer.write(ids, mask)
        key = "val" if to_val else "train"
        stats[f"{key}_conversations"] += 1
        stats[f"{key}_supervised"] += supervised
        stats["kept"] += 1
        if note:
            rejected[note] = rejected.get(note, 0) + 1
    stats["rejected_or_transformed"] = rejected
    return stats


def _allocate_capability(name: str, source_stats: dict[str, dict], target: int,
                         total_target: int, capabilities: dict | None = None,
                         source_rules: dict | None = None,
                         global_caps: dict | None = None) -> tuple[dict[str, int], int]:
    capabilities = capabilities or CAPABILITIES
    source_rules = source_rules or SOURCE_RULES
    global_caps = global_caps or GLOBAL_SOURCE_CAPS
    cfg = capabilities[name]
    desired = {source: round(target * share)
               for source, share in cfg["sources"].items()}
    caps = {}
    for source in cfg["sources"]:
        unique = int(source_stats.get(source, {}).get("train_supervised", 0))
        cap = int(unique * source_rules[source]["max_repeat"])
        global_cap = global_caps.get(source, 0.20)
        cap = min(cap, round(total_target * global_cap))
        caps[source] = cap
    alloc = {source: min(desired[source], caps[source]) for source in desired}
    for _ in range(20):
        missing = target - sum(alloc.values())
        candidates = [s for s in alloc if alloc[s] < caps[s]]
        if missing <= 0 or not candidates:
            break
        weights = cfg["sources"]
        weight_sum = sum(weights[s] for s in candidates)
        progressed = 0
        for source in candidates:
            add = min(caps[source] - alloc[source],
                      max(1, round(missing * weights[source] / weight_sum)))
            alloc[source] += add
            progressed += add
        if progressed == 0:
            break
    return alloc, max(0, target - sum(alloc.values()))


def _append_shard(token_path: Path, mask_path: Path, writer: D.BinWriter) -> None:
    if token_path.stat().st_size == 0:
        return
    tokens = np.memmap(token_path, dtype=D.DTYPE, mode="r")
    masks = np.memmap(mask_path, dtype=np.uint8, mode="r")
    for start in range(0, len(tokens), 1_000_000):
        end = min(len(tokens), start + 1_000_000)
        writer.write(np.asarray(tokens[start:end]), np.asarray(masks[start:end]))
    del tokens, masks


def encode(tok, data_dir: Path, target_supervised: int,
           max_len: int = 2048) -> dict:
    data_dir = Path(data_dir)
    raw_dir = data_dir / "raw"
    ordered_sources = [source for cfg in CAPABILITIES.values()
                       for source in cfg["sources"]]
    paths = {source: raw_dir / f"{source}.jsonl" for source in ordered_sources}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("sources SFT v4.4 absentes : " + ", ".join(missing))

    seen_prompts: set[bytes] = set()
    source_stats: dict[str, dict] = {}
    capabilities_report: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix=".sft-v44-", dir=data_dir) as tmp_name:
        tmp = Path(tmp_name)
        for source in ordered_sources:
            train_w = D.BinWriter(tmp / f"{source}_train.bin", with_mask=True)
            val_w = D.BinWriter(tmp / f"{source}_val.bin", with_mask=True)
            stats = _encode_source(tok, paths[source], source, train_w, val_w,
                                   seen_prompts, max_len=max_len)
            train_w.close()
            val_w.close()
            stats["train_tokens_unique"] = (tmp / f"{source}_train.bin").stat().st_size // 2
            stats["val_tokens_unique"] = (tmp / f"{source}_val.bin").stat().st_size // 2
            source_stats[source] = stats

        aggregate_train = D.BinWriter(tmp / "sft_v44_train.bin", with_mask=True)
        aggregate_val = D.BinWriter(tmp / "sft_v44_val.bin", with_mask=True)
        final_shards: list[tuple[Path, Path]] = []
        for capability, cfg in CAPABILITIES.items():
            target = round(target_supervised * cfg["weight"])
            allocations, shortfall = _allocate_capability(
                capability, source_stats, target, target_supervised
            )
            train_path = tmp / f"sft_v44_train_{capability}.bin"
            val_path = tmp / f"sft_v44_val_{capability}.bin"
            train_w = D.BinWriter(train_path, with_mask=True)
            val_w = D.BinWriter(val_path, with_mask=True)
            source_report = {}
            for source in cfg["sources"]:
                stats = source_stats[source]
                rotation = int.from_bytes(hashlib.sha256(
                    f"v44:{source}".encode()).digest()[:8], "little")
                train_tok, train_sup = D._copy_masked_shard(
                    tmp / f"{source}_train.bin", tmp / f"{source}_train.mask",
                    train_w, allocations[source], stats["train_supervised"], rotation
                )
                val_target = min(VAL_SUPERVISED_PER_SOURCE, stats["val_supervised"])
                val_tok, val_sup = D._copy_masked_shard(
                    tmp / f"{source}_val.bin", tmp / f"{source}_val.mask",
                    val_w, val_target, stats["val_supervised"]
                )
                source_report[source] = {
                    "available_supervised": stats["train_supervised"],
                    "allocated_supervised": train_sup,
                    "train_tokens": train_tok,
                    "val_supervised": val_sup,
                    "val_tokens": val_tok,
                    "effective_repeat": round(train_sup / max(1, stats["train_supervised"]), 3),
                }
            train_tokens, val_tokens = train_w.n, val_w.n
            train_w.close()
            val_w.close()
            _append_shard(train_path, train_path.with_suffix(".mask"), aggregate_train)
            _append_shard(val_path, val_path.with_suffix(".mask"), aggregate_val)
            actual = sum(s["allocated_supervised"] for s in source_report.values())
            capabilities_report[capability] = {
                "weight": cfg["weight"], "target_supervised": target,
                "actual_supervised": actual,
                "shortfall_supervised": max(0, target - actual),
                "train_path": train_path.name, "train_tokens": train_tokens,
                "val_path": val_path.name, "val_tokens": val_tokens,
                "sources": source_report,
            }
        train_tokens, val_tokens = aggregate_train.n, aggregate_val.n
        aggregate_train.close()
        aggregate_val.close()

        # Publication atomique des bins : une préparation interrompue ne laisse
        # jamais un mélange v4.4 partiellement réécrit dans le répertoire final.
        for capability in CAPABILITIES:
            for split in ("train", "val"):
                name = f"sft_v44_{split}_{capability}"
                final_shards.extend(((tmp / f"{name}.bin", data_dir / f"{name}.bin"),
                                     (tmp / f"{name}.mask", data_dir / f"{name}.mask")))
        final_shards.extend(((tmp / "sft_v44_train.bin", data_dir / "sft_v44_train.bin"),
                             (tmp / "sft_v44_train.mask", data_dir / "sft_v44_train.mask"),
                             (tmp / "sft_v44_val.bin", data_dir / "sft_v44_val.bin"),
                             (tmp / "sft_v44_val.mask", data_dir / "sft_v44_val.mask")))
        for source_path, destination in final_shards:
            os.replace(source_path, destination)

    actual_supervised = sum(c["actual_supervised"] for c in capabilities_report.values())
    return {
        "recipe": RECIPE_NAME,
        "target_supervised_tokens": int(target_supervised),
        "actual_supervised_tokens": actual_supervised,
        "shortfall_supervised_tokens": int(target_supervised) - actual_supervised,
        "train_path": "sft_v44_train.bin", "train_tokens": train_tokens,
        "val_path": "sft_v44_val.bin", "val_tokens": val_tokens,
        "capabilities": capabilities_report,
        "sources": source_stats,
        "single_source_cap": 0.20, "openhermes_cap": 0.15,
        "gsm8k_cap": 0.03, "max_repeat": 1.0,
        "direct_answer_fraction_target": [0.65, 0.70],
        "no_cross_capability_redistribution": True,
    }


def prepare(data_dir: Path, target_supervised: int = DEFAULT_TARGET_SUPERVISED,
            max_seq_len: int = 2048, seed: int = 1337,
            skip_download: bool = False) -> dict:
    _validate_recipe()
    data_dir = Path(data_dir)
    tokenizer_path = data_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"tokenizer v4 introuvable : {tokenizer_path}")
    tok = D.load_tokenizer(tokenizer_path)
    raw_report = _ensure_raw(data_dir, seed=seed, skip_download=skip_download)
    report = encode(tok, data_dir, int(target_supervised), max_len=max_seq_len)
    report["raw_build"] = raw_report
    meta_path = data_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta["sft_v44"] = report
    tmp = meta_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, meta_path)
    return report
