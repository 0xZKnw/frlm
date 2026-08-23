"""SFT v4.5 : 24 M tokens assistant, conversations isolées et mix audité.

Cette recette part exclusivement du checkpoint MID v4.3. Elle conserve chaque
conversation entière lors des allocations : aucun shard n'est coupé au milieu
d'un tour et le chargeur ``ConversationCorpus`` empêche tout état inter-document.
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
from frlm import sft_v44 as V44
from frlm import synth_programs


RECIPE_NAME = "v4.5-audited-isolated-24m"
DEFAULT_TARGET_SUPERVISED = 24_000_000
VAL_SUPERVISED_PER_SOURCE = 40_000

SOURCE_RULES = {
    "chat_human": dict(max_repeat=1.0, max_prompt=1500, max_final=1000, max_think=0),
    "croissant": dict(max_repeat=1.0, max_prompt=1400, max_final=900, max_think=0),
    "oasst_fr": dict(max_repeat=1.0, max_prompt=1500, max_final=900, max_think=0),
    "openhermes_fr": dict(max_repeat=1.0, max_prompt=1300, max_final=750, max_think=0),
    "qa_human_v44": dict(max_repeat=1.0, max_prompt=1500, max_final=650, max_think=0),
    "maths_sft": dict(max_repeat=1.0, max_prompt=700, max_final=260, max_think=450),
    "gsm8k": dict(max_repeat=1.0, max_prompt=1200, max_final=320, max_think=500),
    "general_v45": dict(max_repeat=1.0, max_prompt=900, max_final=550, max_think=0),
    "grounded_v45": dict(max_repeat=1.0, max_prompt=1200, max_final=360, max_think=180),
    "reasoning_v45": dict(max_repeat=1.0, max_prompt=800, max_final=180, max_think=360),
    "constraints_v45": dict(max_repeat=1.0, max_prompt=700, max_final=300, max_think=0),
    "multiturn_v45": dict(max_repeat=1.0, max_prompt=900, max_final=260, max_think=0),
    "code_v45": dict(max_repeat=1.0, max_prompt=800, max_final=550, max_think=0),
    "uncertainty_v45": dict(max_repeat=1.0, max_prompt=900, max_final=300, max_think=0),
    "style_v45": dict(max_repeat=1.0, max_prompt=800, max_final=350, max_think=0),
    "identity": dict(max_repeat=1.0, max_prompt=500, max_final=500, max_think=0),
}

CAPABILITIES = {
    "general_response": {
        "weight": 0.30,
        "sources": {
            "chat_human": 0.32, "openhermes_fr": 0.40, "general_v45": 0.21,
            "croissant": 0.06, "oasst_fr": 0.01,
        },
    },
    "grounded_transformation": {
        "weight": 0.18,
        "sources": {"grounded_v45": 0.84, "qa_human_v44": 0.16},
    },
    "verified_reasoning": {
        "weight": 0.18,
        "sources": {"reasoning_v45": 0.75, "maths_sft": 0.20, "gsm8k": 0.05},
    },
    "constraints_structure": {
        "weight": 0.12, "sources": {"constraints_v45": 1.0},
    },
    "multiturn": {
        "weight": 0.08, "sources": {"multiturn_v45": 1.0},
    },
    "verified_short_code": {
        "weight": 0.06, "sources": {"code_v45": 1.0},
    },
    "uncertainty": {
        "weight": 0.05, "sources": {"uncertainty_v45": 1.0},
    },
    "style_identity": {
        "weight": 0.03, "sources": {"style_v45": 0.995, "identity": 0.005},
    },
}

GLOBAL_SOURCE_CAPS = {"openhermes_fr": 0.12, "gsm8k": 0.02}


def _validate_recipe() -> None:
    if not math.isclose(sum(c["weight"] for c in CAPABILITIES.values()), 1.0,
                        abs_tol=1e-9):
        raise ValueError("les poids des capacités SFT v4.5 ne somment pas à 1")
    used: set[str] = set()
    for name, cfg in CAPABILITIES.items():
        if not math.isclose(sum(cfg["sources"].values()), 1.0, abs_tol=1e-9):
            raise ValueError(f"les poids internes de {name} ne somment pas à 1")
        overlap = used & set(cfg["sources"])
        if overlap:
            raise ValueError(f"sources partagées entre capacités : {sorted(overlap)}")
        used.update(cfg["sources"])
    if used - set(SOURCE_RULES):
        raise ValueError(f"sources sans règles : {sorted(used - set(SOURCE_RULES))}")


def _ensure_raw(data_dir: Path, seed: int, skip_download: bool) -> dict:
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {}
    D._write_curated_identity(raw_dir / "identity.jsonl")
    required = {
        "chat_human": 100_000_000, "croissant": 35_000_000,
        "oasst_fr": 20_000_000, "openhermes_fr": 220_000_000,
        "gsm8k": 12_000_000, "maths_sft": 45_000_000,
    }
    for index, (source, budget) in enumerate(required.items()):
        path = raw_dir / f"{source}.jsonl"
        if path.exists() and path.stat().st_size > 0:
            report[source] = {"reused": True, "path": str(path), "bytes": path.stat().st_size}
        elif skip_download:
            raise FileNotFoundError(f"--skip-download mais source absente : {path}")
        else:
            report[source] = D.download_source(source, budget, path,
                                                seed=seed + index * 31)

    qa_mid = raw_dir / "frenchqa_mid_v43.jsonl"
    if not qa_mid.exists() and not skip_download:
        from frlm.mid_v43 import _download_qa_sources
        report["qa_download"] = _download_qa_sources(
            raw_dir, qa_chars=40_000_000, seed=seed + 77, skip_download=False
        )
    report["qa_conversion"] = V44._convert_qa_mid(raw_dir)

    synth_specs = {
        "general_v45": 32_000_000,
        "grounded_v45": 100_000_000,
        "reasoning_v45": 60_000_000,
        "constraints_v45": 50_000_000,
        "multiturn_v45": 38_000_000,
        "code_v45": 32_000_000,
        "uncertainty_v45": 28_000_000,
        "style_v45": 18_000_000,
    }
    for index, (kind, budget) in enumerate(synth_specs.items()):
        path = raw_dir / f"{kind}.jsonl"
        if path.exists() and path.stat().st_size > 0:
            report[kind] = {"reused": True, "path": str(path), "bytes": path.stat().st_size}
        else:
            report[kind] = synth_programs.write_jsonl(
                path, target_chars=budget, seed=seed + 1009 + index * 997, kind=kind
            )
    return report


def _copy_complete_documents(token_path: Path, mask_path: Path, writer: D.BinWriter,
                             target_supervised: int, rotation: int = 0) -> tuple[int, int, int]:
    """Copie des conversations entières, une seule fois, jusqu'au quota."""
    if target_supervised <= 0 or token_path.stat().st_size == 0:
        return 0, 0, 0
    tokens = np.memmap(token_path, dtype=D.DTYPE, mode="r")
    masks = np.memmap(mask_path, dtype=np.uint8, mode="r")
    ends = np.flatnonzero(np.asarray(tokens) == 0).astype(np.int64) + 1
    starts = np.concatenate((np.array([0], dtype=np.int64), ends[:-1]))
    if not len(starts):
        return 0, 0, 0
    offset = rotation % len(starts)
    order = np.concatenate((np.arange(offset, len(starts)), np.arange(0, offset)))
    copied_tokens = copied_supervised = conversations = 0
    for index in order:
        start, end = int(starts[index]), int(ends[index])
        mask = np.asarray(masks[start:end])
        supervised = int(mask.sum())
        if supervised <= 0:
            continue
        if copied_supervised >= target_supervised:
            break
        writer.write(np.asarray(tokens[start:end]), mask)
        copied_tokens += end - start
        copied_supervised += supervised
        conversations += 1
    del tokens, masks
    return copied_tokens, copied_supervised, conversations


def encode(tok, data_dir: Path, target_supervised: int,
           max_len: int = 512) -> dict:
    raw_dir = data_dir / "raw"
    sources = list(dict.fromkeys(
        source for capability in CAPABILITIES.values() for source in capability["sources"]
    ))
    paths = {source: raw_dir / f"{source}.jsonl" for source in sources}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("sources SFT v4.5 absentes : " + ", ".join(missing))

    seen_prompts: set[bytes] = set()
    source_stats: dict[str, dict] = {}
    capabilities_report: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix=".sft-v45-", dir=data_dir) as tmp_name:
        tmp = Path(tmp_name)
        for index, source in enumerate(sources, 1):
            print(f"[i] SFT v4.5 source {index}/{len(sources)} : {source}", flush=True)
            train_w = D.BinWriter(tmp / f"{source}_train.bin", with_mask=True)
            val_w = D.BinWriter(tmp / f"{source}_val.bin", with_mask=True)
            stats = V44._encode_source(
                tok, paths[source], source, train_w, val_w, seen_prompts,
                max_len=max_len, source_rules=SOURCE_RULES,
            )
            train_w.close()
            val_w.close()
            stats["train_tokens_unique"] = (tmp / f"{source}_train.bin").stat().st_size // 2
            stats["val_tokens_unique"] = (tmp / f"{source}_val.bin").stat().st_size // 2
            source_stats[source] = stats
            print(f"    {stats['kept']} conversations, "
                  f"{stats['train_supervised']:,} tokens assistant train", flush=True)

        aggregate_train = D.BinWriter(tmp / "sft_v45_train.bin", with_mask=True)
        aggregate_val = D.BinWriter(tmp / "sft_v45_val.bin", with_mask=True)
        publications: list[tuple[Path, Path]] = []
        for capability, cfg in CAPABILITIES.items():
            target = round(target_supervised * cfg["weight"])
            allocations, _ = V44._allocate_capability(
                capability, source_stats, target, target_supervised,
                capabilities=CAPABILITIES, source_rules=SOURCE_RULES,
                global_caps=GLOBAL_SOURCE_CAPS,
            )
            train_path = tmp / f"sft_v45_train_{capability}.bin"
            val_path = tmp / f"sft_v45_val_{capability}.bin"
            train_w = D.BinWriter(train_path, with_mask=True)
            val_w = D.BinWriter(val_path, with_mask=True)
            source_report = {}
            for source in cfg["sources"]:
                rotation = int.from_bytes(hashlib.sha256(
                    f"v45:{source}".encode()).digest()[:8], "little")
                train_tok, train_sup, train_docs = _copy_complete_documents(
                    tmp / f"{source}_train.bin", tmp / f"{source}_train.mask",
                    train_w, allocations[source], rotation,
                )
                val_target = min(VAL_SUPERVISED_PER_SOURCE,
                                 source_stats[source]["val_supervised"])
                val_tok, val_sup, val_docs = _copy_complete_documents(
                    tmp / f"{source}_val.bin", tmp / f"{source}_val.mask",
                    val_w, val_target,
                )
                source_report[source] = {
                    "available_supervised": source_stats[source]["train_supervised"],
                    "allocated_supervised": train_sup, "train_tokens": train_tok,
                    "train_conversations": train_docs, "val_supervised": val_sup,
                    "val_tokens": val_tok, "val_conversations": val_docs,
                }
            train_tokens, val_tokens = train_w.n, val_w.n
            train_w.close()
            val_w.close()
            V44._append_shard(train_path, train_path.with_suffix(".mask"), aggregate_train)
            V44._append_shard(val_path, val_path.with_suffix(".mask"), aggregate_val)
            actual = sum(item["allocated_supervised"] for item in source_report.values())
            docs = sum(item["train_conversations"] for item in source_report.values())
            capabilities_report[capability] = {
                "weight": cfg["weight"], "target_supervised": target,
                "actual_supervised": actual,
                "shortfall_supervised": max(0, target - actual),
                "train_path": train_path.name, "train_tokens": train_tokens,
                "train_conversations": docs,
                "val_path": val_path.name, "val_tokens": val_tokens,
                "sources": source_report,
            }
            for split in ("train", "val"):
                name = f"sft_v45_{split}_{capability}"
                publications.extend(((tmp / f"{name}.bin", data_dir / f"{name}.bin"),
                                     (tmp / f"{name}.mask", data_dir / f"{name}.mask")))

        train_tokens, val_tokens = aggregate_train.n, aggregate_val.n
        aggregate_train.close()
        aggregate_val.close()
        publications.extend(((tmp / "sft_v45_train.bin", data_dir / "sft_v45_train.bin"),
                             (tmp / "sft_v45_train.mask", data_dir / "sft_v45_train.mask"),
                             (tmp / "sft_v45_val.bin", data_dir / "sft_v45_val.bin"),
                             (tmp / "sft_v45_val.mask", data_dir / "sft_v45_val.mask")))
        for source_path, destination in publications:
            os.replace(source_path, destination)

    actual = sum(item["actual_supervised"] for item in capabilities_report.values())
    conversations = sum(item["train_conversations"] for item in capabilities_report.values())
    return {
        "recipe": RECIPE_NAME,
        "target_supervised_tokens": int(target_supervised),
        "actual_supervised_tokens": actual,
        "shortfall_supervised_tokens": max(0, int(target_supervised) - actual),
        "train_path": "sft_v45_train.bin", "train_tokens": train_tokens,
        "train_conversations": conversations,
        "assistant_tokens_per_conversation": round(actual / max(1, conversations), 2),
        "val_path": "sft_v45_val.bin", "val_tokens": val_tokens,
        "max_conversation_tokens": max_len,
        "sequence_isolation_required": True,
        "loss_normalization": "global_assistant_tokens_per_update",
        "capabilities": capabilities_report, "sources": source_stats,
        "single_source_cap": 0.20, "openhermes_cap": 0.12,
        "max_repeat": 1.0, "max_identical_answer": 100,
        "direct_answer_fraction_target": 0.80,
    }


def prepare(data_dir: Path, target_supervised: int = DEFAULT_TARGET_SUPERVISED,
            max_seq_len: int = 512, seed: int = 1337,
            skip_download: bool = False) -> dict:
    _validate_recipe()
    data_dir = Path(data_dir)
    tokenizer_path = data_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"tokenizer v4 introuvable : {tokenizer_path}")
    raw_report = _ensure_raw(data_dir, seed, skip_download)
    report = encode(D.load_tokenizer(tokenizer_path), data_dir,
                    int(target_supervised), max_len=max_seq_len)
    report["raw_build"] = raw_report
    meta_path = data_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta["sft_v45"] = report
    tmp = meta_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, meta_path)
    return report
