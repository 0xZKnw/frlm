"""Préparation du mid-training v4.3 à budget fixe.

La recette est volontairement séparée de :mod:`frlm.data` : elle réutilise le
tokenizer et les corpus v4 sans rebâtir le pré-entraînement, puis ajoute une
source de QA ancrée et filtrée. Les problèmes OOD des benchmarks ne sont jamais lus.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
import time
from dataclasses import dataclass
from pathlib import Path

from frlm import data as D


RECIPE_NAME = "v4.3-curriculum-1.5b"
DEFAULT_TARGET_TOKENS = 1_500_000_000
STAGE1_FRACTION = 0.80
VAL_FRACTION = 0.005

# 80 % de fondations puis 20 % d'upsampling raisonnement. Le second palier reste
# à 30 % de français naturel/QA/instructions afin de ne pas recuire le modèle sur
# un monocorpus synthétique. Les quotas de chaque étape somment à 1.
STAGE_MIXES: tuple[tuple[str, dict[str, float]], ...] = (
    ("stage1", {
        "fineweb": 0.2575,
        "wiki": 0.2575,
        "books": 0.055,
        "theses": 0.055,
        "oral": 0.020,
        "europarl": 0.020,
        "maths_verified": 0.27,
        "frenchqa": 0.015,
        "instruct": 0.05,
    }),
    ("stage2", {
        "fineweb": 0.0675,
        "wiki": 0.0675,
        "books": 0.03,
        "theses": 0.03,
        "oral": 0.02,
        "europarl": 0.02,
        "maths_verified": 0.70,
        "frenchqa": 0.015,
        "instruct": 0.05,
    }),
)

# Le replay est borné par source. Les gros corpus naturels ne dépassent pas deux
# passages dans le mid, les petits corpus oraux trois, et les données éducatives
# au plus quatre/cinq. Le pré-entraînement antérieur n'est pas compté ici.
MAX_REPEATS = {
    "fineweb": 2,
    "wiki": 2,
    "books": 2,
    "theses": 2,
    "oral": 4,
    "europarl": 4,
    "maths_verified": 4,
    "frenchqa": 5,
    "instruct": 5,
}

RAW_FILES = {
    "fineweb": "fineweb.jsonl",
    "wiki": "wiki.jsonl",
    "books": "books.jsonl",
    "theses": "theses.jsonl",
    "oral": "oral.jsonl",
    "europarl": "europarl.jsonl",
    "maths_verified": "maths_mid.jsonl",
    "frenchqa": "frenchqa_mid_v43.jsonl",
    "instruct": "mid_instruct.jsonl",
}

SOURCE_PROVENANCE = {
    "frenchqa": {
        "repos": ["LsTam/CQuAE (train_v2)", "CATIE-AQ/frenchQA"],
        "license": "CQuAE: CC-BY-NC-4.0; FrenchQA: licences ouvertes par sous-jeu",
        "fields": "contexte/documents + question + réponse; CQuAE corrigé humain en priorité",
    },
}

_WS_RE = re.compile(r"\s+")
_BAD_SPECIAL_RE = re.compile(r"<\|(?:im_start|im_end|endoftext)\|>|</?think>", re.I)
_REPEAT_RE = re.compile(r"(.{30,}?)\1\1", re.S)


@dataclass
class PoolStats:
    train_tokens: int
    val_tokens: int
    docs: int
    rejected: int
    target_train_tokens: int


def _stable_int(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")


def _clean(text: str) -> str:
    return _WS_RE.sub(" ", (text or "").replace("\x00", " ")).strip()


def _quality_pair(question: str, answer: str, min_answer_chars: int = 60) -> tuple[str, str] | None:
    question, answer = _clean(question), _clean(answer)
    if not (35 <= len(question) <= 7000
            and min_answer_chars <= len(answer) <= 5000):
        return None
    if _BAD_SPECIAL_RE.search(question) or _BAD_SPECIAL_RE.search(answer):
        return None
    if not D._looks_french(question) or not D._looks_french(answer):
        return None
    if _REPEAT_RE.search(answer) or answer.count("http") > 1:
        return None
    # Les refus, disclaimers et sorties manifestement tronquées valent peu pour
    # un petit modèle ; on préfère une réponse pédagogique directe.
    low = answer.casefold()
    if any(x in low for x in ("en tant qu'intelligence artificielle",
                              "en tant que modèle de langage",
                              "je ne peux pas répondre")):
        return None
    return question, answer


def _answer_text(value) -> str:
    if isinstance(value, dict):
        value = value.get("text") or value.get("answers") or []
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else ""
    return str(value or "")


def _download_qa_sources(raw_dir: Path, qa_chars: int,
                         seed: int, skip_download: bool) -> dict:
    """Télécharge uniquement les colonnes utiles et écrit le JSONL QA filtré."""
    from datasets import load_dataset
    from tqdm import tqdm

    report: dict[str, dict] = {}

    def write_source(name: str, char_budget: int, iterator, min_answer_chars: int = 60):
        path = raw_dir / RAW_FILES[name]
        if skip_download and path.exists() and path.stat().st_size > 0:
            report[name] = {"path": str(path), "reused": True,
                            "bytes": path.stat().st_size}
            return
        tmp = path.with_suffix(".tmp")
        seen: set[bytes] = set()
        chars = docs = rejected = 0
        started = time.time()
        with tmp.open("w", encoding="utf-8") as stream, tqdm(
            total=char_budget, unit="c", unit_scale=True, desc=f"  {name:12s}", ncols=90
        ) as bar:
            for question, answer in iterator:
                pair = _quality_pair(question, answer, min_answer_chars=min_answer_chars)
                if pair is None:
                    rejected += 1
                    continue
                question, answer = pair
                fingerprint = hashlib.blake2b(question.casefold().encode("utf-8"),
                                               digest_size=16).digest()
                if fingerprint in seen:
                    rejected += 1
                    continue
                seen.add(fingerprint)
                text = f"Question : {question}\nRéponse : {answer}"
                stream.write(json.dumps({"t": text}, ensure_ascii=False) + "\n")
                chars += len(text)
                docs += 1
                bar.update(len(text))
                if chars >= char_budget:
                    break
        os.replace(tmp, path)
        report[name] = {"path": str(path), "docs": docs, "chars": chars,
                        "rejected": rejected, "seconds": round(time.time() - started, 1)}

    def qa_iter():
        # Le petit CQuAE corrigé humain passe en premier afin d'être pris en entier.
        # Sa licence NC convient au projet étudiant privé, mais elle reste inscrite
        # dans meta.json pour empêcher une publication commerciale accidentelle.
        cquae = load_dataset("LsTam/CQuAE", split="train_v2", streaming=True)
        for row in cquae:
            documents = row.get("documents") or []
            if isinstance(documents, str):
                context = documents
            else:
                context = "\n\n".join(str(doc) for doc in documents if doc)
            question = row.get("query") or row.get("question") or ""
            answer = row.get("output") or row.get("answer") or ""
            context = _clean(context)
            if context and len(context) <= 6500:
                yield f"Contexte : {context}\n\n{_clean(question)}", _answer_text(answer)

        ds = load_dataset("CATIE-AQ/frenchQA", split="train", streaming=True)
        ds = ds.shuffle(seed=seed + 71, buffer_size=20_000)
        for row in ds:
            # Les deux blocs pragnakalp sont des traductions automatiques de
            # SQuAD et dominent 79 % du jeu. On garde seulement les collections
            # françaises natives/curées.
            title = row.get("title") or ""
            if title not in {"fquad_v2", "piaf", "piaf_v2",
                             "lincoln/newsquadfr", "lincoln/newsquadfr_v2"}:
                continue
            context = _clean(row.get("context") or "")
            question = _clean(row.get("question") or "")
            answer = _answer_text(row.get("answers") or row.get("answer"))
            if not context or len(context) > 6500:
                continue
            yield f"Contexte : {context}\n\n{question}", answer

    raw_dir.mkdir(parents=True, exist_ok=True)
    # FrenchQA est extractif : une réponse correcte peut légitimement être un nom,
    # une date ou quelques mots. Le contexte reste long et entièrement supervisé.
    write_source("frenchqa", qa_chars, qa_iter(), min_answer_chars=2)
    return report


def _ensure_base_sources(raw_dir: Path, unique_targets: dict[str, int],
                         seed: int, skip_download: bool) -> dict:
    """Rend le build autonome quand le Volume n'a que les anciens bins."""
    report: dict[str, dict] = {}
    chars_per_token = {
        "fineweb": 4.15,
        "wiki": 3.55,
        "books": 3.50,
        "theses": 3.60,
        "oral": 3.55,
        "europarl": 3.70,
        "maths_verified": 2.35,
        "instruct": 3.90,
    }

    for source in ("fineweb", "wiki", "books", "theses", "oral", "europarl"):
        path = raw_dir / RAW_FILES[source]
        if path.exists() and path.stat().st_size > 0:
            report[source] = {"reused": True, "bytes": path.stat().st_size}
            continue
        if skip_download:
            raise FileNotFoundError(f"--skip-download mais source absente : {path}")
        budget = math.ceil(unique_targets[source] * chars_per_token[source])
        report[source] = D.download_source(source, budget, path, seed=seed + 211)

    maths_path = raw_dir / RAW_FILES["maths_verified"]
    if maths_path.exists() and maths_path.stat().st_size > 0:
        report["maths_verified"] = {"reused": True, "bytes": maths_path.stat().st_size}
    elif skip_download:
        raise FileNotFoundError(f"--skip-download mais source absente : {maths_path}")
    else:
        from frlm import synth
        budget = math.ceil(unique_targets["maths_verified"]
                           * chars_per_token["maths_verified"])
        report["maths_verified"] = synth.write_jsonl(
            maths_path, budget, seed=seed + 431, mode="pretrain"
        )

    instruct_path = raw_dir / RAW_FILES["instruct"]
    if instruct_path.exists() and instruct_path.stat().st_size > 0:
        report["instruct"] = {"reused": True, "bytes": instruct_path.stat().st_size}
    elif skip_download:
        raise FileNotFoundError(f"--skip-download mais source absente : {instruct_path}")
    else:
        # Reconstruction minimale : conversations humaines + OpenHermes filtré,
        # GSM8K vérifié et petit corpus Croissant. Aucun ancien SFT n'est requis.
        instruction_inputs = {
            "chat_human": 70_000_000,
            "openhermes_fr": 105_000_000,
            "gsm8k": 12_000_000,
            "croissant": 6_000_000,
        }
        inputs_report = {}
        for source, budget in instruction_inputs.items():
            path = raw_dir / f"{source}.jsonl"
            if path.exists() and path.stat().st_size > 0:
                inputs_report[source] = {"reused": True, "bytes": path.stat().st_size}
            else:
                inputs_report[source] = D.download_source(source, budget, path,
                                                          seed=seed + 509)
        target_chars = math.ceil(unique_targets["instruct"]
                                 * chars_per_token["instruct"])
        stats = D.build_mid_instruct(raw_dir, instruct_path, target_chars)
        stats["inputs"] = inputs_report
        report["instruct"] = stats
    return report


def _allocate(total: int, mix: dict[str, float]) -> dict[str, int]:
    if not math.isclose(sum(mix.values()), 1.0, abs_tol=1e-9):
        raise ValueError(f"mix invalide : somme={sum(mix.values())}")
    exact = {name: total * weight for name, weight in mix.items()}
    out = {name: int(value) for name, value in exact.items()}
    missing = total - sum(out.values())
    for name in sorted(mix, key=lambda n: exact[n] - out[n], reverse=True)[:missing]:
        out[name] += 1
    return out


def _encode_pool(tok, source: str, path: Path, pool_dir: Path,
                 target_train_tokens: int) -> PoolStats:
    """Tokenise une tranche unique et réserve un holdout stable par document."""
    from tqdm import tqdm

    eot = tok.token_to_id(D.EOT)
    train_path = pool_dir / f"{source}_train.bin"
    val_path = pool_dir / f"{source}_val.bin"
    # Une préparation interrompue conserve ses pools atomiques sur le Volume.
    # Les réutiliser rend les ajustements de quotas quasi immédiats sans accepter
    # un pool trop court ni une validation vide.
    if train_path.exists() and val_path.exists():
        train_tokens = train_path.stat().st_size // 2
        val_tokens = val_path.stat().st_size // 2
        if train_tokens >= target_train_tokens and val_tokens > 0:
            print(f"  pool {source:12s} : réutilisé ({train_tokens:,} train, "
                  f"{val_tokens:,} val)")
            return PoolStats(train_tokens, val_tokens, 0, 0, target_train_tokens)
    train_tmp, val_tmp = train_path.with_suffix(".tmp"), val_path.with_suffix(".tmp")
    train_w, val_w = D.BinWriter(train_tmp), D.BinWriter(val_tmp)
    docs = rejected = 0
    buffer: list[tuple[str, bool]] = []

    def flush():
        nonlocal buffer, docs
        if not buffer:
            return
        texts = [text for text, _ in buffer]
        for (text, is_val), enc in zip(buffer, tok.encode_batch(texts)):
            ids = enc.ids + [eot]
            (val_w if is_val else train_w).write(ids)
            docs += 1
        buffer = []

    for rec in tqdm(D.iter_jsonl(path), desc=f"  pool {source:12s}", unit="doc", ncols=90):
        text = (rec.get("t") or "").strip()
        if len(text) < 40:
            rejected += 1
            continue
        is_val = _stable_int(f"{source}\0{text}") % round(1 / VAL_FRACTION) == 0
        buffer.append((text, is_val))
        if len(buffer) >= 512:
            flush()
        if train_w.n >= target_train_tokens:
            break
    flush()
    train_w.close(); val_w.close()
    os.replace(train_tmp, train_path)
    os.replace(val_tmp, val_path)
    return PoolStats(train_w.n, val_w.n, docs, rejected, target_train_tokens)


def _copy_tokens(source_path: Path, out, source_tokens: int, start: int,
                 count: int, eot: int) -> int:
    """Copie exactement ``count`` tokens, avec bouclage déterministe."""
    if source_tokens <= 0:
        raise ValueError(f"pool vide : {source_path}")
    remaining, pos = count, start % source_tokens
    with source_path.open("rb") as stream:
        while remaining:
            take = min(remaining, source_tokens - pos)
            stream.seek(pos * 2)
            left = take * 2
            while left:
                chunk = stream.read(min(left, 16 * 1024 * 1024))
                if not chunk:
                    raise IOError(f"lecture tronquée : {source_path}")
                out.write(chunk)
                left -= len(chunk)
            remaining -= take
            pos = (pos + take) % source_tokens
    # La tranche peut finir au milieu d'un document ; fermer proprement avant la
    # source suivante évite de fusionner deux textes sans séparateur.
    out.seek(-2, os.SEEK_CUR)
    out.write(struct.pack("<H", eot))
    return pos


def _materialize_stages(tok, data_dir: Path, pool_stats: dict[str, PoolStats],
                        stage_tokens: tuple[int, int]) -> tuple[list[dict], dict]:
    pool_dir = data_dir / "mid-v43-pools"
    offsets = {name: 0 for name in pool_stats}
    consumed = {name: 0 for name in pool_stats}
    stage_reports: list[dict] = []
    eot = tok.token_to_id(D.EOT)

    for (stage_name, mix), total in zip(STAGE_MIXES, stage_tokens):
        quotas = _allocate(total, mix)
        path = data_dir / f"mid_v43_{stage_name}_train.bin"
        tmp = path.with_suffix(".tmp")
        with tmp.open("w+b") as out:
            for source, quota in quotas.items():
                stats = pool_stats[source]
                consumed[source] += quota
                repeats = consumed[source] / max(1, stats.train_tokens)
                if repeats > MAX_REPEATS[source] + 1e-9:
                    raise RuntimeError(
                        f"{source}: {consumed[source]:,} tokens demandés pour "
                        f"{stats.train_tokens:,} uniques ({repeats:.2f} passages > "
                        f"{MAX_REPEATS[source]}). Télécharge/génère davantage de données."
                    )
                offsets[source] = _copy_tokens(
                    pool_dir / f"{source}_train.bin", out, stats.train_tokens,
                    offsets[source], quota, eot,
                )
        os.replace(tmp, path)
        stage_reports.append({"name": stage_name, "train_tokens": total,
                              "end_fraction": (STAGE1_FRACTION if stage_name == "stage1" else 1.0),
                              "sources": quotas, "path": path.name})

    val_path = data_dir / "mid_v43_val.bin"
    val_tmp = val_path.with_suffix(".tmp")
    val_sources: dict[str, int] = {}
    with val_tmp.open("w+b") as out:
        for source, stats in pool_stats.items():
            # 1 M/source au maximum : assez pour des évaluations stables sans
            # gonfler le Volume ni réutiliser des documents de train.
            count = min(stats.val_tokens, 1_000_000)
            if count:
                _copy_tokens(pool_dir / f"{source}_val.bin", out, stats.val_tokens,
                             0, count, eot)
            val_sources[source] = count
    os.replace(val_tmp, val_path)
    return stage_reports, {"path": val_path.name, "val_tokens": sum(val_sources.values()),
                           "sources": val_sources}


def prepare(data_dir: Path, target_tokens: int = DEFAULT_TARGET_TOKENS,
            seed: int = 1337, skip_download: bool = False) -> dict:
    """Construit les deux bins curriculum en conservant le tokenizer v4."""
    data_dir = Path(data_dir)
    if target_tokens <= 0 or target_tokens > DEFAULT_TARGET_TOKENS:
        raise ValueError(f"budget mid invalide : {target_tokens:,} (maximum 1 500 000 000)")
    tok_path = data_dir / "tokenizer.json"
    if not tok_path.exists():
        raise FileNotFoundError(f"tokenizer v4 introuvable : {tok_path}")
    tok = D.load_tokenizer(tok_path)
    raw_dir = data_dir / "raw"
    pool_dir = data_dir / "mid-v43-pools"
    pool_dir.mkdir(parents=True, exist_ok=True)

    stage1_tokens = int(target_tokens * STAGE1_FRACTION)
    stage_tokens = (stage1_tokens, target_tokens - stage1_tokens)
    per_stage = [_allocate(total, mix) for total, (_, mix) in zip(stage_tokens, STAGE_MIXES)]
    aggregate = {name: sum(stage[name] for stage in per_stage if name in stage)
                 for name in RAW_FILES}

    # On vise juste assez de tokens uniques pour respecter le plafond de répétition,
    # avec 3 % de marge pour le holdout et les fins de documents.
    unique_targets = {
        name: math.ceil(quota / MAX_REPEATS[name] * 1.03)
        for name, quota in aggregate.items()
    }
    # Estimation prudente pour le téléchargement ; le tokenizer tranche ensuite
    # sur le nombre réel de tokens et l'étape échoue plutôt que de sur-répéter.
    base_sources = _ensure_base_sources(raw_dir, unique_targets, seed, skip_download)
    downloads = _download_qa_sources(
        raw_dir,
        qa_chars=math.ceil(unique_targets["frenchqa"] * 3.7),
        seed=seed,
        skip_download=skip_download,
    )

    missing = [raw_dir / filename for filename in RAW_FILES.values()
               if not (raw_dir / filename).exists()]
    if missing:
        raise FileNotFoundError("sources v4 absentes : " + ", ".join(map(str, missing)))

    pools: dict[str, PoolStats] = {}
    for source, filename in RAW_FILES.items():
        pools[source] = _encode_pool(tok, source, raw_dir / filename, pool_dir,
                                     unique_targets[source])

    stages, validation = _materialize_stages(tok, data_dir, pools, stage_tokens)
    section = {
        "recipe": RECIPE_NAME,
        "target_tokens": target_tokens,
        "stage1_fraction": STAGE1_FRACTION,
        "stages": stages,
        "validation": validation,
        "pools": {name: vars(stats) for name, stats in pools.items()},
        "max_repeats": MAX_REPEATS,
        "downloads": downloads,
        "base_sources": base_sources,
        "provenance": SOURCE_PROVENANCE,
        "ood_excluded": True,
    }
    meta_path = data_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta["midtrain_v43"] = section
    tmp = meta_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, meta_path)
    return section
