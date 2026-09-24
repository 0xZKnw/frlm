"""Préparation locale v5, sans GPU : sources épinglées, BPE, bins et empreintes.

Usage : python -m frlm.prepare_v5 --data-dir data-v5 --target-tokens 4000000000
Les corpus v4 sont indépendants. Une interruption reprend au dernier sous-corpus
publié ; un sous-corpus incomplet est reconstruit, jamais ajouté une seconde fois.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import unicodedata

from frlm import data as D


RECIPE = "v5-fr-math-20260924"
# Pré-tokeniseur Qwen2 reconnu par llama.cpp ; chiffres individuels.
QWEN2_PATTERN = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
SPAM = re.compile(r"(?:write (?:my|a|your) (?:paper|essay)|buy (?:cheap )?(?:essays|viagra)|"
                  r"(?:masterpapers|paperhelp)\.com|veuillez activer javascript|"
                  r"votre compte a bien été créé)", re.I)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def split_for(group: str) -> str:
    bucket = int.from_bytes(hashlib.sha256(group.encode()).digest()[:8], "big") % 1000
    return "val" if bucket == 0 else "sealed" if bucket == 1 else "train"


def token_targets(total: int, weights: list[int]) -> list[int]:
    if total <= 0 or not weights or any(w <= 0 for w in weights):
        raise ValueError("budget et poids doivent être strictement positifs")
    denom = sum(weights)
    result = [total * w // denom for w in weights]
    for i in sorted(range(len(weights)), key=lambda i: (-(total * weights[i] % denom), i))[:total - sum(result)]:
        result[i] += 1
    return result


def clean_document(row: dict, source: dict) -> tuple[str, str] | None:
    text = row.get("text")
    if not isinstance(text, str):
        return None
    text = unicodedata.normalize("NFC", text).strip()
    # Refuser les contrôles de conversation dans le texte brut.
    if any(token in text for token in D.SPECIALS) or SPAM.search(text):
        return None
    if len(text) < 200 or len(text) > 4_000_000:
        return None
    if source["name"] == "books":
        text = D._clean_book(text)
    if text.count("\ufffd") > 2 or sum(c.isalpha() for c in text) < len(text) * 0.30:
        return None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 10 and len(set(lines)) < len(lines) * 0.65:
        return None
    metadata = row.get("metadata") or {}
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    # Même URL / identifiant source => même groupe, y compris les réécritures.
    group = metadata.get("url") or row.get("url") or row.get("id")
    if not group:
        group = hashlib.sha256(" ".join(text.casefold().split()).encode()).hexdigest()
    return text, str(group)


def records(source: dict, cache: Path, used: list):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    for filename in source["files"]:
        path = Path(hf_hub_download(source["repo"], filename, repo_type="dataset",
                                   revision=source["revision"], cache_dir=str(cache)))
        used.append({"path": filename, "bytes": path.stat().st_size, "sha256": sha256(path)})
        parquet = pq.ParquetFile(path)
        columns = [k for k in source.get("columns", ("text", "id", "url", "metadata")) if k in parquet.schema_arrow.names]
        if "kind" not in source and "text" not in columns:
            raise ValueError(f"colonne text absente : {filename}")
        for batch in parquet.iter_batches(batch_size=64, columns=columns):
            yield from batch.to_pylist()


def build_tokenizer(root: Path, sources: list[dict], sample_chars: int):
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

    path = root / "tokenizer.json"
    if path.exists():
        return D.load_tokenizer(path)
    texts = []
    sampling = {}
    for source, target in zip(sources, token_targets(sample_chars, [s["weight"] for s in sources])):
        used, chars = [], 0
        for row in records(source, root / "raw", used):
            doc = clean_document(row, source)
            if doc is None or split_for(doc[1]) != "train":
                continue
            texts.append(doc[0])
            chars += len(doc[0])
            if chars >= target:
                break
        sampling[source["name"]] = {"characters": chars, "files": used}
        if chars < target:
            raise ValueError(f"échantillon tokenizer insuffisant : {source['name']}")
        print(f"tokenizer : {source['name']} {chars:,} caractères", flush=True)
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(QWEN2_PATTERN), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(texts, trainer=trainers.BpeTrainer(
        vocab_size=32768, min_frequency=2, special_tokens=D.SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=True))
    tmp = path.with_suffix(".tmp")
    tok.save(str(tmp))
    os.replace(tmp, path)
    write_json(root / "tokenizer_sampling.json", sampling)
    return tok


def encode_source(root: Path, source: dict, target: int, tok, db):
    name = source["name"]
    folder = root / "shards" / name
    done = folder / "report.json"
    if done.exists():
        report = json.loads(done.read_text())
        if report["target"] != target:
            raise ValueError("changer le budget exige un autre dossier data-dir")
        if report["tokenizer_sha256"] != sha256(root / "tokenizer.json"):
            raise ValueError("tokenizer modifié depuis la binarisation")
        for filename, digest in report["artifacts"].items():
            if sha256(folder / filename) != digest:
                raise ValueError(f"sous-corpus altéré : {name}/{filename}")
        return report
    folder.mkdir(parents=True, exist_ok=True)
    # Après interruption, annuler seulement les clés du sous-corpus inachevé.
    db.execute("DELETE FROM seen WHERE source=?", (name,))
    db.commit()
    writers = {split: D.BinWriter(folder / f"{split}.bin") for split in ("train", "val", "sealed")}
    report = {"target": target, "read": 0, "rejected": 0, "duplicates": 0, "documents": 0,
              "files": [], "source": source, "tokenizer_sha256": sha256(root / "tokenizer.json")}
    batch = []
    eot = tok.token_to_id(D.EOT)

    def flush():
        for (text, group), enc in zip(batch, tok.encode_batch([text for text, _ in batch])):
            fingerprint = hashlib.sha256(" ".join(text.casefold().split()).encode()).digest()
            if not db.execute("INSERT OR IGNORE INTO seen VALUES (?,?,?)", (fingerprint, group, name)).rowcount:
                report["duplicates"] += 1
                continue
            split = split_for(group)
            # Quotas en tokens, dernier document entier (dépassement comptabilisé).
            if split == "train" and writers[split].n >= target:
                continue
            writers[split].write(enc.ids + [eot])
            report["documents"] += 1
        batch.clear()

    try:
        for row in records(source, root / "raw", report["files"]):
            report["read"] += 1
            doc = clean_document(row, source)
            if doc is None:
                report["rejected"] += 1
                continue
            batch.append(doc)
            if len(batch) >= 128 or sum(len(t) for t, _ in batch) >= 2_000_000:
                flush()
            if report["read"] % 10000 == 0:
                print(f"{name} : {writers['train'].n:,}/{target:,} tokens", flush=True)
            if writers["train"].n >= target:
                break
        flush()
    finally:
        for writer in writers.values():
            writer.close()
    if writers["train"].n < target:
        db.rollback()
        raise ValueError(f"{name}: seulement {writers['train'].n:,}/{target:,} tokens ; pas de répétition cachée")
    report["tokens"] = {split: w.n for split, w in writers.items()}
    report["artifacts"] = {f"{split}.bin": sha256(folder / f"{split}.bin") for split in writers}
    db.commit()
    write_json(done, report)
    return report


def prepare(root: Path, recipe: dict, total: int, sample_chars=100_000_000):
    root.mkdir(parents=True, exist_ok=True)
    plan = {"recipe": RECIPE, "sources": recipe["sources"], "tokens": total,
            "sample_chars": sample_chars, "pretokenizer": QWEN2_PATTERN}
    plan_path = root / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("dossier déjà associé à une autre préparation ; utiliser un nouveau data-dir")
    write_json(plan_path, plan)
    tok = build_tokenizer(root, recipe["sources"], sample_chars)
    if tok.get_vocab_size() > 65535:
        raise ValueError("le vocabulaire dépasse uint16")
    db = sqlite3.connect(root / "dedup.sqlite")
    db.execute("CREATE TABLE IF NOT EXISTS seen (fingerprint BLOB PRIMARY KEY, group_key TEXT UNIQUE, source TEXT)")
    try:
        reports = [encode_source(root, source, target, tok, db)
                   for source, target in zip(recipe["sources"], token_targets(total, [s["weight"] for s in recipe["sources"]]))]
    finally:
        db.close()
    artifacts = {}
    for split in ("train", "val", "sealed"):
        final = root / f"{split}.bin"
        with final.with_suffix(".tmp").open("wb") as out:
            for source in recipe["sources"]:
                with (root / "shards" / source["name"] / f"{split}.bin").open("rb") as inp:
                    shutil.copyfileobj(inp, out, length=8 << 20)
        os.replace(final.with_suffix(".tmp"), final)
        artifacts[final.name] = {"sha256": sha256(final), "bytes": final.stat().st_size}
    artifacts["tokenizer.json"] = {"sha256": sha256(root / "tokenizer.json"), "bytes": (root / "tokenizer.json").stat().st_size}
    manifest = {"recipe": RECIPE, "target_tokens": total, "sources": reports,
                "artifacts": artifacts, "split": "group-hash-998/1/1", "sealed_used_for_training": False}
    write_json(root / "manifest.json", manifest)
    write_json(root / "meta.json", {"recipe": RECIPE, "vocab_size": tok.get_vocab_size(),
               "manifest_sha256": sha256(root / "manifest.json"),
               "pretrain_tokens": artifacts["train.bin"]["bytes"] // 2})
    audit(root)


def audit(root: Path):
    import numpy as np
    manifest = json.loads((root / "manifest.json").read_text())
    meta = json.loads((root / "meta.json").read_text())
    if sha256(root / "manifest.json") != meta["manifest_sha256"]:
        raise ValueError("manifest modifié")
    tok = D.load_tokenizer(root / "tokenizer.json")
    if [tok.token_to_id(s) for s in D.SPECIALS] != list(range(len(D.SPECIALS))):
        raise ValueError("ordre des tokens spéciaux modifié")
    for number in ("1234567890", "-12,345", "1.25e-10"):
        if any(sum(c.isdigit() for c in tok.decode([i])) > 1 for i in tok.encode(number).ids):
            raise ValueError("chiffres fusionnés")
    for name, expected in manifest["artifacts"].items():
        path = root / name
        if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"artefact altéré : {name}")
        if name.endswith(".bin"):
            if path.stat().st_size % 2 or path.stat().st_size < 4098:
                raise ValueError(f"bin trop petit ou mal aligné : {name}")
            data = np.memmap(path, dtype=np.uint16, mode="r")
            if int(data.max()) >= tok.get_vocab_size():
                raise ValueError(f"token hors vocabulaire : {name}")
    print("Audit v5 réussi : empreintes, tailles, vocabulaire et chiffres.", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data-v5"))
    p.add_argument("--recipe", type=Path, default=Path("recipes/v5_data.json"))
    p.add_argument("--target-tokens", type=int, default=4_000_000_000)
    p.add_argument("--sample-chars", type=int, default=100_000_000)
    p.add_argument("--audit-only", action="store_true")
    args = p.parse_args()
    if args.audit_only:
        audit(args.data_dir)
    else:
        prepare(args.data_dir, json.loads(args.recipe.read_text()), args.target_tokens, args.sample_chars)


if __name__ == "__main__":
    main()
