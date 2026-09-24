"""SFT v5 : conversations entières, masques assistant et quotas sans répétition."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3

from frlm import data as D
from frlm.prepare_v5 import records, sha256, token_targets, write_json


def conversations(source, root, used):
    kind = source["kind"]
    if kind == "verified":
        from frlm.reason_bootstrap_v45 import make_example, OBJECTIVE_WEIGHTS
        for i in range(100_000):
            # Les programmes train ne traversent jamais les holdouts AST existants.
            row = make_example(source["seed"] + 97 * i, "train",
                               list(OBJECTIVE_WEIGHTS)[i % len(OBJECTIVE_WEIGHTS)])
            yield row["messages"], row["program_key"]
        return
    for row in records(source, root / "raw", used):
        group = None
        if kind == "nemotron":
            messages = json.loads(row["metadata"])["messages"]
        elif kind == "scholar":
            # Garder les solutions courtes ; pas de sélection sur un benchmark.
            messages = row["messages"]
        elif kind == "pleias":
            if row["language"] != "fr":
                continue
            messages = [{"role": "user", "content": row["query"]},
                        {"role": "assistant", "content": row["synthetic_answer"]}]
            group = row.get("query_seed_url") or row["synth_id"]
        elif kind == "openhermes":
            if any(row.get(k) for k in ("bad_entry", "bad_prompt_detected", "bad_response_detected")):
                continue
            messages = [{"role": "user", "content": row["prompt"]},
                        {"role": "assistant", "content": row["accepted_completion"]}]
        elif kind == "human":
            if row.get("author") != "human" or row.get("style") != "human" or row.get("code"):
                continue
            messages = row["conversation"]
            if row.get("context") and messages and messages[0]["role"] == "user":
                messages[0]["text"] = row["context"] + "\n\n" + messages[0]["text"]
        else:
            raise ValueError(f"source SFT inconnue : {kind}")
        yield messages, group


def encode_conversation(tok, messages, max_len, name):
    # Les blocs de raisonnement externes sont retirés : certaines traces restent
    # anglaises. Les raisonnements courts calculés localement sont conservés.
    rules = {name: dict(max_prompt=6000, max_final=6000,
                       max_think=3000 if name == "verified" else 0)}
    clean, fingerprint, note = D._prepare_sft_messages({"m": messages}, name, rules)
    if clean is None:
        return None, note
    if any(s in m["text"] for m in clean for s in (D.EOT, D.IM_START, D.IM_END)):
        return None, "controle_chat_injecte"
    roles = [m["role"] for m in clean if m["role"] != "system"]
    if not roles or roles[-1] != "assistant" or any(r != ("user" if i % 2 == 0 else "assistant") for i, r in enumerate(roles)):
        return None, "ordre_des_roles"
    if name != "verified" and not all(D._looks_french(m["text"]) for m in clean if m["role"] == "assistant"):
        return None, "reponse_non_francaise"
    ids, mask = [], []
    for text, learn in D.chat_segments(clean, ensure_think=False):
        enc = tok.encode(text).ids
        ids.extend(enc)
        mask.extend([int(learn)] * len(enc))
    ids.append(tok.token_to_id(D.EOT))
    mask.append(0)
    if len(ids) > max_len + 1 or not any(mask):
        return None, "trop_long_ou_vide"
    return (ids, mask, fingerprint), note


def prepare(root: Path, recipe: dict):
    path = root / "sft_plan.json"
    if path.exists() and json.loads(path.read_text()) != recipe:
        raise ValueError("recette SFT différente : utiliser un nouveau dossier")
    write_json(path, recipe)
    tok = D.load_tokenizer(root / "tokenizer.json")
    tok_hash = sha256(root / "tokenizer.json")
    db = sqlite3.connect(root / "sft_dedup.sqlite")
    db.execute("CREATE TABLE IF NOT EXISTS prompts (fingerprint BLOB PRIMARY KEY, source TEXT)")
    reports = {}
    try:
        for src, target in zip(recipe["sources"], token_targets(recipe["target_supervised"], [s["weight"] for s in recipe["sources"]])):
            name = src["name"]
            folder = root / "sft_shards" / name
            folder.mkdir(parents=True, exist_ok=True)
            marker = folder / "report.json"
            if marker.exists():
                report = json.loads(marker.read_text())
                if report["tokenizer_sha256"] != tok_hash:
                    raise ValueError("tokenizer SFT modifié")
                for file, digest in report["artifacts"].items():
                    if sha256(folder / file) != digest:
                        raise ValueError(f"sous-corpus SFT altéré : {file}")
                reports[name] = report
                continue
            db.execute("DELETE FROM prompts WHERE source=?", (name,))
            db.commit()
            writers = {s: D.BinWriter(folder / f"{s}.bin", with_mask=True) for s in ("train", "val", "sealed")}
            counts, supervised, rejected = Counter(), Counter(), Counter()
            used = []
            try:
                for messages, group in conversations(src, root, used):
                    encoded, note = encode_conversation(tok, messages, recipe["max_seq_len"], name)
                    if encoded is None:
                        rejected[note] += 1
                        continue
                    ids, mask, fingerprint = encoded
                    if not db.execute("INSERT OR IGNORE INTO prompts VALUES (?,?)", (fingerprint, name)).rowcount:
                        rejected["duplicate_prompt"] += 1
                        continue
                    # 1 % dev + 1 % scellé, par prompt / programme / article source.
                    bucket = int.from_bytes(hashlib.sha256(str(group or fingerprint.hex()).encode()).digest()[:8], "big") % 100
                    split = "val" if bucket == 0 else "sealed" if bucket == 1 else "train"
                    writers[split].write(ids, mask)
                    counts[split] += 1
                    supervised[split] += sum(mask)
                    if counts.total() % 2000 == 0:
                        print(f"SFT {name}: {supervised['train']:,}/{target:,} tokens assistant", flush=True)
                    if supervised["train"] >= target:
                        break
            finally:
                for writer in writers.values():
                    writer.close()
            if min(counts[s] for s in writers) == 0:
                raise ValueError(f"split SFT vide pour {name}")
            report = {"source": src, "target": target, "actual_supervised": supervised["train"],
                      "shortfall": max(0, target - supervised["train"]),
                      "train_conversations": counts["train"], "conversations": dict(counts),
                      "target_token_weight": src["weight"] / 100,
                      "train_tokens": writers["train"].n, "val_tokens": writers["val"].n,
                      "train_path": f"sft_shards/{name}/train.bin", "val_path": f"sft_shards/{name}/val.bin",
                      "tokenizer_sha256": tok_hash, "rejected": dict(rejected), "files": used,
                      "artifacts": {p.name: sha256(p) for p in folder.iterdir() if p.suffix in (".bin", ".mask")}}
            db.commit()
            write_json(marker, report)
            reports[name] = report
    finally:
        db.close()
    for split in ("train", "val", "sealed"):
        for suffix in ("bin", "mask"):
            final = root / f"sft_v5_{split}.{suffix}"
            with final.with_suffix(final.suffix + ".tmp").open("wb") as out:
                for name in reports:
                    with (root / f"sft_shards/{name}/{split}.{suffix}").open("rb") as inp:
                        shutil.copyfileobj(inp, out)
            final.with_suffix(final.suffix + ".tmp").replace(final)
    section = {"recipe": recipe["recipe"], "max_seq_len": recipe["max_seq_len"],
               "sampling_unit": "supervised_tokens", "capabilities": reports,
               "artifacts": {p.name: sha256(p) for p in root.glob("sft_v5_*.*")
                             if p.suffix in (".bin", ".mask")},
               "train_path": "sft_v5_train.bin", "val_path": "sft_v5_val.bin",
               "train_tokens": (root / "sft_v5_train.bin").stat().st_size // 2,
               "val_tokens": (root / "sft_v5_val.bin").stat().st_size // 2}
    write_json(root / "sft_manifest.json", section)
    audit(root)
    # Le manifest prétrain reste immuable pour permettre une reprise exacte.
    print("SFT prêt :", {n: r["actual_supervised"] for n, r in reports.items()}, flush=True)


def audit(root: Path):
    import numpy as np
    section = json.loads((root / "sft_manifest.json").read_text())
    tok_hash = sha256(root / "tokenizer.json")
    for split in ("train", "val", "sealed"):
        for suffix in ("bin", "mask"):
            name = f"sft_v5_{split}.{suffix}"
            if sha256(root / name) != section["artifacts"][name]:
                raise ValueError(f"artefact SFT fusionné altéré : {name}")
    for name, report in section["capabilities"].items():
        if report["tokenizer_sha256"] != tok_hash:
            raise ValueError("tokenizer SFT altéré")
        for filename, digest in report["artifacts"].items():
            path = root / "sft_shards" / name / filename
            if sha256(path) != digest:
                raise ValueError(f"artefact SFT altéré : {path}")
            if path.suffix == ".bin":
                mask = path.with_suffix(".mask")
                if path.stat().st_size != 2 * mask.stat().st_size:
                    raise ValueError("masque SFT mal aligné")
                corpus = D.ConversationCorpus(path, section["max_seq_len"])
                if corpus.dropped_too_long or len(corpus.conversation_starts) == 0:
                    raise ValueError("conversations SFT invalides")
                if int(np.memmap(mask, dtype=np.uint8, mode="r").max()) > 1:
                    raise ValueError("masque SFT non booléen")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data-v5"))
    p.add_argument("--recipe", type=Path, default=Path("recipes/v5_sft.json"))
    a = p.parse_args()
    prepare(a.data_dir, json.loads(a.recipe.read_text()))


if __name__ == "__main__":
    main()
