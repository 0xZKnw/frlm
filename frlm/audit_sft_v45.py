"""Audit CPU bloquant des bins SFT v4.5 avant allocation d'un GPU."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from frlm import data as D
from frlm.sft_v45 import RECIPE_NAME


def _audit_shard(path: Path, expected_tokens: int, seq_len: int) -> dict:
    mask_path = path.with_suffix(".mask")
    if not path.is_file() or not mask_path.is_file():
        raise ValueError(f"shard ou masque absent : {path}")
    tokens = np.memmap(path, dtype=D.DTYPE, mode="r")
    masks = np.memmap(mask_path, dtype=np.uint8, mode="r")
    if len(tokens) != expected_tokens or len(masks) != expected_tokens:
        raise ValueError(f"taille incohérente pour {path}: {len(tokens)}/{len(masks)}")
    if np.any((masks != 0) & (masks != 1)):
        raise ValueError(f"masque non binaire : {mask_path}")
    if np.any(masks[np.asarray(tokens) == 0]):
        raise ValueError(f"token EOT supervisé : {path}")
    ends = np.flatnonzero(np.asarray(tokens) == 0).astype(np.int64) + 1
    starts = np.concatenate((np.array([0], dtype=np.int64), ends[:-1]))
    if not len(ends) or int(ends[-1]) != len(tokens):
        raise ValueError(f"dernier document incomplet : {path}")
    lengths = ends - starts
    if int(lengths.max()) > seq_len + 1:
        raise ValueError(f"conversation > {seq_len + 1} tokens : {path}")
    supervised_per_doc = np.add.reduceat(np.asarray(masks, dtype=np.int64), starts)
    if np.any(supervised_per_doc <= 0):
        raise ValueError(f"conversation sans cible assistant : {path}")
    result = {
        "tokens": len(tokens), "conversations": len(starts),
        "supervised": int(masks.sum()), "max_document_tokens": int(lengths.max()),
        "assistant_mean": round(float(supervised_per_doc.mean()), 3),
        "assistant_p95": round(float(np.percentile(supervised_per_doc, 95)), 3),
    }
    del tokens, masks
    return result


def audit(data_dir: Path, seq_len: int = 512, batch_size: int = 128,
          grad_accum: int = 12, replay_frac: float = 0.12,
          epochs: float = 1.15, samples_per_capability: int = 0) -> dict:
    data_dir = Path(data_dir)
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    section = meta.get("sft_v45") or {}
    if section.get("recipe") != RECIPE_NAME:
        raise ValueError(f"meta SFT v4.5 absent ou ancien : {section.get('recipe')}")
    if int(section.get("max_conversation_tokens", 0)) != seq_len:
        raise ValueError("--seq-len ne correspond pas au build v4.5")

    capabilities = {}
    actual_total = 0
    total_conversations = 0
    for name, capability in section["capabilities"].items():
        report = _audit_shard(
            data_dir / capability["train_path"], int(capability["train_tokens"]), seq_len
        )
        if report["supervised"] != int(capability["actual_supervised"]):
            raise ValueError(f"compte assistant incohérent pour {name}")
        capabilities[name] = report
        actual_total += report["supervised"]
        total_conversations += report["conversations"]

    expected_assistant_per_conversation = actual_total / max(1, total_conversations)

    replay_micros = max(1, round(grad_accum * replay_frac)) if replay_frac else 0
    sft_micros = grad_accum - replay_micros
    assistant_per_update = expected_assistant_per_conversation * batch_size * sft_micros
    recommended_steps = math.ceil(epochs * actual_total / max(1, assistant_per_update))

    # Smoke du vrai sampler : une fois le premier EOT rencontré, tout doit rester EOT.
    corpora = [
        (name, D.ConversationCorpus(data_dir / capability["train_path"], seq_len),
         float(capability["train_conversations"]))
        for name, capability in section["capabilities"].items()
    ]
    mixture = D.SourceMixtureCorpus(corpora)
    x, _, m = mixture.get_batch(17, min(batch_size, 64), seed=451337, device="cpu")
    for row in x.numpy():
        eot = np.flatnonzero(row == 0)
        if len(eot) and np.any(row[eot[0]:] != 0):
            raise ValueError("le sampler mélange deux conversations dans une ligne")
    if int(m.sum()) <= 0:
        raise ValueError("smoke batch sans token assistant")

    samples: dict[str, list[str]] = {}
    if samples_per_capability > 0:
        tok = D.load_tokenizer(data_dir / "tokenizer.json")
        for name, corpus, _ in corpora:
            count = len(corpus.conversation_starts)
            indices = np.linspace(0, count - 1,
                                  min(samples_per_capability, count), dtype=np.int64)
            decoded = []
            for index in indices:
                start = int(corpus.conversation_starts[index])
                end = int(corpus.conversation_ends[index])
                decoded.append(tok.decode(np.asarray(corpus.tokens[start:end]).tolist(),
                                          skip_special_tokens=False))
            samples[name] = decoded

    return {
        "ok": True, "recipe": RECIPE_NAME, "actual_supervised_tokens": actual_total,
        "expected_assistant_tokens_per_conversation": round(
            expected_assistant_per_conversation, 2
        ),
        "batch_size": batch_size, "grad_accum": grad_accum,
        "sft_microbatches": sft_micros, "replay_microbatches": replay_micros,
        "expected_assistant_tokens_per_update": round(assistant_per_update),
        "target_supervised_epochs": epochs,
        "recommended_max_steps": recommended_steps,
        "capabilities": capabilities,
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data-v4")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--grad-accum", type=int, default=12)
    parser.add_argument("--replay-frac", type=float, default=0.12)
    parser.add_argument("--epochs", type=float, default=1.15)
    parser.add_argument("--samples", type=int, default=0,
                        help="conversations décodées par capacité pour l'audit manuel")
    args = parser.parse_args()
    print(json.dumps(audit(Path(args.data_dir), args.seq_len, args.batch_size,
                           args.grad_accum, args.replay_frac, args.epochs,
                           args.samples),
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
