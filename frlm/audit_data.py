"""Audit léger des sources conversationnelles avant de reconstruire le SFT.

Usage : python -m frlm.audit_data --data-dir data-v4

Le script n'importe ni torch ni tokenizers : il peut tourner sur une machine CPU
minimale et ne modifie aucun fichier de données.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


DEFAULT_SOURCES = ("identity", "distill", "gsm8k", "maths_sft",
                   "openhermes_fr", "oasst_fr", "croissant", "chat_human")
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_WS_RE = re.compile(r"\s+")


def _percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    values.sort()
    return values[min(len(values) - 1, round((len(values) - 1) * q))]


def _text(message: dict) -> str:
    return str(message.get("text") or message.get("content") or "").strip()


def _fingerprint(text: str) -> bytes:
    normalized = _WS_RE.sub(" ", text).strip().casefold()
    return hashlib.blake2b(normalized.encode("utf-8"), digest_size=12).digest()


def audit_source(path: Path) -> tuple[dict, set[bytes]]:
    prompts: set[bytes] = set()
    user_lengths: list[int] = []
    answer_lengths: list[int] = []
    think_lengths: list[int] = []
    final_lengths: list[int] = []
    kinds: Counter[str] = Counter()
    stats = Counter()

    with path.open(encoding="utf-8") as stream:
        for line in stream:
            stats["lines"] += 1
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                stats["invalid_json"] += 1
                continue
            messages = record.get("m")
            if not isinstance(messages, list):
                stats["missing_messages"] += 1
                continue
            users = [_text(m) for m in messages if m.get("role") == "user" and _text(m)]
            answers = [_text(m) for m in messages if m.get("role") == "assistant" and _text(m)]
            if not users or not answers:
                stats["missing_turn"] += 1
                continue

            stats["valid"] += 1
            stats["multi_turn"] += int(len(users) > 1 or len(answers) > 1)
            prompt = "\n".join(users)
            answer = "\n".join(answers)
            fp = _fingerprint(prompt)
            stats["duplicate_prompt"] += int(fp in prompts)
            prompts.add(fp)
            user_lengths.append(len(prompt))
            answer_lengths.append(len(answer))

            blocks = _THINK_RE.findall(answer)
            stats["with_think"] += int(bool(blocks))
            stats["bad_think_tags"] += int(answer.lower().count("<think>") != len(blocks)
                                           or answer.lower().count("</think>") != len(blocks))
            think = "\n".join(blocks).strip()
            final = _THINK_RE.sub("", answer).strip()
            think_lengths.append(len(think))
            final_lengths.append(len(final))
            stats["empty_think"] += int(bool(blocks) and not think)
            stats["empty_final"] += int(not final)
            stats["answer_gt_4000"] += int(len(answer) > 4000)
            stats["think_gt_1600"] += int(len(think) > 1600)
            kinds[str(record.get("k") or "-")] += 1

    def lengths(values: list[int]) -> dict:
        return {
            "median": _percentile(values, 0.50),
            "p90": _percentile(values, 0.90),
            "p99": _percentile(values, 0.99),
            "max": max(values, default=0),
        }

    report = dict(stats)
    report.update({
        "file_mib": round(path.stat().st_size / 2**20, 2),
        "user_chars": lengths(user_lengths),
        "answer_chars": lengths(answer_lengths),
        "think_chars": lengths(think_lengths),
        "final_chars": lengths(final_lengths),
        "kinds": dict(kinds),
    })
    return report, prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit des JSONL conversationnels frlm")
    parser.add_argument("--data-dir", default="data-v4")
    parser.add_argument("--sources", nargs="+", default=list(DEFAULT_SOURCES))
    args = parser.parse_args()

    raw_dir = Path(args.data_dir) / "raw"
    all_prompts: set[bytes] = set()
    output = {}
    for source in args.sources:
        path = raw_dir / f"{source}.jsonl"
        if not path.exists():
            output[source] = {"missing": True}
            continue
        report, prompts = audit_source(path)
        report["duplicate_prompt_cross_source"] = len(all_prompts & prompts)
        all_prompts.update(prompts)
        output[source] = report
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
