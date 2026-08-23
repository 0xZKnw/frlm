"""Pipeline RLAIF v4.5 hors-ligne : candidats aveugles puis paires DPO filtrées."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from pathlib import Path

import torch

from frlm import data as D
from frlm.rl_engine_v45 import RolloutEngine, load_policy, resolve_checkpoint, resolve_tokenizer
from frlm.rl_tasks_v45 import TaskSpec
from frlm.verifiers_v45 import AnswerSpec, final_text


def _id(prefix: str, *parts: str) -> str:
    raw = "\0".join(parts).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(raw).hexdigest()[:18]}"


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def _repeat_ratio(text: str, n: int = 4) -> float:
    words = re.findall(r"\w+", text.casefold())
    grams = [tuple(words[i:i + n]) for i in range(max(0, len(words) - n + 1))]
    return 0.0 if not grams else 1.0 - len(set(grams)) / len(grams)


def _read_prompts(paths: list[Path], limit: int, seed: int) -> list[dict]:
    rows, seen = [], set()
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                category = str(raw.get("type") or "general").strip().casefold()
                # Le pool historique mélange calculs, pièges et faits avec OOD v2.
                # La préférence v4.5 est réservée au chat ouvert ; RLVR couvre les
                # capacités vérifiables sur ses propres familles non-OOD.
                if category != "chat":
                    continue
                prompt = str(raw.get("prompt") or raw.get("q") or "").strip()
                folded = _normalize(prompt)
                if len(prompt) < 8 or folded in seen:
                    continue
                seen.add(folded)
                rows.append({"prompt": prompt, "source": path.name, "line": line_no,
                             "category": category})
    random.Random(seed).shuffle(rows)
    return rows[:limit]


def build_candidates(run: str, data_dir: str, out_dir: str, init_stage: str,
                     init_ckpt: str, pools: list[Path], prompts: int, candidates: int,
                     max_new: int, seed: int, device: str) -> dict:
    run_dir = Path(out_dir) / run
    stage_dir = run_dir / "rlaif-v45"
    stage_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = resolve_checkpoint(run_dir, init_stage, init_ckpt)
    tokenizer = D.load_tokenizer(resolve_tokenizer(run_dir, Path(data_dir)))
    model, _cfg, metadata = load_policy(checkpoint, device=device)
    model.eval()
    engine = RolloutEngine(model, tokenizer, device, max_new)
    prompt_rows = _read_prompts(pools, prompts, seed)
    if len(prompt_rows) < prompts:
        raise ValueError(f"seulement {len(prompt_rows)} prompts uniques disponibles")
    private_path = stage_dir / "candidates.private.jsonl"
    judge_path = stage_dir / "judge_packet.jsonl"
    private_tmp = private_path.with_suffix(".jsonl.tmp")
    judge_tmp = judge_path.with_suffix(".jsonl.tmp")
    rng = random.Random(seed + 1)
    digest = hashlib.sha256()
    with private_tmp.open("w", encoding="utf-8") as private, \
            judge_tmp.open("w", encoding="utf-8") as judge:
        for index, source in enumerate(prompt_rows):
            prompt_id = _id("p", source["source"], str(source["line"]), source["prompt"])
            task = TaskSpec(
                task_id=prompt_id, schema_id="rlaif_open", surface_id="private",
                split="dev", capability=source["category"], difficulty=0.5,
                prompt=source["prompt"], answer=AnswerSpec("entity", "__judge_only__"),
                latent_program={}, canonical_trace=(), verifier_version="judge_v45_1",
                seed=seed + index, requires_trace=False,
            )
            torch.manual_seed(seed + index * 10_007)
            _prompt_ids, samples = engine.sample(task, candidates)
            candidate_rows = []
            for candidate_index, sample in enumerate(samples):
                text = final_text(sample.text)
                candidate_id = _id("c", prompt_id, str(candidate_index), text)
                candidate_rows.append({
                    "candidate_id": candidate_id, "text": text,
                    "stopped": sample.stopped, "tokens": len(sample.token_ids),
                    "repeat_ratio": round(_repeat_ratio(text), 4),
                })
            private_row = {"prompt_id": prompt_id, "prompt": source["prompt"],
                           "category": source["category"], "source": source["source"],
                           "candidates": candidate_rows}
            line = json.dumps(private_row, ensure_ascii=False, sort_keys=True)
            private.write(line + "\n")
            digest.update(line.encode("utf-8"))
            blind = [{"candidate_id": row["candidate_id"], "text": row["text"]}
                     for row in candidate_rows]
            rng.shuffle(blind)
            # Aucune réponse canonique, note Python ou provenance privée n'entre
            # dans le paquet du juge : seulement le prompt et les sorties anonymes.
            judge.write(json.dumps({"prompt_id": prompt_id, "prompt": source["prompt"],
                                    "candidates": blind}, ensure_ascii=False) + "\n")
            print(f"candidats {index + 1:3d}/{prompts} · {prompt_id}")
    private_tmp.replace(private_path)
    judge_tmp.replace(judge_path)
    instructions = stage_dir / "JUDGE_INSTRUCTIONS.md"
    instructions.write_text(
        "# Jugement RLAIF v4.5 (aveugle)\n\n"
        "Pour chaque ligne de `judge_packet.jsonl`, classer toutes les sorties selon : "
        "exactitude et ancrage factuel, respect précis de la demande, clarté/concision, "
        "absence d'invention, sécurité et absence de radotage. Ne pas favoriser une sortie "
        "pour sa longueur ou son identifiant.\n\n"
        "Écrire `scores.jsonl`, une ligne JSON par prompt :\n\n"
        "```json\n{\"prompt_id\":\"p_...\",\"ranking\":[\"c_meilleur\",\"c_...\"],"
        "\"scores\":{\"c_meilleur\":4,\"c_...\":1},\"unsafe\":[],\"reason\":\"...\"}\n```\n\n"
        "Les scores sont des entiers 0..4. Une égalité doit recevoir le même score. "
        "Marquer dans `unsafe` toute sortie dangereuse ou manipulatrice.\n",
        encoding="utf-8",
    )
    manifest = {"schema": "frlm-rlaif-candidates-v45-1", "created_unix": time.time(),
                "checkpoint": metadata, "prompts": prompts, "candidates_per_prompt": candidates,
                "seed": seed, "max_new": max_new, "sha256": digest.hexdigest(),
                "private": str(private_path), "judge_packet": str(judge_path)}
    (stage_dir / "candidates.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def import_scores(run: str, out_dir: str, scores_path: Path, min_margin: int = 1,
                  max_pairs_per_prompt: int = 2) -> dict:
    stage_dir = Path(out_dir) / run / "rlaif-v45"
    private_path = stage_dir / "candidates.private.jsonl"
    candidates = {}
    with private_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            candidates[row["prompt_id"]] = row
    scores = {}
    with Path(scores_path).open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            prompt_id = row.get("prompt_id")
            if prompt_id in scores:
                raise ValueError(f"score dupliqué pour {prompt_id}")
            if prompt_id not in candidates:
                raise ValueError(f"prompt inconnu ligne {line_no}: {prompt_id}")
            scores[prompt_id] = row
    pairs, rejected = [], []
    for prompt_id, source in candidates.items():
        judged = scores.get(prompt_id)
        if judged is None:
            rejected.append({"prompt_id": prompt_id, "reason": "missing_judgment"})
            continue
        by_id = {row["candidate_id"]: row for row in source["candidates"]}
        ranking = judged.get("ranking") or []
        if set(ranking) != set(by_id) or len(ranking) != len(by_id):
            raise ValueError(f"classement incomplet ou falsifié pour {prompt_id}")
        score_map = judged.get("scores") or {}
        if score_map and (set(score_map) != set(by_id)
                          or any(type(value) is not int or not 0 <= value <= 4
                                 for value in score_map.values())):
            raise ValueError(f"scores invalides pour {prompt_id}")
        unsafe = set(judged.get("unsafe") or [])
        chosen = by_id[ranking[0]]
        if chosen["candidate_id"] in unsafe or not chosen["stopped"] or not chosen["text"]:
            rejected.append({"prompt_id": prompt_id, "reason": "chosen_invalid"})
            continue
        added = 0
        for rejected_id in reversed(ranking[1:]):
            loser = by_id[rejected_id]
            margin = (score_map.get(chosen["candidate_id"], 4)
                      - score_map.get(rejected_id, 0))
            if margin < min_margin or chosen["text"] == loser["text"]:
                continue
            if chosen["repeat_ratio"] > 0.22 or len(chosen["text"]) > 4000:
                continue
            pairs.append({"pair_id": _id("pair", prompt_id, chosen["candidate_id"], rejected_id),
                          "prompt_id": prompt_id, "prompt": source["prompt"],
                          "category": source["category"], "chosen": chosen["text"],
                          "rejected": loser["text"], "margin": margin,
                          "judge_reason": str(judged.get("reason") or "")[:500]})
            added += 1
            if added >= max_pairs_per_prompt:
                break
        if not added:
            rejected.append({"prompt_id": prompt_id, "reason": "no_clear_margin"})
    if not pairs:
        raise ValueError("aucune paire DPO sûre après filtrage")
    pairs_path = stage_dir / "pairs.sealed.jsonl"
    tmp = pairs_path.with_suffix(".jsonl.tmp")
    digest = hashlib.sha256()
    with tmp.open("w", encoding="utf-8") as stream:
        for pair in pairs:
            line = json.dumps(pair, ensure_ascii=False, sort_keys=True)
            stream.write(line + "\n")
            digest.update(line.encode("utf-8"))
    tmp.replace(pairs_path)
    report = {"schema": "frlm-rlaif-pairs-v45-1", "created_unix": time.time(),
              "prompts_total": len(candidates), "judgments": len(scores),
              "pairs": len(pairs), "rejected_prompts": rejected,
              "sha256": digest.hexdigest(), "path": str(pairs_path)}
    (stage_dir / "pairs.manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def cmd_build(args):
    pools = [Path(part.strip()) for part in args.pool.split(",") if part.strip()]
    print(json.dumps(build_candidates(
        args.run, args.data_dir, args.out_dir, args.init_stage, args.init_ckpt,
        pools, args.prompts, args.candidates, args.max_new, args.seed, args.device,
    ), ensure_ascii=False, indent=2))


def cmd_import(args):
    print(json.dumps(import_scores(args.run, args.out_dir, Path(args.scores),
                                   args.min_margin, args.max_pairs),
                     ensure_ascii=False, indent=2))
