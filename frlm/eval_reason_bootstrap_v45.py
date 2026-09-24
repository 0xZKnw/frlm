"""Profil pass@k du bootstrap raisonnement v4.5 sur des holdouts AST.

Ce profil ne lit jamais OOD v2. Il sert uniquement à choisir le checkpoint de
départ du mini-SFT et à vérifier qu'une frontière d'apprentissage existe avant
de louer un GPU distant.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

from frlm import config_from_dict, data as D, model_from_cfg
from frlm.reason_bootstrap_v45 import FINAL_SPLIT
from frlm.verifiers_v45 import AnswerSpec, final_text, verify


BASE_FEWSHOT = (
    "Question : Calcule la somme de 8 et 5. Réponds uniquement par le nombre final.\n"
    "Réponse : 13\n\n"
)


def _resolve_checkpoint(run_dir: Path, stage: str, ckpt: str) -> Path:
    candidate = Path(ckpt)
    if candidate.is_file():
        return candidate
    name = ckpt if ckpt.endswith(".pt") else f"ckpt_{ckpt}.pt"
    path = run_dir / stage / name
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint introuvable : {path}")
    return path


def _load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def score_output(record: dict, output: str) -> bool:
    target = str(record["target"]).strip()
    if record["objective"] == "order_steps":
        normalized = re.sub(r"\s+", "", final_text(output).upper())
        return normalized in (record.get("valid_orders") or [target])
    strict = record["objective"] in ("number_only", "find_error")
    return verify(AnswerSpec("integer" if strict else "rational", int(target),
                             strict_number_only=strict), output).primary_success


def stratified_sample(records: list[dict], count: int, seed: int) -> list[dict]:
    """Chaque objectif puis difficulté reçoit un tour, sans préfixe aléatoire biaisé."""
    if count <= 0 or count > len(records):
        raise ValueError("nombre de tâches positif et <= taille du manifeste requis")
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for record in records:
        buckets[(record["objective"], record["difficulty"])].append(record)
    for values in buckets.values():
        rng.shuffle(values)
    keys = sorted(buckets)
    objectives = sorted({key[0] for key in keys})
    minimum = len(objectives) * max(sum(key[0] == objective for key in keys)
                                    for objective in objectives)
    if count < minimum:
        raise ValueError(f"au moins {minimum} tâches pour couvrir objectifs/difficultés à poids égal")
    # Répartir les positions d'erreur dans chaque strate, même pour un petit profil.
    for key in keys:
        if key[0] == "find_error":
            positions = defaultdict(list)
            for record in buckets[key]:
                positions[record["target"]].append(record)
            buckets[key] = [row for group in itertools.zip_longest(
                *(positions[pos] for pos in sorted(positions))) for row in group if row]
    selected = []
    turns = Counter()
    while len(selected) < count:
        for objective in objectives:
            available = [key for key in keys if key[0] == objective and buckets[key]]
            if available and len(selected) < count:
                key = available[turns[objective] % len(available)]
                selected.append(buckets[key].pop(0))
                turns[objective] += 1
    return selected


def baseline_report(records: list[dict], seed: int = 455_932, k: int = 4) -> dict:
    """Constantes fixes, oracle majoritaire descriptif et hasard sans accès à la cible."""
    groups = defaultdict(list)
    for row in records:
        for key in (f"objective:{row['objective']}",
                    f"objective:{row['objective']}/{row['difficulty']}"):
            groups[key].append(row)
    report = {}
    for key, rows in sorted(groups.items()):
        rng = random.Random(f"{seed}:{key}")
        objective = rows[0]["objective"]
        constants = (["1", "2", "3"] if objective == "find_error" else
                     ["A,B", "B,A", "A,B,C", "B,A,C", "C,B,A"] if objective == "order_steps"
                     else ["0", "1", "-1"])
        random_hits = 0
        expected = expected_k = 0.0
        for row in rows:
            n = row["operations"]
            if objective == "order_steps":
                labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:n])
                rng.shuffle(labels)
                guess = ",".join(labels)
                probability = len(row.get("valid_orders") or [row["target"]]) / math.factorial(n)
            elif objective == "find_error":
                guess, probability = str(rng.randint(1, n)), 1 / n
            else:
                guess, probability = str(rng.randint(-600, 600)), 1 / 1201
            random_hits += score_output(row, guess)
            expected += probability
            expected_k += 1 - (1 - probability) ** k
        # Calculé a posteriori : ce plafond de constante n'est PAS une baseline apprise.
        candidates = {row["target"] for row in rows}
        best = max(sorted(candidates), key=lambda value: sum(score_output(row, value) for row in rows))
        report[key] = {
            "tasks": len(rows), "target_counts": dict(Counter(row["target"] for row in rows)),
            "fixed_constant_hits": {value: sum(score_output(row, value) for row in rows)
                                    for value in constants},
            "majority_oracle": {"answer": best,
                                "hits": sum(score_output(row, best) for row in rows)},
            "random_hits": random_hits, "random_expected_rate": expected / len(rows),
            f"random_expected_pass@{k}": expected_k / len(rows),
        }
    return report


class Sampler:
    def __init__(self, checkpoint: Path, data_dir: Path, device: str, dtype: str,
                 protocol: str):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = config_from_dict(payload["model_cfg"])
        self.model = model_from_cfg(cfg)
        self.model.load_state_dict(payload["model"])
        torch_dtype = (torch.bfloat16 if dtype == "bf16" and device.startswith("cuda")
                       else torch.float32)
        self.model = self.model.to(device=device, dtype=torch_dtype).eval()
        self.tok = D.load_tokenizer(data_dir / "tokenizer.json")
        self.sp = D.special_ids(self.tok)
        self.device = device
        self.protocol = protocol
        self.description = {
            "checkpoint": str(checkpoint), "step": int(payload.get("step", -1)),
            "phase": payload.get("phase", checkpoint.parent.name),
            "protocol": protocol, "dtype": str(torch_dtype),
            "base_fewshot": BASE_FEWSHOT if protocol == "base" else None,
        }

    def _prompt(self, question: str) -> str:
        if self.protocol == "chat":
            return f"{D.IM_START}user\n{question}{D.IM_END}\n{D.IM_START}assistant\n"
        return BASE_FEWSHOT + f"Question : {question}\nRéponse :"

    @torch.inference_mode()
    def generate(self, question: str, seed: int, max_new: int, greedy: bool) -> str:
        prompt = self._prompt(question)
        ids = torch.tensor([self.tok.encode(prompt).ids], device=self.device)
        torch.manual_seed(seed)
        if self.device.startswith("cuda"):
            torch.cuda.manual_seed_all(seed)
        output = self.model.generate(
            ids, max_new_tokens=max_new,
            temperature=0.0 if greedy else 0.75,
            top_k=40, top_p=0.95, repetition_penalty=1.05,
            stop_ids=(self.sp["im_end"], self.sp["eot"]),
        )
        return self.tok.decode(output[0, ids.shape[1]:].tolist(),
                               skip_special_tokens=False)


def profile(args) -> dict:
    root = Path(args.data_dir)
    splits = tuple(part.strip() for part in args.splits.split(",") if part.strip())
    if not splits or len(splits) != len(set(splits)) or args.k < 1:
        raise ValueError("splits distincts non vides et k >= 1 requis")
    if FINAL_SPLIT in splits and (not args.final_eval or len(splits) != 1):
        raise ValueError("le test final exige --final-eval --splits final_sealed après gel du modèle")
    if FINAL_SPLIT in splits and getattr(args, "baselines_only", False):
        raise ValueError("les baselines de développement ne consultent pas le test final")
    destination = (Path(args.report) if args.report else Path("bench/reports") /
                   f"reason45c_{args.run}_{args.stage}_{Path(args.ckpt).stem}.json")
    if destination.exists():
        raise FileExistsError(f"rapport existant préservé : {destination}")
    section = json.loads((root / "meta.json").read_text(encoding="utf-8"))["reason_bootstrap_v45c"]
    rows = []
    for split in splits:
        path = root / section["eval_manifests"][split]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != section["manifest_sha256"][split]:
            raise ValueError(f"manifeste modifié : {split}")
        rows.extend(stratified_sample(_load_jsonl(path), args.tasks,
                                     args.seed ^ sum(map(ord, split))))
    if getattr(args, "baselines_only", False):
        report = {"schema": "frlm-reason-baselines-3", "recipe": section["recipe"],
                  "model_evaluated": False, "manifest_sha256": section["manifest_sha256"],
                  "settings": {"splits": splits, "tasks_per_split": args.tasks,
                               "seed": args.seed, "k": args.k},
                  "baselines": baseline_report(rows, args.seed, args.k)}
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print(f"[ok] Baselines CPU sauvegardées : {destination}")
        return report
    checkpoint = _resolve_checkpoint(Path(args.out_dir) / args.run,
                                     args.stage, args.ckpt)
    protocol = args.protocol
    if protocol == "auto":
        protocol = "chat" if args.stage not in ("pretrain", "mid") else "base"
    sampler = Sampler(checkpoint, root, args.device, args.dtype, protocol)

    started = time.perf_counter()
    counters = defaultdict(lambda: {"tasks": 0, "greedy": 0, "pass_k": 0,
                                    "samples": 0, "sample_success": 0})
    details = []
    for index, record in enumerate(rows, 1):
        greedy_text = sampler.generate(record["prompt"], args.seed + index,
                                       args.max_new, greedy=True)
        greedy_ok = score_output(record, greedy_text)
        samples = []
        for sample_index in range(args.k):
            text = sampler.generate(record["prompt"],
                                    args.seed + index * 10_003 + sample_index,
                                    args.max_new, greedy=False)
            samples.append({"ok": score_output(record, text), "text": text})
        pass_k = any(sample["ok"] for sample in samples)
        for key in (record["split"], f"objective:{record['objective']}",
                    f"objective:{record['objective']}/{record['difficulty']}", "micro"):
            bucket = counters[key]
            bucket["tasks"] += 1
            bucket["greedy"] += int(greedy_ok)
            bucket["pass_k"] += int(pass_k)
            bucket["samples"] += len(samples)
            bucket["sample_success"] += sum(int(sample["ok"]) for sample in samples)
        details.append({"id": record["id"], "split": record["split"],
                        "prompt": record["prompt"], "difficulty": record["difficulty"],
                        "objective": record["objective"], "target": record["target"],
                        "greedy_ok": greedy_ok, "greedy_text": greedy_text,
                        "pass_k": pass_k, "samples": samples})
        if index == 1 or index % 10 == 0 or index == len(rows):
            done = counters["micro"]
            print(f"profil {index}/{len(rows)} · greedy {done['greedy']}/{done['tasks']} "
                  f"· pass@{args.k} {done['pass_k']}/{done['tasks']}", flush=True)

    metrics = {}
    for key, value in sorted(counters.items()):
        tasks = max(1, value["tasks"])
        samples = max(1, value["samples"])
        metrics[key] = {**value, "greedy_rate": value["greedy"] / tasks,
                        f"pass@{args.k}": value["pass_k"] / tasks,
                        "sample_success_rate": value["sample_success"] / samples}
    report = {
        "schema": "frlm-reason-bootstrap-profile-3", "model": sampler.description,
        "recipe": section["recipe"], "manifest_sha256": section["manifest_sha256"],
        "baselines": baseline_report(rows, args.seed, args.k),
        "settings": {"splits": splits, "tasks_per_split": args.tasks,
                     "k": args.k, "max_new": args.max_new, "seed": args.seed,
                     "sampling": "stratified_objective_operations", "final_eval": args.final_eval,
                     "temperature": 0.75, "top_k": 40, "top_p": 0.95,
                     "repetition_penalty": 1.05},
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "metrics": metrics, "details": details,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] Profil sauvegardé : {destination}")
    for key in sorted(metrics):
        metric = metrics[key]
        print(f"  {key:18s} greedy {metric['greedy']}/{metric['tasks']} · "
              f"pass@{args.k} {metric['pass_k']}/{metric['tasks']} · "
              f"succès samples {metric['sample_success_rate']:.1%}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Profil pass@k AST v4.5 hors OOD")
    parser.add_argument("--run", required=True)
    parser.add_argument("--data-dir", default="data-v4")
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--stage", required=True)
    parser.add_argument("--ckpt", default="best")
    parser.add_argument("--protocol", choices=("auto", "base", "chat"), default="auto")
    parser.add_argument("--splits", default="iid,surface_holdout,structure_holdout")
    parser.add_argument("--final-eval", action="store_true")
    parser.add_argument("--baselines-only", action="store_true",
                        help="calcule les baselines CPU sans charger de checkpoint")
    parser.add_argument("--tasks", type=int, default=30)
    parser.add_argument("-k", type=int, default=4)
    parser.add_argument("--max-new", type=int, default=96)
    parser.add_argument("--seed", type=int, default=455_532)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--report")
    args = parser.parse_args()
    profile(args)


if __name__ == "__main__":
    main()
