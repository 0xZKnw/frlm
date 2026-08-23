"""Profilage pass@k de la frontière RLVR v4.5, sans jeu OOD scellé."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch

from frlm import data as D
from frlm.rl_engine_v45 import RolloutEngine, load_policy, resolve_checkpoint, resolve_tokenizer
from frlm.rl_tasks_v45 import CAPABILITY_WEIGHTS, make_task
from frlm.verifiers_v45 import verify


def _summary(rows: list[dict], k: int, frontier_k: int) -> dict:
    by_capability: dict[str, list[dict]] = defaultdict(list)
    by_schema: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_capability[row["capability"]].append(row)
        by_schema[row["schema_id"]].append(row)

    def aggregate(items: list[dict]) -> dict:
        n = len(items)
        summary = {
            "tasks": n,
            "pass@1": sum(item["first_success"] for item in items) / max(1, n),
            f"pass@{k}": sum(item["initial_successes"] >= 1 for item in items) / max(1, n),
            "success_rate": sum(item["initial_successes"] for item in items) / max(1, n * k),
            "dynamic_rate": sum(0 < item["initial_successes"] < k for item in items) / max(1, n),
            "mean_entropy": sum(item["entropy"] for item in items) / max(1, n),
        }
        if frontier_k > k:
            summary[f"pass@{frontier_k}"] = \
                sum(item["successes"] >= 1 for item in items) / max(1, n)
            summary["frontier_success_rate"] = \
                sum(item["successes"] / item["k"] for item in items) / max(1, n)
            summary["frontier_dynamic_rate"] = \
                sum(0 < item["successes"] < item["k"] for item in items) / max(1, n)
        return summary

    return {
        "overall": aggregate(rows),
        "capabilities": {key: aggregate(value) for key, value in sorted(by_capability.items())},
        "schemas": {key: aggregate(value) for key, value in sorted(by_schema.items())},
    }


def profile(run: str, data_dir: str, out_dir: str, init_stage: str, init_ckpt: str,
            tasks: int, k: int, frontier_k: int, max_new: int, seed: int,
            device: str, output: str = "profile.json", refine_from: str = "",
            output_stage: str = "rlvr-v45") -> dict:
    if k < 2 or frontier_k < k:
        raise ValueError("il faut 2 <= k <= frontier-k")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA demandé mais indisponible")
    run_dir = Path(out_dir) / run
    stage = run_dir / output_stage
    stage.mkdir(parents=True, exist_ok=True)
    previous = None
    if refine_from:
        previous_path = Path(refine_from)
        if not previous_path.is_file():
            previous_path = stage / Path(refine_from).name
        if not previous_path.is_file():
            raise FileNotFoundError(f"profil à raffiner introuvable : {refine_from}")
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        expected = previous.get("config", {})
        for key, value in (("tasks", tasks), ("k", k), ("frontier_k", frontier_k),
                           ("max_new", max_new), ("seed", seed)):
            if int(expected.get(key, -1)) != int(value):
                raise ValueError(f"--refine-from incompatible sur {key}: "
                                 f"{expected.get(key)} != {value}")
        if len(previous.get("rows", [])) != tasks:
            raise ValueError("nombre de lignes incohérent dans le profil à raffiner")
    checkpoint = resolve_checkpoint(run_dir, init_stage, init_ckpt)
    tokenizer = D.load_tokenizer(resolve_tokenizer(run_dir, Path(data_dir)))
    model, _cfg, checkpoint_meta = load_policy(checkpoint, device=device)
    if previous is not None:
        old_checkpoint = previous.get("checkpoint", {})
        for key in ("stage", "step"):
            if old_checkpoint.get(key) != checkpoint_meta.get(key):
                raise ValueError(f"--refine-from provient d'un autre checkpoint ({key})")
    model.eval()
    engine = RolloutEngine(model, tokenizer, device, max_new)
    capabilities = tuple(CAPABILITY_WEIGHTS)
    rows = []
    started = time.time()
    for index in range(tasks):
        capability = capabilities[index % len(capabilities)]
        difficulty = ((index // len(capabilities)) % 9 + 1) / 10
        task = make_task(seed + index * 17, "dev", difficulty, capability)
        old = previous["rows"][index] if previous else None
        if old is not None and old.get("task_id") != task.task_id:
            raise ValueError(f"tâche {index} différente dans le profil à raffiner")
        if old is None:
            torch.manual_seed(seed + index * 20_003 + 1)
            _prompt, samples = engine.sample(task, k)
            results = [verify(task.answer, sample.text) for sample in samples]
            initial_successes = sum(result.primary_success for result in results)
            successes = initial_successes
            used_k = k
            entropy_sum = sum(sample.entropy for sample in samples)
            failure_codes = [result.failure_code for result in results
                             if not result.primary_success]
            first_success = bool(results[0].primary_success)
        else:
            initial_successes = int(old["initial_successes"])
            successes = int(old["successes"])
            used_k = int(old["k"])
            entropy_sum = float(old["entropy"]) * used_k
            failure_codes = list(old.get("failure_codes", []))
            first_success = bool(old["first_success"])

        # Toute tâche non maîtrisée est réellement mesurée jusqu'à frontier_k.
        # En mode raffinement, seuls les rollouts manquants sont générés.
        if successes < used_k and used_k < frontier_k:
            extra_count = frontier_k - used_k
            torch.manual_seed(seed + index * 20_003 + 2)
            _prompt, extra = engine.sample(task, extra_count)
            extra_results = [verify(task.answer, sample.text) for sample in extra]
            successes += sum(result.primary_success for result in extra_results)
            entropy_sum += sum(sample.entropy for sample in extra)
            failure_codes.extend(result.failure_code for result in extra_results
                                 if not result.primary_success)
            used_k = frontier_k
        row = {
            "task_id": task.task_id, "schema_id": task.schema_id,
            "capability": task.capability, "difficulty": task.difficulty,
            "k": used_k, "initial_successes": initial_successes,
            "successes": successes,
            "first_success": first_success,
            "entropy": entropy_sum / used_k,
            "failure_codes": failure_codes,
        }
        rows.append(row)
        print(f"profile {index + 1:3d}/{tasks} · {capability:18s} · {successes}/{used_k}")
    report = {
        "schema": "frlm-rl-profile-v45-2", "created_unix": time.time(),
        "elapsed_s": time.time() - started, "checkpoint": checkpoint_meta,
        "config": {"tasks": tasks, "k": k, "frontier_k": frontier_k,
                   "max_new": max_new, "seed": seed, "split": "dev",
                   "refined_from": refine_from or None},
        "summary": _summary(rows, k, frontier_k), "rows": rows,
    }
    path = stage / Path(output).name
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    print(f"\nProfil sauvegardé : {path}")
    return report


def cmd_profile(args):
    profile(args.run, args.data_dir, args.out_dir, args.init_stage, args.init_ckpt,
            args.tasks, args.k, args.frontier_k, args.max_new, args.seed, args.device,
            args.output, args.refine_from)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="fr-v4-v45-sft")
    parser.add_argument("--data-dir", default="data-v4")
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--init-stage", default="sft")
    parser.add_argument("--init-ckpt", default="best")
    parser.add_argument("--tasks", type=int, default=60)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--frontier-k", type=int, default=32)
    parser.add_argument("--max-new", type=int, default=112)
    parser.add_argument("--seed", type=int, default=455_001)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="profile.json")
    parser.add_argument("--refine-from", default="")
    cmd_profile(parser.parse_args())


if __name__ == "__main__":
    main()
