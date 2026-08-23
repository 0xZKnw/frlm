"""Profil pass@k du bootstrap raisonnement v4.5 sur des holdouts AST.

Ce profil ne lit jamais OOD v2. Il sert uniquement à choisir le checkpoint de
départ du mini-SFT et à vérifier qu'une frontière d'apprentissage existe avant
de louer un GPU distant.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import torch

from frlm import config_from_dict, data as D, model_from_cfg


BASE_FEWSHOT = (
    "Question : Calcule la somme de 8 et 5. Réponds uniquement par le nombre final.\n"
    "Réponse : 13\n\n"
    "Question : Une trace contient une erreur. Étape 1 : r1 = 4 × 3 = 11. "
    "Donne uniquement le numéro de la première étape fausse.\nRéponse : 1\n\n"
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


def _final_zone(text: str) -> str:
    text = text.split("<|im_end|>", 1)[0].split("<|endoftext|>", 1)[0]
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    explicit = list(re.finditer(r"(?i)(?:réponse|conclusion)\s*:\s*", text))
    return text[explicit[-1].end():].strip() if explicit else text.strip()


def score_output(record: dict, output: str) -> bool:
    zone = _final_zone(output)
    target = str(record["target"]).strip()
    if record["objective"] == "order_steps":
        normalized = re.sub(r"[^A-Z,]", "", zone.upper())
        return normalized == target
    numbers = re.findall(r"(?<![\w])[-+]?\d+(?![\w])", zone.replace("−", "-"))
    return bool(numbers) and numbers[-1] == target and len(set(numbers)) == 1


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
    checkpoint = _resolve_checkpoint(Path(args.out_dir) / args.run,
                                     args.stage, args.ckpt)
    protocol = args.protocol
    if protocol == "auto":
        protocol = "chat" if args.stage not in ("pretrain", "mid") else "base"
    sampler = Sampler(checkpoint, root, args.device, args.dtype, protocol)
    splits = tuple(part.strip() for part in args.splits.split(",") if part.strip())
    rows: list[dict] = []
    for split in splits:
        path = root / "raw" / f"reason_bootstrap_v45_{split}.jsonl"
        candidates = _load_jsonl(path)
        rng = random.Random(args.seed ^ sum(map(ord, split)))
        rng.shuffle(candidates)
        rows.extend(candidates[:args.tasks])

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
        for key in (record["split"], f"objective:{record['objective']}", "macro"):
            bucket = counters[key]
            bucket["tasks"] += 1
            bucket["greedy"] += int(greedy_ok)
            bucket["pass_k"] += int(pass_k)
            bucket["samples"] += len(samples)
            bucket["sample_success"] += sum(int(sample["ok"]) for sample in samples)
        details.append({"id": record["id"], "split": record["split"],
                        "objective": record["objective"], "target": record["target"],
                        "greedy_ok": greedy_ok, "greedy_text": greedy_text,
                        "pass_k": pass_k, "samples": samples})
        if index == 1 or index % 10 == 0 or index == len(rows):
            done = counters["macro"]
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
        "schema": "frlm-reason-bootstrap-profile-1", "model": sampler.description,
        "settings": {"splits": splits, "tasks_per_split": args.tasks,
                     "k": args.k, "max_new": args.max_new, "seed": args.seed},
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "metrics": metrics, "details": details,
    }
    destination = (Path(args.report) if args.report else
                   Path("bench/reports") /
                   f"reason_bootstrap_{args.run}_{args.stage}_{args.ckpt}.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] Profil sauvegardé : {destination}")
    for key in (*splits, "macro"):
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
