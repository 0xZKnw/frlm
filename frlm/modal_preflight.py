"""Préflight local/Modal sur CPU, sans importer le SDK ni allouer de GPU."""

from __future__ import annotations

import json
import shlex
from pathlib import Path

def _command_parts(cmd: str) -> list[str]:
    parts = []
    for part in shlex.split(cmd):
        parts.extend(part.split("=", 1) if part.startswith("--") and "=" in part else [part])
    return parts


def with_gpu_peak(cmd: str, peak: float) -> str:
    parts = _command_parts(cmd)
    # Construire la commande localement ne doit pas sonder le disque distant.
    stage = next((parts[i + 1] for i, part in enumerate(parts[:-1])
                  if Path(part).name == "run.py"), None)
    if "--gpu-peak-tflops" not in parts and (
            stage in ("train", "mid", "sft") or "frlm.bench_speed" in parts):
        return cmd + f" --gpu-peak-tflops {peak}"
    return cmd


def _rl_checkpoint(parts: list[str], root: Path, stage: str, spec: str) -> Path:
    explicit = _workspace_path(spec, root)
    if explicit.is_file():
        return explicit
    run_dir = _workspace_path(_arg(parts, "--out-dir", "runs"), root) / _arg(parts, "--run", "fr-v4-v45-sft")
    names = {"best": ("ckpt_best.pt", "ckpt_latest.pt"),
             "latest": ("ckpt_latest.pt", "ckpt_best.pt")}.get(spec, (spec,))
    return next((run_dir / stage / name for name in names
                 if (run_dir / stage / name).is_file()), run_dir / stage / names[0])


def _rl_files(parts: list[str], stage: str, root: Path) -> list[Path]:
    run_dir = _workspace_path(_arg(parts, "--out-dir", "runs"), root) / _arg(parts, "--run", "fr-v4-v45-sft")
    data_dir = _workspace_path(_arg(parts, "--data-dir", "data-v4"), root)
    tokenizer = run_dir / "tokenizer.json"
    required = [tokenizer if tokenizer.is_file() else data_dir / "tokenizer.json"]
    resume = _arg(parts, "--resume", "latest") if "--resume" in parts else None
    resumed_anchor = stage == "rl-v45" and resume and "--keep-reference" not in parts
    if resume and stage == "rl-v45":
        required.append(_rl_checkpoint(parts, root, _arg(parts, "--stage-name", "rlvr-v45"), resume))
    if not resumed_anchor:
        required.append(_rl_checkpoint(parts, root, _arg(parts, "--init-stage", "sft"),
                                       _arg(parts, "--init-ckpt", "best")))
        if stage == "rl-v45":
            required.append(_rl_checkpoint(parts, root, _arg(parts, "--ref-stage", "sft"),
                                           _arg(parts, "--ref-ckpt", "best")))
    if stage == "rl-profile-v45":
        previous = _arg(parts, "--refine-from", "")
        if previous:
            path = _workspace_path(previous, root)
            required.append(path if path.is_file() else
                            run_dir / _arg(parts, "--output-stage", "rlvr-v45") / Path(previous).name)
    elif "--allow-no-profile" not in parts and (not resumed_anchor or "--no-refresh-profile" in parts):
        name = _arg(parts, "--profile-name", "") or ("profile_phase2_v2.json" if resumed_anchor else "profile_v2.json")
        required.append(run_dir / _arg(parts, "--stage-name", "rlvr-v45") / name)
    return list(dict.fromkeys(required))


def _check_rl_files(parts: list[str], stage: str, required: list[Path]) -> None:
    import torch
    from frlm import config_from_dict, model_from_cfg
    from frlm.data import load_tokenizer, special_ids
    from frlm.rl_tasks_v45 import GENERATOR_VERSION

    tok = load_tokenizer(required[0])
    if special_ids(tok)["eot"] != 0:
        raise ValueError("le tokenizer doit conserver EOT=0")
    configs = []
    for path in required[1:]:
        if path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not payload.get("rows") or not payload.get("checkpoint") or not payload.get("config"):
                raise ValueError(f"profil pass@k incomplet : {path}")
            if payload["config"].get("generator_version") != GENERATOR_VERSION:
                raise ValueError(f"générateur du profil obsolète : {path}; refaire un profil")
            for row in payload["rows"]:
                if not 0 <= row["successes"] <= row["k"] or row["k"] < 2:
                    raise ValueError(f"compteurs pass@k invalides : {path}")
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        cfg = config_from_dict(payload["model_cfg"])
        if not payload.get("model") or cfg.vocab_size != tok.get_vocab_size():
            raise ValueError(f"poids/tokenizer incompatibles : {path}")
        with torch.device("meta"):
            expected = model_from_cfg(cfg).state_dict()
        state = payload["model"]
        if set(state) != set(expected) or any(state[key].shape != tensor.shape
                                             for key, tensor in expected.items()):
            raise ValueError(f"forme des poids incompatible avec la config : {path}")
        configs.append(cfg.to_dict())
        if stage == "rl-v45" and "--resume" in parts and path == required[1]:
            if payload.get("stage") != _arg(parts, "--stage-name", "rlvr-v45"):
                raise ValueError(f"phase de reprise incompatible : {path}")
            if "accepted_updates" not in payload or ("--reset-optimizer" not in parts and not payload.get("optimizers")):
                raise ValueError(f"état de reprise incomplet : {path}")
        del payload
    if any(config != configs[0] for config in configs[1:]):
        raise ValueError("configurations de la politique et de l'ancre différentes")

def _workspace_path(value: str, root: Path = Path("/root/app")) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _arg(parts: list[str], name: str, default: str | None = None) -> str | None:
    if name not in parts:
        return default
    index = parts.index(name) + 1
    if index >= len(parts) or parts[index].startswith("--"):
        return default
    return parts[index]


def _resume_candidates(parts: list[str], stage: str, root: Path) -> list[Path]:
    """Reproduit les replis latest/best de Trainer sans importer torch."""
    if "--resume" not in parts:
        return []
    index = parts.index("--resume") + 1
    spec = parts[index] if index < len(parts) and not parts[index].startswith("--") else "latest"
    if spec not in ("latest", "auto", "", "best"):
        path = _workspace_path(spec, root)
        candidates = [path]
        sibling = {"ckpt_best.pt": "ckpt_latest.pt",
                   "ckpt_latest.pt": "ckpt_best.pt"}.get(path.name)
        if sibling:
            candidates.append(path.with_name(sibling))
        return candidates

    out_dir = _workspace_path(_arg(parts, "--out-dir", "runs") or "runs", root)
    run_name = _arg(parts, "--run", "fr-micro") or "fr-micro"
    run_dir = out_dir / run_name
    phases = [stage]
    if stage == "sft":
        phases += ["mid", "pretrain"]
    elif stage == "mid":
        phases += ["pretrain"]
    names = ["ckpt_best.pt"] if spec == "best" else ["ckpt_latest.pt", "ckpt_best.pt"]
    return [run_dir / phase / name for phase in phases for name in names]


def _required_files(cmd: str, root: Path = Path("/root/app")) -> tuple[str | None, list[Path], list[Path]]:
    parts = _command_parts(cmd)
    try:
        run_index = next(i for i, value in enumerate(parts) if Path(value).name == "run.py")
        stage = parts[run_index + 1]
    except (StopIteration, IndexError):
        if "frlm.rl_profile_v45" in parts:
            stage = "rl-profile-v45"
        elif "frlm.eval_reason_bootstrap_v45" in parts:
            data_dir = _workspace_path(_arg(parts, "--data-dir", "data-v4"), root)
            required = [data_dir / "meta.json", data_dir / "tokenizer.json"]
            if "--baselines-only" not in parts:
                spec = _arg(parts, "--ckpt", "best")
                explicit = _workspace_path(spec, root)
                run_dir = _workspace_path(_arg(parts, "--out-dir", "runs"), root) / _arg(parts, "--run", "fr-v4-v45-sft")
                name = spec if spec.endswith(".pt") else f"ckpt_{spec}.pt"
                required.append(explicit if explicit.is_file() else
                                run_dir / _arg(parts, "--stage", "sft") / name)
            return "reason-profile", required, []
        else:
            return None, [], []
    if stage in ("rl-v45", "rl-profile-v45"):
        return stage, _rl_files(parts, stage, root), []
    if stage not in ("train", "mid", "sft"):
        return stage, [], []

    data_dir = _workspace_path(_arg(parts, "--data-dir", "data") or "data", root)
    curriculum = _arg(parts, "--mid-curriculum", "") or ""
    sft_recipe = (_arg(parts, "--sft-recipe", "") or "").casefold().replace("v", "").replace(".", "")
    if stage == "mid" and curriculum:
        required = [data_dir / "tokenizer.json",
                    data_dir / "mid_v43_stage1_train.bin",
                    data_dir / "mid_v43_stage2_train.bin",
                    data_dir / "mid_v43_val.bin"]
    elif stage == "sft" and sft_recipe in ("5", "44", "45", "reason45", "reason45b", "reason45c"):
        prefix = ("reason_v45c" if sft_recipe == "reason45c" else "reason_v45" if sft_recipe in ("reason45", "reason45b", "reason45c")
                  else f"sft_v{sft_recipe}")
        required = [data_dir / "tokenizer.json", data_dir / "sft_v44_train.bin",
                    data_dir / "sft_v44_val.bin"] if sft_recipe == "44" else [
                        data_dir / "tokenizer.json", data_dir / f"{prefix}_train.bin",
                        data_dir / f"{prefix}_val.bin",
                    ]
    else:
        prefix = {"train": "", "mid": "mid_", "sft": "sft_"}[stage]
        required = [data_dir / "tokenizer.json", data_dir / f"{prefix}train.bin",
                    data_dir / f"{prefix}val.bin"]
    if stage == "sft":
        mask_prefix = ("reason_v45c" if sft_recipe == "reason45c" else "reason_v45" if sft_recipe in ("reason45", "reason45b", "reason45c") else
                       f"sft_v{sft_recipe}" if sft_recipe in ("5", "44", "45") else "sft")
        required += [data_dir / f"{mask_prefix}_train.mask",
                     data_dir / f"{mask_prefix}_val.mask"]
        replay_frac = float(_arg(parts, "--replay-frac", "0.15") or "0.15")
        if replay_frac > 0:
            replay_mix = _arg(parts, "--replay-mix", "") or ""
            if replay_mix:
                for item in replay_mix.split(","):
                    replay_path = Path(item.rsplit("=", 1)[0].strip())
                    required.append(replay_path if replay_path.is_absolute()
                                    else data_dir / replay_path)
                replay_val = Path(_arg(parts, "--replay-val", "val.bin") or "val.bin")
                required.append(replay_val if replay_val.is_absolute()
                                else data_dir / replay_val)
            else:
                required += [data_dir / "mid_train.bin", data_dir / "mid_val.bin"]
    return stage, required, _resume_candidates(parts, stage, root)


def _check_command(cmd: str, root: Path = Path("/root/app")) -> None:
    stage, required, resume_candidates = _required_files(cmd, root)
    parts = _command_parts(cmd)
    sft_recipe = (_arg(parts, "--sft-recipe", "") or "").casefold()
    sft_recipe = sft_recipe.replace("v", "").replace(".", "")
    replay_default = "0.15" if stage == "sft" else "0"
    replay_frac = float(_arg(parts, "--replay-frac", replay_default) or replay_default)
    if stage == "sft" and replay_frac > 0 and int(_arg(parts, "--grad-accum", "4")) < 2:
        raise ValueError("le replay SFT exige grad_accum >= 2")
    if stage == "sft" and sft_recipe in ("reason45", "reason45b", "reason45c") and replay_frac != 0:
        raise RuntimeError(
            f"La recette {sft_recipe} contient déjà sa rétention supervisée. "
            "Ajoute `--replay-frac 0` pour éviter un double replay. Aucun GPU n'a été alloué."
        )
    missing = [path for path in required if not path.is_file() or path.stat().st_size == 0]
    if stage in ("mid", "sft") and "--resume" not in _command_parts(cmd):
        raise RuntimeError(f"La phase {stage} exige --resume sur Modal pour éviter un "
                           "démarrage coûteux à zéro.")
    if resume_candidates and not any(path.is_file() and path.stat().st_size > 0
                                     for path in resume_candidates):
        missing.append(resume_candidates[0])
    if missing:
        details = "\n".join(f"  - {path}" for path in missing)
        raise RuntimeError(
            "Préflight Modal échoué, fichiers absents du Volume frlm-vol :\n"
            f"{details}\n"
            "Charge-les avec `modal volume put --force frlm-vol <local> <distant>` "
            "avant de relancer. Aucun GPU n'a été alloué."
        )
    if stage in ("rl-v45", "rl-profile-v45"):
        _check_rl_files(parts, stage, required)
    if stage == "reason-profile":
        from frlm.audit_reason_bootstrap_v45 import audit
        audit(required[0].parent, recipe="reason45c")
        if "--baselines-only" not in parts:
            _check_rl_files(parts, stage, [required[1], required[2]])
    is_v5 = _arg(parts, "--preset", "") == "v5-qwen4exp-350m"
    if is_v5 and stage in ("train", "sft"):
        from frlm.prepare_v5 import audit as audit_v5, sha256
        data_dir = required[0].parent
        audit_v5(data_dir)
        if int(_arg(parts, "--seq-len", "1024")) > 2048:
            raise ValueError("v5 : contexte maximal 2048")
        if stage == "sft":
            from frlm.prepare_sft_v5 import audit as audit_sft
            audit_sft(data_dir)
        if resume_candidates:
            import torch
            path = resume_candidates[0]
            if not path.is_file():
                raise ValueError("v5 exige le checkpoint exact demandé, sans repli best/latest")
            payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
            import argparse
            from run import add_train_args, cfg_from_args
            from frlm.model_v5 import ModelConfigV5, PRESETS_V5, validate_resume
            from frlm.data import load_tokenizer
            parser = argparse.ArgumentParser()
            add_train_args(parser)
            args = parser.parse_args(parts[parts.index(stage) + 1:])
            cfg = cfg_from_args(args, "pretrain" if stage == "train" else stage)
            mcfg = ModelConfigV5(**PRESETS_V5[cfg.preset])
            mcfg.vocab_size = load_tokenizer(data_dir / "tokenizer.json").get_vocab_size()
            mcfg.max_seq_len = max(mcfg.max_seq_len, cfg.seq_len)
            validate_resume(payload, cfg, mcfg.to_dict(), sha256(data_dir / "manifest.json"),
                            sha256(data_dir / "tokenizer.json"),
                            sha256(data_dir / "sft_manifest.json") if stage == "sft" else None,
                            args.init_weights_only)
    if stage in ("mid", "sft") and not is_v5:
        data_dir = required[0].parent
        meta_path = data_dir / "meta.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            sft_recipe = (_arg(_command_parts(cmd), "--sft-recipe", "") or "")
            sft_recipe = sft_recipe.casefold().replace("v", "").replace(".", "")
            if stage == "sft" and sft_recipe in ("44", "45", "reason45", "reason45b", "reason45c"):
                expected_recipe = {
                    "44": "v4.4-balanced-capabilities-18m",
                    "45": "v4.5-audited-isolated-24m",
                    "reason45": "v4.5-reason-bootstrap-ast-1",
                    "reason45b": "v4.5-reason-bootstrap-balanced-2",
                    "reason45c": "v4.5-reason-bootstrap-corrected-3",
                }[sft_recipe]
                key = ({"reason45": "reason_bootstrap_v45",
                        "reason45c": "reason_bootstrap_v45c",
                        "reason45b": "reason_bootstrap_v45b"}.get(
                            sft_recipe, f"sft_v{sft_recipe}"))
                if (meta.get(key) or {}).get("recipe") != expected_recipe:
                    raise ValueError(f"meta.json ne décrit pas la recette {sft_recipe}")
            elif stage == "sft" and (meta.get("sft") or {}).get("recipe") != "v4.2-quality-replay":
                raise ValueError("meta.json ne décrit pas la recette v4.2-quality-replay")
            curriculum = _arg(_command_parts(cmd), "--mid-curriculum", "") or ""
            if stage == "mid" and curriculum:
                section = meta["midtrain_v43"]
                if section.get("recipe") != "v4.3-curriculum-1.5b":
                    raise ValueError("meta.json ne décrit pas la recette mid v4.3")
                expected = {
                    data_dir / stage_meta["path"]: int(stage_meta["train_tokens"]) * 2
                    for stage_meta in section["stages"]
                }
                expected[data_dir / section["validation"]["path"]] = (
                    int(section["validation"]["val_tokens"]) * 2
                )
            elif stage == "sft" and sft_recipe in ("44", "45", "reason45", "reason45b", "reason45c"):
                key = ({"reason45": "reason_bootstrap_v45",
                        "reason45c": "reason_bootstrap_v45c",
                        "reason45b": "reason_bootstrap_v45b"}.get(
                            sft_recipe, f"sft_v{sft_recipe}"))
                section = meta[key]
                expected = {
                    data_dir / section["train_path"]: int(section["train_tokens"]) * 2,
                    data_dir / section["val_path"]: int(section["val_tokens"]) * 2,
                }
                for capability in section["capabilities"].values():
                    train_path = data_dir / capability["train_path"]
                    val_path = data_dir / capability["val_path"]
                    expected[train_path] = int(capability["train_tokens"]) * 2
                    expected[train_path.with_suffix(".mask")] = int(capability["train_tokens"])
                    expected[val_path] = int(capability["val_tokens"]) * 2
                    expected[val_path.with_suffix(".mask")] = int(capability["val_tokens"])
            else:
                section = meta["midtrain"] if stage == "mid" else meta["sft"]
                expected = {
                    required[1]: int(section["train_tokens"]) * 2,
                    required[2]: int(section["val_tokens"]) * 2,
                }
            if stage == "sft":
                expected[required[3]] = int(section["train_tokens"])
                expected[required[4]] = int(section["val_tokens"])
                for source in section.get("eval_sources", []):
                    source_meta = section["sources"][source]
                    source_bin = data_dir / f"sft_val_{source}.bin"
                    source_mask = data_dir / f"sft_val_{source}.mask"
                    expected[source_bin] = int(source_meta["val_tokens_unique"]) * 2
                    expected[source_mask] = int(source_meta["val_tokens_unique"])
                    if not source_bin.is_file() or not source_mask.is_file():
                        raise ValueError(f"validation équilibrée absente pour {source}")
            stale = [path for path, size in expected.items() if path.stat().st_size != size]
            if stale:
                raise ValueError("tailles incompatibles avec meta.json : "
                                 + ", ".join(str(path) for path in stale))
            if stage == "sft" and sft_recipe == "reason45c":
                from frlm.audit_reason_bootstrap_v45 import audit
                audit(data_dir, require_replay_local=True, recipe="reason45c")
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Préflight Modal échoué : les données {stage} du Volume sont anciennes "
                f"ou incohérentes ({exc}). Réuploade les bins, masks et meta.json adaptés ; "
                "aucun GPU n'a été alloué."
            ) from exc
    if required:
        print(f"[ok] Préflight {stage} : {len(required)} fichiers de données et "
              f"{'un checkpoint' if resume_candidates else 'aucun checkpoint requis'} disponibles.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--cmd", required=True)
    args = parser.parse_args()
    _check_command(args.cmd, args.root)
