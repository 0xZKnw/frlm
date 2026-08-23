# --------------------------------------------------------------------------------------
# Wrapper Modal : exécute n'importe quelle commande frlm sur un GPU serverless.
# Le code du dépôt est embarqué dans l'image ; les gros fichiers (bins, checkpoints)
# vivent dans le Volume "frlm-vol", monté sur /vol et relié par symlinks.
#
# Mise en place (une fois) :
#   pip install modal
#   modal setup                        # ouvre le navigateur, lie le compte
#
# Usage :
#   modal run modal_app.py                                        # bench_speed sur L40S
#   modal run modal_app.py --gpu a100                             # bench_speed sur A100
#   modal run modal_app.py --gpu a100 --cmd "python run.py train --preset v4-base ..."
#   modal run modal_app.py --check-only --cmd "python run.py mid ..."  # CPU, aucun GPU
#
# Fichiers vers/depuis le Volume :
#   modal volume put --force frlm-vol data-v4/mid_train.bin /data-v4/mid_train.bin
#   # envoyer de même mid_val.bin, sft_*.bin, sft_*.mask et meta.json uniquement
#   modal volume get frlm-vol /runs/fr-v4 runs/fr-v4              # rapatrier un ckpt
# --------------------------------------------------------------------------------------
import json
import shlex
import shutil
import subprocess
from pathlib import Path

import modal

# le minimum qui donne un MFU fiable : 1 seul preset (celui du vrai run),
# 30 steps mesurés — ~4-5 min de GPU par carte, le compile domine le coût.
# FlexAttention (par défaut depuis 2026-08-21) ne matérialise plus les matrices
# T×T -> bs 32 repasse. Repli anti-OOM auto dans le bench si besoin, et
# FRLM_ATTN=sdpa pour re-mesurer l'ancien chemin masqué.
BENCH = ("python -m frlm.bench_speed --presets v4-base "
         "--batch-size 32 --grad-accum 1 --seq-len 2048 --steps 30 --warmup 8")

# pics bf16 DENSE (pas les chiffres marketing "avec sparsité") pour un MFU honnête
PEAK_TFLOPS = {"l40s": 181.0, "a100": 312.0, "h100": 989.0, "b200": 2250.0}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "tokenizers", "rich", "nvidia-ml-py",
                 "datasets>=2.19", "tqdm>=4.66")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
          "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"})
    .add_local_dir(".", remote_path="/root/app",
                   ignore=["data*/**", "runs/**", ".git/**", "**/__pycache__/**",
                           "*.bin", "*.pt"])
)

app = modal.App("frlm", image=image)
vol = modal.Volume.from_name("frlm-vol", create_if_missing=True)


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
    parts = shlex.split(cmd)
    try:
        run_index = next(i for i, value in enumerate(parts) if Path(value).name == "run.py")
        stage = parts[run_index + 1]
    except (StopIteration, IndexError):
        return None, [], []
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
    elif stage == "sft" and sft_recipe in ("44", "45", "reason45"):
        prefix = "reason_v45" if sft_recipe == "reason45" else f"sft_v{sft_recipe}"
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
        mask_prefix = ("reason_v45" if sft_recipe == "reason45" else
                       f"sft_v{sft_recipe}" if sft_recipe in ("44", "45") else "sft")
        required += [data_dir / f"{mask_prefix}_train.mask",
                     data_dir / f"{mask_prefix}_val.mask"]
        replay_frac = float(_arg(parts, "--replay-frac", "0") or "0")
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


def _mount_workspace() -> None:
    """Recharge le Volume puis remplace les dossiers locaux par des liens fiables."""
    vol.reload()
    Path("/vol/runs").mkdir(parents=True, exist_ok=True)
    for source, target in ((Path("/vol/data-v4"), Path("/root/app/data-v4")),
                           (Path("/vol/runs"), Path("/root/app/runs"))):
        if target.is_symlink():
            target.unlink()
        elif target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        target.symlink_to(source, target_is_directory=True)


def _check_command(cmd: str) -> None:
    stage, required, resume_candidates = _required_files(cmd)
    parts = shlex.split(cmd)
    sft_recipe = (_arg(parts, "--sft-recipe", "") or "").casefold()
    sft_recipe = sft_recipe.replace("v", "").replace(".", "")
    replay_default = "0.15" if stage == "sft" else "0"
    replay_frac = float(_arg(parts, "--replay-frac", replay_default) or replay_default)
    if stage == "sft" and sft_recipe == "reason45" and replay_frac != 0:
        raise RuntimeError(
            "La recette reason45 contient déjà 20 % de rétention supervisée. "
            "Ajoute `--replay-frac 0` pour éviter un double replay. Aucun GPU n'a été alloué."
        )
    missing = [path for path in required if not path.is_file() or path.stat().st_size == 0]
    if stage in ("mid", "sft") and "--resume" not in shlex.split(cmd):
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
    if stage in ("mid", "sft"):
        data_dir = required[0].parent
        meta_path = data_dir / "meta.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            sft_recipe = (_arg(shlex.split(cmd), "--sft-recipe", "") or "")
            sft_recipe = sft_recipe.casefold().replace("v", "").replace(".", "")
            if stage == "sft" and sft_recipe in ("44", "45", "reason45"):
                expected_recipe = {
                    "44": "v4.4-balanced-capabilities-18m",
                    "45": "v4.5-audited-isolated-24m",
                    "reason45": "v4.5-reason-bootstrap-ast-1",
                }[sft_recipe]
                key = ("reason_bootstrap_v45" if sft_recipe == "reason45"
                       else f"sft_v{sft_recipe}")
                if (meta.get(key) or {}).get("recipe") != expected_recipe:
                    raise ValueError(f"meta.json ne décrit pas la recette {sft_recipe}")
            elif stage == "sft" and (meta.get("sft") or {}).get("recipe") != "v4.2-quality-replay":
                raise ValueError("meta.json ne décrit pas la recette v4.2-quality-replay")
            curriculum = _arg(shlex.split(cmd), "--mid-curriculum", "") or ""
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
            elif stage == "sft" and sft_recipe in ("44", "45", "reason45"):
                key = ("reason_bootstrap_v45" if sft_recipe == "reason45"
                       else f"sft_v{sft_recipe}")
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
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Préflight Modal échoué : les données {stage} du Volume sont anciennes "
                f"ou incohérentes ({exc}). Réuploade les bins, masks et meta.json adaptés ; "
                "aucun GPU n'a été alloué."
            ) from exc
    if required:
        print(f"[ok] Préflight {stage} : {len(required)} fichiers de données et "
              f"{'un checkpoint' if resume_candidates else 'aucun checkpoint requis'} disponibles.")


@app.function(volumes={"/vol": vol}, timeout=5 * 60)
def preflight(cmd: str) -> None:
    """Vérifie le Volume sur CPU avant de louer un GPU."""
    _mount_workspace()
    _check_command(cmd)


def _executer(cmd: str, peak: float) -> None:
    import threading

    # Important pour un conteneur GPU réutilisé après un `modal volume put` : sans
    # reload, il peut garder l'ancien snapshot et croire le checkpoint absent.
    _mount_workspace()
    _check_command(cmd)
    if ("--gpu-peak-tflops" not in cmd
            and any(k in cmd for k in ("bench_speed", " train", " mid", " sft"))):
        cmd += f" --gpu-peak-tflops {peak}"

    # commit du Volume toutes les 10 min : une préemption/crash au milieu d'un run
    # de 7 h ne coûte au pire que 10 min de checkpoints
    stop = threading.Event()

    def _committer():
        while not stop.wait(600):
            try:
                vol.commit()
            except Exception:
                pass

    threading.Thread(target=_committer, daemon=True).start()
    try:
        subprocess.run(cmd, shell=True, check=True, cwd="/root/app")
    finally:
        stop.set()
        vol.commit()   # persiste les derniers checkpoints du run


@app.function(gpu="L40S", volumes={"/vol": vol}, timeout=24 * 60 * 60)
def run_l40s(cmd: str) -> None:
    _executer(cmd, PEAK_TFLOPS["l40s"])


@app.function(gpu="A100", volumes={"/vol": vol}, timeout=24 * 60 * 60)
def run_a100(cmd: str) -> None:      # 40 Go (2,10 $/h) — le 80 Go n'apporte rien ici
    _executer(cmd, PEAK_TFLOPS["a100"])


@app.function(gpu="H100", volumes={"/vol": vol}, timeout=24 * 60 * 60)
def run_h100(cmd: str) -> None:      # 3,95 $/h mais ~250 TFLOPS/$ : le favori
    _executer(cmd, PEAK_TFLOPS["h100"])


@app.function(gpu="B200", volumes={"/vol": vol}, timeout=24 * 60 * 60)
def run_b200(cmd: str) -> None:      # 6,25 $/h, ~360 TFLOPS/$ crête — risque Blackwell
    _executer(cmd, PEAK_TFLOPS["b200"])


@app.function(cpu=8.0, memory=32768, volumes={"/vol": vol}, timeout=24 * 60 * 60)
def run_cpu(cmd: str) -> None:
    """Préparation lourde des données sans louer de GPU."""
    _executer(cmd, 0.0)


@app.local_entrypoint()
def main(cmd: str = BENCH, gpu: str = "a100", spawn: bool = False,
         check_only: bool = False):
    fns = {"cpu": run_cpu, "l40s": run_l40s, "a100": run_a100,
           "h100": run_h100, "b200": run_b200}
    # Le préflight tourne sans GPU. Une faute de chemin ou un upload oublié ne
    # consomme donc plus une allocation H100 pour échouer une seconde plus tard.
    preflight.remote(cmd)
    if check_only:
        print("Préflight terminé : Volume et checkpoint cohérents, aucun GPU lancé.")
        return
    if spawn:
        # fire-and-forget : à utiliser avec --detach pour les runs longs.
        # Un Ctrl+C sur un .remote() bloquant ANNULE l'appel en cours (vécu le
        # 2026-08-21, deux fois) ; .spawn() coupe tout lien avec le terminal.
        call = fns[gpu.lower()].spawn(cmd)
        print(f"Job lancé en arrière-plan ({call.object_id}).\n"
              "Terminal fermable immédiatement — suivi sur modal.com ou "
              "`modal app logs <app-id>`.")
    else:
        fns[gpu.lower()].remote(cmd)
