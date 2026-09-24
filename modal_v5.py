"""Modal v5 uniquement. Par défaut : préflight CPU. --go lance le mode demandé.

Les données sont préparées localement ; aucun téléchargement de corpus sur GPU.
Les profils Modal séparent les comptes ; le second reçoit le checkpoint complet.
"""
import math
from pathlib import Path
import shlex
import subprocess

import modal

from frlm.modal_preflight import _check_command

image = (modal.Image.from_registry("nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04", add_python="3.12")
         .apt_install("build-essential", "ninja-build")
         .pip_install_from_requirements("requirements-v5.txt")
         .pip_install("packaging", "ninja", "flash-linear-attention==0.5.2")
         .run_commands("python -m pip install --no-build-isolation causal-conv1d==1.7.0")
         .env({"PYTHONUNBUFFERED": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
         .add_local_dir(".", remote_path="/root/app", ignore=["data*/**", "runs/**", ".git/**",
                         "**/__pycache__/**", "*.bin", "*.pt"]))
app = modal.App("frlm-v5", image=image)
volume = modal.Volume.from_name("frlm-v5", create_if_missing=True)


def command(mode: str, steps: int, seconds: float, resume: str = "") -> list[str]:
    if not math.isfinite(seconds) or not 0 < seconds <= 21600:
        raise ValueError("durée positive, maximum 6 heures par compte")
    if mode == "pilot":
        if seconds > 900:
            raise ValueError("pilote limité à 15 minutes")
        return ["python", "-m", "frlm.bench_speed", "--presets", "v5-qwen4exp-350m",
                "--vocab-size", "32768", "--seq-len", "1024", "--batch-size", "8",
                "--grad-accum", "8", "--warmup", "3", "--steps", "10", "--no-compile",
                "--gpu-peak-tflops", "989"]
    if mode not in ("pretrain", "sft") or steps <= 0:
        raise ValueError("mode pretrain/sft et nombre global de steps explicite requis")
    if mode == "sft" and not resume:
        raise ValueError("le SFT exige un checkpoint prétrain explicite")
    args = ["python", "run.py", "train" if mode == "pretrain" else "sft",
            "--preset", "v5-qwen4exp-350m", "--data-dir", "data-v5",
            "--run", "fr-v5-qwen4exp", "--seq-len", "1024",
            "--batch-size", "8", "--grad-accum", "8", "--max-steps", str(steps),
            "--stop-after-seconds", str(seconds), "--seed", "551337",
            "--optimizer", "muon" if mode == "pretrain" else "adamw",
            "--lr", "0.002" if mode == "pretrain" else "0.00002",
            "--adam-lr", "0.0003", "--weight-decay", "0.1" if mode == "pretrain" else "0.01",
            "--warmup", "100" if mode == "pretrain" else "20", "--schedule", "cosine",
            "--min-lr-frac", "0.1", "--eval-every", "200", "--eval-iters", "20",
            "--sample-every", "200", "--save-every", "200", "--ckpt-every-min", "5",
            "--keep-last", "2", "--no-compile", "--gpu-peak-tflops", "989"]
    if resume:
        args += ["--resume", resume]
    if mode == "sft":
        args += ["--sft-recipe", "v5", "--replay-frac", "0.12",
                 "--replay-mix", "train.bin=1", "--replay-val", "val.bin"]
    return args


def mount():
    volume.reload()
    Path("/vol/runs").mkdir(exist_ok=True)
    for name in ("data-v5", "runs"):
        target = Path("/root/app") / name
        if not target.exists() and not target.is_symlink():
            target.symlink_to(Path("/vol") / name, target_is_directory=True)


@app.function(cpu=2, memory=4096, volumes={"/vol": volume}, timeout=600)
def preflight(args: list[str]):
    mount()
    from frlm.prepare_v5 import audit
    audit(Path("/root/app/data-v5"))
    _check_command(shlex.join(args))


@app.function(gpu="H100", cpu=4, memory=16384, volumes={"/vol": volume}, timeout=22200)
def execute(args: list[str], seconds: float):
    import threading
    # Refuser le repli PyTorch lent avant de mesurer ou d'entraîner.
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: F401
    from causal_conv1d import causal_conv1d_fn  # noqa: F401
    mount()
    stop = threading.Event()
    def commit():
        while not stop.wait(300):
            volume.commit()
    threading.Thread(target=commit, daemon=True).start()
    try:
        subprocess.run(args, cwd="/root/app", check=True, timeout=seconds + 240)
    finally:
        stop.set()
        volume.commit()


@app.local_entrypoint()
def main(mode: str = "pilot", steps: int = 0, seconds: float = 900,
         resume: str = "", go: bool = False, check_only: bool = False):
    args = command(mode, steps, seconds, resume)
    preflight.remote(args)
    if not go or check_only:
        print("Préflight CPU terminé. Aucun GPU lancé ; --go est requis.")
        return
    execute.remote(args, seconds)
