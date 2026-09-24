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
import shutil
import subprocess
from pathlib import Path

import modal

from frlm.modal_preflight import _check_command, with_gpu_peak

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
    cmd = with_gpu_peak(cmd, peak)

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
