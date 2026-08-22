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
#
# Fichiers vers/depuis le Volume :
#   modal volume put frlm-vol data-v4/mid_train.bin /data-v4/mid_train.bin
#   # envoyer de même mid_val.bin, sft_*.bin, sft_*.mask et meta.json uniquement
#   modal volume get frlm-vol /runs/fr-v4 runs/fr-v4              # rapatrier un ckpt
# --------------------------------------------------------------------------------------
import subprocess

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
    .pip_install("torch", "numpy", "tokenizers", "rich", "nvidia-ml-py")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_dir(".", remote_path="/root/app",
                   ignore=["data*/**", "runs/**", ".git/**", "**/__pycache__/**",
                           "*.bin", "*.pt"])
)

app = modal.App("frlm", image=image)
vol = modal.Volume.from_name("frlm-vol", create_if_missing=True)


def _executer(cmd: str, peak: float) -> None:
    import threading

    # les chemins data-v4/ et runs/ du dépôt pointent vers le Volume persistant
    subprocess.run("ln -sfn /vol/data-v4 /root/app/data-v4 && "
                   "mkdir -p /vol/runs && ln -sfn /vol/runs /root/app/runs",
                   shell=True, check=True)
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


@app.local_entrypoint()
def main(cmd: str = BENCH, gpu: str = "a100", spawn: bool = False):
    fns = {"l40s": run_l40s, "a100": run_a100, "h100": run_h100, "b200": run_b200}
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
