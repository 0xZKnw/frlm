# --------------------------------------------------------------------------------------
# Wrapper Beam Cloud : exécute n'importe quelle commande frlm sur un GPU serverless.
# Même philosophie que modal_app.py : le code du dépôt est synchronisé automatiquement
# (voir .beamignore) ; les gros fichiers (bins, checkpoints) vivent dans le Volume
# "frlm-vol", monté sur ./vol et relié par symlinks data-v4/ et runs/.
#
# Mise en place (une fois) :
#   pip install beam-client
#   beam configure default --token <TOKEN>          # token : dashboard > Settings
#   beam volume create frlm-vol
#   beam cp data-v4/tokenizer.json beam://frlm-vol/data-v4/     # + meta.json, bins mid/sft
#   beam cp <ckpt> beam://frlm-vol/runs/fr-v4/pretrain/         # nommé ckpt_latest.pt !
#
# Usage :
#   python beam_app.py --gpu cpu                                 # valide l'image (~0 $)
#   python beam_app.py --gpu 4090 --cmd "python run.py mid --run fr-v4 ..."
#
# Notes vs Modal :
#   - crédits gratuits « serverless only » : ce fichier n'utilise QUE du serverless ;
#     ne jamais passer par `beam machine reserve` (facture même idle, hors crédits).
#   - pas de vol.commit() : le volume distribué persiste tout seul (latence ~60 s).
#   - headless=True : la tâche survit à un Ctrl+C / une coupure du terminal (leçon des
#     deux annulations Modal du 21/08). Suivi : beam task list / beam logs --task-id …
#     Arrêt : beam task stop <task-id>.
#   - retries=0 : un crash ne relance pas la facturation en douce ; on relance à la
#     main avec --resume latest (le mid/SFT reprend alors son propre ckpt_latest).
# --------------------------------------------------------------------------------------
import subprocess

from beam import Image, Volume, function

# GPU dispo en serverless (vérifié 2026-08-21) : T4 / A10G / RTX4090 / RTX5090.
# H100+ = "beam machine reserve" uniquement -> hors crédits gratuits, facture idle : NON.
# 24-32 Go de VRAM -> bs8 x ga4 (le bs32 du H100 prenait 57 Go).
BENCH = ("python -m frlm.bench_speed --presets v4-base "
         "--batch-size 8 --grad-accum 4 --seq-len 2048 --steps 30 --warmup 8")
SMOKE_CPU = "python -c \"import torch; print('torch', torch.__version__, 'OK')\""

# pics bf16 DENSE. Ada consumer (4090) : accumulation fp32 à mi-régime -> 82.6.
# Blackwell consumer (5090) : plein régime vérifié empiriquement le 21/08 (102k tok/s
# sur v4-base = "MFU 110%" avec l'ancien pic 104.8 -> vrai pic ~209.5, MFU réel ~47%).
PEAK_TFLOPS = {"5090": 209.5, "4090": 82.6, "a10g": 62.5}

image = Image(
    python_version="python3.12",
    python_packages=["torch", "numpy", "tokenizers", "rich", "nvidia-ml-py"],
    env_vars={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
              # sans tty, Python bufferise stdout par blocs -> logs en retard par paquets
              "PYTHONUNBUFFERED": "1"},
)
vol = Volume(name="frlm-vol", mount_path="./vol")


def _executer(cmd: str, peak: float | None) -> None:
    # les chemins data-v4/ et runs/ du dépôt pointent vers le Volume persistant
    subprocess.run("ln -sfn vol/data-v4 data-v4 && "
                   "mkdir -p vol/runs && ln -sfn vol/runs runs",
                   shell=True, check=True)
    if (peak and "--gpu-peak-tflops" not in cmd
            and any(k in cmd for k in ("bench_speed", " train", " mid", " sft"))):
        cmd += f" --gpu-peak-tflops {peak}"
    subprocess.run(cmd, shell=True, check=True)


# RTX4090 24 Go (~0,69 $/h) : le cheval de trait — bs8 x ga4, ~3 s/step attendu,
# mid 4000 steps ~ 3h30 ~ 2,5 $.
@function(gpu="RTX4090", cpu=8, memory=32768, image=image, volumes=[vol],
          timeout=24 * 60 * 60, retries=0, headless=True)
def run_4090(cmd: str) -> None:
    _executer(cmd, PEAK_TFLOPS["4090"])


# RTX5090 32 Go : ~25 % plus rapide que la 4090 si dispo ; rester en bs8 x ga4
# (bs16 x ga2 ~ 30 Go = trop juste sur 32 Go).
@function(gpu="RTX5090", cpu=8, memory=32768, image=image, volumes=[vol],
          timeout=24 * 60 * 60, retries=0, headless=True)
def run_5090(cmd: str) -> None:
    _executer(cmd, PEAK_TFLOPS["5090"])


@function(gpu="A10G", cpu=8, memory=32768, image=image, volumes=[vol],
          timeout=24 * 60 * 60, retries=0, headless=True)
def run_a10g(cmd: str) -> None:
    _executer(cmd, PEAK_TFLOPS["a10g"])


# sans GPU : valide l'image + le montage du volume pour ~0 $
@function(cpu=2, memory=4096, image=image, volumes=[vol], timeout=600, retries=0)
def run_cpu(cmd: str) -> None:
    _executer(cmd, None)
    subprocess.run("ls -lh vol/data-v4 vol/runs 2>/dev/null || true", shell=True)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Lance une commande frlm sur Beam serverless")
    ap.add_argument("--gpu", default="4090", choices=["4090", "5090", "a10g", "cpu"])
    ap.add_argument("--cmd", default=None,
                    help="commande à exécuter (défaut : bench_speed sur GPU, smoke sur CPU)")
    args = ap.parse_args()

    fns = {"4090": run_4090, "5090": run_5090, "a10g": run_a10g, "cpu": run_cpu}
    cmd = args.cmd or (SMOKE_CPU if args.gpu == "cpu" else BENCH)
    fns[args.gpu].remote(cmd)
