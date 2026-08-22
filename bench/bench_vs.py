# --------------------------------------------------------------------------------------
# Benchmark : fr-v2 (58M) contre des GPT-2 français publics (~124M, soit 2,15x sa taille).
#
# Trois épreuves, conçues pour être équitables malgré des tokenizers différents :
#   1. bpb        — bits par OCTET sur du texte français tenu à l'écart (la seule façon
#                   honnête de comparer des loss entre tokenizers différents).
#   2. calcul     — exact-match sur des problèmes générés par synth.py avec une seed
#                   jamais utilisée à l'entraînement. Chaque modèle joue avec ses armes :
#                   fr-v2 en format chat (son format natif), les modèles de base en
#                   few-shot 3 exemples (leur meilleur protocole).
#   3. faits      — complétions factuelles en continuation brute, notées par regex.
#
# Usage :  python bench_vs.py                        # tout
#          python bench_vs.py --n-problems 50        # calcul plus court
#          python bench_vs.py --skip-hf              # seulement fr-v2 (sans téléchargement)
# Prérequis concurrents : pip install transformers  (~1,5 Go de téléchargements HF)
# --------------------------------------------------------------------------------------
import argparse
import dataclasses
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]   # racine du dépôt
sys.path.insert(0, str(ROOT))
from frlm import data as D  # noqa: E402
from frlm import synth  # noqa: E402
from frlm.model import ModelConfig, build_model  # noqa: E402

CONCURRENTS = [
    ("asi/gpt-fr-cased-small", "GPT-fr small · 124M (Inria)"),
    ("antoinelouis/belgpt2", "BelGPT-2 · 124M (60 Go)"),
    ("dbddv01/gpt2-french-small", "GPT-2 fr · 124M (transfert)"),
]

FAITS = [
    ("La capitale de la France est", r"\bParis\b"),
    ("La capitale de l'Italie est", r"\bRome\b"),
    ("L'eau bout à une température de", r"100"),
    ("Une semaine compte", r"sept|7"),
    ("Le contraire de grand est", r"\bpetit"),
    ("Les abeilles produisent du", r"\bmiel\b"),
    ("La Seine traverse la ville de", r"\bParis|\bRouen"),
    ("Un triangle possède", r"trois|3"),
]

FEWSHOT = (
    "Question : Calcule : 12 + 7\nRéponse : 19\n\n"
    "Question : On partage 20 billes équitablement entre 4 enfants. "
    "Combien chaque enfant en reçoit-il ?\nRéponse : 5\n\n"
    "Question : Un cahier coûte 3 euros. Combien coûtent 5 cahiers ?\nRéponse : 15\n\n"
)


# --------------------------------------------------------------------------------------
# Outils communs
# --------------------------------------------------------------------------------------
def dernier_nombre(txt: str) -> str | None:
    """Dernier nombre du texte, tolérant aux espaces de groupement ('1 000')."""
    hits = re.findall(r"\d(?:[\d ]*\d)?", txt)
    if not hits:
        return None
    return hits[-1].replace(" ", "")


def problemes_eval(n: int, seed: int) -> list[dict]:
    """Problèmes à réponse numérique, seed hors entraînement (train: 0/1/101)."""
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        p = synth.make_problem(rng)
        rep = dernier_nombre(p["a"])
        if rep is not None:  # on écarte parité/syllogisme/comparaison (réponse texte)
            out.append({"q": p["q"], "attendu": rep})
    return out


def lire_texte_eval(data_dir: Path, n_chars: int) -> str:
    """Texte français 'frais' : derniers documents de wiki+fineweb (au-delà des caps
    de binarisation, qui consomment les fichiers depuis le début)."""
    morceaux = []
    for name in ("wiki", "fineweb"):
        p = data_dir / "raw" / f"{name}.jsonl"
        if not p.exists():
            continue
        size = p.stat().st_size
        with p.open("rb") as f:
            f.seek(max(0, size - 8_000_000))
            brut = f.read().decode("utf-8", errors="ignore")
        lignes = brut.split("\n")[1:]  # la 1re est probablement tronquée
        acc, budget = [], n_chars // 2
        for ligne in reversed(lignes):  # les tout derniers docs du fichier
            ligne = ligne.strip()
            if not ligne:
                continue
            try:
                doc = json.loads(ligne)
            except json.JSONDecodeError:
                continue
            txt = doc.get("text") or next((v for v in doc.values() if isinstance(v, str)), "")
            if txt:
                acc.append(txt)
                budget -= len(txt)
                if budget <= 0:
                    break
        morceaux.extend(reversed(acc))
    texte = "\n\n".join(morceaux)
    assert len(texte) > 50_000, "texte d'évaluation introuvable ou trop court"
    return texte[: n_chars]


# --------------------------------------------------------------------------------------
# Notre modèle
# --------------------------------------------------------------------------------------
class NotreModele:
    def __init__(self, run_dir: Path, data_dir: Path, device: str,
                 stage=None, ckpt_name=None):
        ckpt = None
        # stage/ckpt_name explicites : on NE retombe PAS sur sft en cas d'absence
        # (un repli silencieux produirait un rapport étiqueté rlaif mais mesuré
        # sur le SFT — vécu de justesse le 2026-08-22).
        phases = (stage,) if stage else ("sft", "mid", "pretrain")
        noms = (ckpt_name,) if ckpt_name else ("ckpt_best.pt", "ckpt_latest.pt")
        for phase in phases:
            for name in noms:
                p = run_dir / phase / name
                if p.exists():
                    ckpt = p
                    break
            if ckpt:
                break
        assert ckpt, (f"aucun checkpoint sous {run_dir}"
                      + (f" / {stage}" if stage else "")
                      + (f" / {ckpt_name}" if ckpt_name else ""))
        ck = torch.load(ckpt, map_location=device, weights_only=False)
        from frlm import config_from_dict, model_from_cfg
        mcfg = config_from_dict(ck["model_cfg"])
        self.model = model_from_cfg(mcfg).to(device)
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.tok = D.load_tokenizer(data_dir / "tokenizer.json")
        self.sp = D.special_ids(self.tok)
        self.device = device
        self.phase = ckpt.parent.name
        n_par = sum(p.numel() for p in self.model.parameters())
        self.nom = (f"{run_dir.name} · {n_par / 1e6:.0f}M "
                    f"({self.phase}, step {ck.get('step', '?')})")
        print(f"[i] {run_dir.name} chargé : {ckpt}  (phase {self.phase})")

    @torch.no_grad()
    def bpb(self, texte: str) -> float:
        ids = self.tok.encode(texte).ids
        nll, n_pred = 0.0, 0
        T = self.model.cfg.max_seq_len
        for i in range(0, len(ids) - 1, T):
            fen = ids[i : i + T + 1]
            if len(fen) < 2:
                break
            x = torch.tensor([fen[:-1]], device=self.device)
            y = torch.tensor([fen[1:]], device=self.device)
            m = torch.ones_like(y, dtype=torch.float32)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=self.device.startswith("cuda")):
                _, loss, _ = self.model(x, y, m, z_loss=0.0, diagnostics=False)
            nll += loss.item() * (len(fen) - 1)
            n_pred += len(fen) - 1
        octets = len(texte.encode("utf-8"))
        return nll / math.log(2) / octets, n_pred

    @torch.no_grad()
    def repondre(self, question: str, max_new: int = 220) -> str:
        if self.phase == "sft":
            texte = f"{D.IM_START}user\n{question}{D.IM_END}\n{D.IM_START}assistant\n"
        else:  # checkpoint de base : continuation few-shot comme les concurrents
            texte = FEWSHOT + f"Question : {question}\nRéponse :"
        ids = torch.tensor([self.tok.encode(texte).ids], device=self.device)
        out = self.model.generate(ids, max_new_tokens=max_new, temperature=0.0,
                                  repetition_penalty=1.0,
                                  stop_ids=(self.sp["im_end"], self.sp["eot"]))
        gen = self.tok.decode(out[0, ids.shape[1]:].tolist(), skip_special_tokens=False)
        # on ne note que la réponse finale, pas le brouillon du <think>
        gen = gen.split("</think>")[-1]
        return gen.split("<|im_end|>")[0]

    @torch.no_grad()
    def completer(self, amorce: str, max_new: int = 30) -> str:
        ids = torch.tensor([self.tok.encode(amorce).ids], device=self.device)
        out = self.model.generate(ids, max_new_tokens=max_new, temperature=0.0,
                                  repetition_penalty=1.0,
                                  stop_ids=(self.sp["eot"],))
        return self.tok.decode(out[0, ids.shape[1]:].tolist(), skip_special_tokens=True)


# --------------------------------------------------------------------------------------
# Concurrents Hugging Face (GPT-2 standard)
# --------------------------------------------------------------------------------------
class ModeleHF:
    def __init__(self, repo: str, nom: str, device: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[i] chargement {repo}…")
        self.tok = AutoTokenizer.from_pretrained(repo)
        self.model = AutoModelForCausalLM.from_pretrained(repo).to(device).eval()
        self.device = device
        self.nom = nom
        self.ctx = min(getattr(self.model.config, "n_positions", 1024), 1024)

    @torch.no_grad()
    def bpb(self, texte: str) -> float:
        ids = self.tok(texte, return_tensors="pt").input_ids[0]
        nll, n_pred = 0.0, 0
        # fenêtre de ctx tokens MAX : les GPT-2 ont des embeddings de position appris
        # (1024 pile), une fenêtre de ctx+1 fait déborder la table -> assert CUDA
        T = self.ctx - 1
        for i in range(0, len(ids) - 1, T):
            fen = ids[i : i + T + 1].unsqueeze(0).to(self.device)
            if fen.shape[1] < 2:
                break
            out = self.model(fen, labels=fen)
            nll += out.loss.item() * (fen.shape[1] - 1)
            n_pred += fen.shape[1] - 1
        octets = len(texte.encode("utf-8"))
        return nll / math.log(2) / octets, n_pred

    @torch.no_grad()
    def _greedy(self, prompt: str, max_new: int) -> str:
        enc = self.tok(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                  pad_token_id=self.tok.eos_token_id
                                  or self.tok.pad_token_id or 0)
        return self.tok.decode(out[0, enc.input_ids.shape[1]:],
                               skip_special_tokens=True)

    def repondre(self, question: str, max_new: int = 60) -> str:
        gen = self._greedy(FEWSHOT + f"Question : {question}\nRéponse :", max_new)
        return gen.split("Question")[0]  # on coupe l'exercice suivant qu'il s'invente

    def completer(self, amorce: str, max_new: int = 30) -> str:
        return self._greedy(amorce, max_new)


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="fr-v2")
    ap.add_argument("--data-dir", default="data-v2")
    ap.add_argument("--n-problems", type=int, default=100)
    ap.add_argument("--eval-chars", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=4242, help="seed synth (train: 0/1/101)")
    ap.add_argument("--skip-hf", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = ROOT / args.data_dir

    modeles = [NotreModele(ROOT / "runs" / args.run, data_dir, device)]
    if not args.skip_hf:
        try:
            for repo, nom in CONCURRENTS:
                try:
                    modeles.append(ModeleHF(repo, nom, device))
                except Exception as e:
                    print(f"[!] {repo} indisponible ({e}) — ignoré.")
        except ImportError:
            print("[!] transformers absent : pip install transformers")

    texte = lire_texte_eval(data_dir, args.eval_chars)
    problemes = problemes_eval(args.n_problems, args.seed)
    print(f"[i] épreuves : bpb sur {len(texte):,} chars · "
          f"{len(problemes)} problèmes (seed {args.seed}) · {len(FAITS)} faits\n")

    lignes, details = [], []
    for m in modeles:
        t0 = time.time()
        bpb, n_pred = m.bpb(texte)

        ok_calc = 0
        for p in problemes:
            try:
                brut = m.repondre(p["q"])
            except Exception as e:
                brut = f"<erreur : {e}>"
            rep = dernier_nombre(brut)
            bon = rep == p["attendu"]
            ok_calc += bon
            # réponse complète conservée pour l'audit manuel des notes
            details.append((m.nom, "calc", p["q"], p["attendu"],
                            f"{rep!r} · texte : {brut.strip()[:200]}", bon))

        ok_faits = 0
        for amorce, motif in FAITS:
            try:
                gen = m.completer(amorce)
            except Exception:
                gen = ""
            bon = re.search(motif, gen, re.IGNORECASE) is not None
            ok_faits += bon
            details.append((m.nom, "fait", amorce, motif, gen.strip()[:80], bon))

        lignes.append((m.nom, bpb, ok_calc / len(problemes), ok_faits / len(FAITS)))
        print(f"{m.nom:<42} bpb {bpb:.4f} · calcul {ok_calc}/{len(problemes)} · "
              f"faits {ok_faits}/{len(FAITS)}  ({time.time()-t0:.0f}s)")

        # on libère la VRAM entre deux concurrents
        if not isinstance(m, NotreModele):
            del m.model
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ rapport
    rap = ROOT / "bench" / "reports" / "bench_vs_report.md"
    with rap.open("w", encoding="utf-8") as f:
        f.write("# fr-v2 contre les GPT-2 français publics\n\n")
        f.write(f"bpb sur {len(texte):,} chars tenus à l'écart · "
                f"{len(problemes)} problèmes seed {args.seed} · greedy partout\n\n")
        f.write("| modèle | bpb ↓ | calcul ↑ | faits ↑ |\n|---|---|---|---|\n")
        for nom, bpb, calc, faits in lignes:
            f.write(f"| {nom} | {bpb:.4f} | {calc:.0%} | {faits:.0%} |\n")
        f.write("\n## Détails\n\n")
        for nom, ep, q, attendu, obtenu, bon in details:
            f.write(f"- {'✅' if bon else '❌'} `{ep}` **{nom}** — {q!r} → "
                    f"attendu {attendu!r}, obtenu {obtenu!r}\n")
    print(f"\n[i] rapport détaillé : {rap}")


if __name__ == "__main__":
    main()
