"""
data.py — Téléchargement des corpus français, tokenizer BPE maison, binarisation.

Pourquoi un tokenizer maison plutôt que celui de Qwen3 ?
  Le vocab de Qwen3 fait 151 936 tokens. Sur un modèle de 60M params, la matrice
  d'embedding ferait à elle seule 87M params (soit 60% du modèle pour du chinois,
  de l'arabe et du code qu'on n'utilisera jamais). Un BPE de 16k entraîné sur du
  français pur donne une meilleure compression sur NOS données et libère tout le
  budget de params pour les couches transformer.

Corpus retenus (tous vérifiés existants sur le Hub) :
  fineweb  HuggingFaceFW/fineweb-2  config fra_Latn   -> web français filtré, très gros
  wiki     wikimedia/wikipedia      config 20231101.fr -> français propre et bien écrit
  chat     angeluriot/french_instruct                  -> 275k conversations FR multi-tours
  alpaca   jpacifico/French-Alpaca-dataset-Instruct-110K -> 110k paires instruction/réponse
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---- tokens spéciaux (format ChatML + balises de réflexion, comme Qwen) ---------------
EOT = "<|endoftext|>"
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
THINK = "<think>"
THINK_END = "</think>"
SPECIALS = [EOT, IM_START, IM_END, THINK, THINK_END]

DTYPE = np.uint16  # suffit jusqu'à 65 535 tokens de vocab


# --------------------------------------------------------------------------------------
# Sources de données
# --------------------------------------------------------------------------------------
@dataclass
class Source:
    name: str
    repo: str
    config: str | None
    split: str
    kind: str  # "text" | "chat" | "alpaca"


SOURCES: dict[str, Source] = {
    # v4 (2026-08-21) : FineWeb2-HQ = top 10% de fineweb-2 fra_Latn sélectionné par
    # classifieur de qualité (epfml). Même colonne "text". Le papier mesure : égale
    # fineweb-2 complet avec 6× moins de tokens. data-v2 (déjà binarisé) n'est pas
    # impacté ; l'ancien repo était HuggingFaceFW/fineweb-2.
    "fineweb": Source("fineweb", "epfml/FineWeb2-HQ", "fra_Latn", "train", "text"),
    "wiki": Source("wiki", "wikimedia/wikipedia", "20231101.fr", "train", "text"),
    "chat": Source("chat", "angeluriot/french_instruct", None, "train", "chat"),
    "alpaca": Source("alpaca", "jpacifico/French-Alpaca-dataset-Instruct-110K", None, "train", "alpaca"),
    # dolphin-r1 traduit en français : réponses avec trace de raisonnement complète.
    # Converti au format <think>...</think> de Qwen3 pour le mode "thinking".
    "reasoning": Source("reasoning", "WiroAI/dolphin-r1-french", None, "train", "dolphin_think"),
    # ---- v2 : les sources du "raisonnement léger" -----------------------------------
    # maths : généré LOCALEMENT par synth.py (millions de problèmes FR corrects,
    # solutions calculées en Python). Aucun téléchargement.
    "maths": Source("maths", "(local)", None, "train", "synth"),
    # GSM8K traduit en français : 7,4k problèmes en mots avec étapes -> format <think>.
    "gsm8k": Source("gsm8k", "cmh/gsm8k_fr", None, "train", "gsm8k"),
    # Livres français (Gutenberg via Lucie) : cohérence narrative longue.
    "books": Source("books", "OpenLLM-France/Lucie-Training-Dataset", "Gutenberg-fr", "train", "book"),
    # Français oral spontané (transcriptions Claire) : naturel conversationnel.
    "oral": Source("oral", "OpenLLM-France/Lucie-Training-Dataset", "Claire-fr", "train", "oral"),
    # ---- v4 : registres formels/structurés (Lucie) --------------------------------
    # Thèses françaises : français académique long et soigné.
    "theses": Source("theses", "OpenLLM-France/Lucie-Training-Dataset", "Theses", "train", "text"),
    # Débats du Parlement européen : français oratoire structuré.
    "europarl": Source("europarl", "OpenLLM-France/Lucie-Training-Dataset", "Europarl-fr", "train", "text"),
}

# Budgets de téléchargement (en caractères) des sources SFT hors mix de pré-entraînement.
SFT_BUDGETS = {"alpaca": 40_000_000, "reasoning": 90_000_000,
               "gsm8k": 12_000_000, "maths_sft": 25_000_000}
# Un exemple de raisonnement plus long que ça ne rentrera jamais dans le contexte : on
# le jette au téléchargement plutôt que de gaspiller le budget (~3,5 car/token).
MAX_REASONING_CHARS = 3600

# Recette par défaut du pré-entraînement (proportion du budget de caractères).
# On met déjà 15% de chat DANS le pré-entraînement : sur un tout petit modèle, voir le
# format ChatML dès le début fait une énorme différence sur la capacité à dialoguer.
# v2 : 15% de maths synthétiques dès le pré-entraînement — un modèle ne peut pas
# apprendre à raisonner au SFT si le substrat n'existe pas déjà dans les poids.
DEFAULT_MIX = {"fineweb": 0.40, "wiki": 0.20, "chat": 0.15, "maths": 0.15,
               "books": 0.05, "oral": 0.05}
V1_MIX = {"fineweb": 0.55, "wiki": 0.25, "chat": 0.20}  # l'ancienne recette, si besoin

# Recette v4 (~3B tokens, tokenizer 24k) : qualité par token. Le web passe par
# FineWeb2-HQ (déjà filtré), et les parts des petites sources finies sont calées
# sur leur taille RÉELLE pour éviter le manque silencieux (chat ≈ 0,4B chars,
# oral/europarl ≈ 0,1B chacun — à 13B chars de budget total, 3%/1%/1% les épuisent).
V4_MIX = {"fineweb": 0.55, "wiki": 0.15, "maths": 0.15, "books": 0.06,
          "theses": 0.04, "chat": 0.03, "europarl": 0.01, "oral": 0.01}

# Recette v4.1 du midtrain. Le précédent mix mettait 45 % de maths synthétiques et
# prétendait réserver 5 % à un distillat de 0,5 Mo : sa part réelle était quasi
# nulle. L'audit v4 trouve 156 M caractères d'instructions propres, soit exactement
# 12 % d'un mid à 0,12 du prétrain. Le reste garde assez de français naturel pour
# éviter qu'un recuit trop spécialisé dégrade la langue générale.
MID_MIX = {"maths": 0.30, "wiki": 0.33, "fineweb": 0.15,
           "books": 0.10, "instruct": 0.12}

# Priorité qualité pour la déduplication inter-sources : si le même prompt existe
# dans plusieurs jeux, on garde la version vérifiée/concise avant le gros jeu chat.
SFT_SOURCE_ORDER = ("distill", "gsm8k", "maths_sft", "alpaca", "reasoning", "chat")
MID_INSTRUCT_SOURCES = ("distill", "gsm8k", "alpaca", "reasoning", "chat")

# Parts visées en TOKENS SUPERVISÉS (pas en taille de fichier). Les limites de
# répétition empêchent les petites sources de devenir des tables de mémorisation ;
# tout quota impossible à remplir est redistribué aux sources qui ont de la marge.
SFT_RECIPE = {
    "chat":       dict(weight=0.24, max_repeat=1, max_prompt=1200, max_final=1600, max_think=0),
    "alpaca":     dict(weight=0.22, max_repeat=3, max_prompt=1200, max_final=1600, max_think=0),
    "reasoning":  dict(weight=0.12, max_repeat=2, max_prompt=2000, max_final=1600, max_think=800),
    "gsm8k":      dict(weight=0.10, max_repeat=8, max_prompt=1400, max_final=500,  max_think=1200),
    "maths_sft":  dict(weight=0.30, max_repeat=4, max_prompt=500,  max_final=300,  max_think=500),
    "distill":    dict(weight=0.02, max_repeat=12, max_prompt=600, max_final=500,  max_think=500),
}
SFT_TARGET_SUPERVISED = 50_000_000
SFT_VAL_SUPERVISED_PER_SOURCE = 100_000


def render_chat(messages: list[dict]) -> str:
    """Rend une conversation au format ChatML."""
    out = []
    for m in messages:
        role = m["role"]
        content = (m.get("text") or m.get("content") or "").strip()
        if not content:
            continue
        out.append(f"{IM_START}{role}\n{content}{IM_END}\n")
    return "".join(out)


def chat_segments(messages: list[dict], ensure_think: bool = False) -> list[tuple[str, bool]]:
    """Découpe une conversation en (texte, on_apprend_dessus).

    On n'entraîne QUE sur les réponses de l'assistant : apprendre à prédire les
    questions de l'utilisateur gaspillerait de la capacité et rendrait le modèle
    bavard à la place de l'user.

    ensure_think (v2) : TOUTE réponse assistant commence par un bloc think — plein
    (raisonnement) ou vide (<think>\\n\\n</think>). Le format devient 100 % cohérent :
    le modèle apprend que penser est OPTIONNEL mais que le bloc, lui, est toujours là.
    C'est ce qui fait marcher /think off (la v1 « repensait » hors du bloc vide, car
    elle n'avait jamais vu de bloc vide à l'entraînement) et ce qui apprend au mode
    auto à réserver la réflexion aux questions qui la méritent.
    """
    segs: list[tuple[str, bool]] = []
    for m in messages:
        role = m["role"]
        content = (m.get("text") or m.get("content") or "").strip()
        if not content:
            continue
        segs.append((f"{IM_START}{role}\n", False))
        if role == "assistant":
            if ensure_think and not content.startswith(THINK):
                content = f"{THINK}\n\n{THINK_END}\n{content}"
            segs.append((f"{content}{IM_END}\n", True))
        else:
            segs.append((f"{content}{IM_END}\n", False))
    return segs


def _iter_source(src: Source, seed: int = 0):
    """Générateur de documents. Chaque élément : (texte, messages|None)."""
    from datasets import load_dataset

    kw = dict(split=src.split, streaming=True)
    if src.config:
        kw["name"] = src.config
    ds = load_dataset(src.repo, **kw)
    if src.kind in ("text", "book"):
        # FineWeb2-HQ traîne une colonne embeddings (~2-3× le poids du texte) :
        # la projeter AVANT le shuffle allège le téléchargement ET le buffer
        try:
            ds = ds.select_columns(["text"])
        except Exception:
            pass
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    for row in ds:
        if src.kind == "text":
            t = (row.get("text") or "").strip()
            if len(t) < 200:          # on jette les documents trop courts (souvent du bruit)
                continue
            yield t, None
        elif src.kind == "chat":
            conv = row.get("conversation") or []
            msgs = [{"role": m["role"], "text": m.get("text", "")} for m in conv]
            if len(msgs) < 2:
                continue
            yield render_chat(msgs), msgs
        elif src.kind == "alpaca":
            instr = (row.get("instruction") or "").strip()
            inp = (row.get("input") or "").strip()
            out = (row.get("output") or "").strip()
            if not instr or not out:
                continue
            user = f"{instr}\n\n{inp}" if inp else instr
            msgs = [{"role": "user", "text": user}, {"role": "assistant", "text": out}]
            yield render_chat(msgs), msgs
        elif src.kind == "dolphin_think":
            msgs = _convert_dolphin(row.get("messages") or [])
            if msgs is None:
                continue
            text = render_chat(msgs)
            if len(text) > MAX_REASONING_CHARS:
                continue
            yield text, msgs
        elif src.kind == "gsm8k":
            msgs = _convert_gsm8k(row.get("question") or "", row.get("answer") or "")
            if msgs is None:
                continue
            yield render_chat(msgs), msgs
        elif src.kind == "book":
            t = _clean_book(row.get("text") or "")
            if len(t) < 2000:            # un livre nettoyé trop court = surtout du bruit
                continue
            yield t, None
        elif src.kind == "oral":
            t = (row.get("text") or "").strip()
            # [speaker001:] -> tirets de dialogue à la française
            t = re.sub(r"\[speaker\d+:?\]\s*", "\n— ", t).strip()
            if len(t) < 300:
                continue
            yield t, None


def _convert_gsm8k(question: str, answer: str) -> list[dict] | None:
    """cmh/gsm8k_fr -> format <think>.

    Le champ answer contient les étapes avec des annotations calculatrice <<a+b=c>>
    (parfois cassées par la traduction : un seul '>'), puis '#### réponse_finale'.
    """
    question = question.strip()
    if not question or "####" not in answer:
        return None
    answer = re.sub(r"<<[^<>]*>>?", "", answer)          # vire <<...>> ET <<...> cassés
    answer = re.sub(r">+(?=\d)", "", answer)             # chevrons orphelins collés à un nombre (annotations à 3 '>')
    steps, final = answer.rsplit("####", 1)
    steps, final = steps.strip(), final.strip()
    if not steps or not final:
        return None
    text = f"{THINK}\n{steps}\n{THINK_END}\nLa réponse est {final}."
    return [{"role": "user", "text": question}, {"role": "assistant", "text": text}]


_BOOK_NOISE = re.compile(
    r"gutenberg|proofread|transcrib|e-?text|etext|distributed|online|http|www\.|copyright",
    re.IGNORECASE)


def _clean_book(text: str) -> str:
    """Retire l'en-tête/pied Project Gutenberg (crédits, licences) d'un livre."""
    lines = [ln for ln in text.splitlines() if not _BOOK_NOISE.search(ln)]
    return "\n".join(lines).strip()


def _convert_dolphin(messages: list[dict]) -> list[dict] | None:
    """dolphin-r1-french -> format <think>. Les réponses contiennent
    <|begin_of_thought|>...<|end_of_thought|> puis <|begin_of_solution|>...<|end_of_solution|>."""
    out = []
    for m in messages:
        role, content = m.get("role"), (m.get("content") or "").strip()
        if role == "user":
            out.append({"role": "user", "text": content})
        elif role == "assistant":
            def between(a, b):
                i, j = content.find(a), content.find(b)
                return content[i + len(a):j].strip() if 0 <= i < j else None
            thought = between("<|begin_of_thought|>", "<|end_of_thought|>")
            sol = between("<|begin_of_solution|>", "<|end_of_solution|>")
            if not sol:
                return None
            text = f"{THINK}\n{thought}\n{THINK_END}\n{sol}" if thought else sol
            out.append({"role": "assistant", "text": text})
    return out if len(out) >= 2 else None


# --------------------------------------------------------------------------------------
# Étape 1 : téléchargement -> jsonl brut
# --------------------------------------------------------------------------------------
def download_source(src_name: str, char_budget: int, out_path: Path, seed: int = 0) -> dict:
    from tqdm import tqdm

    # sources synthétiques : générées localement, pas de téléchargement
    if src_name == "maths":
        from frlm import synth
        print(f"  {src_name:8s} : génération locale ({char_budget/1e6:.0f}M chars)…")
        return synth.write_jsonl(out_path, char_budget, seed=seed, mode="pretrain")
    if src_name == "maths_sft":
        from frlm import synth
        print(f"  {src_name:9s} : génération locale ({char_budget/1e6:.0f}M chars)…")
        return synth.write_jsonl(out_path, char_budget, seed=seed + 1, mode="sft")

    src = SOURCES[src_name]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_docs, n_chars = 0, 0
    t0 = time.time()

    with out_path.open("w", encoding="utf-8") as f, tqdm(
        total=char_budget, unit="c", unit_scale=True, desc=f"  {src_name:8s}", ncols=90
    ) as bar:
        for text, msgs in _iter_source(src, seed=seed):
            rec = {"t": text}
            if msgs is not None:
                rec["m"] = msgs
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_docs += 1
            n_chars += len(text)
            bar.update(len(text))
            if n_chars >= char_budget:
                break

    return {"source": src_name, "docs": n_docs, "chars": n_chars, "seconds": round(time.time() - t0, 1)}


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


# --------------------------------------------------------------------------------------
# Étape 2 : tokenizer BPE
# --------------------------------------------------------------------------------------
def train_tokenizer(jsonl_paths: list[Path], vocab_size: int, out_file: Path, sample_chars: int = 400_000_000):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

    tok = Tokenizer(models.BPE(unk_token=None))
    # Digits(individual_digits=True) : chaque chiffre devient son propre token.
    # Sans ça, le BPE fusionne "1234" en un token opaque et l'arithmétique est morte :
    # impossible d'apprendre l'addition posée si "47" et "48" sont des symboles sans
    # rapport. C'est le choix de Llama/Qwen, et il est indispensable pour la v2.
    # ByteLevel = aucun token <unk> possible, tout octet est représentable.
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False),
    ])
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIALS,
        min_frequency=2,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    def corpus_iter():
        seen = 0
        for p in jsonl_paths:
            for rec in iter_jsonl(p):
                t = rec["t"]
                seen += len(t)
                yield t
                if seen >= sample_chars:
                    return

    tok.train_from_iterator(corpus_iter(), trainer=trainer)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_file))
    return tok


def load_tokenizer(path: str | Path):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(path))


def special_ids(tok) -> dict:
    return {
        "eot": tok.token_to_id(EOT),
        "im_start": tok.token_to_id(IM_START),
        "im_end": tok.token_to_id(IM_END),
    }


# --------------------------------------------------------------------------------------
# Étape 3 : binarisation (texte -> tableau plat d'uint16 sur disque)
# --------------------------------------------------------------------------------------
class BinWriter:
    """Écrit un flux de tokens (et optionnellement un masque de loss) en binaire brut."""

    def __init__(self, bin_path: Path, with_mask: bool = False):
        bin_path.parent.mkdir(parents=True, exist_ok=True)
        self.f = bin_path.open("wb")
        self.fm = bin_path.with_suffix(".mask").open("wb") if with_mask else None
        self.n = 0

    def write(self, ids: list[int] | np.ndarray, mask: list[int] | np.ndarray | None = None):
        arr = np.asarray(ids, dtype=DTYPE)
        self.f.write(arr.tobytes())
        if self.fm is not None:
            m = np.asarray(mask if mask is not None else np.ones(len(arr)), dtype=np.uint8)
            self.fm.write(m.tobytes())
        self.n += len(arr)

    def close(self):
        self.f.close()
        if self.fm is not None:
            self.fm.close()


def encode_pretrain(tok, jsonl_paths: list[Path], out_dir: Path, val_frac: float = 0.005,
                    batch: int = 1000, min_val_tokens: int = 32768, prefix: str = "",
                    char_caps: list[int] | None = None):
    """Tokenise un corpus plein-texte. Documents séparés par <|endoftext|>.

    prefix="mid_" produit mid_train.bin / mid_val.bin (phase de midtrain).
    char_caps : plafond de caractères par fichier (aligné sur jsonl_paths) — permet
    au midtrain de ne prendre qu'une TRANCHE des gros jsonl du pré-entraînement."""
    from tqdm import tqdm

    eot = tok.token_to_id(EOT)
    train_w = BinWriter(out_dir / f"{prefix}train.bin")
    val_w = BinWriter(out_dir / f"{prefix}val.bin")
    n_chars = 0

    buf: list[str] = []
    rng = random.Random(1234)
    source_stats: dict[str, dict] = {}

    def flush():
        nonlocal buf
        if not buf:
            return
        for enc in tok.encode_batch(buf):
            ids = enc.ids + [eot]
            # plancher sur la validation : sinon un petit corpus peut produire un
            # val.bin vide et le chargeur ne peut plus former de batch
            to_val = val_w.n < min_val_tokens or rng.random() < val_frac
            (val_w if to_val else train_w).write(ids)
        buf = []

    for i, p in enumerate(jsonl_paths):
        cap = char_caps[i] if char_caps else None
        file_chars = 0
        tokens_before = train_w.n + val_w.n
        for rec in tqdm(iter_jsonl(p), desc=f"  tokenize {p.stem:10s}", unit="doc", ncols=90):
            buf.append(rec["t"])
            n_chars += len(rec["t"])
            file_chars += len(rec["t"])
            if len(buf) >= batch:
                flush()
            if cap is not None and file_chars >= cap:
                break
        flush()
        source_stats[p.stem] = {
            "chars": file_chars,
            "tokens": train_w.n + val_w.n - tokens_before,
            "char_cap": cap,
        }

    n_train, n_val = train_w.n, val_w.n
    train_w.close()
    val_w.close()
    return {"train_tokens": n_train, "val_tokens": n_val, "chars": n_chars,
            "chars_per_token": round(n_chars / max(1, n_train + n_val), 3),
            "sources": source_stats}


_THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_WORDS_RE = re.compile(r"[\wÀ-ÿ]+", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def _prompt_fingerprint(messages: list[dict]) -> bytes:
    users = [str(m.get("text") or m.get("content") or "")
             for m in messages if m.get("role") == "user"]
    normalized = _WS_RE.sub(" ", "\n".join(users)).strip().casefold()
    return hashlib.blake2b(normalized.encode("utf-8"), digest_size=12).digest()


def _repetition_ratio(text: str, n: int = 4) -> float:
    words = [w.casefold() for w in _WORDS_RE.findall(text)]
    if len(words) < 80:
        return 0.0
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - len(set(grams)) / max(1, len(grams))


def _prepare_sft_messages(record: dict, source: str) -> tuple[list[dict] | None, bytes | None, str | None]:
    """Nettoie un exemple et raccourcit les traces de raisonnement trop longues.

    Les longues traces Dolphin ne sont pas jetées : on garde leur réponse finale,
    mais avec un bloc think vide ajouté plus tard par ``chat_segments``. Le modèle
    profite ainsi de la couverture instructionnelle sans apprendre les monologues
    qui ont provoqué les ``think-fleuves`` de la première v4.
    """
    cfg = SFT_RECIPE.get(source)
    messages = record.get("m")
    if cfg is None or not isinstance(messages, list):
        return None, None, "messages_absents"
    clean = []
    user_chars = 0
    stripped_think = False
    n_user = n_assistant = 0

    for message in messages:
        role = message.get("role")
        content = str(message.get("text") or message.get("content") or "").strip()
        if role not in ("system", "user", "assistant") or not content:
            continue
        if role == "user":
            n_user += 1
            user_chars += len(content)
        elif role == "assistant":
            n_assistant += 1
            opens = content.lower().count(THINK)
            closes = content.lower().count(THINK_END)
            blocks = _THINK_BLOCK_RE.findall(content)
            if opens != closes or opens != len(blocks):
                return None, None, "balises_think_invalides"
            final = _THINK_BLOCK_RE.sub("", content).strip()
            if not final:
                return None, None, "reponse_finale_vide"
            think_chars = sum(len(block.strip()) for block in blocks)
            if blocks and think_chars > cfg["max_think"]:
                content = final
                stripped_think = True
            if len(final) > cfg["max_final"]:
                return None, None, "reponse_trop_longue"
            if _repetition_ratio(final) > 0.35:
                return None, None, "repetition"
        clean.append({"role": role, "text": content})

    if n_user == 0 or n_assistant == 0:
        return None, None, "tour_manquant"
    if user_chars > cfg["max_prompt"]:
        return None, None, "prompt_trop_long"
    return clean, _prompt_fingerprint(clean), "think_retire" if stripped_think else None


def _allocate_sft_targets(source_reports: dict[str, dict], target_supervised: int) -> dict[str, int]:
    weights = {s: SFT_RECIPE[s]["weight"] for s in source_reports}
    weight_sum = sum(weights.values()) or 1.0
    caps = {s: source_reports[s]["train_supervised"] * SFT_RECIPE[s]["max_repeat"]
            for s in source_reports}
    alloc = {s: min(caps[s], round(target_supervised * weights[s] / weight_sum))
             for s in source_reports}

    # Redistribue les quotas plafonnés (surtout le petit distillat) sans dépasser
    # la répétition maximale de chaque source.
    for _ in range(20):
        missing = target_supervised - sum(alloc.values())
        candidates = [s for s in alloc if alloc[s] < caps[s]]
        if missing <= 0 or not candidates:
            break
        wsum = sum(weights[s] for s in candidates) or 1.0
        progressed = 0
        for source in candidates:
            add = min(caps[source] - alloc[source],
                      max(1, round(missing * weights[source] / wsum)))
            alloc[source] += add
            progressed += add
        if progressed == 0:
            break
    return alloc


def _copy_masked_shard(token_path: Path, mask_path: Path, writer: BinWriter,
                       target_supervised: int, source_supervised: int,
                       rotation: int = 0) -> tuple[int, int]:
    """Copie/répète un shard jusqu'au quota supervisé, par blocs sans charger en RAM."""
    if target_supervised <= 0 or source_supervised <= 0:
        return 0, 0
    tokens = np.memmap(token_path, dtype=DTYPE, mode="r")
    masks = np.memmap(mask_path, dtype=np.uint8, mode="r")
    copied_tokens = copied_sup = 0
    chunk_size = 1_000_000
    rotation %= max(1, len(tokens))

    while copied_sup < target_supervised:
        before_pass = copied_sup
        # Une rotation propre à la source évite que les corpus sous-échantillonnés
        # gardent systématiquement leur préfixe. Elle est stable entre deux builds.
        ranges = ((rotation, len(tokens)), (0, rotation)) if rotation else ((0, len(tokens)),)
        for range_start, range_end in ranges:
            for start in range(range_start, range_end, chunk_size):
                end = min(range_end, start + chunk_size)
                m = np.asarray(masks[start:end])
                remaining = target_supervised - copied_sup
                chunk_sup = int(m.sum())
                if chunk_sup > remaining:
                    cutoff = int(np.searchsorted(np.cumsum(m, dtype=np.int64), remaining,
                                                 side="left")) + 1
                    end = start + cutoff
                    m = np.asarray(masks[start:end])
                    chunk_sup = int(m.sum())
                writer.write(np.asarray(tokens[start:end]), m)
                copied_tokens += end - start
                copied_sup += chunk_sup
                if copied_sup >= target_supervised:
                    break
            if copied_sup >= target_supervised:
                break
        if copied_sup == before_pass:  # filet contre un shard au masque vide/corrompu
            break
    del tokens, masks
    return copied_tokens, copied_sup


def encode_sft(tok, jsonl_paths: list[Path], out_dir: Path, max_len: int = 1024,
               target_supervised: int = SFT_TARGET_SUPERVISED):
    """Construit un SFT dédupliqué et pondéré par tokens de réponse assistant."""
    from tqdm import tqdm

    out_dir = Path(out_dir)
    eot = tok.token_to_id(EOT)
    paths = {p.stem: p for p in jsonl_paths if p.exists() and p.stem in SFT_RECIPE}
    ordered = [s for s in SFT_SOURCE_ORDER if s in paths]
    seen_prompts: set[bytes] = set()
    source_reports: dict[str, dict] = {}

    with tempfile.TemporaryDirectory(prefix=".sft-build-", dir=out_dir) as tmp_name:
        tmp = Path(tmp_name)
        for source in ordered:
            train_w = BinWriter(tmp / f"{source}_train.bin", with_mask=True)
            val_w = BinWriter(tmp / f"{source}_val.bin", with_mask=True)
            stats: dict[str, int] = {"read": 0, "kept": 0, "duplicates": 0,
                                     "train_conversations": 0, "val_conversations": 0,
                                     "train_supervised": 0, "val_supervised": 0}
            rejected: dict[str, int] = {}
            for record in tqdm(iter_jsonl(paths[source]), desc=f"  sft {source:14s}",
                               unit="conv", ncols=90):
                stats["read"] += 1
                messages, fingerprint, note = _prepare_sft_messages(record, source)
                if messages is None or fingerprint is None:
                    rejected[note or "invalide"] = rejected.get(note or "invalide", 0) + 1
                    continue
                if fingerprint in seen_prompts:
                    stats["duplicates"] += 1
                    continue
                seen_prompts.add(fingerprint)

                ids: list[int] = []
                mask: list[int] = []
                for text, learn in chat_segments(messages, ensure_think=True):
                    enc = tok.encode(text).ids
                    ids += enc
                    mask += [1 if learn else 0] * len(enc)
                supervised = sum(mask)
                if len(ids) > max_len or supervised == 0:
                    rejected["trop_de_tokens"] = rejected.get("trop_de_tokens", 0) + 1
                    continue
                ids.append(eot)
                mask.append(0)
                # Split stable par contenu : changer l'ordre des sources ne déplace
                # jamais un exemple entre train et validation.
                to_val = int.from_bytes(fingerprint[:8], "little") % 1000 < 5
                writer = val_w if to_val else train_w
                writer.write(ids, mask)
                key = "val" if to_val else "train"
                stats[f"{key}_conversations"] += 1
                stats[f"{key}_supervised"] += supervised
                stats["kept"] += 1
                if note:
                    rejected[note] = rejected.get(note, 0) + 1
            train_w.close()
            val_w.close()
            stats["train_tokens_unique"] = (tmp / f"{source}_train.bin").stat().st_size // 2
            stats["val_tokens_unique"] = (tmp / f"{source}_val.bin").stat().st_size // 2
            stats["rejected_or_transformed"] = rejected
            source_reports[source] = stats

        allocations = _allocate_sft_targets(source_reports, int(target_supervised))
        train_w = BinWriter(out_dir / "sft_train.bin", with_mask=True)
        val_w = BinWriter(out_dir / "sft_val.bin", with_mask=True)
        for source in ordered:
            stats = source_reports[source]
            rotation = int.from_bytes(hashlib.sha256(source.encode()).digest()[:8], "little")
            train_tok, train_sup = _copy_masked_shard(
                tmp / f"{source}_train.bin", tmp / f"{source}_train.mask", train_w,
                allocations[source], stats["train_supervised"], rotation=rotation)
            val_target = min(SFT_VAL_SUPERVISED_PER_SOURCE, stats["val_supervised"])
            val_tok, val_sup = _copy_masked_shard(
                tmp / f"{source}_val.bin", tmp / f"{source}_val.mask", val_w,
                val_target, stats["val_supervised"])
            stats.update({"target_supervised": allocations[source],
                          "mixed_train_tokens": train_tok, "mixed_train_supervised": train_sup,
                          "mixed_val_tokens": val_tok, "mixed_val_supervised": val_sup,
                          "effective_repeat": round(train_sup / max(1, stats["train_supervised"]), 2)})
        train_tokens, val_tokens = train_w.n, val_w.n
        train_w.close()
        val_w.close()

    return {
        "recipe": "v4.1-quality",
        "target_supervised_tokens": int(target_supervised),
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "supervised_tokens": sum(s["mixed_train_supervised"] for s in source_reports.values()),
        "val_supervised_tokens": sum(s["mixed_val_supervised"] for s in source_reports.values()),
        "sources": source_reports,
    }


# --------------------------------------------------------------------------------------
# Étape 4 : chargeur de batchs
# --------------------------------------------------------------------------------------
class BinCorpus:
    """Corpus tokenisé en mémoire virtuelle (np.memmap) + échantillonnage déterministe.

    Déterministe = le batch du step N ne dépend QUE de (seed, N). Donc reprendre un
    entraînement au step N redonne exactement la même suite de batchs : pas de
    doublon, pas de trou, sans avoir à sauvegarder l'état du dataloader.
    """

    def __init__(self, bin_path: str | Path, seq_len: int, with_mask: bool = False):
        self.path = Path(bin_path)
        self.tokens = np.memmap(self.path, dtype=DTYPE, mode="r")
        self.mask = None
        mask_path = self.path.with_suffix(".mask")
        if with_mask and mask_path.exists():
            self.mask = np.memmap(mask_path, dtype=np.uint8, mode="r")
        self.seq_len = seq_len
        self.n_tokens = len(self.tokens)
        if self.n_tokens < seq_len + 1:
            raise ValueError(f"{bin_path} ne contient que {self.n_tokens} tokens (< seq_len+1)")
        self.doc_starts = None
        if self.mask is not None:
            # En SFT, démarrer chaque fenêtre au début d'une conversation évite de
            # superviser une réponse dont le prompt se trouve avant la fenêtre.
            # EOT vaut 0 par invariant du tokenizer ; le scan par blocs borne la RAM.
            starts = [np.array([0], dtype=np.int64)]
            chunk_size = 10_000_000
            for start in range(0, self.n_tokens, chunk_size):
                chunk = np.asarray(self.tokens[start:start + chunk_size])
                positions = np.flatnonzero(chunk == 0).astype(np.int64) + start + 1
                starts.append(positions[positions < self.n_tokens - self.seq_len - 1])
            self.doc_starts = np.concatenate(starts)

    def __len__(self):
        return self.n_tokens

    def get_batch(self, step: int, batch_size: int, seed: int = 1337, device: str = "cuda"):
        import torch

        rng = np.random.default_rng(seed * 1_000_003 + step)
        hi = self.n_tokens - self.seq_len - 1
        if self.doc_starts is not None:
            offsets = rng.choice(self.doc_starts, size=batch_size)
        else:
            offsets = rng.integers(0, hi, size=batch_size)

        x = np.stack([self.tokens[o: o + self.seq_len] for o in offsets]).astype(np.int64)
        y = np.stack([self.tokens[o + 1: o + 1 + self.seq_len] for o in offsets]).astype(np.int64)
        xt = torch.from_numpy(x)
        yt = torch.from_numpy(y)
        mt = None
        if self.mask is not None:
            m = np.stack([self.mask[o + 1: o + 1 + self.seq_len] for o in offsets]).astype(np.uint8)
            mt = torch.from_numpy(m)

        if device.startswith("cuda"):
            xt = xt.pin_memory().to(device, non_blocking=True)
            yt = yt.pin_memory().to(device, non_blocking=True)
            if mt is not None:
                mt = mt.pin_memory().to(device, non_blocking=True)
        else:
            xt, yt = xt.to(device), yt.to(device)
            if mt is not None:
                mt = mt.to(device)
        return xt, yt, mt


# --------------------------------------------------------------------------------------
# Orchestration : la commande `prepare`
# --------------------------------------------------------------------------------------
def prepare_all(data_dir: Path, target_tokens: int, vocab_size: int, mix: dict[str, float],
                sft: bool = True, chars_per_token: float = 3.6, max_seq_len: int = 1024,
                skip_download: bool = False, seed: int = 0,
                mid_frac: float = 0.2,
                sft_target_supervised: int = SFT_TARGET_SUPERVISED) -> dict:
    """Prépare les trois phases : pretrain, midtrain (mid_frac du budget), SFT.

    mid_frac=0 désactive le midtrain. Le midtrain réutilise les jsonl déjà
    téléchargés (revoir wiki/livres pendant le recuit est voulu — c'est un recuit,
    pas de la donnée neuve) + une fournée FRAÎCHE de maths synthétiques (autre graine).
    """
    data_dir = Path(data_dir)
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"target_tokens": target_tokens, "vocab_size": vocab_size, "mix": mix}
    n_steps = 5 if (sft and mid_frac > 0) else 4

    char_budget = int(target_tokens * chars_per_token)
    print(f"\n[1/{n_steps}] Téléchargement (~{char_budget/1e9:.2f} G caractères visés pour ~{target_tokens/1e6:.0f}M tokens)")
    paths = []
    dl = []
    for name, w in mix.items():
        p = raw_dir / f"{name}.jsonl"
        budget = int(char_budget * w)
        if skip_download and p.exists():
            print(f"  {name:8s} : déjà là ({p.stat().st_size/1e6:.0f} Mo), on garde")
        else:
            dl.append(download_source(name, budget, p, seed=seed))
        paths.append(p)
    report["download"] = dl

    print(f"\n[2/{n_steps}] Entraînement du tokenizer BPE ({vocab_size} tokens, chiffres séparés)")
    tok_path = data_dir / "tokenizer.json"
    tok = train_tokenizer(paths, vocab_size, tok_path)
    report["tokenizer"] = {"path": str(tok_path), "vocab_size": tok.get_vocab_size(), **special_ids(tok)}

    print(f"\n[3/{n_steps}] Binarisation du pré-entraînement")
    report["pretrain"] = encode_pretrain(tok, paths, data_dir)

    if sft:
        # Préparer le SFT avant le mid rend alpaca/reasoning/gsm8k disponibles pour
        # le shard instructionnel du mid. L'ordre de préparation ne change pas
        # l'ordre d'entraînement : pretrain -> mid -> SFT reste obligatoire.
        step = 4 if mid_frac > 0 else n_steps
        print(f"\n[{step}/{n_steps}] Sources de dialogue + raisonnement, binarisation avec masque de loss")
        build_sft(tok, data_dir, mix, max_seq_len, sft_target_supervised,
                  skip_download=skip_download, seed=seed, report=report)

    if mid_frac > 0:
        print(f"\n[{n_steps}/{n_steps}] Corpus de midtrain (recuit dense en raisonnement)")
        build_mid(tok, data_dir, target_tokens, mid_frac, chars_per_token,
                  skip_download=skip_download, seed=seed, report=report)
    (data_dir / "meta.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def build_mid_instruct(raw_dir: Path, out_path: Path, char_budget: int) -> dict:
    """Crée un shard ChatML propre, dédupliqué et assez grand pour le midtrain."""
    seen: set[bytes] = set()
    counts: dict[str, int] = {}
    rejected: dict[str, int] = {}
    n_chars = n_docs = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as stream:
        for source in MID_INSTRUCT_SOURCES:
            path = raw_dir / f"{source}.jsonl"
            if not path.exists():
                continue
            for record in iter_jsonl(path):
                messages, fingerprint, note = _prepare_sft_messages(record, source)
                if messages is None or fingerprint is None:
                    rejected[note or "invalide"] = rejected.get(note or "invalide", 0) + 1
                    continue
                if fingerprint in seen:
                    rejected["doublon"] = rejected.get("doublon", 0) + 1
                    continue
                seen.add(fingerprint)
                text = render_chat(messages)
                if len(text) > 6000:
                    rejected["document_trop_long"] = rejected.get("document_trop_long", 0) + 1
                    continue
                stream.write(json.dumps({"t": text}, ensure_ascii=False) + "\n")
                counts[source] = counts.get(source, 0) + 1
                n_docs += 1
                n_chars += len(text)
                if n_chars >= char_budget:
                    return {"docs": n_docs, "chars": n_chars, "sources": counts,
                            "rejected_or_transformed": rejected}
    return {"docs": n_docs, "chars": n_chars, "sources": counts,
            "rejected_or_transformed": rejected}


def build_mid(tok, data_dir: Path, target_tokens: int, mid_frac: float,
              chars_per_token: float = 3.6, skip_download: bool = False,
              seed: int = 0, report: dict | None = None) -> dict:
    """Construit mid_train.bin / mid_val.bin selon MID_MIX."""
    report = report if report is not None else {}
    raw_dir = Path(data_dir) / "raw"
    mid_budget = int(target_tokens * mid_frac * chars_per_token)
    mid_paths, mid_caps = [], []
    for name, w in MID_MIX.items():
        budget_src = int(mid_budget * w)
        if name == "maths":
            # fournée fraîche : autre graine que le pretrain, donc problèmes inédits
            p = raw_dir / "maths_mid.jsonl"
            if not (skip_download and p.exists()):
                from frlm import synth
                synth.write_jsonl(p, budget_src, seed=seed + 101, mode="pretrain")
        elif name == "instruct":
            p = raw_dir / "mid_instruct.jsonl"
            stats = build_mid_instruct(raw_dir, p, budget_src)
            report["mid_instruct"] = stats
            if stats["chars"] < budget_src:
                print(f"  instruct : seulement {stats['chars']/1e6:.1f}M / "
                      f"{budget_src/1e6:.1f}M caractères propres disponibles")
        else:
            p = raw_dir / f"{name}.jsonl"
            if not p.exists():
                dl_mid = download_source(name, budget_src, p, seed=seed + 7)
                report.setdefault("download_mid", []).append(dl_mid)
        mid_paths.append(p)
        mid_caps.append(budget_src)
    report["midtrain"] = encode_pretrain(tok, mid_paths, Path(data_dir), prefix="mid_",
                                         min_val_tokens=16384, char_caps=mid_caps)
    return report


def build_sft(tok, data_dir: Path, mix: dict[str, float] | None = None,
              max_seq_len: int = 1024,
              target_supervised: int = SFT_TARGET_SUPERVISED,
              skip_download: bool = False,
              seed: int = 0, report: dict | None = None) -> dict:
    """Télécharge (si besoin) les sources SFT et construit sft_train.bin / sft_val.bin."""
    report = report if report is not None else {}
    mix = mix or {}
    raw_dir = Path(data_dir) / "raw"
    for name, budget in SFT_BUDGETS.items():
        p = raw_dir / f"{name}.jsonl"
        if name in mix:
            continue                     # déjà téléchargé pour le pré-entraînement
        if skip_download and p.exists():
            print(f"  {name:9s} : déjà là ({p.stat().st_size/1e6:.0f} Mo), on garde")
        else:
            report.setdefault("download_sft", []).append(download_source(name, budget, p, seed=seed))
    chat_paths = [raw_dir / f"{n}.jsonl"
                  for n in ("chat", "alpaca", "reasoning", "gsm8k", "maths_sft", "distill")
                  if (raw_dir / f"{n}.jsonl").exists()]
    if not chat_paths:
        print("  (aucune source de dialogue téléchargée — SFT ignoré)")
    else:
        report["sft"] = encode_sft(tok, chat_paths, Path(data_dir), max_len=max_seq_len,
                                   target_supervised=target_supervised)
    return report


def rebin_mid_sft(data_dir: Path, mid_frac: float = 0.2, max_seq_len: int = 1024,
                  seed: int = 0,
                  sft_target_supervised: int = SFT_TARGET_SUPERVISED) -> dict:
    """Re-binarise UNIQUEMENT mid + SFT avec le tokenizer EXISTANT.

    À utiliser quand les recettes (MID_MIX, sur-échantillonnage, nouveau distill.jsonl)
    changent après le pré-entraînement : réentraîner le tokenizer invaliderait le
    checkpoint (les ids changeraient), ici on ne touche ni au tokenizer ni aux bins
    du pretrain. target_tokens est relu depuis meta.json pour garder le même budget.
    """
    data_dir = Path(data_dir)
    tok_path = data_dir / "tokenizer.json"
    if not tok_path.exists():
        raise SystemExit(f"[!] {tok_path} introuvable — lance d'abord un prepare complet.")
    tok = load_tokenizer(tok_path)
    meta = {}
    meta_path = data_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    target_tokens = int(meta.get("target_tokens") or 300e6)
    cpt = float((meta.get("pretrain") or {}).get("chars_per_token") or 3.6)

    report: dict = {}
    print(f"[1/2] Re-binarisation du midtrain (target {target_tokens/1e6:.0f}M tok, "
          f"{cpt} c/tok, tokenizer conservé)")
    build_mid(tok, data_dir, target_tokens, mid_frac, chars_per_token=cpt,
              skip_download=True, seed=seed, report=report)
    print("\n[2/2] Re-binarisation du SFT")
    build_sft(tok, data_dir, mix=meta.get("mix"), max_seq_len=max_seq_len,
              target_supervised=sft_target_supervised,
              skip_download=True, seed=seed, report=report)

    meta.update(report)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
