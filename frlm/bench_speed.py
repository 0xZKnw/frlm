# --------------------------------------------------------------------------------------
# Banc de vitesse : mesure le débit d'ENTRAÎNEMENT réel (tok/s, ms/step, VRAM, MFU)
# d'un ou plusieurs presets, v2 ou v3, dans les conditions exactes de run.py :
# bf16 autocast, Muon + AdamW, grad clip, torch.compile.
#
# À lancer AVANT de s'engager sur 8 h : 5 minutes suffisent pour choisir la
# géométrie. Données aléatoires (le débit ne dépend pas du contenu).
#
# Usage :
#   python -m frlm.bench_speed                          # mini (v2) vs v3-mini vs v3-base
#   python -m frlm.bench_speed --presets v3-base --batch-size 24
#   python -m frlm.bench_speed --no-compile             # pour isoler l'effet compile
# --------------------------------------------------------------------------------------
import argparse
import time
from types import SimpleNamespace

import torch

from frlm.model import PRESETS, ModelConfig, build_model
from frlm.model_v3 import PRESETS_V3, ModelConfigV3, build_model_v3
from frlm.optim import build_optimizers


def construire(preset: str, seq_len: int, vocab: int):
    if preset in PRESETS_V3:
        cfg = ModelConfigV3(**PRESETS_V3[preset])
        cfg.vocab_size = vocab
        cfg.max_seq_len = max(cfg.max_seq_len, seq_len)
        return build_model_v3(cfg), cfg
    cfg = ModelConfig(**PRESETS[preset])
    cfg.vocab_size = vocab
    cfg.max_seq_len = max(cfg.max_seq_len, seq_len)
    return build_model(cfg), cfg


def mesurer(preset: str, a, batch_size: int, grad_accum: int) -> dict:
    device = "cuda"
    torch.manual_seed(1337)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model, cfg = construire(preset, a.seq_len, a.vocab_size)
    model = model.to(device)
    n_params = model.num_params()
    n_ne = model.num_params(non_embedding=True)
    fpt = model.flops_per_token()

    opt_args = SimpleNamespace(optimizer="muon", lr=0.02, adam_lr=1.5e-3,
                               beta1=0.9, beta2=0.95, weight_decay=0.05)
    opts, _ = build_optimizers(model, opt_args)

    if a.compile:
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"  [!] compile indisponible ({e}) — eager")

    # pool de batchs aléatoires pré-chargés (la data ne doit pas fausser la mesure)
    pool = [torch.randint(0, a.vocab_size, (batch_size, a.seq_len), device=device)
            for _ in range(8)]
    tokens_step = batch_size * grad_accum * a.seq_len

    def step(i: int):
        for micro in range(grad_accum):
            idx = pool[(i * grad_accum + micro) % len(pool)]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss, _ = model(idx[:, :-1], idx[:, 1:], z_loss=1e-4, diagnostics=False)
            (loss / grad_accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for opt in opts:
            opt.step()
        for opt in opts:
            opt.zero_grad(set_to_none=True)

    print(f"\n=== {preset} : {n_params/1e6:.1f}M params ({n_ne/1e6:.1f}M hors emb.) "
          f"· {model._orig_mod.describe() if hasattr(model, '_orig_mod') else model.describe()} ===")
    t0 = time.time()
    for i in range(a.warmup):
        step(i)
    torch.cuda.synchronize()
    print(f"  warmup {a.warmup} steps (compile inclus) : {time.time()-t0:.0f}s")

    t0 = time.time()
    for i in range(a.steps):
        step(i)
    torch.cuda.synchronize()
    dt = time.time() - t0

    tok_s = tokens_step * a.steps / dt
    tflops = fpt * tok_s / 1e12
    mfu = tflops / a.gpu_peak_tflops
    vram = torch.cuda.max_memory_allocated() / 1e9
    res = {"preset": preset, "params": n_params, "non_emb": n_ne, "tok_s": tok_s,
           "ms_step": dt / a.steps * 1000, "tflops": tflops, "mfu": mfu, "vram": vram,
           "tokens_8h": tok_s * 8 * 3600, "bs": batch_size, "ga": grad_accum}
    print(f"  {tok_s/1e3:6.1f}k tok/s · {res['ms_step']:6.0f} ms/step · "
          f"{tflops:5.2f} TFLOPS · MFU {mfu:5.1%} · pic VRAM {vram:.2f} Go")
    print(f"  -> en 8 h : {res['tokens_8h']/1e9:.2f}B tokens "
          f"({res['tokens_8h']/n_ne:.0f} tok par param hors emb.)")

    del model, opts, pool
    torch.cuda.empty_cache()
    return res


def mesurer_avec_repli(preset: str, a) -> dict:
    """Filet anti-OOM : divise le batch (et double l'accum — tokens/step constants)
    jusqu'à ce que ça tienne. Un bench lancé sur du GPU payant DOIT rendre un chiffre."""
    bs, ga = a.batch_size, a.grad_accum
    while True:
        try:
            res = mesurer(preset, a, bs, ga)
            if bs != a.batch_size:
                print(f"  [i] mesuré à bs {bs} × accum {ga} (repli OOM depuis bs {a.batch_size})")
            return res
        except torch.cuda.OutOfMemoryError:
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            try:
                torch._dynamo.reset()   # purge les artefacts compile de la tentative ratée
            except Exception:
                pass
            if bs <= 2:
                raise
            bs, ga = bs // 2, ga * 2
            print(f"  [!] OOM — repli automatique : bs {bs} × accum {ga} "
                  f"(mêmes tokens/step), nouvelle tentative...")


def main():
    ap = argparse.ArgumentParser(description="Banc de vitesse d'entraînement v2/v3")
    ap.add_argument("--presets", nargs="+", default=["mini", "v3-mini", "v3-base"],
                    choices=list(PRESETS) + list(PRESETS_V3))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--vocab-size", type=int, default=16384)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=12)
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    ap.add_argument("--gpu-peak-tflops", type=float, default=30.0)
    a = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("[!] CUDA indisponible — ce banc mesure le débit GPU.")
    print(f"[i] batch {a.batch_size} × {a.grad_accum} accum × {a.seq_len} tok · "
          f"compile {'ON' if a.compile else 'OFF'} · pic GPU {a.gpu_peak_tflops:.0f} TFLOPS")

    resultats = [mesurer_avec_repli(p, a) for p in a.presets]

    print("\n" + "=" * 78)
    print(f"{'preset':<10} {'params':>8} {'bs×ga':>7} {'tok/s':>9} {'MFU':>7} {'VRAM':>7} {'tokens/8h':>11}")
    for r in resultats:
        print(f"{r['preset']:<10} {r['params']/1e6:>7.1f}M {r['bs']:>4}×{r['ga']:<2} {r['tok_s']/1e3:>8.1f}k "
              f"{r['mfu']:>6.1%} {r['vram']:>5.2f}Go {r['tokens_8h']/1e9:>10.2f}B")


if __name__ == "__main__":
    main()
