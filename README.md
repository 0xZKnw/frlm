# frlm

**A 58M-parameter French reasoning LLM, trained from scratch in one afternoon on an
RTX 4060 (8 GB).**

frlm speaks French, works through problems in a `<think>…</think>` scratchpad before
answering, and beats the public French GPT-2s at 124M (2× its size) on both language
quality **and** arithmetic — for ~8 hours of gaming-GPU time total.

> **v4 experimental release (August 2026).** The repository now also contains the
> 229M-parameter `v4-base` speedrun architecture, a 24k digit-split tokenizer, a
> 3B-token data recipe, filtered distillation, improved GRPO, file-mediated RLAIF,
> secret-family OOD benchmarks, and Modal/Beam runners. The original 58M model and
> results documented below remain the stable published baseline.

```
you   › Léa a 15 pommes. Elle en mange 7 et en récupère 3. Combien de pommes a-t-elle ?
model › <think>
        Au départ : 15 pommes.
        15 − 7 = 8
        8 + 3 = 11
        </think>
        Léa a maintenant 11 pommes.
```

**Qwen3.5-style** architecture (2026), **Muon** optimizer, **WSD** schedule, custom
digit-split BPE tokenizer, full 3-stage pipeline (pretrain → midtrain → SFT), a live
stats dashboard, and a chat mode to poke the model without stopping training.

---

## v4 (experimental)

v4 is a larger research iteration focused on learning efficiency and post-training,
not a replacement claim for the carefully audited v1 result. Its main additions are:

- `model_v3.py`: Canon layers, value embeddings, U-net skips, ReLU² MLPs,
  3:1 sliding/global attention, logit softcap, and v3/v4 presets;
- a 3B-token French recipe with a 24,576-token digit-split tokenizer and new
  synthetic families for time, fractions, money, inverse problems, and packing;
- sequence-level distillation, DAPO/Dr. GRPO refinements, verifiable instructions,
  and an RLAIF stage with anti-rambling probes;
- `bench_ood_v2.py`, whose secret problem families are kept out of training, plus
  reports for pretrain, SFT, and RLAIF checkpoints;
- reproducible GPU throughput runners for local CUDA, Modal, and Beam.

The latest checked-in report for the RLAIF2 checkpoint scores **4/40** on the
strict secret-family OOD benchmark and **9/12** elementary facts. These modest
numbers are published intentionally: v4 is an experiment with useful infrastructure
and documented failure modes, not a benchmark victory. See
[`bench/reports/bench_ood_v2_fr-v4_rlaif2_latest.md`](bench/reports/bench_ood_v2_fr-v4_rlaif2_latest.md).

The Git repository includes the v4 code, tokenizer, data metadata, RLAIF prompt
pools, and benchmark reports. It deliberately excludes raw corpora, tokenized
binaries, masks, run logs, judge exchanges, optimizer states, and multi-gigabyte
checkpoints. Regenerate data with `python run.py prepare`, or use your own compatible
checkpoint under `runs/<name>/<phase>/ckpt_latest.pt` or `ckpt_best.pt`.

Example v4 commands:

```bash
python run.py prepare --data-dir data-v4 --target-tokens 3e9 --vocab-size 24576 --seq-len 2048 \
  --mid-frac 0.12 --sft-target-supervised 50e6 \
  --mix "fineweb:0.55,wiki:0.15,maths:0.15,books:0.06,theses:0.04,chat:0.03,europarl:0.01,oral:0.01"
python run.py train --data-dir data-v4 --run fr-v4 --preset v4-base --seq-len 2048
python run.py mid --data-dir data-v4 --run fr-v4 --preset v4-base --seq-len 2048
python run.py sft --data-dir data-v4 --run fr-v4 --preset v4-base --seq-len 2048
python run.py rl --run fr-v4
python run.py rlaif --run fr-v4
python bench/bench_ood_v2.py --run fr-v4 --data-dir data-v4 --hf none
```

### v4.1 post-training recipe

The audited v4.1 recipe removes duplicate prompts across all SFT sources, rejects
broken or highly repetitive answers, strips overlong reasoning traces while keeping
their final answers, and balances sources by **assistant tokens** instead of JSONL
file size. SFT batches now start at conversation boundaries, so a supervised answer
never loses its prompt at the left edge of a random training window. The midtrain is
a gentler 12% annealing pass: 30% fresh synthetic math, 12% clean deduplicated
instructions, and 58% natural French. Run the read-only source audit with:

```bash
python -m frlm.audit_data --data-dir data-v4
```

To rebuild only the derived mid/SFT files without touching the tokenizer or the
pretrain binaries/checkpoint:

```bash
python run.py prepare --data-dir data-v4 --rebin --skip-download --mid-frac 0.12 \
  --seq-len 2048 --sft-target-supervised 50e6
```

Use a new run name so the first v4 post-training remains intact. On an H100, a
conservative reproducible launch is:

```bash
modal run --detach modal_app.py --gpu h100 --spawn --cmd \
  "python run.py mid --data-dir data-v4 --run fr-v4-v41 --preset v4-base --seq-len 2048 --batch-size 16 --grad-accum 2 --max-steps 6000 --optimizer muon --lr 0.004 --adam-lr 0.0002 --schedule cosine --warmup 50 --eval-every 100 --resume /vol/runs/fr-v4/pretrain/ckpt_latest.pt"

modal run --detach modal_app.py --gpu h100 --spawn --cmd \
  "python run.py sft --data-dir data-v4 --run fr-v4-v41 --preset v4-base --seq-len 2048 --batch-size 16 --grad-accum 2 --max-steps 2500 --optimizer adamw --lr 0.00008 --weight-decay 0.01 --schedule cosine --warmup 50 --eval-every 100 --resume latest"
```

Upload only `mid_train.bin`, `mid_val.bin`, the four `sft_*.bin`/`sft_*.mask`
files and `meta.json` with `modal volume put --force`. The wrapper runs a CPU
preflight, reloads reused containers, validates the v4.1 metadata/file sizes, and
checks the resume checkpoint before allocating a GPU. Raw JSONL corpora and
pretrain binaries are not needed for these two Modal phases.

Use `--check-only` to validate the exact command and Volume contents without
starting a GPU function.

---

## Results

Against the closest public French causal LMs — all **2.15× its size**
([GPT-fr small](https://huggingface.co/asi/gpt-fr-cased-small),
[BelGPT-2](https://huggingface.co/antoinelouis/belgpt2),
[gpt2-french-small](https://huggingface.co/dbddv01/gpt2-french-small)):

| model | params | bpb ↓ ¹ | arithmetic ↑ ² | facts ↑ |
|---|---|---|---|---|
| **frlm (sft)** | **58M** | **1.412** | **100/100** | 5/8 |
| GPT-fr small (Inria) | 124M | 1.567 | ~1/100 | 3/8 |
| BelGPT-2 (60 GB corpus) | 124M | 1.851 | 0/100 | 5/8 |
| French GPT-2 (transfer) | 124M | 1.812 | 0/100 | 3/8 |

¹ *bits per byte* on 300k characters of held-out French — the only honest way to
compare losses across different tokenizers.
² 100 exact-answer arithmetic problems (a generation seed never used in training),
greedy decoding, scores **audited by hand** (the baselines' "lucky" points from
degenerate loops were removed). Full details: `bench/reports/bench_vs_report.md`.

### The honest benchmark

On 40 **out-of-distribution** problems (reworded phrasings, novel contexts, novel
concepts — `bench_ood.py`), frlm drops to **8/40** while the baselines get 0-1/40.
At 58M, skill is still largely indexed on phrasing: the model masters its procedures,
not yet the abstraction behind them. Both benchmarks ship with the repo, because
either one alone would tell a lie. Details: `bench/reports/bench_ood_report.md`.

| | reworded | novel context | novel concept |
|---|---|---|---|
| frlm | 3/15 | 3/15 | 2/10 |
| best 124M baseline | 0/15 | 0/15 | 1/10 |

---

## Try it (2 minutes)

Weights are in the [release](../../releases/latest): `frlm-58m-sft.pt` (chat,
recommended), `frlm-58m-base.pt` (base model), `frlm-58m-mid.pt` (after annealing),
plus the `tokenizer.json`. fp16, weights-only, ~132 MB each.

```bash
pip install -r requirements.txt
```

Lay the files out like this (the `ckpt_latest.pt` name matters):

```
runs/frlm/tokenizer.json
runs/frlm/sft/ckpt_latest.pt      <- frlm-58m-sft.pt renamed
```

Then:

```bash
python run.py chat --run frlm
```

Chat commands: `/think on|off|auto`, `/temp 0.8`, `/topp 0.95`, `/topk 50`,
`/max 300`, `/raw <text>` (raw completion, no chat template), `/reset`, `/quit`.
CPU is plenty for inference at this size.

---

## The repo

```
run.py               CLI: prepare / train / mid / sft / rl / chat / info
frlm/
  model.py           the Qwen3.5-style architecture: gated attention (GQA +
                     zero-centered QK-Norm + partial RoPE + output gate), Gated
                     DeltaNet (optional 3:1 hybrid), SwiGLU, KV-cached generation
  data.py            French corpora download, custom BPE tokenizer (digit-split),
                     binarization, reasoning-trace conversion to <think> format
  synth.py           French math/logic problem generator: 17 families, multiple
                     phrasings per concept, Python-computed solutions (never wrong)
  optim.py           Muon (Newton-Schulz orthogonalization, batched) + LR schedules
  rl.py              GRPO with verifiable rewards (see Roadmap)
bench/
  bench_vs.py        benchmark vs the public French GPT-2s (bpb, arithmetic, facts)
  bench_ood.py       out-of-distribution benchmark (40 novel hand-written problems)
  reports/           full benchmark reports with every model answer
```

---

## Reproduce frlm from scratch (~8 h on a 4060)

```bash
python run.py prepare --data-dir data-v2 --target-tokens 500e6
```

```bash
python run.py train --data-dir data-v2 --run frlm --preset deep --batch-size 12 --grad-accum 3 --max-steps 24000
```

```bash
python run.py mid --data-dir data-v2 --run frlm --preset deep --batch-size 12 --grad-accum 3
```

```bash
python run.py sft --data-dir data-v2 --run frlm --preset deep --batch-size 12 --grad-accum 3 --max-steps 3000
```

```bash
python run.py chat --run frlm
```

Measured reference points: pretrain 24,000 steps / 885M tokens / **6 h 40 min**
(val 2.30, ppl 10); midtrain 4,000 steps / **1 h 09 min** (val 1.01 on its mix);
SFT 3,000 steps / **1 h** (val 1.46). ~38.5k tokens/s, ~51% MFU, 6.6 GB peak VRAM.

**Ctrl+C = clean stop** (checkpoint written before exiting), auto-checkpoint every
5 min, `--resume` picks up at the exact step (same batch order, RNG restored).
Dropping a `STOP` file in the run directory does the same thing remotely.

### The data

| source | Hugging Face dataset | role |
|---|---|---|
| `fineweb` | `HuggingFaceFW/fineweb-2` (`fra_Latn`) | filtered French web |
| `wiki` | `wikimedia/wikipedia` (`20231101.fr`) | clean, factual French |
| `books` | `OpenLLM-France/Lucie-Training-Dataset` (`Gutenberg-fr`) | literature |
| `oral` | `OpenLLM-France/Lucie-Training-Dataset` (`Claire-fr`) | spoken French |
| `chat` | `angeluriot/french_instruct` | 275k French conversations |
| `alpaca` | `jpacifico/French-Alpaca-dataset-Instruct-110K` | French instructions |
| `reasoning` | `WiroAI/dolphin-r1-french` | 128k reasoning traces → `<think>` |
| `gsm8k` | `cmh/gsm8k_fr` | translated grade-school word problems, step by step |
| `maths` / `maths_sft` | **generated locally by `synth.py`** | French arithmetic/logic, Python-computed solutions |

The v4.1 midtrain (annealing) phase re-runs a balanced mix — 30% fresh synthetic
math, 12% clean instructions, and 58% natural French — while a low learning rate
decays. It concentrates reasoning late without sacrificing the language substrate.

---

## The four ideas that make a 58M model reason

1. **Digit-by-digit tokenizer.** A standard BPE merges `1234` into one opaque token
   and arithmetic is dead on arrival. Here `pre_tokenizers.Digits` gives every digit
   its own token: the model can carry out an addition column by column, like a human.

2. **Correct math from pretraining on.** `synth.py` generates problems whose
   solutions are **computed by Python** — zero errors in the data, unlike the web.
   17 families: column arithmetic, money, sharing, sequences, unit conversions,
   syllogisms, transitivity, parity, state tracking, groupings, parenthesized
   expressions, liquid volumes…

3. **The midtrain.** A short annealing phase on concentrated data between pretraining
   and SFT — the modern-lab recipe, at RTX 4060 scale.

4. **`<think>` discipline.** During SFT every answer opens with a thinking block —
   **filled with steps for problems, empty for small talk**. The model learns *when*
   to think, not just how: no 500-token detour to say hello, no answering a math
   problem without a scratchpad.

---

## The architecture, brick by brick

**Qwen3.5-style** ([analysis](https://huggingface.co/blog/mlabonne/qwen35),
[Qwen3-Next](https://www.alibabacloud.com/blog/602580)), minus the MoE (pointless at
this size). The `deep` preset: **16 layers × d512**, 8Q/2KV (4:1 GQA), SwiGLU 1408,
context 1024, 57.7M params of which 49.3M non-embedding.

- **Output gate** — `out * sigmoid(W_g x)` before the projection: suppresses
  attention sinks ([NeurIPS 2025](https://arxiv.org/abs/2505.06708)).
- **Zero-centered QK-Norm** — gain stored as `1 + w` with weight decay on `w`:
  pulled back toward 1, never toward 0.
- **Partial RoPE** — 16 of 64 head dims rotated, the rest position-free.
- **z-loss** (1e-4) against logit drift, **pre-norm**, no biases, tied embeddings,
  SDPA/FlashAttention.
- **Gated DeltaNet** (`--hybrid` flag): Qwen3.5's linear attention, implemented
  exactly in pure PyTorch (verified to 1e-15 against the fp64 recurrence) —
  educational, but 3.5× slower without the dedicated Triton kernels, so not the default.

Why **deep rather than wide**: at equal parameter count, each layer is roughly one
sequential computation step, and reasoning benefits more from depth. `deep` (16 × 512)
costs only 3% throughput vs `mini` (12 × 576) for 33% more layers.

### Muon + WSD

Muon orthogonalizes the momentum gradient (5 Newton-Schulz iterations, all matmuls)
instead of scaling it coefficient by coefficient: every singular direction advances
at the same speed. ~1.3-1.5× fewer steps to the same loss. Applied to the 2D matrices
in the blocks, stacked by shape so everything orthogonalizes in one call (measured
overhead: ~2% of the step). Embeddings and head stay on AdamW.

WSD (Warmup-Stable-Decay): constant LR for 80% of the run, then a drop over the last
20%. During the plateau, val loss stalls in a staircase — that's normal, the model
"orbits" the minimum — then the decay cashes it all in at once: measured **2.65 →
2.30 val over the final 4,800 steps** of pretraining. Practical bonus: during the
plateau, `--max-steps` can change without distorting the schedule.

---

## The dashboard

All stats live in the terminal (Rich), also written to
`runs/<name>/<phase>/metrics.jsonl`, with periodic generations in `samples.txt`:

| panel | contents |
|---|---|
| quality | raw/EMA loss, perplexity, bits/token, val loss, top-1 acc, prediction entropy, logit RMS |
| optimization | LR and % of max, grad norm, % clipped, weight norm, Δp/p, z-loss |
| throughput & hardware | tokens/s, time/step, MFU, data/fwd/bwd/opt breakdown, VRAM alloc/reserved/peak, GPU °C/W (with `nvidia-ml-py`) |
| progress | step, tokens seen, tokens/param, corpus epochs, ETA, checkpoints |

Health markers: entropy > 1 nat (below that the model loops), stable logit RMS (the
z-loss keeps watch), clipping < 20% at steady state, `bwd ≈ 2× fwd`, corpus
epochs < 4 (beyond that it memorizes).

---

## 8 GB / Windows notes

- **bf16** everywhere (autocast), TF32 on, no GradScaler (pointless with bf16).
- **`torch.compile` is the single biggest win**: +94% throughput and −46% VRAM
  measured. Triton on Windows: `triton-windows<3.3` (in requirements; the version
  bound matters). `--no-compile` to debug.
- ⚠️ **Do not overflow VRAM.** Windows silently spills into shared system RAM
  instead of raising OOM, and throughput collapses 3.5× (measured: `deep` at
  bs16 = 7.74 GB → 11.3k tok/s, vs bs12 = 6.06 GB → 38.4k). If throughput suddenly
  tanks, check "VRAM peak". Lower `--batch-size`, compensate with `--grad-accum`.
- Close heavy browser tabs: a browser eats 1-2 GB of VRAM.
- The benchmark GPT-2s need `pip install transformers` (only for
  `python bench/bench_vs.py` / `python bench/bench_ood.py`, run from the repo root).

---

## Limitations, honestly

A 58M model trained on 1.1B tokens is not ChatGPT:

- **yes**: correct French, short structured answers, arithmetic worked out in its
  format (100/100), readable scratchpad, stops cleanly;
- **fragile**: problems reworded outside its templates (8/40), multi-turn tracking;
- **no**: reliable factual knowledge (it once claimed bees produce milk), long
  context, anything that isn't French.

## Roadmap

- **v2.2 — in the repo**: `synth.py` now emits several phrasings per concept —
  machine symbols (`10*50 =`), parenthesized expressions, elliptical forms
  ("Le double de 16 ?"), liters/containers state-tracking, more objects. Feeds
  the RL stage below; a ~1 h re-SFT on regenerated data is the next cheap win.
- **RL (GRPO) — in the repo**: `python run.py rl` — group-relative policy
  optimization with verifiable rewards (`synth.py` knows every answer, a Python
  verifier scores each rollout), a brevity bonus against rambling scratchpads,
  and a KL anchor to the frozen SFT to preserve language quality. Live dashboard
  with %-correct and held-out OOD curves. Results published after the first full run.

## License

MIT. The listed datasets keep their respective licenses.
