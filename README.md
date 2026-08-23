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
  --mid-frac 0.12 --sft-target-supervised 60e6 \
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

### v4.2 quality-first SFT

The v4.2 SFT keeps the v4.1 mid checkpoint and rebuilds only the supervised
phase. It drops the noisy chatbot-style and long translated-reasoning sources,
keeps human/human-style French conversations, ranked French OASST2 paths,
short translation instructions, verified math, and a locally re-filtered French
OpenHermes subset. Ordinary answers are no longer forced to start with an empty
`<think>` block.

The 60M assistant-token target is a quality cap: after filtering and global
prompt deduplication, the retained mixture contains 51.46M supervised tokens in
79.39M total tokens. Training uses AdamW at `1e-5`, an effective batch of 64
sequences, 15% mid-data replay, and a macro validation loss across sources. At
1,800 steps this processes 235.93M tokens (1.03 token per model parameter), of
which roughly 200.5M come from the SFT mixture, or about 2.5 effective sweeps.

```bash
python run.py prepare --data-dir data-v4 --rebin --sft-only --skip-download \
  --seq-len 2048 --sft-target-supervised 60e6

modal run --detach modal_app.py --gpu h100 --spawn --cmd \
  "python run.py sft --data-dir data-v4 --run fr-v4-v42 --preset v4-base --seq-len 2048 --batch-size 16 --grad-accum 4 --max-steps 1800 --optimizer adamw --lr 0.00001 --weight-decay 0.01 --schedule cosine --warmup 50 --replay-frac 0.15 --eval-every 100 --eval-iters 42 --sample-every 100 --resume /vol/runs/fr-v4-v41/mid/ckpt_best.pt"
```

Only upload the tokenizer, `sft_train.bin`/mask, the global and per-source SFT
validation files, and `meta.json`. Raw JSONL files remain local and untracked.

### v4.3 1.5B-token curriculum midtrain

The v4.3 midtrain keeps the original v4 tokenizer and pretrain weights. It builds
an exact 1.5B-token, two-stage curriculum: 80% broad educational/natural French,
then 20% late reasoning upsampling. The aggregate mixture is 57.9% natural French
(FineWeb2-HQ, Wikipedia, books, theses, oral transcripts and Europarl), 35.6%
Python-verified exercises, 1.5% grounded FrenchQA/PIAF/CQuAE and 5% clean
instructions. No benchmark prompt or reserved OOD seed is read by the builder.

CQuAE is CC-BY-NC-4.0, so this recipe is suitable for the private student run
described here but must be reviewed before any commercial redistribution. Source
provenance and repetition counts are recorded in `meta.json`.

```bash
# CPU Modal : télécharge, filtre et tokenise sans louer de GPU.
modal run --detach modal_app.py --gpu cpu --cmd \
  "python run.py prepare-mid-v43 --data-dir data-v4 --target-tokens 1.5e9"

# Préflight seul, puis lancement H100 détaché.
modal run modal_app.py --check-only --gpu h100 --cmd \
  "python run.py mid --data-dir data-v4 --run fr-v4-v43 --preset v4-base --mid-curriculum v4.3 --seq-len 2048 --batch-size 16 --grad-accum 4 --max-steps 11444 --optimizer muon --lr 0.002 --adam-lr 0.0001 --schedule wsd --warmup 100 --decay-frac 0.10 --min-lr-frac 0.02 --eval-every 1000 --sample-every 1000 --save-every 1000 --ckpt-every-min 10000 --keep-last 12 --resume /vol/runs/fr-v4/pretrain/ckpt_latest.pt"

modal run --detach modal_app.py --gpu h100 --spawn --cmd \
  "python run.py mid --data-dir data-v4 --run fr-v4-v43 --preset v4-base --mid-curriculum v4.3 --seq-len 2048 --batch-size 16 --grad-accum 4 --max-steps 11444 --optimizer muon --lr 0.002 --adam-lr 0.0001 --schedule wsd --warmup 100 --decay-frac 0.10 --min-lr-frac 0.02 --eval-every 1000 --sample-every 1000 --save-every 1000 --ckpt-every-min 10000 --keep-last 12 --resume /vol/runs/fr-v4/pretrain/ckpt_latest.pt"
```

At 131,072 tokens/step, 11,444 steps process 1,499,987,968 tokens, staying
12,032 tokens below the hard budget. Evaluation and rolling checkpoints happen
every 1,000 steps; `ckpt_best.pt` and the final checkpoint are also preserved.

### v4.4 capability-balanced SFT

The v4.4 SFT replaces the v4.2 fallback allocation that accidentally produced
78.07% OpenHermes-FR. Its 18M assistant-token budget is split into six closed
capabilities: 25% native French chat, 20% student-oriented distillation, 30%
verified reasoning, 12% grounded QA, 8% constraints/calibration and 5% short
multi-turn/identity examples. A missing source creates an explicit shortfall;
it is never replaced by a different capability. OpenHermes-FR is capped at 15%,
GSM8K-FR at 3%, every source at 20%, and every record at one effective pass.

`frlm/synth_programs.py` generates fresh executable programs independently from
their French renderers. It also reserves complete renderer families for validation,
adds grounded QA and explicit unanswerable/contradictory prompts, and keeps 65-70%
of generated answers direct. Bins remain separated by capability and the trainer
samples them with explicit probabilities. The 15% anti-forgetting replay is 70%
clean pretrain and 30% v4.3 mid data.

```bash
# CPU only: build and audit the v4.4 bins without changing pretrain or mid files.
modal run --detach modal_app.py --gpu cpu --spawn --cmd \
  "python run.py prepare-sft-v44 --data-dir data-v4 --target-supervised 18e6 --seq-len 2048"

# Three 120-step pilots must be compared before the 300-450 step full SFT.
# Repeat with --lr 3e-5, 1e-4 and 3e-4; do not start all three blindly.
modal run modal_app.py --check-only --gpu h100 --cmd \
  "python run.py sft --data-dir data-v4 --run fr-v4-v44-sft-lr1e4 --preset v4-base --sft-recipe v4.4 --seq-len 2048 --batch-size 16 --grad-accum 4 --max-steps 120 --optimizer adamw --lr 1e-4 --weight-decay 0.01 --schedule cosine --warmup 20 --min-lr-frac 0.10 --replay-frac 0.15 --replay-mix mid_v43_stage1_train.bin=0.80,mid_v43_stage2_train.bin=0.20 --replay-val mid_v43_val.bin --eval-every 25 --eval-iters 48 --sample-every 25 --save-every 25 --keep-last 8 --resume /vol/runs/fr-v4-v43/mid/ckpt_latest.pt"
```

OOD v2 is run on the v4.3 mid **before** this SFT. The new curriculum deliberately
teaches several structural capabilities measured by OOD v2, so post-SFT quality
must use its held-out schema/renderer validation and a newly sealed OOD generation;
reporting OOD v2 after v4.4 as untouched generalization would be misleading.

### v4.5 audited, conversation-isolated SFT

The v4.4 pilot exposed two trainer bugs: a nominally conversation-aligned 2,048-token
window still crossed every following EOT boundary, and gradient accumulation averaged
microbatches instead of assistant tokens. v4.5 fixes both. Every SFT row now contains
one complete conversation padded with unsupervised EOT tokens, so neither attention nor
the Canon convolution can carry state between documents. Cross-entropy is summed and
normalized by the global assistant-token count of the full update; replay has its own
explicit 12% weight.

The 24M-token mix follows the second audit: 30% general response/explanation, 18%
grounded transformation, 18% verified reasoning, 12% constraints/structure, 8%
multi-turn, 6% tested short code, 5% uncertainty and 3% style/identity. Sources are
single-pass, OpenHermes-FR is capped at 12%, the overall mix targets 80% direct
answers and 18% short traces, and allocations preserve whole conversations. The H100 run starts only from the final
v4.3 MID checkpoint at step 11,444; it never reruns midtrain.

```bash
# CPU Modal: generate, filter and binarize only the new SFT.
modal run --detach modal_app.py --gpu cpu --spawn --cmd \
  "python run.py prepare-sft-v45 --data-dir data-v4 --target-supervised 24e6 --seq-len 512 --seed 451337"

# Validate Volume metadata, masks and the exact MID checkpoint without a GPU.
modal run modal_app.py --check-only --gpu h100 --cmd \
  "python run.py sft --data-dir data-v4 --run fr-v4-v45-sft --preset v4-base --sft-recipe v4.5 --seq-len 512 --batch-size 128 --grad-accum 12 --max-steps 736 --optimizer adamw --lr 2e-5 --weight-decay 0.01 --schedule cosine --warmup 30 --min-lr-frac 0.10 --replay-frac 0.12 --replay-mix mid_v43_stage1_train.bin=0.80,mid_v43_stage2_train.bin=0.20 --replay-val mid_v43_val.bin --eval-every 100 --eval-iters 48 --sample-every 100 --save-every 100 --keep-last 12 --resume /vol/runs/fr-v4-v43/mid/ckpt_latest.pt"
```

With 37.5k supervised assistant tokens per update, 736 steps expose roughly
27.6M assistant tokens, or 1.15 passes over the 24M-token corpus. Re-run
`python -m frlm.audit_sft_v45` after every corpus rebuild and adjust this value if
the measured density changes.

The H100 run completed cleanly at step 736 (578.81M padded training tokens). Its
final checkpoint is also `ckpt_best.pt`: validation macro reached 0.10197,
`verified_reasoning` 0.13704 and replay MID validation 2.08871. These internal
losses confirm that the recipe learned its supervised capabilities without visible
MID forgetting; they do not by themselves establish an OOD reasoning gain.

### v4 development observations (OOD v2)

OOD v2 reports the exact raw generations rather than only an aggregate. The automatic
score for the final v4.3 MID is 7/40; manual inspection finds one false negative on
the transitive-age question: the extractor selects `Marc`, while the generated final
sentence correctly says `Tom`. The corrected total is therefore **8/40**. No automatic
false positive was found in that run. This is a clear improvement over v4.1 MID
(3/40 automatic) and the v4.1/v4.2 SFT checkpoints (3-4/40 automatic).

The same local report also contains modern models in the same parameter class. Base
models use the benchmark few-shot prompt; the RL/instruct model uses its chat template.
Competitor values below are the report's automatic scores, while the frlm row includes
the one manually verified correction described above.

| model | stage | transitif | piège | intervalle | branches | reste | cycle | composition | OOD v2 | facts |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **frlm v4.3 · 229M** | **MID, step 11444** | **2/6** | 0/6 | 1/6 | **3/5** | 0/5 | **2/5** | 0/7 | **8/40 manual** (7 auto) | 6/12 |
| Gemma 3 · 270M | base | 2/6 | **2/6** | 0/6 | 2/5 | 0/5 | 1/5 | 0/7 | **7/40 auto** | 6/12 |
| LFM2.5 · 230M | base, 28T tokens | 2/6 | 1/6 | 1/6 | 2/5 | 0/5 | 1/5 | 1/7 | **8/40 auto** | **7/12** |
| LFM2.5 · 230M | RL/instruct, 28T tokens | **3/6** | 1/6 | 1/6 | 2/5 | 0/5 | 1/5 | **4/7** | **12/40 auto** | **7/12** |

This puts the **MID-only** frlm checkpoint level with LFM2.5 Base on the corrected
total and one point above Gemma 3 on this small benchmark, before v4.5 SFT or RL.
The 40 questions are too few for a general leaderboard claim; every answer remains
auditable in `bench/reports/bench_ood_v2_fr-v3.md` and
`bench/reports/bench_ood_v2_fr-v4-v43_mid_latest.md`.

| checkpoint | automatic OOD v2 | facts | observation |
|---|---:|---:|---|
| v4.1 MID, step 6000 | 3/40 | 8/12 | older 404M-token curriculum |
| v4.1 SFT, best step 500 | 4/40 | 8/12 | modest instruction gain, weak transfer |
| v4.1 SFT, step 2500 | 3/40 | 6/12 | longer SFT regresses both OOD and facts |
| v4.2 SFT, best step 1700 | 4/40 | 8/12 | cleaner text, no OOD breakthrough |
| v4.3 MID, step 4000 | 7/40 | 9/12 | best factual checkpoint in this sweep |
| v4.3 MID, final step 11444 | 7/40 (**8/40 manual**) | 6/12 | strongest manually audited OOD base |
| v4.4 pilot SFT, step 120 | 4/40 | 5/12 | rejected: unstable and off-task generations |
| v4.5 SFT, step 300 | 5/40 | 4/12 (**3/12 manual**) | best internal reasoning loss, no OOD gain |
| v4.5 SFT, final/best step 736 | **5/40 manual** | 6/12 (**5/12 manual**) | better instruction tuning, but below the MID on OOD v2 |

These runs suggest that the 1.5B-token v4.3 curriculum materially improved transfer,
while the old SFT recipes damaged part of that gain. v4.5 was the corrective experiment:
lower LR, larger assistant-token budget, isolated conversations, token-correct loss,
capability-balanced sampling and 12% MID replay. It fixed the trainer/data path and
preserved MID validation, but its final checkpoint scores only **5/40** on OOD v2,
versus **8/40 after manual correction** for its source MID. Manual inspection found no
hidden reasoning success in either the step-300 or final SFT report. It did find one
factual false positive in each: the final mentions `Lisbonne` inside an unrelated
hallucination, while step 300 first answers `rouge vif` before later mentioning green
leaves. A newly sealed benchmark is still needed for a clean post-SFT generalization
claim. Full raw outputs and decoding settings are in `bench/reports/bench_ood_v2_*.md`.

### What v4.5 means for the local RL stage

v4.5 is not the strongest standalone reasoning checkpoint, but it remains the preferred
policy initialization for RL. It learned the chat/instruction interface, produces shorter
and more directly scorable answers than the MID, and retained the MID distribution during
training (`mid_val` improved from 2.097 at the first evaluation to 2.089 at the end). The
OOD regression therefore looks more like a change in answer policy and weak multi-step
execution than catastrophic loss of the v4.3 representation.

The final/best step 736 is preferred over step 300: both score 5/40 on manually audited
OOD v2, while step 736 retains more factual completions (5/12 corrected versus 3/12).
The next RL/RLAIF experiment must run locally rather than on Modal after the SFT budget.
It uses exact Python-verifiable rewards on task families deliberately disjoint from OOD
v2 (one-step equations, exact percentages, means, powers, grounded synthetic records,
structured constraints, contradictory sources and non-numeric state updates). It keeps
an explicit KL reference to the v4.5 SFT, replays curated supervised examples, penalizes
missing final answers and repetition, and selects checkpoints on held-out generated
tasks. Internal reward alone must never choose the release checkpoint.

A recovery from 5/40 to roughly 7-9/40 is a plausible experimental target, not a promised
result. Reaching 10-12/40 would be a strong outcome for a 229M model and the available
local compute. The MID checkpoint remains the mandatory no-regression reference throughout
RL, because its manually corrected 8/40 is still the best demonstrated v4 reasoning score.

### v4.5 local RLVR and offline RLAIF

All post-training in this section remains **v4.5**. `v5` is reserved for a future
model trained from a new pretrain checkpoint; an RL or DPO pass does not rename the
base model.

The new local pipeline lives beside the historical `rl.py`/`rlaif.py` so old runs stay
reproducible. It first profiles the SFT at pass@6 and reruns only frontier tasks at
pass@32. RLVR then uses groups of six, three dynamic prompts per accepted update,
DrGRPO advantages centered without standard-deviation scaling, a fixed token budget
denominator, one on-policy epoch, typed verifiers and 5% supervised replay. Zero-success
tasks are written to `needs_sft.jsonl` instead of receiving a misleading shaped reward.

The preliminary pre-RL profile of the final v4.5 SFT contains **3/60 pass@1**, **11/60
pass@6** and **10/60 groups already known to be dynamic**. The initial usable frontier is concentrated in
`grounded_rooms`, `code_1`, `constraint_number_only` and one `state_update` task;
`reasoning_program` and `uncertainty` are 0/6 throughout. Profiles created before the
full-frontier correction must be refined because a 0/6 task can still succeed between
samples 7 and 32. The trainer therefore reads
and hashes `profile.json`, spends 80% of on-policy prompts on measured frontier rows,
and reserves 20% for zero-success or unmeasured schemas. Those schemas additionally
receive canonical supervised bridge replay. A resume is refused if the profile changed.
This is a curriculum decision, not a post-RL quality result.

```bash
# 1. Required frontier profile (no OOD v2 prompt is read).
python run.py rl-profile-v45 --run fr-v4-v45-sft --data-dir data-v4 \
  --init-stage sft --init-ckpt best --tasks 60 --k 6 --frontier-k 32 \
  --max-new 112 --output profile.json

# Migration only: complete an older partial profile without replaying its first samples.
python run.py rl-profile-v45 --run fr-v4-v45-sft --data-dir data-v4 \
  --init-stage sft --init-ckpt best --tasks 60 --k 6 --frontier-k 32 \
  --max-new 112 --refine-from profile.json --output profile.json

# 2. Main local RTX 4060 run. Start with a short 10-update pilot, then resume.
python run.py rl-v45 --run fr-v4-v45-sft --data-dir data-v4 \
  --init-stage sft --init-ckpt best --ref-stage sft --ref-ckpt best \
  --updates 10 --prompts 3 --group 6 --max-new 112 --micro-bs 2 \
  --lr 2e-6 --kl-beta 0.018 --kl-target 0.012 --replay-weight 0.05 \
  --eval-tasks 60
python run.py rl-v45 --run fr-v4-v45-sft --data-dir data-v4 \
  --updates 200 --resume latest
```

The frozen reference is intentionally fp32. A real RTX 4060 smoke test measured an
initial token KL of about **0.96** with a bf16 reference despite identical source
weights, versus **0.00084** in fp32. The generic bf16-memory recommendation from the
audit is therefore unsafe for this v3 architecture. Policy parameters and AdamW remain
fp32, forward passes use bf16 autocast, reference gradients are disabled, optimizer
foreach is disabled, and only one CPU microbatch is moved to CUDA at a time.

RLAIF is offline: candidate generation never waits for an external judge while the GPU
is allocated. Two blind packets contain only prompts and anonymous outputs, with the
candidate order exactly reversed in packet B; they exclude canonical answers, Python
rewards and private provenance. The two passes must be judged independently and agree
on the winner. Imported rankings must cover exactly the issued candidate IDs, use integer
scores 0..4, clear the margin in both passes, and pass truncation/repetition/safety filters
before pairs are sealed for DPO. DPO validation is split by `prompt_id`, never by pair.

```bash
python run.py rlaif-build-v45 --run fr-v4-v45-sft --data-dir data-v4 \
  --init-stage rlvr-v45 --init-ckpt best --prompts 40 --candidates 6
# Fill scores_a.jsonl and scores_b.jsonl independently using JUDGE_INSTRUCTIONS.md.
python run.py rlaif-import-v45 --run fr-v4-v45-sft \
  --scores runs/fr-v4-v45-sft/rlaif-v45/scores_a.jsonl \
  --scores-reverse runs/fr-v4-v45-sft/rlaif-v45/scores_b.jsonl
python run.py dpo-v45 --run fr-v4-v45-sft --data-dir data-v4 \
  --init-stage rlvr-v45 --init-ckpt best --ref-stage rlvr-v45 --ref-ckpt best \
  --epochs 1 --grad-accum 8 --lr 5e-7 --beta 0.10
```

After training, regenerate the same dev profile with `--output profile_post.json`,
then compare it to the baseline. OOD v2 is evaluated only once on the candidate
checkpoint. The benchmark writes a `.raw.json`; prepare and complete an exhaustive
human correction before applying the final 7/40 reasoning and 5/12 factual floors.
This prevents the substring-based false positives and answer-extraction false negatives
observed in earlier reports.

```bash
python run.py rl-profile-v45 --run fr-v4-v45-sft --data-dir data-v4 \
  --init-stage dpo-v45 --init-ckpt best --tasks 60 --k 6 --frontier-k 32 \
  --output profile_post.json
python -m frlm.posttrain_gates_v45 \
  --baseline runs/fr-v4-v45-sft/rlvr-v45/profile.json \
  --candidate runs/fr-v4-v45-sft/rlvr-v45/profile_post.json \
  --output runs/fr-v4-v45-sft/posttrain_gates_dev.json

python bench/bench_ood_v2.py --run fr-v4-v45-sft --data-dir data-v4 \
  --stage dpo-v45 --ckpt ckpt_best.pt --hf none
python -m bench.adjudicate_ood_v2 bench/reports/bench_ood_v2_fr-v4-v45-sft_dpo-v45_best.raw.json
# Fill every manual_correct field in the generated .manual.json, then:
python -m bench.adjudicate_ood_v2 \
  bench/reports/bench_ood_v2_fr-v4-v45-sft_dpo-v45_best.raw.manual.json --finalize
```

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

4. **`<think>` discipline.** The historical v2-v4.1 recipe opened every answer with
   a thinking block, including an empty one for small talk. The v4.2 SFT is stricter:
   only examples with a genuine verified reasoning trace contain `<think>`; ordinary
   dialogue starts directly with the answer.

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
