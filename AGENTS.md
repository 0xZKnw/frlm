# AGENTS.md

## Objet du projet

`frlm` est un framework Python/PyTorch pour entraîner de zéro un petit modèle de
langue français orienté raisonnement. Le pipeline couvre la préparation des
données, le pré-entraînement, le mid-training, le SFT, le RL vérifiable (GRPO),
le RLAIF, l'inférence interactive et les benchmarks. Le code vise d'abord un GPU
grand public sous Windows, mais peut aussi être lancé sur Modal ou Beam.

Le code, les commentaires, les messages CLI et la documentation sont
majoritairement en français. Conserver cette langue pour les ajouts destinés aux
utilisateurs et pour les explications métier.

## Carte du dépôt

- `run.py` : point d'entrée principal. Commandes `prepare`, `train`, `mid`,
  `sft`, `rl`, `rlaif`, `chat` et `info`.
- `frlm/model.py` : architecture v2, presets historiques et interface commune de
  génération.
- `frlm/model_v3.py` : architecture speedrun v3/v4. Elle doit rester compatible
  avec l'interface de `QwenLikeLM` utilisée par le CLI, le chat, le RL et les
  benchmarks.
- `frlm/__init__.py` : routage des configs et checkpoints entre architectures v2
  et v3.
- `frlm/data.py` : téléchargement, conversion des corpus, tokenizer BPE,
  binarisation et masques SFT.
- `frlm/synth.py` : génération déterministe de problèmes français dont les
  réponses sont calculées par Python.
- `frlm/optim.py` : Muon, AdamW et schedules de learning rate.
- `frlm/rl.py` : GRPO avec récompenses vérifiables.
- `frlm/rlaif.py` : GRPO avec juge LLM et protocole d'échange par fichiers.
- `frlm/rl_tasks_v45.py`, `frlm/verifiers_v45.py` : tâches RLVR non-OOD et
  vérificateurs typés stricts du post-training v4.5.
- `frlm/rl_profile_v45.py`, `frlm/rl_v45.py` : profil pass@k et DrGRPO local v4.5.
- `frlm/reason_bootstrap_v45.py`, `frlm/eval_reason_bootstrap_v45.py` : mini-SFT
  AST exécutable et profil pass@k hors OOD pour restaurer une frontière avant RLVR.
- `frlm/rlaif_offline_v45.py`, `frlm/dpo_v45.py` : candidats aveugles, paires
  scellées et préférence hors-ligne v4.5.
- `frlm/distill.py` : génération et filtrage de données de distillation via API.
- `frlm/bench_speed.py` : mesure GPU du débit, de la VRAM et du MFU.
- `bench/` : benchmarks de qualité et rapports Markdown.
- `modal_app.py`, `beam_app.py` : wrappers d'exécution distante.
- `data-v4/` : données et tokenizer de la recette v4. Les gros artefacts dérivés
  restent hors de Git.
- `runs/` : checkpoints, métriques, échantillons et fichiers du juge. Ne pas
  versionner ni modifier sans demande explicite.

Le `README.md` décrit bien la v1/v2 publiée, mais le code courant contient aussi
les architectures v3/v4, la distillation et le RLAIF. En cas d'écart, vérifier le
CLI et les docstrings des modules concernés avant de reprendre une commande du
README telle quelle.

## Installation et environnement

Depuis la racine du dépôt :

```powershell
python -m pip install -r requirements.txt
```

PyTorch 2.5 ou plus récent est attendu. L'entraînement est conçu pour CUDA et le
bf16. Sous Windows, `triton-windows<3.3` est volontairement borné pour rester
compatible avec la version de PyTorch documentée. `transformers` n'est nécessaire
que pour comparer le modèle aux baselines Hugging Face. Les wrappers distants
demandent séparément `modal` ou `beam-client`, et la distillation utilise
`requests` ainsi qu'une clé d'API fournie par variable d'environnement.

Ne jamais ajouter de clé, token, checkpoint ou corpus lourd au dépôt.

## Commandes utiles

Inspection sans entraînement :

```powershell
python run.py --help
python run.py info --run fr-v4 --data-dir data-v4
python run.py chat --run fr-v4
```

Pipeline principal :

```powershell
python run.py prepare --data-dir data-v4 --target-tokens 3e9 --vocab-size 24576 --seq-len 2048 --mid-frac 0.12 --sft-target-supervised 60e6 --mix "fineweb:0.55,wiki:0.15,maths:0.15,books:0.06,theses:0.04,chat:0.03,europarl:0.01,oral:0.01"
python run.py train --data-dir data-v4 --run fr-v4 --preset v4-base --seq-len 2048
python run.py mid --data-dir data-v4 --run fr-v4 --preset v4-base --seq-len 2048
python run.py sft --data-dir data-v4 --run fr-v4 --preset v4-base --seq-len 2048
python run.py rl --run fr-v4
python run.py rlaif --run fr-v4
```

Pour refaire uniquement le SFT v4.2 à partir du tokenizer et du mid v4.1 existants :

```powershell
python -m frlm.audit_data --data-dir data-v4
python run.py prepare --data-dir data-v4 --rebin --sft-only --skip-download --seq-len 2048 --sft-target-supervised 60e6
```

Cette recette déduplique globalement les prompts, retire les sources chatbot/bruitées,
filtre les réponses cassées ou répétitives, n'impose plus de bloc `<think>` vide,
pondère le SFT par tokens assistant et aligne les fenêtres sur le début des
conversations. Le trainer rejoue 15 % du corpus mid et choisit le meilleur checkpoint
sur une validation macro par source. Utiliser un nouveau nom de run et reprendre
explicitement `runs/fr-v4-v41/mid/ckpt_best.pt` ; ne pas relancer le mid.

Pour le mid v4.3 à budget maximal de 1,5 milliard de tokens :

```powershell
python run.py prepare-mid-v43 --data-dir data-v4 --target-tokens 1.5e9
python run.py mid --data-dir data-v4 --run fr-v4-v43 --preset v4-base --mid-curriculum v4.3 --seq-len 2048 --batch-size 16 --grad-accum 4 --max-steps 11444 --optimizer muon --lr 0.002 --adam-lr 0.0001 --schedule wsd --warmup 100 --decay-frac 0.10 --min-lr-frac 0.02 --eval-every 1000 --sample-every 1000 --save-every 1000 --ckpt-every-min 10000 --keep-last 12 --resume runs/fr-v4/pretrain/ckpt_latest.pt
```

Cette recette conserve le tokenizer/pré-entraînement v4, utilise deux bins 80/20,
exclut explicitement les prompts/seeds OOD et inscrit les licences/provenances dans
`meta.json`. Elle inclut CQuAE sous CC-BY-NC-4.0 : ne pas présenter le modèle dérivé
comme commercial ni publier les corpus bruts sans revue de licence.

Pour préparer le SFT v4.4 équilibré sans modifier les bins précédents :

```powershell
python run.py prepare-sft-v44 --data-dir data-v4 --target-supervised 18e6 --seq-len 2048
python run.py sft --data-dir data-v4 --run fr-v4-v44-sft-lr1e4 --preset v4-base --sft-recipe v4.4 --seq-len 2048 --batch-size 16 --grad-accum 4 --max-steps 120 --optimizer adamw --lr 1e-4 --weight-decay 0.01 --schedule cosine --warmup 20 --min-lr-frac 0.10 --replay-frac 0.15 --replay-mix "mid_v43_stage1_train.bin=0.80,mid_v43_stage2_train.bin=0.20" --replay-val mid_v43_val.bin --eval-every 25 --eval-iters 48 --sample-every 25 --save-every 25 --keep-last 8 --resume runs/fr-v4-v43/mid/ckpt_latest.pt
```

La v4.4 alloue strictement 18M tokens assistant par capacités, sans redistribution
inter-capacité : 25 % chat français natif, 20 % distillation, 30 % raisonnement
vérifié, 12 % QA ancrée, 8 % contraintes/calibration et 5 % multi-tour/identité.
OpenHermes-FR est plafonné à 15 %, GSM8K-FR à 3 %, toute source à 20 % et les
shortfalls restent visibles. Le trainer échantillonne les bins par capacité et le
replay conseillé mélange 70 % de prétrain propre avec 30 % du mid v4.3. Comparer
d'abord trois pilotes de 120 pas à 3e-5, 1e-4 et 3e-4 ; le full SFT de 300-450 pas
ne part que du meilleur pilote.

Le SFT v4.4 enseigne volontairement des compétences structurelles proches des
familles OOD v2. Exécuter OOD v2 sur le mid avant SFT, puis utiliser les splits
retenus par schéma/surface et un nouveau benchmark scellé pour mesurer le SFT.

Le pilote v4.4 ne doit pas servir de recette finale : il a révélé que les fenêtres
concaténées traversaient les conversations et que l'accumulation pondérait chaque
microbatch également. La v4.5 corrige ces deux défauts et reprend uniquement le MID
v4.3 final au step 11444 :

```powershell
python run.py prepare-sft-v45 --data-dir data-v4 --target-supervised 24e6 --seq-len 512 --seed 451337
python run.py sft --data-dir data-v4 --run fr-v4-v45-sft --preset v4-base --sft-recipe v4.5 --seq-len 512 --batch-size 128 --grad-accum 12 --max-steps 736 --optimizer adamw --lr 2e-5 --weight-decay 0.01 --schedule cosine --warmup 30 --min-lr-frac 0.10 --replay-frac 0.12 --replay-mix "mid_v43_stage1_train.bin=0.80,mid_v43_stage2_train.bin=0.20" --replay-val mid_v43_val.bin --eval-every 100 --eval-iters 48 --sample-every 100 --save-every 100 --keep-last 12 --resume runs/fr-v4-v43/mid/ckpt_latest.pt
```

Les bins v4.5 contiennent une conversation entière par document (maximum 512 tokens)
et doivent être chargés avec `ConversationCorpus`. La loss SFT est une somme divisée
par le nombre global de tokens assistant de l'update ; le replay conserve un poids
séparé de 12 %. Le mix cible 24M tokens assistant en 30/18/18/12/8/6/5/3 et plafonne
OpenHermes-FR à 12 %. Les 736 steps correspondent à environ 1,15 passe pour 24M
tokens assistant et 37,5k tokens supervisés par update. Réexécuter l'audit après
chaque rebuild et ajuster ce nombre si la densité change, sans relancer le mid.

Si le profil RLVR reste à 0/pass@k sur le raisonnement, intercaler le bootstrap
supervisé `reason45` au lieu de lancer davantage de GRPO. Il génère 20k programmes
AST exécutés par Python, réserve des holdouts de surface et de topologie, et ne lit
jamais OOD v2 :

```powershell
python run.py prepare-reason-bootstrap-v45 --data-dir data-v4 --examples 20000 --seq-len 256 --eval-per-split 120 --seed 455500
python -m frlm.audit_reason_bootstrap_v45 --data-dir data-v4
python -m frlm.eval_reason_bootstrap_v45 --run fr-v4-v45-sft --data-dir data-v4 --stage sft --ckpt best --tasks 30 -k 4
python run.py sft --data-dir data-v4 --run fr-v4-v45-reason --preset v4-base --sft-recipe reason45 --seq-len 256 --batch-size 128 --grad-accum 4 --max-steps 160 --optimizer adamw --lr 5e-6 --weight-decay 0.01 --schedule cosine --warmup 10 --min-lr-frac 0.10 --replay-frac 0 --eval-every 20 --eval-iters 30 --sample-every 20 --save-every 40 --ckpt-every-min 10000 --keep-last 6 --resume runs/fr-v4-v45-sft/sft/ckpt_best.pt --init-weights-only
```

La recette contient déjà 20 % de rétention v4.5 : conserver `--replay-frac 0`.
`--init-weights-only` est obligatoire pour repartir au step 0 avec un AdamW neuf ;
une reprise normale entre deux checkpoints `stage=sft` restaurerait sinon le step
736 et l'ancien optimiseur. Les bins de rétention peuvent rester uniquement sur le
Volume Modal ; l'audit local les reporte alors comme différés et le préflight Modal
valide leur présence et leur taille avant toute allocation GPU.

Le run H100 `fr-v4-v45-reason` a fini à 160 steps / 20,97M tokens traités. Son
minimum de validation est `ckpt_best.pt` au step 120 (`0,2246`) ; les steps 140 et
160 remontent légèrement à `0,2251` et `0,2256`. Toujours comparer ce best sur les
holdouts AST de surface/structure et sur des générations brutes avant de reprendre
RLVR. Les losses de rétention stables ne suffisent pas à conclure à un gain OOD.
Le profil strict du best donne 11/30 greedy et 17/30 pass@4, mais OOD v2 régresse à
2/40. Une interpolation de poids à 25 % bootstrap atteint 7/40 après revue manuelle
mais ne conserve que 2/30 pass@4 ; 35 % donne 4/40 et 7/30, 50 % donne 3/40 et 9/30.
Ces mesures font désormais d'OOD v2 un benchmark de développement, pas un holdout
propre. Pour une seconde recette, réduire fortement les sorties numériques et ajouter
des programmes symboliques/état à cibles textuelles ainsi que plus de rétention ; ne
pas lancer directement RLVR sur le best 120 pur.

La seconde recette courte est `reason45b`. Elle ne régénère aucun corpus : elle
réutilise les bins vérifiés de `reason45`, mais les échantillonne à 35 % avec 65 %
de rétention v4.5 diversifiée. Elle repart toujours du SFT original avec un AdamW
neuf, LR `3e-6`, 120 steps et `--replay-frac 0` :

```powershell
python run.py prepare-reason-bootstrap-v45b --data-dir data-v4
python -m frlm.audit_reason_bootstrap_v45 --data-dir data-v4 --recipe reason45b
python run.py sft --data-dir data-v4 --run fr-v4-v45-reason-balanced --preset v4-base --sft-recipe reason45b --seq-len 256 --batch-size 128 --grad-accum 4 --max-steps 120 --optimizer adamw --lr 3e-6 --weight-decay 0.01 --schedule cosine --warmup 8 --min-lr-frac 0.10 --replay-frac 0 --eval-every 20 --eval-iters 39 --sample-every 20 --save-every 40 --ckpt-every-min 10000 --keep-last 6 --resume runs/fr-v4-v45-sft/sft/ckpt_best.pt --init-weights-only
```

Le run H100 a fini à 15,73M tokens ; le best est le step 120 (val `0,22492`).
Sur le même sous-ensemble AST que `reason45`, il obtient 10/30 greedy et 15/30
pass@4 contre 11/30 et 17/30, tout en remontant OOD v2 de 2/40 à 4/40 corrigé.
Le profil AST élargi donne 22/90 greedy et 34/90 pass@4. Préférer ce checkpoint
équilibré au best pur `reason45` comme point de départ d'un futur RLVR gardé.

Le post-training local suivant reste nommé **v4.5** ; réserver `v5` à un futur
pré-entraînement neuf. Le pipeline corrigé ne remplace pas les anciens `rl.py` et
`rlaif.py`, afin de garder leurs runs reproductibles :

```powershell
python run.py rl-profile-v45 --run fr-v4-v45-sft --data-dir data-v4 --init-stage sft --init-ckpt best --tasks 60 --k 6 --frontier-k 32 --max-new 112 --output profile.json
python run.py rl-profile-v45 --run fr-v4-v45-sft --data-dir data-v4 --init-stage sft --init-ckpt best --tasks 60 --k 6 --frontier-k 32 --max-new 112 --refine-from profile.json --output profile.json
python run.py rl-v45 --run fr-v4-v45-sft --data-dir data-v4 --init-stage sft --init-ckpt best --ref-stage sft --ref-ckpt best --updates 10 --prompts 3 --group 6 --max-new 112 --micro-bs 2 --lr 2e-6 --kl-beta 0.018 --kl-target 0.012 --replay-weight 0.05 --eval-tasks 60
python run.py rl-v45 --run fr-v4-v45-sft --data-dir data-v4 --updates 200 --resume best --reset-optimizer --lr 5e-7 --kl-beta 0.10 --kl-beta-max 1.0 --kl-target 0.012 --kl-soft-max 0.05 --kl-hard-max 0.12 --replay-weight 0.15 --retention-kl-weight 0.05 --eval-every 5 --max-dev-drop 0.05 --max-capability-drop 0.10 --max-auto-recoveries 6
python run.py rlaif-build-v45 --run fr-v4-v45-sft --data-dir data-v4 --init-stage rlvr-v45 --init-ckpt best --prompts 40 --candidates 6
python run.py rlaif-import-v45 --run fr-v4-v45-sft --scores runs/fr-v4-v45-sft/rlaif-v45/scores_a.jsonl --scores-reverse runs/fr-v4-v45-sft/rlaif-v45/scores_b.jsonl
python run.py dpo-v45 --run fr-v4-v45-sft --data-dir data-v4 --init-stage rlvr-v45 --init-ckpt best --ref-stage rlvr-v45 --ref-ckpt best --epochs 1 --grad-accum 8 --lr 5e-7 --beta 0.10
```

Le profil pass@6/pass@32 est obligatoire avant RLVR. Toute tâche non maîtrisée, y
compris à 0/6, doit être mesurée jusqu'à 32 ; `--refine-from` complète les anciens
profils sans refaire leurs premiers rollouts. Le trainer retient un groupe
uniquement si son nombre de réussites primaires est strictement entre 0 et G ; il
centre les avantages sans division par l'écart-type, impose T=1/top-p=1 et un seul
epoch on-policy. Une update acceptée contient exactement trois groupes dynamiques et
la sélection du checkpoint utilise les 60 tâches dev. Les groupes sans aucune réussite vont dans `needs_sft.jsonl`.
Le pilote historique utilisait un replay de 5 % ; la continuation stabilisée utilise
0,15 avec rétention équilibrée et exemples conversationnels propres.
Le premier run lit et hache `profile.json`. Une reprise utilise par défaut le checkpoint
repris comme ancre KL fp32 et génère/réutilise automatiquement `profile_phase2.json`
depuis ce même checkpoint. N'accepter un hash de profil différent du checkpoint que si
le nouveau profil déclare exactement la même phase et la même update. `--keep-reference`
est réservé aux reproductions historiques. En phase 2, seuls les schémas réellement
dynamiques à pass@32 reçoivent des rollouts RL ; les 0/32 restent dans le bridge SFT.
Pondérer les frontières par le déficit de leur capacité et conserver une rétention
équilibrée pour les capacités acquises.

Après la réévaluation d'une reprise avec un nouveau profil/verifier, matérialiser
atomiquement ce checkpoint dans `ckpt_phase_anchor.pt` et `ckpt_best.pt`, puis remettre
les pics par capacité sur cette baseline. Un rollback ne doit jamais résoudre un
`ckpt_best.pt` physique hérité de la phase précédente.

La branche longue initiale a culminé au step 60 à 0,550 (33/60), puis oscillé à
27/60, 32/60 et 29/60 aux steps 70/80/90 avec des KL ponctuelles de 0,09 à 0,14.
Pour une reprise stabilisée, repartir de `best` au step 60, réinitialiser AdamW,
utiliser LR 5e-7 et replay 0,15. Le replay par défaut contient une ligne de rétention
cyclique équilibrée entre les six capacités et une ligne conversationnelle curée.
Une KL <=0,05 est acceptée normalement ; entre 0,05 et 0,12, snapshotter modèle et
Adam sur CPU, mesurer la KL post-step et restaurer les deux si elle franchit 0,12 ;
au-delà de 0,12 avant le step, refuser sans mutation. Trois excursions rapprochées
divisent le LR par deux. Évaluer toutes les 5 updates. Une baisse dev stricte supérieure
à 0,05 ou une capacité plus de 0,10 sous son meilleur historique provoque un rollback
vers `ckpt_best`, une baisse de LR, une hausse de beta et un replay renforcé, puis une
nouvelle branche automatique. Après le budget de reprises, arrêter sur le best. Les snapshots transactionnels ne doivent être créés
que dans la zone d'alerte car ils occupent environ la taille d'un checkpoint en RAM CPU.

Le replay phase 2 alterne les bridges de déficit avec un cycle équilibré sur les six
capacités et ajoute une KL sélectionnée sur les tokens vers le teacher figé du checkpoint
repris. Ne jamais déplacer automatiquement cette référence après chaque nouveau best :
elle reste l'ancre de la phase courante jusqu'à une nouvelle invocation explicite.

Seul `reasoning_program` possède une difficulté scalaire réellement utilisée par le
générateur. Pour les autres capacités, adapter en ligne le poids de chaque ligne
frontière à partir de son taux de succès observé ; conserver échelles et historiques
dans les checkpoints. Le pilote local de 10 updates a fait passer la macro dev fixe de
0,067 (4/60) à 0,183 (11/60), entièrement grâce au code (0/10 vers 7/10), avec KL finale
0,0058 et aucune réussite initiale perdue. Ne pas présenter ce pilote comme un gain de
raisonnement : quatre capacités restent à zéro.

`verify(..., kind="abstain")` applique le regex après suppression Unicode des accents :
le motif doit donc rester intégralement non accentué. Tester à la fois les formulations
accentuées et non accentuées lors de toute modification.

Les tâches `constraint_number_only` utilisent `AnswerSpec.strict_number_only` : la
réponse visible doit être exactement un entier signé, sans explication ni ponctuation.
Ne pas réintroduire un raccourci par nombre marqué (`donc 121`) pour ce schéma.
La version du vérificateur est inscrite dans les nouveaux checkpoints. Une reprise
d'un checkpoint plus ancien doit recalculer la macro dev avant toute nouvelle update
et remplacer en mémoire le meilleur score devenu incomparable.
Une reprise depuis un checkpoint antérieur à `metrics.jsonl` doit isoler les lignes
postérieures dans `metrics_abandoned_after_*.jsonl` et repointer atomiquement
`ckpt_latest.pt`, sans supprimer l'historique abandonné.

Le RLAIF v4.5 produit deux paquets aveugles A/B dont l'ordre des candidats est inversé.
Les jugements doivent être réalisés indépendamment, couvrir tous les IDs et désigner le
même gagnant ; la marge minimale doit tenir dans les deux passes. Le split DPO est fait
par `prompt_id` afin qu'aucune paire d'un même prompt ne traverse train/validation.

Sur model_v3, une ancre KL bf16 a produit une KL initiale artificielle d'environ
0,96 contre 0,00084 en fp32 lors du smoke test RTX 4060. Conserver donc la référence
gelée en fp32 malgré la recommandation mémoire générique ; policy/AdamW restent fp32,
les forwards utilisent autocast bf16, `foreach=False` et `micro_bs<=2`.

Les familles générées par `rl_tasks_v45.py` doivent rester structurellement disjointes
des familles OOD v2. En particulier, ne jamais y ajouter transitivité, information
absente/hors sujet, fencepost/intervalles, comparaison de branches, reste euclidien,
cycle de jours ou composition arithmétique multi-opérations. Le test
`test_no_ood_v2_family_is_generated` protège la liste connue mais ne remplace pas une
revue humaine de toute nouvelle famille.

Benchmarks :

```powershell
python -m frlm.bench_speed --presets v3-mini --no-compile --steps 2 --warmup 1
python bench/bench_ood_v2.py --run fr-v4 --data-dir data-v4 --hf none
python bench/bench_vs.py --run fr-v4 --data-dir data-v4 --skip-hf
```

Le RLVR v4.5 possède deux contrôleurs distincts : KL des rollouts et KL de
rétention sur replay. Conserver leurs seuils séparés. Une reprise sûre repart du
`best`, réinitialise Adam pour la nouvelle phase et laisse le CLI reprofiler ce
checkpoint avant de l'ancrer :

```powershell
python run.py rl-v45 --run fr-v4-v45-reason-balanced --data-dir data-v4 --updates 200 --resume best --reset-optimizer --lr 5e-7 --eval-every 5
```

Au-dessus de la cible de rétention, le trainer augmente automatiquement sa
pression et le replay. Dans la zone risquée il snapshotte policy et Adam sur CPU,
puis rejette transactionnellement une aggravation au-delà du seuil dur. Ne pas
retirer ce rollback exact ni fusionner `retention_kl` avec la KL rollout : les
deux métriques portent sur des distributions différentes.

Les commandes de préparation téléchargent et écrivent beaucoup de données. Les
entraînements, `bench_speed` et certains benchmarks exigent un GPU et peuvent être
longs ou coûteux. Ne pas les lancer comme simple validation sans accord explicite.
Avant un entraînement Modal, utiliser l'option `--check-only` du wrapper pour
valider sur CPU les bins, métadonnées et checkpoints du Volume sans allouer de GPU.

## Invariants à préserver

### Données et tokenizer

- Les tokens spéciaux sont définis dans `frlm/data.py` : `<|endoftext|>`,
  `<|im_start|>`, `<|im_end|>`, `<think>` et `</think>`. Ne pas changer leur ordre
  ou leur rendu sans migration explicite des données et checkpoints.
- Le tokenizer sépare les chiffres ; c'est une propriété centrale pour
  l'arithmétique. Toute modification de tokenisation doit être testée sur les
  nombres et suivie d'une rebinarisation.
- Les fichiers `.bin` utilisent des identifiants `uint16`; un vocabulaire supérieur
  à 65 535 tokens exige une modification coordonnée du format.
- `render_chat`, les masques SFT et les prompts de génération doivent garder le
  même format de conversation.
- Les générateurs synthétiques doivent rester déterministes pour une seed donnée,
  calculer la réponse avec Python et éviter toute fuite des seeds d'évaluation.

### Modèles et checkpoints

- Les configs v3 écrivent `{"arch": "v3"}`. Les anciens checkpoints sans champ
  `arch` sont traités comme v2. Toute nouvelle architecture doit passer par les
  helpers de `frlm/__init__.py` et conserver la rétrocompatibilité.
- Préserver l'interface commune des modèles : `forward` avec `targets`,
  `loss_mask` et `z_loss`, ainsi que les méthodes de cache et `generate` utilisées
  par `run.py`, le RL et les benchmarks.
- Un checkpoint contient au minimum la config modèle, les poids, le step et les
  informations nécessaires à une reprise exacte. Ne pas charger seulement les
  poids dans un chemin de reprise d'entraînement.
- Chaque phase écrit dans `runs/<run>/<phase>/`. Ne jamais faire écraser par une
  phase les checkpoints d'une autre phase.
- Les écritures de checkpoints doivent rester atomiques. Conserver la rotation de
  `ckpt_latest.pt`, des checkpoints numérotés et de `ckpt_best.pt`.
- `mid` et `sft` reprennent par défaut les poids précédents ; `rl` part du SFT et
  le RLAIF part normalement du meilleur checkpoint RL. Vérifier cette chaîne lors
  de tout changement au chargement.

### Entraînement et évaluation

- Préserver les seeds explicites et la restauration des RNG lors des reprises.
- Ne jamais entraîner sur les problèmes OOD écrits à la main ni sur les seeds
  réservées à l'évaluation.
- Garder `Ctrl+C`, le fichier `STOP`, les checkpoints périodiques et les sorties
  propres fonctionnels.
- Attention à la VRAM sous Windows : un dépassement peut provoquer du spill en RAM
  partagée au lieu d'un OOM clair. Ne pas augmenter les batch sizes par défaut sans
  mesure de VRAM et de débit.
- Les chiffres de performance doivent préciser le matériel, le preset, la longueur
  de séquence, le batch, l'accumulation, la compilation et le nombre de steps.
  Distinguer le pic bf16 dense des chiffres marketing avec sparsité.
- Les rapports de benchmark doivent conserver les réponses brutes et les réglages
  utilisés. Ne pas annoncer uniquement un score agrégé.

### RLAIF

- Le protocole complet et le barème de jugement sont documentés en tête de
  `frlm/rlaif.py`; les lire avant toute modification.
- Le trainer écrit `pending_step_N.jsonl` puis attend un
  `scores_step_N.json` valide dans le dossier `judge/`. Respecter l'ordre des
  groupes et des rollouts ainsi que les dimensions attendues.
- Une génération sans réponse finale après `</think>` vaut zéro. Le juge doit aussi
  corriger les faux positifs et faux négatifs du vérificateur Python.
- Une nouvelle passe doit utiliser un nouveau `--stage-name` (`rlaif2`, etc.) pour
  ne pas écraser le meilleur checkpoint de la passe précédente.
- Pour le pipeline v4.5, préférer `rlaif_offline_v45.py` : générer les candidats,
  libérer le GPU, faire juger le paquet aveugle, valider/sceller les paires, puis
  lancer `dpo-v45`. Le paquet juge ne doit jamais contenir réponse canonique, score
  Python, provenance privée ou ordre non randomisé.
- Un classement v4.5 doit reprendre exactement tous les IDs émis. Refuser les IDs
  inventés, candidats dangereux/tronqués/répétitifs, égalités sans marge et fichiers
  de paires dont le hash diffère du manifest.

## Style de code

- Cibler Python 3.10+ et utiliser les annotations modernes déjà présentes
  (`list[str]`, `X | None`, dataclasses).
- Respecter le style existant : quatre espaces, imports standard/tiers/local,
  noms Python en anglais ou en français selon le module, commentaires concis en
  français et séparateurs de sections pour les gros modules.
- Préférer `pathlib.Path`, les context managers et les écritures atomiques pour les
  artefacts importants.
- Garder les imports lourds ou optionnels au plus près de leur utilisation quand
  cela permet aux commandes CPU de fonctionner sans dépendance GPU/cloud.
- Ne pas introduire un nouveau framework, formatter ou linter pour une correction
  locale. Si un outil de qualité est ajouté au projet, le configurer à la racine et
  documenter sa commande ici.
- Éviter les refactorings massifs dans `run.py` ou les modèles en même temps qu'une
  modification algorithmique : les écarts de qualité et de débit doivent rester
  attribuables.

## Validation des changements

Il n'existe actuellement ni suite `pytest` ni configuration de linter. Adapter la
validation à la portée du changement et signaler clairement ce qui n'a pas pu être
exécuté.

Minimum pour tout changement Python :

```powershell
python -c "import ast, pathlib; files=list(pathlib.Path('.').glob('*.py'))+list(pathlib.Path('frlm').glob('*.py'))+list(pathlib.Path('bench').glob('*.py')); [ast.parse(p.read_text(encoding='utf-8'), filename=str(p)) for p in files]; print(f'{len(files)} fichiers Python valides')"
python run.py --help
```

Puis, selon la zone touchée :

- modèle : construire un petit preset sur CPU, faire un forward sur un lot court et
  vérifier la forme des logits et une loss finie ; tester aussi sérialisation et
  rechargement de la config v2/v3 ;
- données/tokenizer : tester quelques conversations, nombres et masques sur un
  répertoire temporaire, sans modifier `data-v4/` ;
- synthèse/RL : vérifier plusieurs centaines d'exemples avec des seeds fixes et
  confirmer que `verifier` accepte la réponse canonique ;
- checkpoints : tester sauvegarde, résolution `latest`/`best` et reprise dans un
  répertoire temporaire ;
- performance CUDA : faire d'abord un smoke test très court, puis seulement un
  benchmark représentatif accepté par l'utilisateur ;
- benchmark : utiliser `--skip-hf` ou `--hf none` pour le smoke test local, puis
  exécuter les baselines uniquement si `transformers`, les poids et le temps sont
  disponibles ;
- cloud : valider localement la commande construite. Ne jamais lancer un job payant,
  envoyer des données ou arrêter un job existant sans demande explicite.

## Discipline de modification

- Examiner `git status` avant et après le travail. Le dépôt peut contenir des
  expériences et rapports non commités : ne pas les supprimer, les reformater ou
  les inclure par accident.
- Limiter chaque changement aux fichiers nécessaires et préserver les résultats
  d'expérience existants.
- Mettre à jour `README.md` et ce fichier quand une commande, une architecture, un
  format de données ou la chaîne des phases change.
- Dans le compte rendu final, donner les fichiers modifiés, les validations
  réellement exécutées et les validations GPU/cloud non exécutées.
