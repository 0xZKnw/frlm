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

Benchmarks :

```powershell
python -m frlm.bench_speed --presets v3-mini --no-compile --steps 2 --warmup 1
python bench/bench_ood_v2.py --run fr-v4 --data-dir data-v4 --hf none
python bench/bench_vs.py --run fr-v4 --data-dir data-v4 --skip-hf
```

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
