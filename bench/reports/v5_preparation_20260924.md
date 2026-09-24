# V5 Qwen4-exp 350M : préparation du 24 septembre 2026

Les correctifs antérieurs du pipeline 229M, leurs contre-exemples et leurs
vérifications sont détaillés dans [le rapport reason45c](pipeline_corrections_20260924.md).

## Décision et état

V5 = nouveau modèle de **350 011 504 paramètres**, initialisés aléatoirement,
avec tokenizer français propre. Le 229M n'est pas relancé. Aucun entraînement
350M, job Modal, envoi cloud ni dépense de crédits n'a été effectué.
Les étapes d'optimisation CPU des tests portent sur des mini-modèles jetables.

L'architecture choisie est celle de [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next),
nommée `qwen4_exp_text` dans Transformers et `qwen4exp` dans llama.cpp.
L'adaptateur utilise Transformers 5.17.0 : 20 couches, largeur 512, Gated DeltaNet
et attention en 3:1, 8 experts dont 2 activés + un partagé, GR4, PLE à n-grammes,
embeddings liés, vocabulaire 32 768. Le contexte maximal est 2048 ; lancement
proposé à 1024. Le budget QSA couvrant toute cette fenêtre, aucun gain de
sparsité longue portée n'est revendiqué. Aucun MTP n'a été ajouté.

**Ce choix reste expérimental à cette échelle et à ce budget.** Une architecture
récente n'est pas automatiquement supérieure à un Transformer dense 350M.
Le décompte total moins experts inactifs donne 161 267 824 paramètres, mais inclut
des tables de lookup : ce chiffre n'est ni un équivalent dense ni un coût exact.
Le calcul matriciel estimé est 0,867 GFLOP/token entraîné ; il omet certains coûts
récurrents, routage, mémoire et kernels. Seul le pilote donnera des tokens/$.

La préparation locale est terminée : **4 000 141 714 tokens prétrain** (dernier
document de chaque source conservé entier), audit complet et préflight local
réussis. Le SFT et ses masques sont également audités. Le dossier de transfert
`data-v5/modal-upload/` contient **58 fichiers / 8 134 700 916 octets** ; ce sont
des liens locaux vers les artefacts figés, sans copie des corpus bruts.
[Inventaire et empreintes](v5_prepared_artifacts_20260924.json).

Les étapes suivantes restent soumises au « go » : connexion des comptes,
chargement du Volume, préflight distant et pilote GPU. La préparation locale ne
prétend pas remplacer ce dernier contrôle sur H100.

## Recherche des données

Les fiches, schémas, accès, licences et petits échantillons ont été examinés avant
la sélection. Les recettes versionnées fixent les commits HF et l'ordre des
fichiers. Les manifests locaux enregistrent aussi les SHA-256 des fichiers
réellement consommés, les rejets, les doublons et les quantités obtenues.

### Pré-entraînement : réserve de 4 milliards de tokens distincts

Les six sous-corpus proviennent de la version préparée
[Luciole](https://huggingface.co/datasets/OpenLLM-France/Luciole-Training-Dataset),
commit `dc89cbe9b500eb934f198e7b4718926132238117`. Ce choix réutilise les textes
convertis et évite notamment de récupérer les grosses colonnes d'embeddings
FineWeb2-HQ. Il ne reprend pas sans tri tout le mélange Luciole.

| Source | Cible en tokens | Pourquoi la retenir | Limite |
|---|---:|---|---|
| FineWeb2-HQ français | 50 % / 2 000 M | Texte informatif sélectionné par qualité | Un score amont ne garantit pas chaque document |
| FineWeb2-3plus français | 15 % / 600 M | Contenu éducatif, complément de couverture | Déduplication entre sources nécessaire |
| Wikipédia français | 10 % / 400 M | Français encyclopédique et connaissances | Le web contient déjà des copies |
| Gutenberg français | 5 % / 200 M | Syntaxe et narration longues | Langue ancienne, en-têtes à enlever |
| FineMath-4plus anglais | 15 % / 600 M | Explications mathématiques et calculs | Anglais assumé ; du bruit subsiste |
| SynthFineweb2 français | 5 % / 200 M | Réécritures structurées | Faible part pour limiter les tics synthétiques |

Ce mélange vise **85 % français / 15 % maths anglais** en tokens, et non en
nombre de documents. Il reste une hypothèse de recette, sans ablation locale.
[FineWeb2-HQ](https://huggingface.co/datasets/epfml/FineWeb2-HQ) publie des expériences
multilingues favorables à une sélection informative ; cela ne démontre pas un
facteur de gain pour notre 350M. [FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath)
fournit des textes éducatifs mathématiques filtrés, plus utiles ici qu'une masse
de traductions automatiques de problèmes avec réponses parfois cassées.

Les licences sont consignées par constituant : ODC-BY pour plusieurs corpus web
et FineMath, CC-BY-SA/GFDL pour Wikipédia, conditions des éditions et provenance
Gutenberg pour les livres. Luciole affiche CC-BY-SA-4.0 au niveau de l'agrégat.
La licence MIT du code n'annule pas les obligations des données.

### SFT : conversations françaises, solutions courtes et cibles vérifiées

| Source | Poids cible en tokens assistant | Tokens effectivement préparés | Sélection |
|---|---:|---:|---|
| [Nemotron post-training v2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2) via Luciole | 35 % | 4 200 382 | Sous-ensemble français, traces externes retirées |
| [Scholar](https://huggingface.co/datasets/kurakurai/scholar) via Luciole | 5 % | 600 039 | Solutions françaises tenant entièrement dans 1024 tokens |
| [PleIAs/SYNTH](https://huggingface.co/datasets/PleIAs/SYNTH) | 30 % | 3 600 186 | `language=fr`, champs query/answer, regroupement par article source |
| [OpenHermes-FR](https://huggingface.co/datasets/legmlai/openhermes-fr) | 12 % | 1 440 499 | Retrait des entrées signalées mauvaises, plafond explicite |
| [french_instruct](https://huggingface.co/datasets/angeluriot/french_instruct) | 3 % | 360 098 | `author=human`, `style=human`, sans code |
| Programmes AST frlm corrigés | 15 % | 1 164 702 | Réponses exécutées en Python, programmes du split train |

Total : **11 365 906 tokens assistant, 118 195 conversations train**. La cible de
12 M présente un déficit de **635 298 tokens**, uniquement sur les AST : les
100 000 candidats ne suffisent pas après déduplication. Il est conservé dans le
manifest, sans gonfler le fichier par répétition. Le sampler conserve les poids
cibles ; le nombre de passages propre à chaque source sera donc différent.

Nemotron et SYNTH indiquent CC-BY-4.0, Scholar/OpenHermes ODC-BY, french_instruct
MIT. Les fiches et licences des textes sources restent à conserver. Pour SYNTH,
le commit original choisi et ses champs de provenance font foi ; les métadonnées
de licences de certaines versions dérivées ne sont pas interchangeables.

Les traces de raisonnement externes sont retirées : plusieurs sources présentent
une réponse française mais un long `<think>` anglais. Les solutions françaises
visibles, explications et petits raisonnements locaux sont conservés. Il n'y a
ni teacher API payé, ni pseudo-preuve qu'une réponse synthétique est correcte.

Avec la pondération en tokens, une conversation AST ne supervise que 13,3 tokens
en moyenne contre 463 pour Nemotron. Les AST représentent donc environ **81,4 %
des conversations tirées**, mais 15 % des tokens assistant en espérance. La
moyenne attendue est 72,36 tokens assistant par conversation, soit 7,07 % d'une
fenêtre paddée de 1024. À batch 8 × accumulation 8, avec un microbatch de replay,
une exposition équivalente au total unique demande environ **2805 updates SFT**.
Ce nombre doit être confronté au débit réel et au budget ; il ne constitue pas
une promesse de faire un epoch complet avec 6 $.

### Alternatives examinées, non intégrées

| Candidat | Motif de non-sélection pour cette préparation |
|---|---|
| [Luciole post-training complet](https://huggingface.co/datasets/OpenLLM-France/Luciole-PostTraining-Dataset-1.1) | Mélange hétérogène ; certaines lignes sont en anglais malgré le contexte français |
| [OASST2](https://huggingface.co/datasets/OpenAssistant/oasst2) | Conversations humaines utiles, mais arbres et annotations à reconstruire ; faible sous-ensemble FR |
| [ComparIA](https://huggingface.co/datasets/ministere-culture/comparia-fr-arena) | Accès soumis à acceptation ; préférences réelles, pas automatiquement des réponses SFT de référence |
| [Nemotron-CC-Math](https://huggingface.co/datasets/nvidia/Nemotron-CC-Math-v1) | Accès gated et conditions supplémentaires ; FineMath déjà accessible |
| [Python-Edu / SmolLM corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) | Le sous-ensemble inspecté contient identifiants/chemins/scores, pas directement le code |
| [Stack-Edu](https://huggingface.co/datasets/HuggingFaceTB/stack-edu) | Code éducatif pertinent si le code devient une priorité ; non prioritaire dans ces 4 B |
| [French-PD-Books](https://huggingface.co/datasets/PleIAs/French-PD-Books) | OCR historique ; Gutenberg est retenu pour limiter ce bruit |
| [GSM8K-FR](https://huggingface.co/datasets/cmh/gsm8k_fr) | Échantillons contenant faux amis et formulations mathématiques dégradées ; pas de split test incorporé |
| [Mille-Pensées](https://huggingface.co/datasets/GLauzza/Mille-Pensees-Dataset) | Provenance/licence à clarifier, longues traces mixtes FR/EN |
| [SmolTalk français communautaire](https://huggingface.co/datasets/Volko76/smol-smoltalk-french-instruction-dataset) | Conditions et qualité de traduction insuffisamment établies pour le mélange principal |
| [SmolTalk2](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2) | Gros corpus essentiellement anglophone ; pas une source française par défaut |
| [WildChat](https://huggingface.co/datasets/allenai/WildChat-1M) | Chats bruts demandant une curation distincte ; pas de bénéfice démontré ici |

Les travaux [SmolLM2](https://arxiv.org/abs/2502.02737), la
[fiche du 360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M) et les
[recettes publiques SmolLM](https://github.com/huggingface/smollm) confortent le
rôle des données éducatives, des mélanges explicites et d'une phase conversation.
Ils ne valident pas notre budget : SmolLM2-360M annonce 4 000 milliards de tokens,
soit environ 1000 fois notre réserve locale. Les résultats de grands modèles de
raisonnement ne permettent pas non plus de promettre les mêmes gains à 350M.

## Préparation, séparation et limites de l'audit

- BPE 32 768, appris sur 100 M de caractères du train ; chiffres individuels,
  mêmes tokens spéciaux et regex de pré-tokenisation compatible Qwen2.
- Normalisation NFC ; bornes de longueur, rejet des contrôles chat dans le brut,
  de certains spams observés, des répétitions de lignes et du texte très corrompu.
- Déduplication exacte normalisée globale prétrain et par prompt pour le SFT.
  Pas de déduplication sémantique exhaustive ; les quasi-doublons restent possibles.
- Prétrain : split par hash d'URL/identifiant, 99,8/0,1/0,1 %. SFT : 98/1/1 %
  par prompt/programme/article. Une conversation entière par document, aucun
  tronquage pour faire tenir une solution, masque uniquement sur l'assistant.
- Les articles web déjà vus en prétrain peuvent réapparaître comme connaissances
  dans le SFT ; ces holdouts ne constituent pas une garantie de décontamination
  inter-corpus ou vis-à-vis de benchmarks publics.
- Aucun chargement d'OOD v2 pour préparer ces corpus. Les AST réutilisent le
  générateur corrigé sur son split train ; aucune ancienne donnée v4 n'est réécrite.
- Les fichiers `sealed` ne sont pas utilisés par Trainer. Leurs tailles, IDs et
  empreintes sont audités, sans évaluation du modèle ni sélection sur leur loss.
  Ils sont locaux et accessibles : scellement procédural, pas secret chiffré.
- Le filtrage français du SFT est heuristique. Les samples consultés ne sont pas
  un audit humain exhaustif des millions de documents.

## Compatibilité GGUF effectivement testée

Le runtime officiel contient [Qwen4-exp](https://github.com/ggml-org/llama.cpp/blob/a72e04abe0fe9b36e203033ac71bd5f379c35bc5/src/models/qwen4exp.cpp).
Version testée : `a72e04abe0fe9b36e203033ac71bd5f379c35bc5`, build CPU local macOS,
Metal désactivé, Accelerate activé, 2 threads. L'adaptation du convertisseur se
limite au BPE personnalisé vérifié structurellement, au nom canonique QSA et à
la table PLE non fragmentée de Transformers. Le reste est le convertisseur amont.

[Mesure complète](v5_gguf_350m_20260924.json) sur **350 011 504 paramètres**, poids
aléatoires, normes perturbées et convolution PLE non nulle :

- 6 cas de tokenisation identiques (français, accents, chiffres, Unicode, LaTeX,
  espaces et tokens ChatML) avec le tokenizer réellement préparé ;
- deux chunks de 64 tokens, comparaison de toutes les log-probabilités de leur
  seconde moitié : erreur maximale **0,0000228882**, tolérance 0,003 ;
- conversion Q4_K_M, **230 143 744 octets**, chargement et perplexité finie.

Un premier test avec le cache/Flash Attention par défaut présentait un écart de
0,248. L'analyse a localisé un changement d'expert près d'une égalité de routage,
après des arrondis d'attention. La référence numérique utilise donc explicitement
`-ctk f32 -ctv f32 -fa off`. Cela confirme la conversion en pleine précision,
**pas** une identité numérique sous tous les kernels, matériels ou quantifications.
La qualité Q4 devra être comparée au checkpoint entraîné. La compatibilité de
chaque version d'Ollama/LM Studio n'a pas été vérifiée ; il faut un moteur récent
incluant cette architecture. GGUF n'implique pas une compatibilité universelle.

## Budget et passage entre comptes

Au [tarif Modal consulté le 24 septembre 2026](https://modal.com/pricing), H100 =
3,9492 $/h, CPU = 0,04716 $/cœur/h et mémoire = 0,007992 $/GiB/h. Avec 4 cœurs et
16 GiB, enveloppe indicative **4,265712 $/h**, hors construction, transferts et
surcoûts éventuels. B200 dans la même configuration serait 6,566112 $/h : il doit
être au moins 1,539× plus rapide pour améliorer les tokens/$. Aucun débit n'a été
mesuré sur ces GPU dans cette tâche.

Répartition proposée : 2 $ pilote/préflights, 48 $ prétrain, 6 $ SFT, 4 $ marge.
Par compte : premier 2 + 26 $, second 22 + 6 $, avec 2 $ de marge chacun. Les
plafonds temporels proposés, à ajuster aux soldes réels, sont 21 600 s puis
18 000 s de prétrain (environ 46,92 $ de calcul nominal au total). Ils laissent
une marge pour l'arrêt et les checkpoints ; ils ne sont pas un plafond de facture.

Avec 48 $ entièrement consacrés au calcul nominal, 11,25 h permettent environ
1,01 B / 2,03 B / 4,05 B tokens à 25k / 50k / 100k tokens/s respectivement.
Évaluations, sauvegardes et démarrages réduiront ces chiffres. La réserve de 4 B
n'est donc pas une annonce de volume effectivement entraîné.

Le pilote proposé : H100, bf16, contexte 1024, batch 8, accumulation 8, sans
compilation, Muon 0,002 + AdamW 0,0003, 3 updates de chauffe puis 10 mesurées.
Timeout demandé 900 s, plus 240 s maximum pour démarrage/fin du subprocess :
ordre de grandeur inférieur à 1,36 $ de calcul GPU/CPU/RAM nominal. Il faut aussi
vérifier la loss finie et les kernels FLA/causal-conv1d. Aucun repli lent ne doit
être accepté silencieusement.

Après mesure, fixer le nombre **global** d'updates (65 536 tokens/update prétrain),
le schedule cosine et la durée propre à chaque compte. Ne pas changer
`--max-steps` à mi-parcours pour « ajouter une seconde moitié » : cela change le
schedule. Le second compte reçoit `ckpt_latest.pt` entier, pas le seul export HF.
Le préflight vérifie config, manifests, tokenizer, optimiseurs, RNG et réglages
stables avant location du GPU. Un test CPU vérifie l'identité bit à bit d'une
trajectoire 3 updates avec une trajectoire 1 + reprise de 2 updates.

Les profils Modal et les soldes réels ne sont pas configurés/vérifiés ici.
Après « go », connecter les deux profils, envoyer seulement les fichiers requis
par les manifests (pas `raw/`, ni les bases SQLite), puis exécuter le préflight
CPU du Volume et le petit pilote. Un résultat défavorable du pilote demande une
correction de la recette avant le prétrain complet.

### Commandes de lancement préparées, non exécutées

```bash
python -m pip install modal==1.5.5
# Authentification distincte des deux comptes, après accord.
modal token new --profile compte1
modal token new --profile compte2
modal volume create --profile compte1 frlm-v5
modal volume create --profile compte2 frlm-v5
modal volume put --profile compte1 frlm-v5 data-v5/modal-upload /data-v5
modal volume put --profile compte2 frlm-v5 data-v5/modal-upload /data-v5
# Préflight cloud : CPU uniquement, consomme quand même un peu de crédit.
modal run --profile compte1 modal_v5.py --mode pilot --check-only
# GPU uniquement après accord explicite.
modal run --profile compte1 modal_v5.py --mode pilot --seconds 900 --go
# GLOBAL_STEPS est fixé une fois à partir du pilote, et reste identique.
modal run --profile compte1 modal_v5.py --mode pretrain --steps "$GLOBAL_STEPS" --seconds 21600 --go
modal volume get --profile compte1 frlm-v5 runs/fr-v5-qwen4exp/pretrain/ckpt_latest.pt transfert/ckpt_latest.pt
shasum -a 256 transfert/ckpt_latest.pt
modal volume put --profile compte2 frlm-v5 transfert/ckpt_latest.pt runs/fr-v5-qwen4exp/pretrain/ckpt_latest.pt
modal run --profile compte2 modal_v5.py --mode pretrain --steps "$GLOBAL_STEPS" --seconds 18000 --resume runs/fr-v5-qwen4exp/pretrain/ckpt_latest.pt --go
```

Ne pas déplacer/supprimer les anciens runs. Copier aussi le journal du premier
compte pour conserver l'historique ; vérifier le SHA-256 du checkpoint téléchargé
et celui relu depuis le second Volume avant reprise.

## Vérification locale

- `python -m unittest discover -s tests -v` : **83 tests réussis** dans
  l'environnement v5 ; 83 collectés, 7 sauts attendus dans l'environnement
  historique PyTorch 2.6 / sans dépendances v5.
- Tests v5 : schéma Parquet imbriqué, quotas, filtrage, déduplication, reprise
  de préparation, corruption, masques, causalité, cache, gradients/routeurs,
  Muon sur experts, sérialisation, reprise exacte, préflight et transition SFT.
- Tests de garde Modal : durées invalides rejetées, mode par défaut sans GPU,
  `check_only` sans GPU même avec `go`, schedule conservé entre commandes.
- CrossHair sur `verification/v5_contracts.py`, `--analysis_kind=asserts
  --per_condition_timeout=15 --report_all` : **Not confirmed**, aucun
  contre-exemple rapporté, résultat inconclusif. Domaine : budget 1..10^12,
  trois poids 1..100. Complément exhaustif par tests : budgets 1..29 et tous
  les triplets de poids 1..3. Aucune preuve complète du pipeline revendiquée.
- `.github/workflows/cpu.yml` : job historique et nouveau job v5, suite CPU et
  conversion GGUF miniature (exécutée aussi localement : erreur F32 0,00001097). `actionlint` valide le YAML. Aucun push ni run
  GitHub distant n'a été lancé.
- `python -m pip check` : aucune dépendance cassée après épinglage du solveur
  de vérification Z3 à 4.15.1.0 (wheel 5.1 incompatible avec les tags macOS locaux).
- Analyse AST : 49 fichiers valides ; `python run.py --help` et `git diff --check` réussis.
- Validation CUDA, compilation de l'image Modal, débit, VRAM, qualité du modèle
  et dégradation Q4 après entraînement restent **non exécutés**.

Fichiers principaux : `frlm/model_v5.py`, `frlm/prepare_v5.py`,
`frlm/prepare_sft_v5.py`, `frlm/export_v5.py`, `modal_v5.py`, `recipes/v5_*.json`,
`requirements-v5.txt`, `bench/verify_v5_gguf.py`, `tests/test_v5.py`,
`verification/v5_contracts.py`. Adaptations communes : routage dans
`frlm/__init__.py`, optimiseur, trainer, banc de vitesse, préflight, CI et docs.
