# Correctifs du pipeline reason45c — 24 septembre 2026

Ces changements corrigent des erreurs de données et de mesure. **Aucun gain de
qualité du 229M n'a été mesuré, aucun entraînement de ce modèle n'a été relancé.**
La nouvelle demande v5 et son budget sont traités dans
[le rapport v5](v5_preparation_20260924.md).

## Erreurs confirmées et corrections

| Erreur / cause | Correction | Régression reproductible |
|---|---|---|
| `9 moins 4 moins 2` masquait deux AST donnant 3 et 7 | Parenthèses explicites, nombres négatifs conservés dans `_natural` | Les deux AST et interprétation indépendante de 2250 arbres binaires à deux opérations |
| Numéros et noms `r1`, `r2` révélaient l'ordre d'exécution | Noms aléatoires sans rang, labels après mélange, acceptation de tous les tris topologiques valides | Toutes les permutations de 750 exercices, train et quatre holdouts |
| Position d'erreur fortement concentrée sur 1 et traces triviales | Au moins deux opérations, équilibre par longueur et contrôle d'une seule erreur locale | 600 exemples pour chacun des cinq splits, écarts de fréquence au plus 1 |
| Mix 35/65 interprété en conversations malgré la cible en tokens | Conversion par nombre moyen de tokens assistant avant le sampler | Grille exhaustive des longueurs 1..512 ; contributions attendues équilibrées |
| Identifiants de surface différents mais mêmes formulations train/dev | Textes effectivement distincts pour contraintes, incertitude, suivi d'état RL | 1000 seeds pour chaque split/capacité concernés |
| Moyenne de trois valeurs contenait toujours la réponse au milieu | Triplet permuté dont aucune entrée n'est égale à sa moyenne | 2000 exemples train/dev, moyenne recalculée et vérificateur inchangé |
| Remise de 25 % sur certains prix tronquée par division entière | Prix garantissant un résultat entier exact | 2000 seeds du générateur ; égalité rationnelle contrôlée |
| Consigne JSON n'explicitant pas le double attendu | Consigne précise sur `resultat` et sur la parité du nombre initial | 2000 seeds du générateur de contraintes |
| Profil RL réutilisable après changement du générateur | Version `rl-tasks-v45-2`, nouveaux noms de profils, rejet des anciens profils | Tests de validation de version et reprise |
| Préflight limité aux chemins train/mid/SFT | Module CPU partagé, fichiers RL/profilage, configs, poids, tokenizer, profils | Fichiers absents/corrompus, reprise et arguments du wrapper |
| Corpus conversationnel mal formé parfois toléré | Rejet des masques non binaires, frontière supervisée et dernier document non terminé | Cas invalides temporaires et test de padding/isolation |

Les regroupements commutatifs sont canonicalisés pour identifier les programmes.
Le split dépend du programme, indépendamment de sa formulation et de son objectif.
Les formes réservées au holdout de surface et les topologies réservées à celui de
structure sont réellement séparées dans les nouveaux manifests. OOD v2 n'est pas
utilisé pour produire les exemples.

Le profilage ajoute baselines constantes, majorité oracle, attentes aléatoires et
résultats par objectif/difficulté. Une bonne réponse constante reste correcte :
le score n'est pas artificiellement diminué dans le vérificateur.
Les sorties `number_only` imposent réellement un entier seul, signé si nécessaire.

Les masques SFT, conversations isolées et normalisation globale des tokens
assistant de v4.5 ont été vérifiés : les anciens défauts de concaténation et de
normalisation ne sont pas présentés comme toujours présents. Les tests comparent
les gradients d'une accumulation à ceux du batch global sur mini-modèles v2/v3.
Les rollbacks modèle/optimiseur et les deux contrôleurs KL restent couverts par la
suite existante. L'écriture atomique de best est testée en injectant un échec de
copie, sans corruption du précédent fichier.

## Artefacts distincts et données à régénérer

- [Rapport de pipeline](reason45c_pipeline_20260924.json) : 20 000 exemples,
  600 tâches par split, empreintes, mix réel et référence historique inchangée.
- [Baselines sans modèle](reason45c_baselines_20260924.json) : résultats par
  objectif/difficulté. Ces nombres décrivent des stratégies triviales et le
  générateur corrigé, pas les performances d'un checkpoint.
- Corpus d'audit généré localement dans `/tmp/frlm-reason45c-real-20260924` ;
  les bins de rétention absents localement sont signalés comme différés.
  Le préflight cloud les exige avant GPU. Aucun ancien corpus n'est remplacé.
- Pour une utilisation v4 ultérieure, publier une fois `reason_v45c_*` avec
  `prepare-reason-bootstrap-v45c`, longueur 512, seed 455900, puis auditer.
  Ne pas écraser `reason45`/`reason45b`. La republication c est refusée.
- Les sources utilisant les remises/consignes JSON corrigées nécessitent de
  nouveaux bins avant réutilisation. Les anciens profils RL ne doivent pas être
  raffinés comme s'ils provenaient du nouveau générateur : produire `profile_v2.json`
  et, à la reprise, `profile_phase2_v2.json`.
- `final_sealed` est réservé à une seule évaluation après gel du protocole et du
  checkpoint, avec `--final-eval --splits final_sealed`. Son hash est fixé ; son
  contenu local n'est pas chiffré. OOD v2 reste du développement historique.

## Vérifications exécutées et limites

```bash
python -m unittest discover -s tests -v
python -m unittest tests.test_pipeline_corrections -v
python run.py --help
python -m frlm.audit_reason_bootstrap_v45 --data-dir /tmp/frlm-reason45c-real-20260924 --recipe reason45c
python -m crosshair check verification/reason45c_contracts.py --analysis_kind=asserts --per_condition_timeout=15 --report_all
```

Suite finale comprenant v5 : **83 tests**, succès sous PyTorch 2.9.1 ; sous
PyTorch 2.6, 83 collectés et 7 sauts correspondant aux dépendances v5 optionnelles.
Analyse AST de 49 fichiers Python réussie, CLI valide. CI CPU créée et validée par
`actionlint`, jobs sur pull requests et pushes main/master, sans GPU/identifiants.
Aucun push ni exécution de cette nouvelle CI sur GitHub.

CrossHair a retourné **Not confirmed** pour les deux contrats : aucune preuve
complète. Bornes : soustractions avec trois entiers entre -600 et 600 et dernier
opérande non nul ; mix de deux longueurs entre 1 et 512. Les tests exhaustifs et
génératifs complètent cette recherche inconclusive, sans prouver tous les AST,
toutes les formulations ni l'absence de fuite sémantique générale.

Fichiers corrigés : `frlm/reason_bootstrap_v45.py`, audit et profileur associés,
`frlm/data.py`, `frlm/synth_programs.py`, générateur/profileur/trainer RL v4.5,
`frlm/modal_preflight.py`, `modal_app.py`, `run.py`, tests, contrats, CI et docs.
Les réglages d'entraînement, benchmarks et contrôles CUDA/cloud restent non
exécutés. Le petit pilote chiffré désormais proposé concerne uniquement la v5 :
H100, 3 updates de chauffe et 10 mesurées, enveloppe initiale de 2 $ incluant
préflight/démarrage, **seulement après « go »**.

## Publication GitHub et correction de portabilité

Le push demandé ensuite a publié `98120f6` sur main. Le premier run GitHub
`36039297989` a révélé un accès inutile à `/root/app` dans `with_gpu_peak` :
construire une commande appelait le résolveur de fichiers distants. Le runner
Linux non privilégié levait `PermissionError`, absent lors du test macOS.
La construction de commande analyse maintenant uniquement les arguments.

Le test avec `Path.stat` interdit reproduit l'échec avant correction et réussit
après. Les deux suites locales repassent : 83 tests, dont 7 sauts attendus dans
l'environnement historique. Le contrat de commande couvre quatre modes et
un profileur, idempotence comprise, pour tous les pics entiers de 1 à 2000 :
vérification exhaustive finie réussie. CrossHair, 15 s par condition, retourne
`Not confirmed` sans contre-exemple : aucune preuve complète revendiquée.
La CI est relancée par le commit correctif ; son résultat est consultable dans
[GitHub Actions](https://github.com/0xZKnw/frlm/actions/workflows/cpu.yml).
