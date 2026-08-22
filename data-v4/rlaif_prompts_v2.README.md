# Pool RLAIF v2 — 70 prompts ciblés sur les murs du run 1

Complément de `rlaif_prompts.jsonl` (153 prompts). Les deux fichiers s'utilisent
ensemble : `--pool data-v4/rlaif_prompts.jsonl,data-v4/rlaif_prompts_v2.jsonl`
(c'est le défaut). Total 223 prompts, tirés sans remise.

## Pourquoi ces prompts

Chaque famille vise un échec précis, observé pendant les 100 steps du run 1
(3 600 rollouts jugés). Rien n'a été ajouté « au cas où ».

| Famille | n | Mur visé (constat du run 1) |
|---|---|---|
| `calc` taux implicites | 12 | « 6 litres aux 100 km, combien pour 300 km ? » → 5/6 répondent « 600 litres ». Le modèle attrape les deux nombres visibles et multiplie ; il ne représente pas le taux. Même mécanisme sur le taxi (5 € + 2 €/km → « 5+2=7 ») et le restaurant (90 ÷ 6 → 0/6, alors que la division est acquise ailleurs). |
| `calc` deux opérations | 10 | Pattern « première opération puis stop » : café 3×2+2×1 → « 6 euros » (6/6), pommes 5+7−3 → « 12 » (4/6), métro 2×2×5 → « 10 ». La deuxième opération ne vient jamais. |
| `piege` comptage/mesure | 14 | **0 % de refus** sur cette famille : « depuis 1584 » pour la bougie, « 350 kilomètres » inventés pour l'autoroute, pseudo-calculs pour les fichiers (3×100=300). Les refus n'ont émergé que sur les pièges *personnels* (métier du père ~4/6) — aucun transfert. |
| `chat` description de soi | 17 | Trou complet : « Qu'est-ce que tu sais faire ? », « Tu parles quelles langues ? » (listes qui bouclent), et l'empathie qui produit un cours de médecine au lieu de « repose-toi ». Plusieurs prompts visent aussi la calibration (« Tu es sûr de ta réponse ? ») et l'honnêteté sur l'absence de vie intérieure. |
| `fait` élémentaires | 9 | « Quel animal aboie ? » → zéro « chien » sur tout le run. Faits de base absents du pré-entraînement. |
| `consigne` | 8 | Renfort des formats courts, déjà bons (6/6) mais fragiles hors des formulations vues. |

## Contamination : ce qu'il faut savoir pour lire les scores

- **Familles secrètes du bench v2** (cycle, intervalle, composition, transitif,
  branches, reste) : **aucune n'a été ajoutée**, vérifié prompt par prompt. En
  particulier, pas de « double du double » ni de comparaison transitive.
- **Piège** reste *semi-in-dist* par décision explicite (entraîner le refus était
  le but) → garder l'astérisque dans les rapports bench v2.
- **Transitif** doit aussi porter un astérisque, mais pour une raison antérieure à
  ce fichier : `synth.make_problem` produit déjà des problèmes de transitivité, et
  le flux synth représente 35 % des tirages RLAIF.
- **Bench v1 OOD** (les 40 problèmes qui servent d'éval *pendant* le run) : aucune
  collision littérale, vérifiée automatiquement. Mais il contient déjà des
  problèmes de taux simples (« un kilo de pommes coûte 3 euros », « 10 pages par
  jour → une semaine ») et deux problèmes à deux opérations. Entraîner sur les
  taux devrait donc faire monter le score OOD v1 en partie par transfert légitime,
  en partie parce qu'on se rapproche de sa distribution. **La mesure honnête reste
  bench_ood_v2**, dont les familles sont intactes.

## Format

```json
{"type": "calc",     "q": "...", "ans": 14}
{"type": "piege",    "q": "..."}
{"type": "fait",     "q": "...", "attendu": "regex"}
{"type": "chat",     "q": "...", "note": "ce que le juge doit attendre"}
{"type": "consigne", "q": "...", "ans": 42, "check": "nombre_seul"}
```
`check` ∈ `nombre_seul` | `un_mot` | `oui_non` | `max_mots`. Les huit consignes ont
été auto-testées : le vérificateur Python accepte bien la réponse attendue.
