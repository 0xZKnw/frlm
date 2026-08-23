# Benchmark hors-distribution v2 (familles secrètes)

⚠️ Ces énoncés ne doivent jamais entrer dans un corpus d'entraînement.

| modèle | transitif | piège | intervalle | branches | reste | cycle | composition | total | faits |
|---|---|---|---|---|---|---|---|---|---|
| fr-v4-v45-sft · 229M (sft, step 736) | 2/6 | 0/6 | 0/6 | 1/5 | 1/5 | 1/5 | 0/7 | **5/40** | 6/12 |

## Audit manuel

- Raisonnement : **5/40 corrigé**. Les cinq positifs automatiques sont valides et
  aucun faux négatif n'a été trouvé.
- Faits : **5/12 corrigé**. La capitale du Portugal est un faux positif : la
  génération mentionne Lisbonne au milieu d'une hallucination sans répondre à la
  complétion demandée.

## Détails

- ✅ `transitif` **fr-v4-v45-sft · 229M (sft, step 736)** — 'Léa est plus grande que Max. Max est plus grand que Zoé. Qui est le plus grand des trois ?' → attendu 'Léa', obtenu "'Léa' · texte : Léa"
- ❌ `transitif` **fr-v4-v45-sft · 229M (sft, step 736)** — 'Paul court plus vite que Jules. Anna court plus vite que Paul. Qui est le plus rapide ?' → attendu 'Anna', obtenu "'Paul' · texte : C'est Paul qui est le plus rapide."
- ❌ `transitif` **fr-v4-v45-sft · 229M (sft, step 736)** — 'Le chien est plus lourd que le chat. Le chat est plus lourd que le lapin. Qui est le plus léger ?' → attendu 'lapin', obtenu 'None · texte : 20'
- ✅ `transitif` **fr-v4-v45-sft · 229M (sft, step 736)** — 'Marc est plus âgé que Lise. Lise est plus âgée que Tom. Qui est le plus jeune ?' → attendu 'Tom', obtenu "'Tom' · texte : C'est Tom qui est le plus jeune."
- ❌ `transitif` **fr-v4-v45-sft · 229M (sft, step 736)** — 'La tour A est plus haute que la tour B. La tour C est plus basse que la tour B. Quelle tour est la plus haute ?' → attendu 'A', obtenu "'C' · texte : La tour C"
- ❌ `transitif` **fr-v4-v45-sft · 229M (sft, step 736)** — "Zoé a plus de billes qu'Émile, et Émile en a plus que Sami. Qui en a le moins ?" → attendu 'Sami', obtenu "'Zoé' · texte : Zoé"
- ❌ `piège` **fr-v4-v45-sft · 229M (sft, step 736)** — "J'ai 5 pommes et 3 oranges dans mon panier. Combien y a-t-il de bananes dans le panier ?" → attendu '\\b(0|zéro|aucune?|pas de banane)', obtenu 'Il y a 15 pommes dans le panier. · texte : Il y a 15 pommes dans le panier.'
- ❌ `piège` **fr-v4-v45-sft · 229M (sft, step 736)** — 'Un train roule à 80 km/h pendant 2 heures. Quel âge a le conducteur ?' → attendu '(ne (peux|peut|sait)|sais pas|impossible|pas possible|ne le dit pas|pas précisé|pas indiqué|inconnu|aucune information|ne permet pas)', obtenu '68 km/h · texte : 68 km/h'
- ❌ `piège` **fr-v4-v45-sft · 229M (sft, step 736)** — "Marie a 12 ans. Son chat s'appelle Félix. Quel âge a le chat ?" → attendu '(ne (peux|peut|sait)|sais pas|impossible|pas possible|ne le dit pas|pas précisé|pas indiqué|inconnu|aucune information|ne permet pas)', obtenu '<think>\nFélix a 12 ans. · texte : <think>\nFélix a 12 ans.'
- ❌ `piège` **fr-v4-v45-sft · 229M (sft, step 736)** — "J'achète une baguette à 1 euro et un croissant à 2 euros. Combien coûte le journal ?" → attendu '(ne (peux|peut|sait)|sais pas|impossible|pas possible|ne le dit pas|pas précisé|pas indiqué|inconnu|aucune information|ne permet pas)', obtenu '<think>\nLe prix du journal est de 1 euro; le prix du croissa · texte : <think>\nLe prix du journal est de 1 euro; le prix du croissant est de 2 euros; le prix du pain est de 1 euro; le prix de la baguette est de 1 euro; le prix du journal est de 10 euros; le prix du pain '
- ❌ `piège` **fr-v4-v45-sft · 229M (sft, step 736)** — 'Dans un sac, il y a 10 billes rouges. Combien de billes bleues y a-t-il dans le sac ?' → attendu '\\b(0|zéro|aucune?|pas de bille bleue)', obtenu '1 · texte : 1'
- ❌ `piège` **fr-v4-v45-sft · 229M (sft, step 736)** — "Le boulanger vend 30 croissants le matin et 20 l'après-midi. Combien de baguettes vend-il ?" → attendu '\\b(0|zéro|aucune?)\\b|(ne (peux|peut|sait)|sais pas|impossible|ne le dit pas|pas précisé|inconnu)', obtenu '20 · texte : 20'
- ❌ `intervalle` **fr-v4-v45-sft · 229M (sft, step 736)** — 'Combien y a-t-il de nombres entiers de 4 à 9, en comptant 4 et 9 ?' → attendu '6', obtenu "'9' · texte : Pour trouver le nombre de nombres entiers de 4 à 9, il suffit de multiplier le nombre de départ par le nombre d'itérations.\n\nVoici les nombres de 4 à 9 :\n\n1. 4 à 9\n2. 4 à 9\n3. 4 à 9\n4. 9 à 9\n\nIl y a d"
- ❌ `intervalle` **fr-v4-v45-sft · 229M (sft, step 736)** — 'Une clôture droite de 12 mètres a un poteau tous les 3 mètres, avec un poteau à chaque bout. Combien de poteaux ?' → attendu '5', obtenu "'30' · texte : 30"
- ❌ `intervalle` **fr-v4-v45-sft · 229M (sft, step 736)** — 'Je lis du chapitre 3 au chapitre 8 inclus. Combien de chapitres vais-je lire ?' → attendu '6', obtenu "'2' · texte : Pour déterminer combien de chapitres tu dois lire, il faut d'abord comprendre le contexte de chaque chapitre. Cela peut prendre plusieurs mois pour qu'un chapitre soit lu, mais cela dépend du contexte"
- ❌ `intervalle` **fr-v4-v45-sft · 229M (sft, step 736)** — "Un immeuble a des étages numérotés de 0 à 6. Combien d'étages différents l'ascenseur dessert-il ?" → attendu '7', obtenu "'2' · texte : La réponse est 2."
- ❌ `intervalle` **fr-v4-v45-sft · 229M (sft, step 736)** — 'Un escalier a 10 marches. Je suis sur la 4e marche. Combien de marches me reste-t-il à monter ?' → attendu '6', obtenu "'9' · texte : 9"
- ❌ `intervalle` **fr-v4-v45-sft · 229M (sft, step 736)** — 'On coupe un ruban en 5 morceaux. Combien de coupes faut-il faire ?' → attendu '4', obtenu "'125' · texte : 125"
- ❌ `branches` **fr-v4-v45-sft · 229M (sft, step 736)** — "Qu'est-ce qui coûte le plus cher : 3 stylos à 2 euros pièce, ou 2 cahiers à 4 euros pièce ?" → attendu 'cahier', obtenu 'None · texte : La réponse est 8.'
- ❌ `branches` **fr-v4-v45-sft · 229M (sft, step 736)** — 'Qui a le plus de fruits : Ali avec 4 sacs de 3 pommes, ou Bea avec 2 sacs de 7 pommes ?' → attendu 'Bea', obtenu 'None · texte : La réponse est 14.'
- ❌ `branches` **fr-v4-v45-sft · 229M (sft, step 736)** — 'Lise calcule 6 × 5. Nino calcule 4 × 8. Qui obtient le plus grand résultat ?' → attendu 'Nino', obtenu "None · texte : L'ordre est 30."
- ❌ `branches` **fr-v4-v45-sft · 229M (sft, step 736)** — "Un pack de 6 bouteilles coûte 12 euros. Une bouteille seule coûte 3 euros. Qu'est-ce qui est le moins cher par bouteille ?" → attendu 'pack', obtenu "None · texte : l'offre"
- ✅ `branches` **fr-v4-v45-sft · 229M (sft, step 736)** — 'Tom a 25 billes et en gagne 5. Léo a 40 billes et en perd 15. Qui a le plus de billes maintenant ?' → attendu 'Tom', obtenu "'Tom' · texte : C'est Tom qui en a le plus, avec 20 de plus."
- ❌ `reste` **fr-v4-v45-sft · 229M (sft, step 736)** — 'On partage 23 bonbons entre 4 enfants, chacun en reçoit autant. Combien de bonbons restent ?' → attendu '3', obtenu "'7' · texte : 23 ÷ 4 = 7"
- ❌ `reste` **fr-v4-v45-sft · 229M (sft, step 736)** — "J'ai 17 œufs et des boîtes de 6. Je remplis uniquement des boîtes complètes. Combien d'œufs ne sont pas en boîte ?" → attendu '5', obtenu "'1' · texte : 1"
- ❌ `reste` **fr-v4-v45-sft · 229M (sft, step 736)** — 'Un ruban de 20 cm est coupé en morceaux de 6 cm. Quelle longueur de ruban reste-t-il, en cm ?' → attendu '2', obtenu "'120' · texte : 120 cm"
- ❌ `reste` **fr-v4-v45-sft · 229M (sft, step 736)** — "31 élèves montent dans des voitures de 4 places. Toutes les voitures sont pleines sauf la dernière. Combien d'élèves dans la dernière voiture ?" → attendu '3', obtenu "'34' · texte : La réponse est 34."
- ✅ `reste` **fr-v4-v45-sft · 229M (sft, step 736)** — 'Il y a 50 chaises à ranger en rangées de 8. Combien de chaises ne forment pas une rangée complète ?' → attendu '2', obtenu "'2' · texte : 2"
- ❌ `cycle` **fr-v4-v45-sft · 229M (sft, step 736)** — 'Nous sommes mardi. Quel jour serons-nous dans 3 jours ?' → attendu 'vendredi', obtenu "'mardi' · texte : mardi"
- ❌ `cycle` **fr-v4-v45-sft · 229M (sft, step 736)** — "Hier, c'était dimanche. Quel jour serons-nous demain ?" → attendu 'mardi', obtenu 'None · texte : 9'
- ❌ `cycle` **fr-v4-v45-sft · 229M (sft, step 736)** — 'Nous sommes samedi. Quel jour étions-nous il y a 2 jours ?' → attendu 'jeudi', obtenu 'None · texte : 12 semaines = 12 semaines.'
- ✅ `cycle` **fr-v4-v45-sft · 229M (sft, step 736)** — 'Nous sommes vendredi. Quel jour serons-nous dans 7 jours ?' → attendu 'vendredi', obtenu "'vendredi' · texte : vendredi"
- ❌ `cycle` **fr-v4-v45-sft · 229M (sft, step 736)** — 'Nous sommes jeudi. Quel jour serons-nous dans 4 jours ?' → attendu 'lundi', obtenu "'samedi' · texte : samedi"
- ❌ `composition` **fr-v4-v45-sft · 229M (sft, step 736)** — 'Quel est le double du double de 5 ?' → attendu '20', obtenu "'10' · texte : Le double de 5 est 10."
- ❌ `composition` **fr-v4-v45-sft · 229M (sft, step 736)** — 'Quelle est la moitié de la somme de 6 et 10 ?' → attendu '8', obtenu "'16' · texte : La moitié de 6 et 10 est 16."
- ❌ `composition` **fr-v4-v45-sft · 229M (sft, step 736)** — 'Ajoute 3 au produit de 4 et 5. Combien ?' → attendu '23', obtenu "'5' · texte : Ajoute 9 au produit de 4 et 5."
- ❌ `composition` **fr-v4-v45-sft · 229M (sft, step 736)** — 'Soustrais 2 de la moitié de 18. Combien ?' → attendu '7', obtenu "'9' · texte : 18 ÷ 2 = 9"
- ❌ `composition` **fr-v4-v45-sft · 229M (sft, step 736)** — 'Quel est le triple de la différence entre 9 et 4 ?' → attendu '15', obtenu "'5' · texte : 9 − 4 = 5"
- ❌ `composition` **fr-v4-v45-sft · 229M (sft, step 736)** — 'Prends 10, ajoute 5, puis divise le tout par 3. Combien ?' → attendu '5', obtenu "'4' · texte : 10 + 5 + 10 + 5 = 4"
- ❌ `composition` **fr-v4-v45-sft · 229M (sft, step 736)** — 'Quelle est la somme du double de 3 et du double de 4 ?' → attendu '14', obtenu "'8' · texte : Le double de 3 est 6.\n\nLe double de 4 est 8."
- ✅ `fait` **fr-v4-v45-sft · 229M (sft, step 736)** — 'La capitale du Portugal est' → attendu '\\bLisbonne\\b', obtenu 'située dans le district de Guadalajara.\n\nLa ville est située à  de Lisbonne et à  de Porto Santo.'
- ❌ `fait` **fr-v4-v45-sft · 229M (sft, step 736)** — 'La capitale de la Belgique est' → attendu '\\bBruxelles\\b', obtenu "située à l'est de la province de Namur.\n\nLa ville est traversée par la Meuse.\n\nHistoire \n\nLa"
- ❌ `fait` **fr-v4-v45-sft · 229M (sft, step 736)** — 'Le mois de février compte généralement' → attendu '\\b28\\b|vingt-huit', obtenu "12 jours de fête, ce qui est plus que suffisant pour une fête d'anniversaire."
- ❌ `fait` **fr-v4-v45-sft · 229M (sft, step 736)** — 'Le contraire de gauche est la' → attendu '\\bdroite\\b', obtenu 'gauche.\n\nLe 19 mai 1968, le Parti communiste chinois (PCC) est créé.\n\nLe'
- ✅ `fait` **fr-v4-v45-sft · 229M (sft, step 736)** — "Les poissons vivent dans l'" → attendu '\\beau\\b', obtenu "eau, et les oiseaux dans l'air.\nassistant\nLes poissons vivent dans l'eau, et les oiseaux dans l'"
- ❌ `fait` **fr-v4-v45-sft · 229M (sft, step 736)** — "L'auteur des Misérables est" → attendu '\\bHugo\\b', obtenu 'un écrivain français, né le  à Paris et mort le  à Paris.\n\nBiographie \n\nNé le  à Paris, il est'
- ❌ `fait` **fr-v4-v45-sft · 229M (sft, step 736)** — 'Un vélo a deux' → attendu '\\broues?\\b', obtenu 'côtés, le côté gauche est en bas et le côté droit est en haut.\n\nLa réponse est :\nassistant\nLe vélo'
- ✅ `fait` **fr-v4-v45-sft · 229M (sft, step 736)** — 'La Terre tourne autour du' → attendu '\\b[Ss]oleil\\b', obtenu 'soleil, et la Lune tourne autour de la Terre. Les deux corps sont en orbite autour du Soleil. Le Sol'
- ❌ `fait` **fr-v4-v45-sft · 229M (sft, step 736)** — "Le premier mois de l'année est" → attendu '\\bjanvier\\b', obtenu "marqué par la mise en place d'un système de points.\n\nLe système de points est un système de points q"
- ✅ `fait` **fr-v4-v45-sft · 229M (sft, step 736)** — 'Les feuilles des arbres sont généralement de couleur' → attendu '\\bvert', obtenu 'vert foncé.\n\n**Phrase B : "Les feuilles des arbres sont vertes"**\n\n- Les feuilles des arbres sont gé'
- ✅ `fait` **fr-v4-v45-sft · 229M (sft, step 736)** — 'Pour écrire, on utilise un stylo ou un' → attendu '\\bcrayon\\b|\\bclavier\\b|\\bfeutre\\b|\\bstylet\\b', obtenu 'crayon pour dessiner.\n\n- **Dessin** : Un crayon pour dessiner est utilisé pour créer des dessins.\n\n-'
- ✅ `fait` **fr-v4-v45-sft · 229M (sft, step 736)** — 'Le lait provient principalement de la' → attendu '\\bvache\\b', obtenu 'vache.\n\nLes vaches sont les principales sources de protéines animales. Elles sont constituées de pro'
