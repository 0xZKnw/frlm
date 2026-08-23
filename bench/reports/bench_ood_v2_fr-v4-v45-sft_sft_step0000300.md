# Benchmark hors-distribution v2 (familles secrètes)

⚠️ Ces énoncés ne doivent jamais entrer dans un corpus d'entraînement.

| modèle | transitif | piège | intervalle | branches | reste | cycle | composition | total | faits |
|---|---|---|---|---|---|---|---|---|---|
| fr-v4-v45-sft · 229M (sft, step 300) | 2/6 | 0/6 | 0/6 | 1/5 | 0/5 | 1/5 | 1/7 | **5/40** | 4/12 |

## Audit manuel

- Raisonnement : **5/40 corrigé**. Les cinq positifs automatiques sont valides et
  aucun faux négatif n'a été trouvé.
- Faits : **3/12 corrigé**. La complétion sur la couleur des feuilles commence par
  `rouge vif`; sa mention ultérieure de feuilles vertes ne rend pas la réponse
  correcte.

## Détails

- ✅ `transitif` **fr-v4-v45-sft · 229M (sft, step 300)** — 'Léa est plus grande que Max. Max est plus grand que Zoé. Qui est le plus grand des trois ?' → attendu 'Léa', obtenu "'Léa' · texte : Léa"
- ❌ `transitif` **fr-v4-v45-sft · 229M (sft, step 300)** — 'Paul court plus vite que Jules. Anna court plus vite que Paul. Qui est le plus rapide ?' → attendu 'Anna', obtenu "'Paul' · texte : C'est Paul qui est le plus rapide."
- ❌ `transitif` **fr-v4-v45-sft · 229M (sft, step 300)** — 'Le chien est plus lourd que le chat. Le chat est plus lourd que le lapin. Qui est le plus léger ?' → attendu 'lapin', obtenu 'None · texte : 4'
- ✅ `transitif` **fr-v4-v45-sft · 229M (sft, step 300)** — 'Marc est plus âgé que Lise. Lise est plus âgée que Tom. Qui est le plus jeune ?' → attendu 'Tom', obtenu "'Tom' · texte : C'est Tom qui est le plus jeune."
- ❌ `transitif` **fr-v4-v45-sft · 229M (sft, step 300)** — 'La tour A est plus haute que la tour B. La tour C est plus basse que la tour B. Quelle tour est la plus haute ?' → attendu 'A', obtenu 'None · texte : La réponse est 2.'
- ❌ `transitif` **fr-v4-v45-sft · 229M (sft, step 300)** — "Zoé a plus de billes qu'Émile, et Émile en a plus que Sami. Qui en a le moins ?" → attendu 'Sami', obtenu "'Zoé' · texte : Zoé"
- ❌ `piège` **fr-v4-v45-sft · 229M (sft, step 300)** — "J'ai 5 pommes et 3 oranges dans mon panier. Combien y a-t-il de bananes dans le panier ?" → attendu '\\b(0|zéro|aucune?|pas de banane)', obtenu 'Il y a 15 pommes dans le panier. · texte : Il y a 15 pommes dans le panier.'
- ❌ `piège` **fr-v4-v45-sft · 229M (sft, step 300)** — 'Un train roule à 80 km/h pendant 2 heures. Quel âge a le conducteur ?' → attendu '(ne (peux|peut|sait)|sais pas|impossible|pas possible|ne le dit pas|pas précisé|pas indiqué|inconnu|aucune information|ne permet pas)', obtenu '68 km/h · texte : 68 km/h'
- ❌ `piège` **fr-v4-v45-sft · 229M (sft, step 300)** — "Marie a 12 ans. Son chat s'appelle Félix. Quel âge a le chat ?" → attendu '(ne (peux|peut|sait)|sais pas|impossible|pas possible|ne le dit pas|pas précisé|pas indiqué|inconnu|aucune information|ne permet pas)', obtenu '<think>\nFélix a 12 ans. · texte : <think>\nFélix a 12 ans.'
- ❌ `piège` **fr-v4-v45-sft · 229M (sft, step 300)** — "J'achète une baguette à 1 euro et un croissant à 2 euros. Combien coûte le journal ?" → attendu '(ne (peux|peut|sait)|sais pas|impossible|pas possible|ne le dit pas|pas précisé|pas indiqué|inconnu|aucune information|ne permet pas)', obtenu '<think>\nLe prix du journal est de 1 euro; le prix du croissa · texte : <think>\nLe prix du journal est de 1 euro; le prix du croissant est de 2 euros; le prix du journal est de 1 euro; le prix du journal est de 10 euros; le prix du journal est de 10 euros; le prix du jour'
- ❌ `piège` **fr-v4-v45-sft · 229M (sft, step 300)** — 'Dans un sac, il y a 10 billes rouges. Combien de billes bleues y a-t-il dans le sac ?' → attendu '\\b(0|zéro|aucune?|pas de bille bleue)', obtenu '1 · texte : 1'
- ❌ `piège` **fr-v4-v45-sft · 229M (sft, step 300)** — "Le boulanger vend 30 croissants le matin et 20 l'après-midi. Combien de baguettes vend-il ?" → attendu '\\b(0|zéro|aucune?)\\b|(ne (peux|peut|sait)|sais pas|impossible|ne le dit pas|pas précisé|inconnu)', obtenu '10 · texte : 10'
- ❌ `intervalle` **fr-v4-v45-sft · 229M (sft, step 300)** — 'Combien y a-t-il de nombres entiers de 4 à 9, en comptant 4 et 9 ?' → attendu '6', obtenu "'9' · texte : Pour trouver le nombre de nombres entiers de 4 à 9, il faut d'abord déterminer le nombre de départ.\n\nIl y a 9 nombres de 4 à 9.\n\nOn additionne ces nombres :\n\\[ 9 \\times 4 \\times 9 = 36 \\]\n\nDonc, il y "
- ❌ `intervalle` **fr-v4-v45-sft · 229M (sft, step 300)** — 'Une clôture droite de 12 mètres a un poteau tous les 3 mètres, avec un poteau à chaque bout. Combien de poteaux ?' → attendu '5', obtenu "'30' · texte : 30"
- ❌ `intervalle` **fr-v4-v45-sft · 229M (sft, step 300)** — 'Je lis du chapitre 3 au chapitre 8 inclus. Combien de chapitres vais-je lire ?' → attendu '6', obtenu "'10' · texte : Je vais lire 10 chapitres."
- ❌ `intervalle` **fr-v4-v45-sft · 229M (sft, step 300)** — "Un immeuble a des étages numérotés de 0 à 6. Combien d'étages différents l'ascenseur dessert-il ?" → attendu '7', obtenu "'36' · texte : La réponse est 36."
- ❌ `intervalle` **fr-v4-v45-sft · 229M (sft, step 300)** — 'Un escalier a 10 marches. Je suis sur la 4e marche. Combien de marches me reste-t-il à monter ?' → attendu '6', obtenu "'9' · texte : La réponse est 9."
- ❌ `intervalle` **fr-v4-v45-sft · 229M (sft, step 300)** — 'On coupe un ruban en 5 morceaux. Combien de coupes faut-il faire ?' → attendu '4', obtenu "'15' · texte : <think>\nIl y a 125 morceaux dans un ruban et 5 de plus que 125, ce qui fait 125+5 = 130 morceaux. Il y a 130 morceaux dans un ruban et 5 de plus que 125, ce qui fait 130+5 = 135 morceaux. Il y a 135 m"
- ❌ `branches` **fr-v4-v45-sft · 229M (sft, step 300)** — "Qu'est-ce qui coûte le plus cher : 3 stylos à 2 euros pièce, ou 2 cahiers à 4 euros pièce ?" → attendu 'cahier', obtenu 'None · texte : 6'
- ❌ `branches` **fr-v4-v45-sft · 229M (sft, step 300)** — 'Qui a le plus de fruits : Ali avec 4 sacs de 3 pommes, ou Bea avec 2 sacs de 7 pommes ?' → attendu 'Bea', obtenu 'None · texte : La réponse est 14.'
- ❌ `branches` **fr-v4-v45-sft · 229M (sft, step 300)** — 'Lise calcule 6 × 5. Nino calcule 4 × 8. Qui obtient le plus grand résultat ?' → attendu 'Nino', obtenu "'Lise' · texte : Lise calcule 30."
- ❌ `branches` **fr-v4-v45-sft · 229M (sft, step 300)** — "Un pack de 6 bouteilles coûte 12 euros. Une bouteille seule coûte 3 euros. Qu'est-ce qui est le moins cher par bouteille ?" → attendu 'pack', obtenu 'None · texte : 15'
- ✅ `branches` **fr-v4-v45-sft · 229M (sft, step 300)** — 'Tom a 25 billes et en gagne 5. Léo a 40 billes et en perd 15. Qui a le plus de billes maintenant ?' → attendu 'Tom', obtenu "'Tom' · texte : C'est Tom qui en a le plus, avec 20 de plus."
- ❌ `reste` **fr-v4-v45-sft · 229M (sft, step 300)** — 'On partage 23 bonbons entre 4 enfants, chacun en reçoit autant. Combien de bonbons restent ?' → attendu '3', obtenu "'7' · texte : 23 ÷ 4 = 7"
- ❌ `reste` **fr-v4-v45-sft · 229M (sft, step 300)** — "J'ai 17 œufs et des boîtes de 6. Je remplis uniquement des boîtes complètes. Combien d'œufs ne sont pas en boîte ?" → attendu '5', obtenu "'1' · texte : 1"
- ❌ `reste` **fr-v4-v45-sft · 229M (sft, step 300)** — 'Un ruban de 20 cm est coupé en morceaux de 6 cm. Quelle longueur de ruban reste-t-il, en cm ?' → attendu '2', obtenu "'120' · texte : 120 cm"
- ❌ `reste` **fr-v4-v45-sft · 229M (sft, step 300)** — "31 élèves montent dans des voitures de 4 places. Toutes les voitures sont pleines sauf la dernière. Combien d'élèves dans la dernière voiture ?" → attendu '3', obtenu "'136' · texte : La réponse est 136."
- ❌ `reste` **fr-v4-v45-sft · 229M (sft, step 300)** — 'Il y a 50 chaises à ranger en rangées de 8. Combien de chaises ne forment pas une rangée complète ?' → attendu '2', obtenu "'1' · texte : 1"
- ❌ `cycle` **fr-v4-v45-sft · 229M (sft, step 300)** — 'Nous sommes mardi. Quel jour serons-nous dans 3 jours ?' → attendu 'vendredi', obtenu 'None · texte : Cela fait 9 jours.'
- ❌ `cycle` **fr-v4-v45-sft · 229M (sft, step 300)** — "Hier, c'était dimanche. Quel jour serons-nous demain ?" → attendu 'mardi', obtenu 'None · texte : 21'
- ❌ `cycle` **fr-v4-v45-sft · 229M (sft, step 300)** — 'Nous sommes samedi. Quel jour étions-nous il y a 2 jours ?' → attendu 'jeudi', obtenu 'None · texte : 12 semaines = 12 semaines.'
- ✅ `cycle` **fr-v4-v45-sft · 229M (sft, step 300)** — 'Nous sommes vendredi. Quel jour serons-nous dans 7 jours ?' → attendu 'vendredi', obtenu "'vendredi' · texte : vendredi"
- ❌ `cycle` **fr-v4-v45-sft · 229M (sft, step 300)** — 'Nous sommes jeudi. Quel jour serons-nous dans 4 jours ?' → attendu 'lundi', obtenu 'None · texte : Cela fait 16 jours.'
- ❌ `composition` **fr-v4-v45-sft · 229M (sft, step 300)** — 'Quel est le double du double de 5 ?' → attendu '20', obtenu "'10' · texte : 10"
- ❌ `composition` **fr-v4-v45-sft · 229M (sft, step 300)** — 'Quelle est la moitié de la somme de 6 et 10 ?' → attendu '8', obtenu "'16' · texte : La moitié de 6 est 16."
- ❌ `composition` **fr-v4-v45-sft · 229M (sft, step 300)** — 'Ajoute 3 au produit de 4 et 5. Combien ?' → attendu '23', obtenu "'9' · texte : 9"
- ❌ `composition` **fr-v4-v45-sft · 229M (sft, step 300)** — 'Soustrais 2 de la moitié de 18. Combien ?' → attendu '7', obtenu "'9' · texte : 18 ÷ 2 = 9"
- ❌ `composition` **fr-v4-v45-sft · 229M (sft, step 300)** — 'Quel est le triple de la différence entre 9 et 4 ?' → attendu '15', obtenu "'5' · texte : 9 − 4 = 5"
- ✅ `composition` **fr-v4-v45-sft · 229M (sft, step 300)** — 'Prends 10, ajoute 5, puis divise le tout par 3. Combien ?' → attendu '5', obtenu "'5' · texte : 15 − 10 = 5"
- ❌ `composition` **fr-v4-v45-sft · 229M (sft, step 300)** — 'Quelle est la somme du double de 3 et du double de 4 ?' → attendu '14', obtenu "'6' · texte : La somme du double de 3 et du double de 4 est 6."
- ❌ `fait` **fr-v4-v45-sft · 229M (sft, step 300)** — 'La capitale du Portugal est' → attendu '\\bLisbonne\\b', obtenu "Porto, située dans la région de l'Algarve.\n\nHistoire \n\nLa ville a été fondée par les Portugais en"
- ❌ `fait` **fr-v4-v45-sft · 229M (sft, step 300)** — 'La capitale de la Belgique est' → attendu '\\bBruxelles\\b', obtenu 'Anvers.\n\nLa ville est située dans la région de la Flandre-Occidentale, dans le département de la Fla'
- ❌ `fait` **fr-v4-v45-sft · 229M (sft, step 300)** — 'Le mois de février compte généralement' → attendu '\\b28\\b|vingt-huit', obtenu '10 jours de congé.\n\nLe mois de mars compte généralement 10 jours de congé.\n\nLe mois de mai compte gé'
- ❌ `fait` **fr-v4-v45-sft · 229M (sft, step 300)** — 'Le contraire de gauche est la' → attendu '\\bdroite\\b', obtenu 'gauche.\n\nLa gauche est une tendance politique, et la gauche est une tendance économique.\n\nLa gauche '
- ✅ `fait` **fr-v4-v45-sft · 229M (sft, step 300)** — "Les poissons vivent dans l'" → attendu '\\beau\\b', obtenu "eau, ils sont donc des poissons.\nassistant\n\nLes poissons vivent dans l'eau, et ils sont donc des poi"
- ❌ `fait` **fr-v4-v45-sft · 229M (sft, step 300)** — "L'auteur des Misérables est" → attendu '\\bHugo\\b', obtenu 'un écrivain français, né à Paris le  et mort à Paris le .\n\nBiographie \n\nNé à Paris le , il est le fi'
- ❌ `fait` **fr-v4-v45-sft · 229M (sft, step 300)** — 'Un vélo a deux' → attendu '\\broues?\\b', obtenu 'côtés, le côté gauche est en bas et le côté droit est en haut.\n\nLa réponse est :\nassistant\nLe vélo'
- ✅ `fait` **fr-v4-v45-sft · 229M (sft, step 300)** — 'La Terre tourne autour du' → attendu '\\b[Ss]oleil\\b', obtenu 'soleil, et la Lune tourne autour de la Terre. Les deux corps sont en orbite autour du Soleil. Le Sol'
- ❌ `fait` **fr-v4-v45-sft · 229M (sft, step 300)** — "Le premier mois de l'année est" → attendu '\\bjanvier\\b', obtenu 'un peu plus difficile, mais il y a une tendance à la baisse.\n\nEn résumé, la saison se termine généra'
- ✅ `fait` **fr-v4-v45-sft · 229M (sft, step 300)** — 'Les feuilles des arbres sont généralement de couleur' → attendu '\\bvert', obtenu 'rouge vif.\n\n**Phrase B :**\n\nLes feuilles des arbres sont généralement de couleur verte.\n\n**Analyse :'
- ✅ `fait` **fr-v4-v45-sft · 229M (sft, step 300)** — 'Pour écrire, on utilise un stylo ou un' → attendu '\\bcrayon\\b|\\bclavier\\b|\\bfeutre\\b|\\bstylet\\b', obtenu 'crayon.\n\n- **Stylo** : Un stylo est utilisé pour écrire. Il est souvent utilisé pour écrire des lett'
- ❌ `fait` **fr-v4-v45-sft · 229M (sft, step 300)** — 'Le lait provient principalement de la' → attendu '\\bvache\\b', obtenu 'laiterie de la rue de la Paix, à Paris.\n\nHistorique \n\nLe lait de la laiterie de la rue de la'
