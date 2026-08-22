# Benchmark hors-distribution v2 (familles secrètes)

⚠️ Ces énoncés ne doivent jamais entrer dans un corpus d'entraînement.

| modèle | transitif | piège | intervalle | branches | reste | cycle | composition | total | faits |
|---|---|---|---|---|---|---|---|---|---|
| fr-v4 · 229M (rlaif2, step 38) | 1/6 | 1/6 | 0/6 | 1/5 | 0/5 | 1/5 | 0/7 | **4/40** | 9/12 |

## Détails

- ✅ `transitif` **fr-v4 · 229M (rlaif2, step 38)** — 'Léa est plus grande que Max. Max est plus grand que Zoé. Qui est le plus grand des trois ?' → attendu 'Léa', obtenu "'Léa' · texte : Léa<|im_end|>"
- ❌ `transitif` **fr-v4 · 229M (rlaif2, step 38)** — 'Paul court plus vite que Jules. Anna court plus vite que Paul. Qui est le plus rapide ?' → attendu 'Anna', obtenu "'Paul' · texte : Paul"
- ❌ `transitif` **fr-v4 · 229M (rlaif2, step 38)** — 'Le chien est plus lourd que le chat. Le chat est plus lourd que le lapin. Qui est le plus léger ?' → attendu 'lapin', obtenu "'chien' · texte : Le chien.<|im_end|>"
- ❌ `transitif` **fr-v4 · 229M (rlaif2, step 38)** — 'Marc est plus âgé que Lise. Lise est plus âgée que Tom. Qui est le plus jeune ?' → attendu 'Tom', obtenu "'Marc' · texte : Marc"
- ❌ `transitif` **fr-v4 · 229M (rlaif2, step 38)** — 'La tour A est plus haute que la tour B. La tour C est plus basse que la tour B. Quelle tour est la plus haute ?' → attendu 'A', obtenu "'B' · texte : La tour B est la plus haute.<|im_end|>"
- ❌ `transitif` **fr-v4 · 229M (rlaif2, step 38)** — "Zoé a plus de billes qu'Émile, et Émile en a plus que Sami. Qui en a le moins ?" → attendu 'Sami', obtenu "'Zoé' · texte : Zoé."
- ❌ `piège` **fr-v4 · 229M (rlaif2, step 38)** — "J'ai 5 pommes et 3 oranges dans mon panier. Combien y a-t-il de bananes dans le panier ?" → attendu '\\b(0|zéro|aucune?|pas de banane)', obtenu 'Il y a 3 bananes dans le panier.<|im_end|> · texte : Il y a 3 bananes dans le panier.<|im_end|>'
- ✅ `piège` **fr-v4 · 229M (rlaif2, step 38)** — 'Un train roule à 80 km/h pendant 2 heures. Quel âge a le conducteur ?' → attendu '(ne (peux|peut|sait)|sais pas|impossible|pas possible|ne le dit pas|pas précisé|pas indiqué|inconnu|aucune information|ne permet pas)', obtenu 'On ne peut pas le savoir. · texte : On ne peut pas le savoir.'
- ❌ `piège` **fr-v4 · 229M (rlaif2, step 38)** — "Marie a 12 ans. Son chat s'appelle Félix. Quel âge a le chat ?" → attendu '(ne (peux|peut|sait)|sais pas|impossible|pas possible|ne le dit pas|pas précisé|pas indiqué|inconnu|aucune information|ne permet pas)', obtenu '12 ans<|im_end|> · texte : 12 ans<|im_end|>'
- ❌ `piège` **fr-v4 · 229M (rlaif2, step 38)** — "J'achète une baguette à 1 euro et un croissant à 2 euros. Combien coûte le journal ?" → attendu '(ne (peux|peut|sait)|sais pas|impossible|pas possible|ne le dit pas|pas précisé|pas indiqué|inconnu|aucune information|ne permet pas)', obtenu 'Le journal coûte 2 euros.<|im_end|> · texte : Le journal coûte 2 euros.<|im_end|>'
- ❌ `piège` **fr-v4 · 229M (rlaif2, step 38)** — 'Dans un sac, il y a 10 billes rouges. Combien de billes bleues y a-t-il dans le sac ?' → attendu '\\b(0|zéro|aucune?|pas de bille bleue)', obtenu 'Il y a 10 billes bleues.<|im_end|> · texte : Il y a 10 billes bleues.<|im_end|>'
- ❌ `piège` **fr-v4 · 229M (rlaif2, step 38)** — "Le boulanger vend 30 croissants le matin et 20 l'après-midi. Combien de baguettes vend-il ?" → attendu '\\b(0|zéro|aucune?)\\b|(ne (peux|peut|sait)|sais pas|impossible|ne le dit pas|pas précisé|inconnu)', obtenu "Le boulanger vend 30 le matin et 20 l'après-midi. Combien de · texte : Le boulanger vend 30 le matin et 20 l'après-midi. Combien de baguettes vend-il ?\n\nRéponse : Le boulanger vend 30 baguettes le matin.<|im_end|>"
- ❌ `intervalle` **fr-v4 · 229M (rlaif2, step 38)** — 'Combien y a-t-il de nombres entiers de 4 à 9, en comptant 4 et 9 ?' → attendu '6', obtenu "'4' · texte : Il y a 4 nombres entiers de 4 à 9.<|im_end|>\n\nRéponse : 4<|im_end|>"
- ❌ `intervalle` **fr-v4 · 229M (rlaif2, step 38)** — 'Une clôture droite de 12 mètres a un poteau tous les 3 mètres, avec un poteau à chaque bout. Combien de poteaux ?' → attendu '5', obtenu "'1' · texte : Il y a 1 poteau à chaque bout.<|im_end|>"
- ❌ `intervalle` **fr-v4 · 229M (rlaif2, step 38)** — 'Je lis du chapitre 3 au chapitre 8 inclus. Combien de chapitres vais-je lire ?' → attendu '6', obtenu "'8' · texte : Nous allons lire le chapitre 8.<|im_end|>\n\nRéponse : Nous allons lire le chapitre 8.<|im_end|>\n\nJ'espère que cela vous aidera à comprendre la question. Faites-moi savoir si vous avez d'autres question"
- ❌ `intervalle` **fr-v4 · 229M (rlaif2, step 38)** — "Un immeuble a des étages numérotés de 0 à 6. Combien d'étages différents l'ascenseur dessert-il ?" → attendu '7', obtenu "None · texte : Il n'y a pas d'étages.<|im_end|>"
- ❌ `intervalle` **fr-v4 · 229M (rlaif2, step 38)** — 'Un escalier a 10 marches. Je suis sur la 4e marche. Combien de marches me reste-t-il à monter ?' → attendu '6', obtenu "'10' · texte : Il reste 10 marches.<|im_end|>"
- ❌ `intervalle` **fr-v4 · 229M (rlaif2, step 38)** — 'On coupe un ruban en 5 morceaux. Combien de coupes faut-il faire ?' → attendu '4', obtenu 'None · texte : Nous ne pouvons pas le savoir.<|im_end|>'
- ❌ `branches` **fr-v4 · 229M (rlaif2, step 38)** — "Qu'est-ce qui coûte le plus cher : 3 stylos à 2 euros pièce, ou 2 cahiers à 4 euros pièce ?" → attendu 'cahier', obtenu "'stylo' · texte : Les stylos coûtent 4 euros.<|im_end|>"
- ❌ `branches` **fr-v4 · 229M (rlaif2, step 38)** — 'Qui a le plus de fruits : Ali avec 4 sacs de 3 pommes, ou Bea avec 2 sacs de 7 pommes ?' → attendu 'Bea', obtenu 'None · texte : Nous.'
- ❌ `branches` **fr-v4 · 229M (rlaif2, step 38)** — 'Lise calcule 6 × 5. Nino calcule 4 × 8. Qui obtient le plus grand résultat ?' → attendu 'Nino', obtenu "'Lise' · texte : Lise."
- ❌ `branches` **fr-v4 · 229M (rlaif2, step 38)** — "Un pack de 6 bouteilles coûte 12 euros. Une bouteille seule coûte 3 euros. Qu'est-ce qui est le moins cher par bouteille ?" → attendu 'pack', obtenu "'seule' · texte : Une bouteille seule.<|im_end|>"
- ✅ `branches` **fr-v4 · 229M (rlaif2, step 38)** — 'Tom a 25 billes et en gagne 5. Léo a 40 billes et en perd 15. Qui a le plus de billes maintenant ?' → attendu 'Tom', obtenu "'Tom' · texte : Tom.<|im_end|>"
- ❌ `reste` **fr-v4 · 229M (rlaif2, step 38)** — 'On partage 23 bonbons entre 4 enfants, chacun en reçoit autant. Combien de bonbons restent ?' → attendu '3', obtenu "'4' · texte : Nous partageons 23 bonbons entre 4 enfants.<|im_end|>"
- ❌ `reste` **fr-v4 · 229M (rlaif2, step 38)** — "J'ai 17 œufs et des boîtes de 6. Je remplis uniquement des boîtes complètes. Combien d'œufs ne sont pas en boîte ?" → attendu '5', obtenu "'6' · texte : Il y a 17 œufs et 6 boîtes.<|im_end|>"
- ❌ `reste` **fr-v4 · 229M (rlaif2, step 38)** — 'Un ruban de 20 cm est coupé en morceaux de 6 cm. Quelle longueur de ruban reste-t-il, en cm ?' → attendu '2', obtenu "'20' · texte : Il reste 20 cm.<|im_end|>"
- ❌ `reste` **fr-v4 · 229M (rlaif2, step 38)** — "31 élèves montent dans des voitures de 4 places. Toutes les voitures sont pleines sauf la dernière. Combien d'élèves dans la dernière voiture ?" → attendu '3', obtenu "'4' · texte : Il y a 31 voitures de 4 places.<|im_end|>"
- ❌ `reste` **fr-v4 · 229M (rlaif2, step 38)** — 'Il y a 50 chaises à ranger en rangées de 8. Combien de chaises ne forment pas une rangée complète ?' → attendu '2', obtenu "'50' · texte : Il y a 50 chaises à ranger en rangées de 8.<|im_end|>\n\nRéponse : Il y a 50 chaises à ranger en rangées de 8.<|im_end|>\n\nRéponse : Il y a 50 chaises à ranger en rangées de 8.<|im_end|>\n\nRéponse : Il y "
- ❌ `cycle` **fr-v4 · 229M (rlaif2, step 38)** — 'Nous sommes mardi. Quel jour serons-nous dans 3 jours ?' → attendu 'vendredi', obtenu "'mardi' · texte : Nous sommes mardi."
- ❌ `cycle` **fr-v4 · 229M (rlaif2, step 38)** — "Hier, c'était dimanche. Quel jour serons-nous demain ?" → attendu 'mardi', obtenu "'lundi' · texte : Nous sommes lundi.<|im_end|>"
- ❌ `cycle` **fr-v4 · 229M (rlaif2, step 38)** — 'Nous sommes samedi. Quel jour étions-nous il y a 2 jours ?' → attendu 'jeudi', obtenu "'mardi' · texte : Nous le sommes mardi."
- ✅ `cycle` **fr-v4 · 229M (rlaif2, step 38)** — 'Nous sommes vendredi. Quel jour serons-nous dans 7 jours ?' → attendu 'vendredi', obtenu "'vendredi' · texte : Nous sommes vendredi."
- ❌ `cycle` **fr-v4 · 229M (rlaif2, step 38)** — 'Nous sommes jeudi. Quel jour serons-nous dans 4 jours ?' → attendu 'lundi', obtenu "'jeudi' · texte : jeudi"
- ❌ `composition` **fr-v4 · 229M (rlaif2, step 38)** — 'Quel est le double du double de 5 ?' → attendu '20', obtenu "'10' · texte : 10."
- ❌ `composition` **fr-v4 · 229M (rlaif2, step 38)** — 'Quelle est la moitié de la somme de 6 et 10 ?' → attendu '8', obtenu "'10' · texte : 10"
- ❌ `composition` **fr-v4 · 229M (rlaif2, step 38)** — 'Ajoute 3 au produit de 4 et 5. Combien ?' → attendu '23', obtenu "'5' · texte : Ajoute 3 au produit de 5."
- ❌ `composition` **fr-v4 · 229M (rlaif2, step 38)** — 'Soustrais 2 de la moitié de 18. Combien ?' → attendu '7', obtenu "'16' · texte : 18 - 2 = 16.\n\nRéponse : 16<|im_end|>"
- ❌ `composition` **fr-v4 · 229M (rlaif2, step 38)** — 'Quel est le triple de la différence entre 9 et 4 ?' → attendu '15', obtenu "'5' · texte : 9 - 4 = 5."
- ❌ `composition` **fr-v4 · 229M (rlaif2, step 38)** — 'Prends 10, ajoute 5, puis divise le tout par 3. Combien ?' → attendu '5', obtenu "'15' · texte : 10 + 5 = 15.<|im_end|>"
- ❌ `composition` **fr-v4 · 229M (rlaif2, step 38)** — 'Quelle est la somme du double de 3 et du double de 4 ?' → attendu '14', obtenu "'36' · texte : 12 + 24 = 36.<|im_end|>\n\nRéponse : 36<|im_end|>"
- ✅ `fait` **fr-v4 · 229M (rlaif2, step 38)** — 'La capitale du Portugal est' → attendu '\\bLisbonne\\b', obtenu "Lisbonne.\n\nLa ville de Lisbonne est située dans la région de l'Est de l'Europe, près de la côte atla"
- ✅ `fait` **fr-v4 · 229M (rlaif2, step 38)** — 'La capitale de la Belgique est' → attendu '\\bBruxelles\\b', obtenu 'Bruxelles.\n\nLa capitale de la Belgique est Bruxelles.\n\nLa capitale de la Belgique est Bruxelles.\n\nLa'
- ❌ `fait` **fr-v4 · 229M (rlaif2, step 38)** — 'Le mois de février compte généralement' → attendu '\\b28\\b|vingt-huit', obtenu 'entre février et mars.\n\nEn résumé, février est le mois le plus courant pour les mois de mars et avri'
- ❌ `fait` **fr-v4 · 229M (rlaif2, step 38)** — 'Le contraire de gauche est la' → attendu '\\bdroite\\b', obtenu 'gauche.\n\nEn résumé, la gauche est la plus populaire parmi les gauche.\n\nEn ce qui concerne les autres'
- ✅ `fait` **fr-v4 · 229M (rlaif2, step 38)** — "Les poissons vivent dans l'" → attendu '\\beau\\b', obtenu 'eau salée.\n\nEn résumé, la réponse à la question est 2.\n\nEn résumé, la réponse à la question est'
- ✅ `fait` **fr-v4 · 229M (rlaif2, step 38)** — "L'auteur des Misérables est" → attendu '\\bHugo\\b', obtenu "Victor Hugo.\n\nL'auteur des Misérables est Victor Hugo.\n\nL'auteur des Misérables est Victor Hugo."
- ❌ `fait` **fr-v4 · 229M (rlaif2, step 38)** — 'Un vélo a deux' → attendu '\\broues?\\b', obtenu "pédésésésésésés et deux pédésés. Les pédés sont reliés par des pédés d'accoudoirs. Les péd"
- ✅ `fait` **fr-v4 · 229M (rlaif2, step 38)** — 'La Terre tourne autour du' → attendu '\\b[Ss]oleil\\b', obtenu 'Soleil, ce qui fait que la Terre tourne autour du Soleil.\n\nLa Terre tourne autour du Soleil en raiso'
- ✅ `fait` **fr-v4 · 229M (rlaif2, step 38)** — "Le premier mois de l'année est" → attendu '\\bjanvier\\b', obtenu 'le 1er janvier.\n\nLe deuxième mois est le 21 janvier.\n\nLe troisième mois est le 31'
- ✅ `fait` **fr-v4 · 229M (rlaif2, step 38)** — 'Les feuilles des arbres sont généralement de couleur' → attendu '\\bvert', obtenu 'vert foncé. Les arbres ont des feuilles vertes et des branches qui sont vertes et qui se courbent po'
- ✅ `fait` **fr-v4 · 229M (rlaif2, step 38)** — 'Pour écrire, on utilise un stylo ou un' → attendu '\\bcrayon\\b|\\bclavier\\b|\\bfeutre\\b|\\bstylet\\b', obtenu 'crayon. On commence par dessiner le dessin sur le stylo. Ensuite, on dessine les lignes du dessin su'
- ✅ `fait` **fr-v4 · 229M (rlaif2, step 38)** — 'Le lait provient principalement de la' → attendu '\\bvache\\b', obtenu 'vache, mais il existe également de nombreux autres produits laitiers.\n\nEn conclusion, le lait est un'
