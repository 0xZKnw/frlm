# Benchmark hors-distribution v2 (familles secrètes)

⚠️ Ces énoncés ne doivent jamais entrer dans un corpus d'entraînement.

| modèle | transitif | piège | intervalle | branches | reste | cycle | composition | total | faits |
|---|---|---|---|---|---|---|---|---|---|
| fr-v4-v44-sft-lr1e4 · 229M (sft, step 120) | 2/6 | 0/6 | 0/6 | 1/5 | 1/5 | 0/5 | 0/7 | **4/40** | 5/12 |

## Détails

- ✅ `transitif` **fr-v4-v44-sft-lr1e4 · 229M (sft, step 120)** — 'Léa est plus grande que Max. Max est plus grand que Zoé. Qui est le plus grand des trois ?' → attendu 'Léa', obtenu "'Léa' · texte : Léa"
- ❌ `transitif` **fr-v4-v44-sft-lr1e4 · 229M (sft, step 120)** — 'Paul court plus vite que Jules. Anna court plus vite que Paul. Qui est le plus rapide ?' → attendu 'Anna', obtenu "'Paul' · texte : C'est Paul qui est le plus rapide."
- ❌ `transitif` **fr-v4-v44-sft-lr1e4 · 229M (sft, step 120)** — 'Le chien est plus lourd que le chat. Le chat est plus lourd que le lapin. Qui est le plus léger ?' → attendu 'lapin', obtenu 'None · texte : 12 livres'
- ✅ `transitif` **fr-v4-v44-sft-lr1e4 · 229M (sft, step 120)** — 'Marc est plus âgé que Lise. Lise est plus âgée que Tom. Qui est le plus jeune ?' → attendu 'Tom', obtenu "'Tom' · texte : C'est Tom qui est le plus jeune."
- ❌ `transitif` **fr-v4-v44-sft-lr1e4 · 229M (sft, step 120)** — 'La tour A est plus haute que la tour B. La tour C est plus basse que la tour B. Quelle tour est la plus haute ?' → attendu 'A', obtenu "'B' · texte : La tour B"
- ❌ `transitif` **fr-v4-v44-sft-lr1e4 · 229M (sft, step 120)** — "Zoé a plus de billes qu'Émile, et Émile en a plus que Sami. Qui en a le moins ?" → attendu 'Sami', obtenu "'Zoé' · texte : Zoé"
- ❌ `piège` **fr-v4-v44-sft-lr1e4 · 229M (sft, step 120)** — "J'ai 5 pommes et 3 oranges dans mon panier. Combien y a-t-il de bananes dans le panier ?" → attendu '\\b(0|zéro|aucune?|pas de banane)', obtenu '1 · texte : 1'
- ❌ `piège` **fr-v4-v44-sft-lr1e4 · 229M (sft, step 120)** — 'Un train roule à 80 km/h pendant 2 heures. Quel âge a le conducteur ?' → attendu '(ne (peux|peut|sait)|sais pas|impossible|pas possible|ne le dit pas|pas précisé|pas indiqué|inconnu|aucune information|ne permet pas)', obtenu '0.6666 · texte : 0.6666'
- ❌ `piège` **fr-v4-v44-sft-lr1e4 · 229M (sft, step 120)** — "Marie a 12 ans. Son chat s'appelle Félix. Quel âge a le chat ?" → attendu '(ne (peux|peut|sait)|sais pas|impossible|pas possible|ne le dit pas|pas précisé|pas indiqué|inconnu|aucune information|ne permet pas)', obtenu '12 ans · texte : 12 ans'
- ❌ `piège` **fr-v4-v44-sft-lr1e4 · 229M (sft, step 120)** — "J'achète une baguette à 1 euro et un croissant à 2 euros. Combien coûte le journal ?" → attendu '(ne (peux|peut|sait)|sais pas|impossible|pas possible|ne le dit pas|pas précisé|pas indiqué|inconnu|aucune information|ne permet pas)', obtenu '<think>\nLe prix de 10 croissants est de 1 euro; celui de 2 c · texte : <think>\nLe prix de 10 croissants est de 1 euro; celui de 2 croissants est de 2 euros; le prix de 1 croissant est de 2 euros; le prix de 1 décade est de 1 euro; le prix de 1 décade vaut 1 euro; le prix'
- ❌ `piège` **fr-v4-v44-sft-lr1e4 · 229M (sft, step 120)** — 'Dans un sac, il y a 10 billes rouges. Combien de billes bleues y a-t-il dans le sac ?' → attendu '\\b(0|zéro|aucune?|pas de bille bleue)', obtenu '1 · texte : 1'
- ❌ `piège` **fr-v4-v44-sft-lr1e4 · 229M (sft, step 120)** — "Le boulanger vend 30 croissants le matin et 20 l'après-midi. Combien de baguettes vend-il ?" → attendu '\\b(0|zéro|aucune?)\\b|(ne (peux|peut|sait)|sais pas|impossible|ne le dit pas|pas précisé|inconnu)', obtenu '50 · texte : 50'
- ❌ `intervalle` **fr-v4-v44-sft-lr1e4 · 229M (sft, step 120)** — 'Combien y a-t-il de nombres entiers de 4 à 9, en comptant 4 et 9 ?' → attendu '6', obtenu "'11' · texte : 11"
- ❌ `intervalle` **fr-v4-v44-sft-lr1e4 · 229M (sft, step 120)** — 'Une clôture droite de 12 mètres a un poteau tous les 3 mètres, avec un poteau à chaque bout. Combien de poteaux ?' → attendu '5', obtenu "'4' · texte : 4"
- ❌ `intervalle` **fr-v4-v44-sft-lr1e4 · 229M (sft, step 120)** — 'Je lis du chapitre 3 au chapitre 8 inclus. Combien de chapitres vais-je lire ?' → attendu '6', obtenu "'20' · texte : 20"
- ❌ `intervalle` **fr-v4-v44-sft-lr1e4 · 229M (sft, step 120)** — "Un immeuble a des étages numérotés de 0 à 6. Combien d'étages différents l'ascenseur dessert-il ?" → attendu '7', obtenu "'1' · texte : 1"
- ❌ `intervalle` **fr-v4-v44-sft-lr1e4 · 229M (sft, step 120)** — 'Un escalier a 10 marches. Je suis sur la 4e marche. Combien de marches me reste-t-il à monter ?' → attendu '6', obtenu "'1' · texte : <think>\nLa première marche a 10 marches. La deuxième marche a 10 - 4 = 6 marches. La troisième marche a 6 marches. La quatrième marche a 6 + 1 = 7 marches. La cinquième marche a 7 - 1 = 6 marches. La "
- ❌ `intervalle` **fr-v4-v44-sft-lr1e4 · 229M (sft, step 120)** — 'On coupe un ruban en 5 morceaux. Combien de coupes faut-il faire ?' → attendu '4', obtenu "'500' · texte : 10 × 5 = 500"
- ❌ `branches` **fr-v4-v44-sft-lr1e4 · 229M (sft, step 120)** — "Qu'est-ce qui coûte le plus cher : 3 stylos à 2 euros pièce, ou 2 cahiers à 4 euros pièce ?" → attendu 'cahier', obtenu 'None · texte : 6 ÷ 2 = 3'
- ❌ `branches` **fr-v4-v44-sft-lr1e4 · 229M (sft, step 120)** — 'Qui a le plus de fruits : Ali avec 4 sacs de 3 pommes, ou Bea avec 2 sacs de 7 pommes ?' → attendu 'Bea', obtenu "'Ali' · texte : Ali a 26 pommes et Bea en a 12."
- ❌ `branches` **fr-v4-v44-sft-lr1e4 · 229M (sft, step 120)** — 'Lise calcule 6 × 5. Nino calcule 4 × 8. Qui obtient le plus grand résultat ?' → attendu 'Nino', obtenu 'None · texte : 6 + 5 = 11'
- ❌ `branches` **fr-v4-v44-sft-lr1e4 · 229M (sft, step 120)** — "Un pack de 6 bouteilles coûte 12 euros. Une bouteille seule coûte 3 euros. Qu'est-ce qui est le moins cher par bouteille ?" → attendu 'pack', obtenu 'None · texte : 144'
- ✅ `branches` **fr-v4-v44-sft-lr1e4 · 229M (sft, step 120)** — 'Tom a 25 billes et en gagne 5. Léo a 40 billes et en perd 15. Qui a le plus de billes maintenant ?' → attendu 'Tom', obtenu "'Tom' · texte : C'est Tom qui est le plus de billes."
- ❌ `reste` **fr-v4-v44-sft-lr1e4 · 229M (sft, step 120)** — 'On partage 23 bonbons entre 4 enfants, chacun en reçoit autant. Combien de bonbons restent ?' → attendu '3', obtenu "'7' · texte : 28 ÷ 4 = 7"
- ❌ `reste` **fr-v4-v44-sft-lr1e4 · 229M (sft, step 120)** — "J'ai 17 œufs et des boîtes de 6. Je remplis uniquement des boîtes complètes. Combien d'œufs ne sont pas en boîte ?" → attendu '5', obtenu "'1' · texte : 1"
- ❌ `reste` **fr-v4-v44-sft-lr1e4 · 229M (sft, step 120)** — 'Un ruban de 20 cm est coupé en morceaux de 6 cm. Quelle longueur de ruban reste-t-il, en cm ?' → attendu '2', obtenu "'120' · texte : Pour résoudre ce problème, nous devons déterminer la longueur de chaque côté du ruban.\n\n1. **Longueur du ruban :**\n   - Le ruban mesure 20 cm.\n\n2. **Longueur du ruban :**\n   - La longueur totale du ru"
- ❌ `reste` **fr-v4-v44-sft-lr1e4 · 229M (sft, step 120)** — "31 élèves montent dans des voitures de 4 places. Toutes les voitures sont pleines sauf la dernière. Combien d'élèves dans la dernière voiture ?" → attendu '3', obtenu 'None · texte : La dernière phrase donne sa réponse.'
- ✅ `reste` **fr-v4-v44-sft-lr1e4 · 229M (sft, step 120)** — 'Il y a 50 chaises à ranger en rangées de 8. Combien de chaises ne forment pas une rangée complète ?' → attendu '2', obtenu "'2' · texte : 2"
- ❌ `cycle` **fr-v4-v44-sft-lr1e4 · 229M (sft, step 120)** — 'Nous sommes mardi. Quel jour serons-nous dans 3 jours ?' → attendu 'vendredi', obtenu "'mercredi' · texte : mercredi"
- ❌ `cycle` **fr-v4-v44-sft-lr1e4 · 229M (sft, step 120)** — "Hier, c'était dimanche. Quel jour serons-nous demain ?" → attendu 'mardi', obtenu "'samedi' · texte : Samedi, c'était dimanche."
- ❌ `cycle` **fr-v4-v44-sft-lr1e4 · 229M (sft, step 120)** — 'Nous sommes samedi. Quel jour étions-nous il y a 2 jours ?' → attendu 'jeudi', obtenu "'samedi' · texte : Samedi"
- ❌ `cycle` **fr-v4-v44-sft-lr1e4 · 229M (sft, step 120)** — 'Nous sommes vendredi. Quel jour serons-nous dans 7 jours ?' → attendu 'vendredi', obtenu "'lundi' · texte : lundi"
- ❌ `cycle` **fr-v4-v44-sft-lr1e4 · 229M (sft, step 120)** — 'Nous sommes jeudi. Quel jour serons-nous dans 4 jours ?' → attendu 'lundi', obtenu "'mercredi' · texte : mercredi"
- ❌ `composition` **fr-v4-v44-sft-lr1e4 · 229M (sft, step 120)** — 'Quel est le double du double de 5 ?' → attendu '20', obtenu "'10' · texte : Le double de 5 est 10."
- ❌ `composition` **fr-v4-v44-sft-lr1e4 · 229M (sft, step 120)** — 'Quelle est la moitié de la somme de 6 et 10 ?' → attendu '8', obtenu "'10' · texte : La moitié de 6 et 10 est 10."
- ❌ `composition` **fr-v4-v44-sft-lr1e4 · 229M (sft, step 120)** — 'Ajoute 3 au produit de 4 et 5. Combien ?' → attendu '23', obtenu "'4' · texte : 4"
- ❌ `composition` **fr-v4-v44-sft-lr1e4 · 229M (sft, step 120)** — 'Soustrais 2 de la moitié de 18. Combien ?' → attendu '7', obtenu "'9' · texte : 18 ÷ 2 = 9"
- ❌ `composition` **fr-v4-v44-sft-lr1e4 · 229M (sft, step 120)** — 'Quel est le triple de la différence entre 9 et 4 ?' → attendu '15', obtenu "'13' · texte : 9 + 4 = 13"
- ❌ `composition` **fr-v4-v44-sft-lr1e4 · 229M (sft, step 120)** — 'Prends 10, ajoute 5, puis divise le tout par 3. Combien ?' → attendu '5', obtenu "'11' · texte : 11"
- ❌ `composition` **fr-v4-v44-sft-lr1e4 · 229M (sft, step 120)** — 'Quelle est la somme du double de 3 et du double de 4 ?' → attendu '14', obtenu "'8' · texte : Le double de 3 est 6. Le double de 4 est 8."
- ❌ `fait` **fr-v4-v44-sft-lr1e4 · 229M (sft, step 120)** — 'La capitale du Portugal est' → attendu '\\bLisbonne\\b', obtenu 'située dans le district de Guadalajara.\n\nLa ville est la plus grande ville du Portugal, avec une pop'
- ❌ `fait` **fr-v4-v44-sft-lr1e4 · 229M (sft, step 120)** — 'La capitale de la Belgique est' → attendu '\\bBruxelles\\b', obtenu 'Anvers.\n\nLa ville est située dans le département de la Meuse, dans la région de la province de Luxem'
- ❌ `fait` **fr-v4-v44-sft-lr1e4 · 229M (sft, step 120)** — 'Le mois de février compte généralement' → attendu '\\b28\\b|vingt-huit', obtenu '12 jours, soit 48 semaines.\nassistant\n\nLe mois de février compte 12 jours, soit'
- ✅ `fait` **fr-v4-v44-sft-lr1e4 · 229M (sft, step 120)** — 'Le contraire de gauche est la' → attendu '\\bdroite\\b', obtenu 'droite.\n\nLa droite est un parti politique, et le parti politique est un parti politique.\n\nLa droite '
- ✅ `fait` **fr-v4-v44-sft-lr1e4 · 229M (sft, step 120)** — "Les poissons vivent dans l'" → attendu '\\beau\\b', obtenu "eau, et les oiseaux dans leur nid.\nassistant\n\nLes poissons vivent dans l'eau, et les oiseaux dans le"
- ❌ `fait` **fr-v4-v44-sft-lr1e4 · 229M (sft, step 120)** — "L'auteur des Misérables est" → attendu '\\bHugo\\b', obtenu 'un écrivain français, né le  à Paris et mort le  à Paris.\n\nBiographie \n\nIl est le fils de Jean-Franç'
- ❌ `fait` **fr-v4-v44-sft-lr1e4 · 229M (sft, step 120)** — 'Un vélo a deux' → attendu '\\broues?\\b', obtenu 'côtés. Le côté gauche est un vélo de course, et le côté droit est un vélo de course.\nassistant\n\nLe'
- ✅ `fait` **fr-v4-v44-sft-lr1e4 · 229M (sft, step 120)** — 'La Terre tourne autour du' → attendu '\\b[Ss]oleil\\b', obtenu 'Soleil. Le Soleil est plus proche du centre de la Terre que la Lune. La Terre tourne autour de la Lu'
- ❌ `fait` **fr-v4-v44-sft-lr1e4 · 229M (sft, step 120)** — "Le premier mois de l'année est" → attendu '\\bjanvier\\b', obtenu "marqué par la sortie de l'album  de la chanteuse américaine Jessie J.\n\nEn 2012, elle sort son premie"
- ✅ `fait` **fr-v4-v44-sft-lr1e4 · 229M (sft, step 120)** — 'Les feuilles des arbres sont généralement de couleur' → attendu '\\bvert', obtenu 'rouge, jaune ou vert.\n\n3. **Feuilles** : Les feuilles des arbres sont généralement de couleur jaune,'
- ✅ `fait` **fr-v4-v44-sft-lr1e4 · 229M (sft, step 120)** — 'Pour écrire, on utilise un stylo ou un' → attendu '\\bcrayon\\b|\\bclavier\\b|\\bfeutre\\b|\\bstylet\\b', obtenu 'crayon pour dessiner.\n\n### Dessiner\n\n- **Pinceaux** : Utilisez un pinceau pour dessiner des formes.'
- ❌ `fait` **fr-v4-v44-sft-lr1e4 · 229M (sft, step 120)** — 'Le lait provient principalement de la' → attendu '\\bvache\\b', obtenu 'laiterie de la rue de la Paix, à Paris.\n\nHistorique \n\nLe lait de la laiterie de la rue de la'
