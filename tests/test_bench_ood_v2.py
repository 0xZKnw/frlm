from __future__ import annotations

import unittest

from bench.bench_ood_v2 import noter_fait, noter_strict


class StrictOODScorerTests(unittest.TestCase):
    def test_transitif_accepte_la_vraie_conclusion_finale(self):
        problem = {"type": "choix", "options": ["Marc", "Lise", "Tom"],
                   "attendu": "Tom"}
        text = ("Marc passe devant Lise, et Lise passe devant Tom. "
                "L'ordre est donc : Marc, puis Lise, puis Tom. "
                "C'est Tom qui est le plus jeune.")
        self.assertEqual(noter_strict(problem, text)["status"], "correct")

    def test_mention_du_bon_choix_dans_un_raisonnement_faux_ne_vaut_plus_un_point(self):
        problem = {"type": "choix", "options": ["Lise", "Nino"],
                   "attendu": "Nino"}
        text = ("Lina passe devant Lina, et Lina passe devant Nino. "
                "L'ordre est donc : Lina, puis Lina, puis Nino. "
                "C'est Lina qui est le plus grand.")
        self.assertEqual(noter_strict(problem, text)["status"], "incorrect")

    def test_choix_cite_sans_conclusion_est_envoye_en_revue(self):
        problem = {"type": "choix", "options": ["pack", "seule"],
                   "attendu": "pack"}
        result = noter_strict(problem, "6 × 3 = 18. Un pack de 6 bouteilles coûte 18 euros.")
        self.assertEqual(result["status"], "manual_review")

    def test_reponse_courte_et_reponse_marquee_restent_acceptables(self):
        choice = {"type": "choix", "options": ["lundi", "mardi"], "attendu": "mardi"}
        number = {"type": "num", "attendu": "23"}
        self.assertEqual(noter_strict(choice, "mardi")["status"], "correct")
        self.assertEqual(noter_strict(number, "Calculs divers. Réponse : 23")["status"],
                         "correct")

    def test_nombres_de_calcul_non_isoles_ne_sont_plus_un_resultat_final(self):
        problem = {"type": "num", "attendu": "5"}
        result = noter_strict(problem, "9 − 4 = 5. 9 − 4 = 5")
        self.assertEqual(result["status"], "manual_review")

    def test_fait_note_uniquement_la_premiere_proposition(self):
        result = noter_fait(r"\bdroite\b", "gauche. Le contraire de gauche est la droite.")
        self.assertEqual(result["status"], "incorrect")


if __name__ == "__main__":
    unittest.main()
