from __future__ import annotations

import unittest

from frlm.reason_bootstrap_v45 import (
    BALANCED_AST_WEIGHTS, BALANCED_REPLAY_WEIGHTS, EVAL_SPLITS,
    evaluate_ast, make_example, operator_signature,
)


class ReasonBootstrapV45Tests(unittest.TestCase):
    def test_generation_est_deterministe_et_ast_executable(self):
        first = make_example(455_501, "train", "execute")
        second = make_example(455_501, "train", "execute")
        self.assertEqual(first, second)
        self.assertEqual(str(evaluate_ast(first["program_ast"])), first["target"])

    def test_holdout_structurel_ne_partage_aucune_topologie_avec_train(self):
        train = {operator_signature(make_example(500_000 + index, "train")
                                    ["program_ast"]) for index in range(500)}
        structure = {operator_signature(make_example(600_000 + index,
                                                      "structure_holdout")["program_ast"])
                     for index in range(500)}
        self.assertTrue(train.isdisjoint(structure), train & structure)

    def test_surface_holdout_change_la_formulation_pas_la_classe_de_topologie(self):
        train_shapes = {operator_signature(make_example(700_000 + index, "train")
                                           ["program_ast"]) for index in range(200)}
        surface_shapes = {operator_signature(make_example(800_000 + index,
                                                           "surface_holdout")["program_ast"])
                          for index in range(200)}
        self.assertTrue(train_shapes & surface_shapes)
        prompts = " ".join(make_example(810_000 + index, "surface_holdout")["prompt"]
                           for index in range(30))
        self.assertIn("augmenté", prompts)

    def test_tous_les_objectifs_ont_une_cible_non_vide(self):
        objectives = ("execute", "masked_step", "order_steps", "find_error", "number_only")
        for index, objective in enumerate(objectives):
            row = make_example(900_000 + index, EVAL_SPLITS[index % len(EVAL_SPLITS)],
                               objective)
            self.assertTrue(row["target"])
            self.assertTrue(row["answer"])
            self.assertEqual(row["messages"][-1]["text"], row["answer"])

    def test_aucune_famille_ood_nommee_n_est_generee(self):
        forbidden = ("jour de la semaine", "poteau", "reste euclidien",
                     "plus grand des trois", "boîtes complètes")
        text = " ".join(make_example(1_000_000 + index, "train")["prompt"].casefold()
                        for index in range(300))
        self.assertFalse(any(term in text for term in forbidden))

    def test_mix_equilibre_conserve_une_majorite_de_retention(self):
        ast = sum(BALANCED_AST_WEIGHTS.values())
        replay = sum(BALANCED_REPLAY_WEIGHTS.values())
        self.assertAlmostEqual(ast + replay, 1.0)
        self.assertAlmostEqual(ast, 0.35)
        self.assertAlmostEqual(replay, 0.65)


if __name__ == "__main__":
    unittest.main()
