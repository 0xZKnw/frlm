from __future__ import annotations

import json
import random
import tempfile
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import Mock, patch

from frlm.dpo_v45 import _load_pairs
from frlm.rlaif_offline_v45 import _read_prompts, import_scores
from frlm.posttrain_gates_v45 import evaluate_gates
from frlm.rl_profile_v45 import _summary
from frlm.rl_v45 import RLVRConfig, RLVRTrainer, _canonical_answer
from frlm.rl_tasks_v45 import SCHEMAS_BY_CAPABILITY, make_task
from frlm.verifiers_v45 import VERIFIER_VERSION, AnswerSpec, final_text, verify


class VerifierTests(unittest.TestCase):
    def test_integer_signed_fraction_and_ambiguity(self):
        self.assertTrue(verify(AnswerSpec("integer", -12), "Réponse : -12").primary_success)
        self.assertTrue(verify(AnswerSpec("rational", "4"), "12/3").primary_success)
        self.assertFalse(verify(AnswerSpec("integer", 12), "12,0").primary_success)
        self.assertEqual(verify(AnswerSpec("integer", 12), "12 puis 3").failure_code,
                         "number_ambiguous")

    def test_number_only_refuse_le_faux_positif_du_step_50(self):
        spec = AnswerSpec("integer", 121, strict_number_only=True)
        self.assertTrue(verify(spec, "121").primary_success)
        bad = ("44 + 77 = 121.\n8 × 15 = 121, donc 121 ÷ 8 = 15\n"
               "� 8 = 15")
        result = verify(spec, bad)
        self.assertFalse(result.primary_success)
        self.assertEqual(result.failure_code, "number_only_format")
        self.assertFalse(verify(spec, "Réponse : 121").primary_success)
        self.assertFalse(verify(spec, "121.").primary_success)

    def test_answer_spec_reste_compatible_avec_les_anciens_payloads(self):
        spec = AnswerSpec.from_dict({"kind": "integer", "value": 121})
        self.assertFalse(spec.strict_number_only)

    def test_choice_word_boundaries(self):
        spec = AnswerSpec("choice", "Paul", choices=("Paul", "Pauline"))
        self.assertTrue(verify(spec, "Paul").primary_success)
        self.assertFalse(verify(spec, "Pauline").primary_success)
        self.assertFalse(verify(spec, "Paul et Pauline").primary_success)
        self.assertFalse(verify(AnswerSpec("entity", "chat"), "chatière").primary_success)

    def test_abstention(self):
        spec = AnswerSpec("abstain")
        accepted = (
            "Le contexte ne permet pas de le déterminer.",
            "Impossible à déterminer avec les informations fournies.",
            "Ce n'est pas précisé dans l'énoncé.",
            "Les informations sont insuffisantes.",
            "C'est indéterminable : les deux sources se contredisent.",
        )
        for answer in accepted:
            self.assertTrue(verify(spec, answer).primary_success, answer)
        self.assertFalse(verify(spec, "Je pense que la réponse est 4.").primary_success)

    def test_json_schema(self):
        spec = AnswerSpec("json", {"operation": "addition", "resultat": 5},
                          json_schema={"operation": "string", "resultat": "integer"})
        self.assertTrue(verify(spec, json.dumps(spec.value)).primary_success)
        self.assertFalse(verify(spec, '{"operation":"addition","resultat":"5"}').primary_success)
        self.assertFalse(verify(spec, '{"operation":"addition","resultat":5,"x":1}').primary_success)

    def test_code_sandbox(self):
        spec = AnswerSpec("code", function_name="double", tests=(([3], 6), ([-2], -4)))
        self.assertTrue(verify(spec, "def double(n):\n    return 2 * n").primary_success)
        self.assertFalse(verify(spec, "import os\ndef double(n):\n    return 2*n").primary_success)
        self.assertFalse(verify(spec, "def double(n):\n    return n").primary_success)

    def test_final_after_think(self):
        self.assertEqual(final_text("<think>2+2=4</think>\n4<|im_end|>"), "4")


class TaskTests(unittest.TestCase):
    def test_resume_pass32_revele_une_frontiere_cachee(self):
        rows = [
            {"capability": "code", "schema_id": "code_0", "first_success": False,
             "initial_successes": 0, "successes": 1, "k": 32, "entropy": 1.0},
            {"capability": "grounded", "schema_id": "grounded_year", "first_success": True,
             "initial_successes": 6, "successes": 6, "k": 6, "entropy": 0.1},
        ]
        overall = _summary(rows, 6, 32)["overall"]
        self.assertEqual(overall["pass@6"], 0.5)
        self.assertEqual(overall["pass@32"], 1.0)
        self.assertEqual(overall["frontier_dynamic_rate"], 0.5)

    def test_deterministic_and_split_surfaces(self):
        a = make_task(123, "train", 0.4)
        b = make_task(123, "train", 0.4)
        dev = make_task(123, "dev", 0.4)
        self.assertEqual(a.to_dict(), b.to_dict())
        self.assertNotEqual(a.surface_id, dev.surface_id)
        self.assertNotEqual(a.task_id, dev.task_id)

    def test_all_capabilities_have_canonical_success(self):
        responses = {
            "reasoning_program": lambda t: str(t.answer.value),
            "grounded": lambda t: str(t.answer.value),
            "constraints": lambda t: json.dumps(t.answer.value, ensure_ascii=False)
            if t.answer.kind == "json" else str(t.answer.value),
            "uncertainty": lambda _t: "Le contexte ne permet pas de le déterminer.",
            "state_tracking": lambda t: str(t.answer.value),
        }
        for index, (capability, response) in enumerate(responses.items()):
            task = make_task(900 + index, capability=capability)
            self.assertTrue(verify(task.answer, response(task)).primary_success, task)

    def test_no_ood_v2_family_is_generated(self):
        forbidden = {"transitive", "missing_information", "inclusive_count", "remainder",
                     "weekly_cycle", "compare_totals", "stock_three_ops"}
        schemas = {make_task(20_000 + index).schema_id for index in range(600)}
        self.assertTrue(schemas.isdisjoint(forbidden), schemas & forbidden)

    def test_schema_force_garde_une_reponse_verifiable(self):
        schemas = {
            "reasoning_program": "linear_equation", "grounded": "grounded_rooms",
            "constraints": "constraint_number_only", "code": "code_1",
            "uncertainty": "contradictory_sources", "state_tracking": "state_update",
        }
        for index, (capability, schema) in enumerate(schemas.items()):
            task = make_task(30_000 + index, capability=capability, schema_id=schema)
            self.assertEqual(task.schema_id, schema)
            self.assertTrue(verify(task.answer, _canonical_answer(task)).primary_success)

    def test_constraint_number_only_transporte_le_contrat_strict(self):
        task = make_task(455_851, "dev", 0.9, "constraints",
                         "constraint_number_only")
        self.assertTrue(task.answer.strict_number_only)
        self.assertEqual(task.verifier_version, VERIFIER_VERSION)
        self.assertTrue(verify(task.answer, str(task.answer.value)).primary_success)
        self.assertFalse(verify(task.answer, f"Donc {task.answer.value}.").primary_success)

    def test_curriculum_rl_est_pilote_par_le_profil(self):
        trainer = RLVRTrainer.__new__(RLVRTrainer)
        trainer.cfg = RLVRConfig(require_profile=True)
        trainer.profile = {"config": {"k": 6, "frontier_k": 32}, "rows": [
            {"capability": "code", "schema_id": "code_1", "difficulty": 0.3,
             "initial_successes": 1, "successes": 7, "k": 32, "task_id": "a"},
            {"capability": "reasoning_program", "schema_id": "linear_equation",
             "difficulty": 0.1, "initial_successes": 0, "successes": 1,
             "k": 32, "task_id": "b"},
            {"capability": "grounded", "schema_id": "grounded_year", "difficulty": 0.2,
             "initial_successes": 6, "successes": 6, "k": 6, "task_id": "c"},
        ]}
        frontier, bridge = trainer._profile_curriculum()
        self.assertEqual({row["schema_id"] for row in frontier},
                         {"code_1", "linear_equation"})
        bridge_schemas = {row["schema_id"] for row in bridge}
        self.assertIn("small_power", bridge_schemas)
        self.assertNotIn("code_1", bridge_schemas)
        self.assertNotIn("linear_equation", bridge_schemas)
        self.assertNotIn("grounded_year", bridge_schemas)

        trainer.frontier_specs, trainer.bridge_specs = frontier, bridge
        trainer.frontier_scales = {row["key"]: 1.0 for row in frontier}
        trainer.rng = random.Random(45)
        trainer.difficulty = {"reasoning_program": 0.25}
        trainer.rollout_index = 0
        sampled = [trainer._next_task().schema_id for _ in range(100)]
        self.assertGreaterEqual(sum(schema in {"code_1", "linear_equation"}
                                    for schema in sampled), 65)

    def test_curriculum_refuse_un_zero_sur_six_non_raffine(self):
        trainer = RLVRTrainer.__new__(RLVRTrainer)
        trainer.cfg = RLVRConfig(require_profile=True)
        trainer.profile = {"config": {"k": 6, "frontier_k": 32}, "rows": [{
            "task_id": "zero", "capability": "code", "schema_id": "code_0",
            "difficulty": 0.2, "initial_successes": 0, "successes": 0, "k": 6,
        }]}
        with self.assertRaises(RuntimeError):
            trainer._profile_curriculum()

    def test_curriculum_adapte_raisonnement_et_poids_frontiere(self):
        trainer = RLVRTrainer.__new__(RLVRTrainer)
        trainer.difficulty = {"reasoning_program": 0.25}
        trainer.cap_history = {"reasoning_program": deque([0.0] * 20, maxlen=40)}
        trainer.frontier_history = {"frontier": deque([0.5] * 12, maxlen=24)}
        trainer.frontier_base_weights = {"frontier": 0.05}
        trainer.frontier_scales = {"frontier": 1.0}
        trainer._adjust_curriculum()
        self.assertEqual(trainer.difficulty["reasoning_program"], 0.20)
        self.assertFalse(trainer.cap_history["reasoning_program"])
        self.assertGreater(trainer.frontier_scales["frontier"], 1.0)
        self.assertFalse(trainer.frontier_history["frontier"])

    def test_reprise_checkpoint_pilote_sans_etat_frontiere(self):
        trainer = RLVRTrainer.__new__(RLVRTrainer)
        trainer.cfg = RLVRConfig()
        trainer.run_dir = Path("unused")
        trainer.profile_sha256 = "profile"
        trainer.model, trainer.optimizer = Mock(), Mock()
        trainer.use_cuda = False
        trainer.rng = random.Random(1)
        trainer.difficulty = {"reasoning_program": 0.25}
        trainer.frontier_scales = {"frontier": 1.0}
        trainer.frontier_history = {"frontier": deque(maxlen=24)}
        trainer.cap_history = {"reasoning_program": deque(maxlen=40)}
        checkpoint = {
            "stage": "rlvr-v45", "profile_sha256": "profile", "model": {},
            "optimizers": [{}], "accepted_updates": 10, "rollout_index": 65,
            "tokens_generated": 14156, "kl_beta": 0.0035, "best_score": 0.183,
            "baseline_score": 0.067,
            "difficulty": {"reasoning_program": 0.25, "code": 0.20}, "rng": {},
        }
        with patch("frlm.rl_v45.resolve_checkpoint", return_value=Path("pilot.pt")), \
                patch("frlm.rl_v45.torch.load", return_value=checkpoint):
            trainer._resume("latest")
        self.assertEqual(trainer.update, 10)
        self.assertEqual(trainer.frontier_scales, {"frontier": 1.0})
        self.assertEqual(trainer.difficulty, {"reasoning_program": 0.25})
        self.assertTrue(trainer._resume_revalidate)

    def test_reprise_recalcule_un_meilleur_score_de_verificateur_obsolete(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = RLVRTrainer.__new__(RLVRTrainer)
            trainer.cfg = RLVRConfig(accepted_updates=50)
            trainer.stage_dir = Path(directory)
            trainer.run_dir = Path(directory)
            trainer.baseline_score = 0.067
            trainer.best_score = 0.533
            trainer.update = 50
            trainer._resume_revalidate = True
            trainer.stop_requested = False
            trainer.profile = None
            trainer._evaluate = Mock(return_value={"macro": 31 / 60, "rows": []})
            trainer._save = Mock()
            with patch("builtins.print"):
                trainer.train()
            self.assertAlmostEqual(trainer.best_score, 31 / 60)
            self.assertFalse(trainer._resume_revalidate)
            self.assertTrue((Path(directory) / "eval_resume_000050.json").is_file())
            trainer._save.assert_called_once_with(best=False)


class OfflineRLAIFTests(unittest.TestCase):
    def test_pool_exclut_les_familles_ood_historiques(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pool.jsonl"
            path.write_text(
                json.dumps({"type": "calc", "q": "Une composition arithmétique secrète."}) + "\n"
                + json.dumps({"type": "piege", "q": "Une information manque ici."}) + "\n"
                + json.dumps({"type": "chat", "q": "Rédige un message de bienvenue."}) + "\n",
                encoding="utf-8",
            )
            rows = _read_prompts([path], 10, 1)
            self.assertEqual([row["category"] for row in rows], ["chat"])

    def _fixture(self, root: Path):
        stage = root / "run" / "rlaif-v45"
        stage.mkdir(parents=True)
        candidates = {
            "prompt_id": "p_1", "prompt": "Combien font 2 + 2 ?",
            "category": "calc", "source": "fixture",
            "candidates": [
                {"candidate_id": "c_good", "text": "4", "stopped": True,
                 "tokens": 2, "repeat_ratio": 0.0},
                {"candidate_id": "c_bad", "text": "5", "stopped": True,
                 "tokens": 2, "repeat_ratio": 0.0},
            ],
        }
        (stage / "candidates.private.jsonl").write_text(
            json.dumps(candidates, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return stage

    def test_import_scelle_une_preference_claire(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage = self._fixture(root)
            scores_a = stage / "scores_a.jsonl"
            scores_b = stage / "scores_b.jsonl"
            judgment = {
                "prompt_id": "p_1", "ranking": ["c_good", "c_bad"],
                "scores": {"c_good": 4, "c_bad": 1}, "unsafe": [],
            }
            scores_a.write_text(json.dumps(judgment) + "\n", encoding="utf-8")
            scores_b.write_text(json.dumps(judgment) + "\n", encoding="utf-8")
            report = import_scores("run", str(root), scores_a, scores_b)
            self.assertEqual(report["pairs"], 1)
            self.assertTrue(report["double_judgment"])
            self.assertTrue((stage / "pairs.sealed.jsonl").is_file())

    def test_import_refuse_un_identifiant_invente(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage = self._fixture(root)
            scores_a = stage / "scores_a.jsonl"
            scores_b = stage / "scores_b.jsonl"
            scores_a.write_text(json.dumps({
                "prompt_id": "p_1", "ranking": ["c_good", "c_bad"],
                "scores": {"c_good": 4, "c_bad": 1}, "unsafe": [],
            }) + "\n", encoding="utf-8")
            scores_b.write_text(json.dumps({
                "prompt_id": "p_1", "ranking": ["c_good", "c_forge"],
                "scores": {"c_good": 4, "c_forge": 0}, "unsafe": [],
            }) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                import_scores("run", str(root), scores_a, scores_b)

    def test_import_refuse_un_desaccord_sur_le_meilleur(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage = self._fixture(root)
            a, b = stage / "a.jsonl", stage / "b.jsonl"
            a.write_text(json.dumps({"prompt_id": "p_1", "ranking": ["c_good", "c_bad"],
                                     "scores": {"c_good": 4, "c_bad": 1}, "unsafe": []}) + "\n")
            b.write_text(json.dumps({"prompt_id": "p_1", "ranking": ["c_bad", "c_good"],
                                     "scores": {"c_bad": 4, "c_good": 1}, "unsafe": []}) + "\n")
            with self.assertRaises(ValueError):
                import_scores("run", str(root), a, b)

    def test_split_dpo_est_isole_par_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.jsonl"
            rows = []
            for prompt in range(6):
                for pair in range(2):
                    rows.append({"pair_id": f"pair_{prompt}_{pair}",
                                 "prompt_id": f"prompt_{prompt}", "prompt": f"Question {prompt}",
                                 "chosen": f"bonne {pair}", "rejected": f"mauvaise {pair}"})
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            train, val = _load_pairs(path, 45)
            self.assertFalse({row["prompt_id"] for row in train}
                             & {row["prompt_id"] for row in val})


class GateTests(unittest.TestCase):
    def test_gate_refuse_un_profil_de_taches_different(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            common = {"summary": {"overall": {"tasks": 1, "pass@1": 0.0,
                                                "pass@6": 0.5, "success_rate": 0.1,
                                                "dynamic_rate": 1.0, "mean_entropy": 2.0}}}
            baseline = {**common, "rows": [{"task_id": "a"}]}
            candidate = {**common, "rows": [{"task_id": "b"}]}
            bp, cp = root / "base.json", root / "candidate.json"
            bp.write_text(json.dumps(baseline), encoding="utf-8")
            cp.write_text(json.dumps(candidate), encoding="utf-8")
            with self.assertRaises(ValueError):
                evaluate_gates(bp, cp)


if __name__ == "__main__":
    unittest.main()
