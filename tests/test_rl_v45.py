from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

from frlm.rlaif_offline_v45 import _read_prompts, import_scores
from frlm.posttrain_gates_v45 import evaluate_gates
from frlm.rl_v45 import RLVRConfig, RLVRTrainer, _canonical_answer
from frlm.rl_tasks_v45 import SCHEMAS_BY_CAPABILITY, make_task
from frlm.verifiers_v45 import AnswerSpec, final_text, verify


class VerifierTests(unittest.TestCase):
    def test_integer_signed_fraction_and_ambiguity(self):
        self.assertTrue(verify(AnswerSpec("integer", -12), "Réponse : -12").primary_success)
        self.assertTrue(verify(AnswerSpec("rational", "4"), "12/3").primary_success)
        self.assertFalse(verify(AnswerSpec("integer", 12), "12,0").primary_success)
        self.assertEqual(verify(AnswerSpec("integer", 12), "12 puis 3").failure_code,
                         "number_ambiguous")

    def test_choice_word_boundaries(self):
        spec = AnswerSpec("choice", "Paul", choices=("Paul", "Pauline"))
        self.assertTrue(verify(spec, "Paul").primary_success)
        self.assertFalse(verify(spec, "Pauline").primary_success)
        self.assertFalse(verify(spec, "Paul et Pauline").primary_success)
        self.assertFalse(verify(AnswerSpec("entity", "chat"), "chatière").primary_success)

    def test_abstention(self):
        spec = AnswerSpec("abstain")
        self.assertTrue(verify(spec, "Le contexte ne permet pas de le déterminer.").primary_success)
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

    def test_curriculum_rl_est_pilote_par_le_profil(self):
        trainer = RLVRTrainer.__new__(RLVRTrainer)
        trainer.cfg = RLVRConfig(require_profile=True)
        trainer.profile = {"config": {"k": 6}, "rows": [
            {"capability": "code", "schema_id": "code_1", "difficulty": 0.3,
             "initial_successes": 1},
            {"capability": "reasoning_program", "schema_id": "linear_equation",
             "difficulty": 0.1, "initial_successes": 0},
            {"capability": "grounded", "schema_id": "grounded_year", "difficulty": 0.2,
             "initial_successes": 6},
        ]}
        frontier, bridge = trainer._profile_curriculum()
        self.assertEqual({row["schema_id"] for row in frontier}, {"code_1"})
        bridge_schemas = {row["schema_id"] for row in bridge}
        self.assertIn("linear_equation", bridge_schemas)
        self.assertIn("small_power", bridge_schemas)
        self.assertNotIn("code_1", bridge_schemas)
        self.assertNotIn("grounded_year", bridge_schemas)

        trainer.frontier_specs, trainer.bridge_specs = frontier, bridge
        trainer.rng = random.Random(45)
        trainer.difficulty = {capability: 0.25 for capability in SCHEMAS_BY_CAPABILITY}
        trainer.rollout_index = 0
        sampled = [trainer._next_task().schema_id for _ in range(100)]
        self.assertGreaterEqual(sampled.count("code_1"), 65)
        self.assertGreaterEqual(len(sampled) - sampled.count("code_1"), 10)


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
            scores = stage / "scores.jsonl"
            scores.write_text(json.dumps({
                "prompt_id": "p_1", "ranking": ["c_good", "c_bad"],
                "scores": {"c_good": 4, "c_bad": 1}, "unsafe": [],
            }) + "\n", encoding="utf-8")
            report = import_scores("run", str(root), scores)
            self.assertEqual(report["pairs"], 1)
            self.assertTrue((stage / "pairs.sealed.jsonl").is_file())

    def test_import_refuse_un_identifiant_invente(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage = self._fixture(root)
            scores = stage / "scores.jsonl"
            scores.write_text(json.dumps({
                "prompt_id": "p_1", "ranking": ["c_good", "c_forge"],
                "scores": {"c_good": 4, "c_forge": 0}, "unsafe": [],
            }) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                import_scores("run", str(root), scores)


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
