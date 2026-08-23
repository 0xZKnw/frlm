from __future__ import annotations

import json
import unittest

from frlm.rl_tasks_v45 import make_task
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


if __name__ == "__main__":
    unittest.main()
