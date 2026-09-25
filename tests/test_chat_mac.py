"""Chargement et génération du chat sur CPU ou GPU Apple."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from tokenizers import Tokenizer, models

from frlm.model_v3 import ModelConfigV3, SpeedLM
from run import chat_device


ROOT = Path(__file__).resolve().parents[1]


class ChatMacTests(unittest.TestCase):
    def test_device_priority(self):
        self.assertEqual(chat_device(False, False), "cpu")
        self.assertEqual(chat_device(False, True), "mps")
        self.assertEqual(chat_device(True, False), "cuda")
        self.assertEqual(chat_device(True, True), "cuda")

    def test_chat_loads_best_and_generates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            data_dir.mkdir()
            vocab = {"Bonjour": 0, "<|endoftext|>": 1, "<|im_start|>": 2,
                     "<|im_end|>": 3, "<think>": 4, "</think>": 5, "[UNK]": 6}
            Tokenizer(models.WordLevel(vocab, unk_token="[UNK]")).save(str(data_dir / "tokenizer.json"))

            cfg = ModelConfigV3(vocab_size=len(vocab), n_layer=2, n_head=2,
                                n_kv_head=1, d_model=32, head_dim=16, d_ff=64,
                                max_seq_len=32, window=16, n_value_embeds=0,
                                canon_kernel=0, unet_skips=False)
            run_dir = root / "runs" / "smoke" / "pretrain"
            run_dir.mkdir(parents=True)
            torch.save({"model_cfg": cfg.to_dict(), "model": SpeedLM(cfg).state_dict(),
                        "step": 7, "tokens_seen": 42, "val_loss": 1.0,
                        "stage": "pretrain"}, run_dir / "ckpt_best.pt")

            result = subprocess.run(
                [sys.executable, str(ROOT / "run.py"), "chat", "--run", "smoke",
                 "--out-dir", str(root / "runs"), "--data-dir", str(data_dir),
                 "--stage", "pretrain", "--ckpt", "best", "--max-new-tokens", "2",
                 "--temperature", "0"],
                input="/raw Bonjour\n/quit\n", text=True, capture_output=True, timeout=90,
                cwd=ROOT,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            expected_device = chat_device(torch.cuda.is_available(), torch.backends.mps.is_available())
            self.assertIn(f"d=32 · {expected_device}", result.stdout)
            self.assertIn("step 7", result.stdout)
            self.assertIn("Bonjour", result.stdout)


if __name__ == "__main__":
    unittest.main()
