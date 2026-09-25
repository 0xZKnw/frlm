"""Régressions v5 : formats, quotas, gradients, causalité et cache sur CPU."""
import itertools
import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import torch

from frlm import config_from_dict, model_from_cfg
from frlm import data as D
from frlm.prepare_v5 import clean_document, encode_source, split_for, token_targets

try:
    from transformers import Qwen4ExpForCausalLM
except ImportError:
    Qwen4ExpForCausalLM = None


class DataV5Tests(unittest.TestCase):
    def test_pretrain_batches_cover_blocks_once_and_resume(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.bin"
            np.arange(41, dtype=np.uint16).tofile(path)
            corpus = D.BinCorpus(path, 4, without_replacement=True)
            first = [int(v) for step in range(4)
                     for v in corpus.get_batch(step, 3, seed=19, device="cpu")[0][:, 0]]
            self.assertEqual(set(first[:10]), set(range(0, 40, 4)))
            self.assertEqual(len(set(first[:10])), 10)
            self.assertEqual(len(set(first[10:])), 2)
            self.assertEqual(first[9:12], [int(v) for v in corpus.get_batch(3, 3, seed=19, device="cpu")[0][:, 0]])

    def test_v4_preset_on_v5_data_audits_and_resumes_exactly(self):
        import contextlib
        import io
        import shlex
        import time
        import numpy as np
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers
        from frlm.prepare_v5 import sha256, write_json
        from frlm.modal_preflight import _check_command
        from run import TrainConfig, Trainer
        import run

        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel()
        tok.train_from_iterator(["Les nombres et les fractions."], trainers.BpeTrainer(
            vocab_size=300, special_tokens=D.SPECIALS, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp) / "data-v5"
            root.mkdir()
            tok.save(str(root / "tokenizer.json"))
            for split in ("train", "val"):
                np.random.default_rng(55).integers(5, tok.get_vocab_size(), 3000, dtype=np.uint16).tofile(root / f"{split}.bin")
            artifacts = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
                         for p in root.iterdir()}
            write_json(root / "manifest.json", {"artifacts": artifacts})
            write_json(root / "meta.json", {"manifest_sha256": sha256(root / "manifest.json")})
            tiny = dict(n_layer=4, n_head=4, n_kv_head=2, d_model=64, head_dim=16,
                        d_ff=128, max_seq_len=32, window=16, n_value_embeds=1)
            cfg = TrainConfig(run_name="v4-v5", data_dir=str(root), out_dir=str(root / "runs"),
                              preset="v4-base", batch_size=1, grad_accum=1, seq_len=12,
                              max_steps=2, device="cpu", compile=False, sample_every=0)
            with patch.dict(run.PRESETS_V3, {"v4-base": tiny}):
                trainer = Trainer(cfg)
                self.assertTrue(trainer.train_data.without_replacement)
                trainer.t_start = time.time()
                checkpoint = root / "checkpoint.pt"
                torch.save(trainer.state_payload(), checkpoint)
                Trainer(cfg, resume=str(checkpoint))
                cli = ["python", "run.py", "train", "--preset", "v4-base",
                       "--data-dir", str(root), "--run", "v4-v5", "--seq-len", "12",
                       "--batch-size", "1", "--grad-accum", "1", "--max-steps", "2",
                       "--resume", str(checkpoint)]
                _check_command(shlex.join(cli), root)
                cli[cli.index("--max-steps") + 1] = "3"
                with self.assertRaisesRegex(ValueError, "reprise v5 non exacte"):
                    _check_command(shlex.join(cli), root)
                with self.assertRaisesRegex(ValueError, "reprise v5 non exacte"):
                    Trainer(TrainConfig(**(vars(cfg) | {"max_steps": 3})), resume=str(checkpoint))
                (root / "train.bin").write_bytes(b"bad")
                with self.assertRaisesRegex(ValueError, "artefact altéré"):
                    _check_command(shlex.join(cli), root)

    @unittest.skipUnless(importlib.util.find_spec("modal"), "SDK Modal optionnel")
    def test_modal_command_budget_and_explicit_handoff(self):
        from modal_v5 import command
        for seconds in (0, -1, float("nan"), float("inf"), 21601):
            with self.assertRaises(ValueError):
                command("pretrain", 100, seconds)
        for args in [("pilot", 0, 901), ("pretrain", 0, 900), ("sft", 100, 900)]:
            with self.assertRaises(ValueError):
                command(*args)
        first = command("pretrain", 3000, 21600)
        second = command("pretrain", 3000, 18000, "runs/fr-v5-qwen35-230m/pretrain/ckpt_latest.pt")
        self.assertEqual(first[first.index("--preset") + 1], "v5-qwen35-230m")
        self.assertEqual(first[first.index("--run") + 1], "fr-v5-qwen35-230m")
        self.assertEqual(first[first.index("--batch-size") + 1], "32")
        self.assertEqual(first[first.index("--grad-accum") + 1], "2")
        self.assertIn("--no-compile", first)
        sft = command("sft", 2800, 3600, "runs/fr-v5-qwen35-230m/pretrain/ckpt_best.pt")
        self.assertEqual(sft[sft.index("--batch-size") + 1], "8")
        self.assertEqual(sft[sft.index("--grad-accum") + 1], "8")
        self.assertIn("--no-compile", sft)
        self.assertIn("v5-qwen35-230m", command("pilot", 0, 900))
        pilot = command("pilot", 0, 900, pilot_batch=16, pilot_compile=True)
        self.assertEqual(pilot[pilot.index("--batch-size") + 1], "16")
        self.assertEqual(pilot[pilot.index("--grad-accum") + 1], "4")
        self.assertNotIn("--no-compile", pilot)
        v4_pilot = command("pilot-v4", 0, 900, pilot_batch=32, pilot_compile=True)
        self.assertEqual(v4_pilot[v4_pilot.index("--presets") + 1], "v4-base")
        self.assertEqual(v4_pilot[v4_pilot.index("--vocab-size") + 1], "32768")
        self.assertEqual(v4_pilot[v4_pilot.index("--grad-accum") + 1], "2")
        self.assertNotIn("--no-compile", v4_pilot)
        v4_train = command("pretrain-v4", 91553, 18000)
        self.assertEqual(v4_train[v4_train.index("--preset") + 1], "v4-base")
        self.assertEqual(v4_train[v4_train.index("--run") + 1], "fr-v5-v4base-252m")
        self.assertNotIn("--no-compile", v4_train)
        with self.assertRaises(ValueError):
            command("pilot", 0, 900, pilot_batch=3)
        self.assertEqual(first[first.index("--max-steps") + 1], second[second.index("--max-steps") + 1])
        self.assertNotIn("--init-weights-only", second)
        import modal_v5
        with patch.object(modal_v5, "preflight") as cpu, patch.object(modal_v5, "execute") as gpu:
            modal_v5.main()
            cpu.remote.assert_called_once()
            gpu.remote.assert_not_called()
            gpu.spawn.assert_not_called()
            modal_v5.main(go=True, check_only=True)
            gpu.remote.assert_not_called()
            gpu.spawn.assert_not_called()
            modal_v5.main(mode="pretrain", steps=3000, go=True)
            gpu.spawn.assert_called_once()
            gpu.spawn.return_value.get.assert_called_once()
            gpu.remote.assert_not_called()
            self.assertNotIn("--no-compile", command("pilot", 0, 900,
                                                      pilot_batch=32, pilot_compile=True))

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "pyarrow v5 optionnel")
    def test_parquet_nested_messages(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from frlm.prepare_v5 import records
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested.parquet"
            messages = [{"role": "user", "content": "Bonjour"}]
            pq.write_table(pa.Table.from_pylist([{"messages": messages}]), path)
            source = {"name": "scholar", "kind": "scholar", "columns": ["messages"],
                      "files": ["data.parquet"], "revision": "pinned", "repo": "test/repo"}
            with patch("huggingface_hub.hf_hub_download", return_value=str(path)):
                self.assertEqual(list(records(source, Path(tmp), [])), [{"messages": messages}])

    def test_sft_masks_reject_truncation_and_bad_roles(self):
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers
        from frlm.prepare_sft_v5 import encode_conversation
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel()
        tok.train_from_iterator(["Les fractions permettent de calculer un nombre."], trainers.BpeTrainer(
            vocab_size=300, special_tokens=D.SPECIALS, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
        messages = [{"role": "user", "text": "Calcule deux plus deux."},
                    {"role": "assistant", "text": "Deux plus deux font quatre."}]
        encoded, _ = encode_conversation(tok, messages, 1024, "verified")
        ids, mask, _ = encoded
        self.assertEqual(ids[-1], 0)
        self.assertEqual(mask[-1], 0)
        self.assertEqual(mask[0], 0)
        self.assertTrue(any(mask))
        self.assertIsNone(encode_conversation(tok, messages, 4, "verified")[0])
        self.assertIsNone(encode_conversation(tok, messages[::-1], 1024, "verified")[0])
        messages[0]["text"] += D.IM_START
        self.assertIsNone(encode_conversation(tok, messages, 1024, "verified")[0])

    def test_quotas(self):
        self.assertEqual(token_targets(7, [5, 3, 2]), [4, 2, 1])
        self.assertEqual(token_targets(10, [5, 3, 2]), [5, 3, 2])
        for total, weights in itertools.product(range(1, 30), itertools.product(range(1, 4), repeat=3)):
            counts = token_targets(total, list(weights))
            self.assertEqual(sum(counts), total)
            self.assertTrue(all(abs(n - total * w / sum(weights)) < 1 for n, w in zip(counts, weights)))
        for total, weights in [(0, [1]), (3, []), (3, [-1, 2])]:
            with self.assertRaises(ValueError):
                token_targets(total, weights)

    def test_filter_and_group(self):
        src = {"name": "web_hq"}
        text = "Voici une explication française assez longue et utile pour comprendre les fractions. " * 4
        self.assertEqual(clean_document({"text": text, "id": "doc"}, src)[1], "doc")
        for bad in ["court", text + D.IM_START, text + "buy cheap essays", "\ufffd" * 300]:
            self.assertIsNone(clean_document({"text": bad}, src))
        self.assertEqual(split_for("doc"), split_for("doc"))
        self.assertEqual({split_for(str(i)) for i in range(10000)}, {"train", "val", "sealed"})

    def test_encoding_dedup_resume_and_corruption(self):
        from tokenizers import Tokenizer, models, pre_tokenizers, trainers
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel()
        text = "Une explication utile sur les nombres et les fractions permet de calculer correctement. " * 4
        tok.train_from_iterator([text], trainers.BpeTrainer(vocab_size=300, special_tokens=D.SPECIALS,
                                initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tok.save(str(root / "tokenizer.json"))
            db = sqlite3.connect(root / "dedup.sqlite")
            db.execute("CREATE TABLE seen (fingerprint BLOB PRIMARY KEY, group_key TEXT UNIQUE, source TEXT)")
            src = {"name": "web_hq"}
            rows = [{"text": text, "id": "doc"}, {"text": text, "id": "duplicate"},
                    {"text": text.replace("fractions", "multiplications"), "id": "other"}]
            target = len(tok.encode(text).ids) + 2
            with patch("frlm.prepare_v5.records", return_value=iter(rows)):
                report = encode_source(root, src, target, tok, db)
            self.assertEqual(report["duplicates"], 1)
            self.assertEqual(report["documents"], 2)
            self.assertEqual(encode_source(root, src, target, tok, db), json.loads((root / "shards/web_hq/report.json").read_text()))
            (root / "shards/web_hq/train.bin").write_bytes(b"bad")
            with self.assertRaises(ValueError):
                encode_source(root, src, target, tok, db)
            db.close()


@unittest.skipUnless(Qwen4ExpForCausalLM, "installer requirements-v5.txt")
class ModelV5Tests(unittest.TestCase):
    def setUp(self):
        from frlm.model_v5 import ModelConfigV5
        torch.set_num_threads(1)
        torch.manual_seed(5501)
        self.cfg = ModelConfigV5(vocab_size=300, d_model=32, n_layer=4, n_head=2,
                                n_kv_head=1, head_dim=8, d_ff=16, linear_heads=2,
                                linear_key_heads=1, ngram_vocab=31, num_experts=2,
                                experts_per_token=1, max_seq_len=32)
        self.preset = "v5-qwen4exp-350m"
        self.model = model_from_cfg(config_from_dict(self.cfg.to_dict()))
        self.x = torch.randint(5, 300, (1, 12))

    def test_full_context_qsa_matches_reference_and_skips_token_loop(self):
        import copy
        reference = copy.deepcopy(self.model)
        for module in reference.modules():
            if type(module).__name__ == "Qwen4ExpTextQSAIndexer":
                del module.forward
        indexer = self.model.hf.model.layers[3].self_attn.indexer
        for implementation in ("eager", "sdpa"):
            self.model.hf.set_attn_implementation(implementation)
            reference.hf.set_attn_implementation(implementation)
            self.model.zero_grad(set_to_none=True)
            reference.zero_grad(set_to_none=True)
            expected, expected_loss, _ = reference(self.x, self.x.roll(-1, 1))
            # Une régression vers la boucle amont doit échouer même si les logits restent justes.
            with patch.object(type(indexer), "forward", side_effect=AssertionError("boucle QSA")):
                actual, actual_loss, _ = self.model(self.x, self.x.roll(-1, 1))
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
            actual_loss.backward()
            expected_loss.backward()
            for a, b in zip(self.model.parameters(), reference.parameters()):
                if b.grad is None:
                    self.assertIsNone(a.grad)
                else:
                    torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)

    def test_qsa_cache_crosses_budget_without_losing_keys(self):
        import copy
        reference = copy.deepcopy(self.model).eval()
        self.model.eval()
        for model in (self.model, reference):
            indexer = model.hf.model.layers[3].self_attn.indexer
            indexer.token_budget = 8
            indexer.block_topk = 2
        del reference.hf.model.layers[3].self_attn.indexer.forward
        actual_cache = self.model._alloc_caches(1, 32, "cpu", torch.float32)
        expected_cache = reference._alloc_caches(1, 32, "cpu", torch.float32)
        with torch.no_grad():
            for start, end in ((0, 7), (7, 8), (8, 9), (9, 12)):
                actual = self.model._forward_cached(self.x[:, start:end], actual_cache, start)
                expected = reference._forward_cached(self.x[:, start:end], expected_cache, start)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(actual_cache.layers[3].indexer_keys,
                                           expected_cache.layers[3].indexer_keys, rtol=0, atol=0)

    def test_qsa_masks_padding_empty_rows_and_budget_boundaries(self):
        indexer = self.model.hf.model.layers[3].self_attn.indexer
        indexer.token_budget, indexer.block_topk = 8, 2
        for length in (1, 4, 7, 8, 9, 12):
            hidden = torch.randn(2, length, self.cfg.d_model)
            positions = (torch.ones(2, length, indexer.index_head_dim),
                         torch.zeros(2, length, indexer.index_head_dim))
            visible = torch.rand(2, 1, length, length) > 0.35
            visible &= torch.ones(length, length, dtype=torch.bool).tril()
            visible[:, :, 0] = False
            for dtype in (torch.bool, torch.float32, torch.bfloat16):
                mask = visible if dtype == torch.bool else torch.where(
                    visible, torch.tensor(0, dtype=dtype), torch.finfo(dtype).min)
                expected = type(indexer).forward(indexer, hidden, positions, mask, None)
                actual = indexer(hidden, positions, mask, None)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_gradient_mask_and_already_shifted_targets(self):
        targets = self.x.roll(-1, 1)
        mask = torch.zeros_like(targets)
        mask[:, 4:9] = 1
        self.model.eval()
        logits, loss, _ = self.model(self.x, targets, mask)
        expected = torch.nn.functional.cross_entropy(logits[mask.bool()].float(), targets[mask.bool()])
        torch.testing.assert_close(loss, expected)
        _, summed, _ = self.model(self.x, targets, mask, loss_reduction="sum")
        torch.testing.assert_close(summed, loss * mask.sum())
        self.model.train()
        _, loss, _ = self.model(self.x, targets, mask)
        loss.backward()
        router = self.model.hf.model.layers[0].mlp.gate.weight
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(router.grad.norm()), 0)

    def test_causality_cache_and_roundtrip(self):
        self.model.eval()
        with torch.no_grad():
            full = self.model(self.x)[0]
            changed = self.x.clone()
            changed[:, 9:] = 42
            torch.testing.assert_close(full[:, :9], self.model(changed)[0][:, :9])
            cache = self.model._alloc_caches(1, 32, "cpu", torch.float32)
            self.model._forward_cached(self.x[:, :8], cache, 0)
            for i in range(8, 12):
                cached = self.model._forward_cached(self.x[:, i:i+1], cache, i)
                torch.testing.assert_close(cached[:, -1], full[:, i], atol=1e-6, rtol=1e-5)
            restored = model_from_cfg(config_from_dict(self.cfg.to_dict())).eval()
            restored.load_state_dict(self.model.state_dict())
            torch.testing.assert_close(restored(self.x)[0], full)

    def test_parameter_budget_and_context_guard(self):
        from frlm.model_v5 import ModelConfigV5
        with torch.device("meta"):
            full = model_from_cfg(ModelConfigV5())
        self.assertEqual(full.num_params(), 350_011_504)
        with self.assertRaises(ValueError):
            ModelConfigV5(max_seq_len=4096).hf_config()

    def test_optimizer_experts_and_exact_reload(self):
        from frlm.optim import build_optimizers
        from run import TrainConfig
        cfg = TrainConfig(optimizer="muon", lr=0.001, adam_lr=0.0001)
        opts, _ = build_optimizers(self.model, cfg)
        expert = self.model.hf.model.layers[0].mlp.experts.gate_up_proj
        router = self.model.hf.model.layers[0].mlp.gate.weight
        self.assertTrue(any(expert is p for g in opts[0].param_groups for p in g["params"]))
        self.assertTrue(any(router is p for g in opts[1].param_groups for p in g["params"]))
        _, loss, _ = self.model(self.x, self.x.roll(-1, 1))
        loss.backward()
        for opt in opts:
            opt.step()
            opt.zero_grad(set_to_none=True)
        import copy
        resumed = model_from_cfg(self.cfg)
        resumed.load_state_dict(self.model.state_dict())
        resumed_opts, _ = build_optimizers(resumed, cfg)
        for opt, previous in zip(resumed_opts, opts):
            opt.load_state_dict(copy.deepcopy(previous.state_dict()))
        for model, optimizers in [(self.model, opts), (resumed, resumed_opts)]:
            model(self.x, self.x.roll(-1, 1))[1].backward()
            for opt in optimizers:
                opt.step()
        for a, b in zip(self.model.parameters(), resumed.parameters()):
            torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_trainer_handoff_preserves_global_schedule(self):
        import contextlib
        import io
        import numpy as np
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers
        from frlm.prepare_v5 import sha256, write_json
        from run import TrainConfig, Trainer
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel()
        tok.train_from_iterator(["Les nombres et les fractions."], trainers.BpeTrainer(
            vocab_size=300, special_tokens=D.SPECIALS, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            tok.save(str(root / "tokenizer.json"))
            for split in ("train", "val", "sealed"):
                np.random.default_rng(55).integers(5, tok.get_vocab_size(), 3000, dtype=np.uint16).tofile(root / f"{split}.bin")
            artifacts = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
                         for p in root.iterdir()}
            write_json(root / "manifest.json", {"artifacts": artifacts})
            write_json(root / "meta.json", {"manifest_sha256": sha256(root / "manifest.json")})
            preset = self.cfg.to_dict()
            preset["max_seq_len"] = 128
            cfg = dict(data_dir=str(root), out_dir=str(root / "runs"), preset=self.preset,
                       batch_size=1, grad_accum=1, seq_len=12, max_steps=3, warmup=2,
                       dtype="float32", device="cpu", compile=False, sample_every=0,
                       eval_every=2, eval_iters=1, optimizer="adamw", lr=1e-4)
            with patch.dict("run.PRESETS_V5", {self.preset: preset}):
                uninterrupted = Trainer(TrainConfig(run_name="whole", **cfg))
                self.assertEqual(uninterrupted.train_data.without_replacement,
                                 self.preset == "v5-qwen35-230m")
                uninterrupted.train()
                interrupted = Trainer(TrainConfig(run_name="split", stop_after_seconds=1e-9, **cfg))
                interrupted.train()
                self.assertEqual(interrupted.step, 1)
                checkpoint = root / "runs/split/pretrain/ckpt_latest.pt"
                resumed = Trainer(TrainConfig(run_name="split", **cfg), resume=str(checkpoint))
                self.assertEqual(resumed.cfg.max_steps, 3)
                resumed.train()
                for a, b in zip(uninterrupted.model.parameters(), resumed.model.parameters()):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)
                altered = cfg | {"max_steps": 4}
                with self.assertRaisesRegex(ValueError, "reprise v5 non exacte"):
                    Trainer(TrainConfig(run_name="split", **altered), resume=str(checkpoint))
                # Le même contrat est exécuté avant une allocation Modal.
                import shlex
                from frlm.modal_preflight import _check_command
                from frlm.model_v5 import validate_resume
                cli = ["python", "run.py", "train", "--preset", self.preset,
                       "--data-dir", str(root), "--seq-len", "12", "--batch-size", "1",
                       "--grad-accum", "1", "--max-steps", "3", "--warmup", "2",
                       "--dtype", "float32", "--optimizer", "adamw", "--lr", "0.0001",
                       "--resume", str(checkpoint)]
                _check_command(shlex.join(cli), root)
                cli[cli.index("--max-steps") + 1] = "4"
                with self.assertRaisesRegex(ValueError, "reprise v5 non exacte"):
                    _check_command(shlex.join(cli), root)
                payload = torch.load(checkpoint, weights_only=False)
                with self.assertRaisesRegex(ValueError, "état complet"):
                    validate_resume(payload, resumed.cfg, resumed.mcfg.to_dict(),
                                    resumed.data_manifest_sha256, resumed.tokenizer_sha256, None, True)

                from frlm.prepare_sft_v5 import prepare, audit
                import hashlib
                groups = {}
                for i in range(1000):
                    bucket = int.from_bytes(hashlib.sha256(str(i).encode()).digest()[:8], "big") % 100
                    groups.setdefault("val" if bucket == 0 else "sealed" if bucket == 1 else "train", str(i))
                rows = [([{"role": "user", "text": f"Calcule le nombre {i} plus deux."},
                          {"role": "assistant", "text": f"Le résultat est {i + 2}."}], groups[s])
                        for i, s in enumerate(("val", "sealed", "train"))]
                recipe = {"recipe": "v5-sft-20260924", "max_seq_len": 128,
                          "target_supervised": 1, "sources": [{"name": "verified", "weight": 100}]}
                with patch("frlm.prepare_sft_v5.conversations", return_value=iter(rows)):
                    prepare(root, recipe)
                sft_cfg = cfg | {"stage": "sft", "sft_recipe": "v5", "seq_len": 128,
                                 "stop_after_seconds": 1e-9}
                sft = Trainer(TrainConfig(run_name="split", **sft_cfg), resume=str(checkpoint))
                self.assertEqual(sft.step, 0)
                sft.train()
                self.assertEqual(sft.step, 1)
                (root / "sft_v5_val.mask").write_bytes(b"bad")
                with self.assertRaisesRegex(ValueError, "fusionné altéré"):
                    audit(root)
                from frlm.export_v5 import export_hf
                from transformers import AutoTokenizer, GenerationConfig
                exported = root / "hf"
                export_hf(checkpoint, root / "tokenizer.json", exported)
                fast = AutoTokenizer.from_pretrained(exported, local_files_only=True)
                messages = [{"role": "user", "content": "  Bonjour !  "},
                            {"role": "assistant", "content": "  Bonjour.  "}]
                self.assertEqual(fast.apply_chat_template(messages, tokenize=False), D.render_chat(messages))
                self.assertEqual(GenerationConfig.from_pretrained(exported).eos_token_id, [0, 2])
                with self.assertRaises(FileExistsError):
                    export_hf(checkpoint, root / "tokenizer.json", exported)
                if self.preset in ("v5-dense-350m", "v5-qwen35-230m"):
                    from transformers import AutoModelForCausalLM
                    hf = AutoModelForCausalLM.from_pretrained(
                        exported, local_files_only=True, trust_remote_code=True).eval()
                    reference = model_from_cfg(config_from_dict(payload["model_cfg"])).eval()
                    reference.load_state_dict(payload["model"])
                    probe = self.x % reference.cfg.vocab_size
                    torch.testing.assert_close(hf(probe).logits, reference(probe)[0])
                    # Un ancien MoE doit être refusé avant l'allocation GPU et au chargement.
                    from frlm.model_v5 import ModelConfigV5
                    wrong = root / "old-moe.pt"
                    torch.save(payload | {"model_cfg": ModelConfigV5().to_dict()}, wrong)
                    cli[cli.index("--max-steps") + 1] = "3"
                    cli[cli.index("--resume") + 1] = str(wrong)
                    with self.assertRaisesRegex(ValueError, "configuration v5 différente"):
                        _check_command(shlex.join(cli), root)
                    with self.assertRaisesRegex(ValueError, "configuration v5 différente"):
                        Trainer(TrainConfig(run_name="split", **cfg), resume=str(wrong))


@unittest.skipUnless(Qwen4ExpForCausalLM, "installer requirements-v5.txt")
class DenseV5Tests(unittest.TestCase):
    def setUp(self):
        from frlm.model_v5 import ModelConfigV5Dense
        torch.set_num_threads(1)
        torch.manual_seed(5501)
        self.cfg = ModelConfigV5Dense(vocab_size=300, d_model=32, n_layer=4, n_head=4,
                                     n_kv_head=2, head_dim=8, d_ff=64, max_seq_len=32,
                                     linear_heads=4, linear_key_heads=2, ngram_vocab=31)
        self.preset = "v5-dense-350m"
        self.model = model_from_cfg(config_from_dict(self.cfg.to_dict()))
        self.x = torch.randint(5, 300, (1, 12))

    # Même contrat utilisateur pour le dense et les checkpoints MoE historiques.
    test_causality_cache_and_roundtrip = ModelV5Tests.test_causality_cache_and_roundtrip
    test_trainer_handoff_preserves_global_schedule = ModelV5Tests.test_trainer_handoff_preserves_global_schedule
    test_full_context_qsa_matches_reference_and_skips_token_loop = ModelV5Tests.test_full_context_qsa_matches_reference_and_skips_token_loop

    def test_hf_config_and_export_guard(self):
        from huggingface_hub.errors import StrictDataclassClassValidationError
        from frlm.qwen4_dense import Qwen4ExpDenseConfig
        from frlm.export_v5 import convert, main
        cfg = self.cfg.hf_config().to_dict()
        for changed in ({"num_experts": 1}, {"num_experts_per_tok": 1},
                        {"output_router_logits": True}, {"router_aux_loss_coef": 0.01},
                        {"intermediate_size": 0}, {"hc_count": 1},
                        {"indexer_kv_heads": 2}, {"ple_layer_ids": [4]}):
            with self.subTest(changed=changed), self.assertRaises(StrictDataclassClassValidationError):
                Qwen4ExpDenseConfig.from_dict(cfg | changed)
        self.assertEqual(Qwen4ExpDenseConfig.from_dict(cfg).num_experts, 0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError, "GGUF Qwen4-exp dense non pris en charge"):
                convert(root, root / "out.gguf", root / "llama")
            self.assertFalse((root / "out.gguf").exists())
        with patch("sys.argv", ["export", "--hf-only", "--checkpoint", "ckpt.pt",
                                "--hf-dir", "exported"]), \
                patch("frlm.export_v5.export_hf") as export, patch("frlm.export_v5.convert") as gguf:
            main()
            export.assert_called_once_with(Path("ckpt.pt"), Path("data-v5/tokenizer.json"), Path("exported"))
            gguf.assert_not_called()

    def test_dense_budget_routing_and_invalid_dimensions(self):
        from frlm.bench_speed import construire
        from frlm.model_v5 import ModelConfigV5Dense
        from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextMLP
        with torch.device("meta"):
            full, cfg = construire(self.preset, 1024, 32768)
        self.assertEqual(full.num_params(), 351_046_320)
        self.assertEqual(cfg.to_dict()["arch"], "v5-dense")
        self.assertEqual(full.hf.config.model_type, "frlm_qwen4exp_dense")
        self.assertEqual(full.hf.config.num_experts, 0)
        self.assertEqual(full.hf.config.hc_count, 4)
        self.assertEqual(full.hf.config.ple_layer_ids, [2])
        self.assertFalse(full.has_router)
        self.assertTrue(all(type(layer.mlp) is Qwen4ExpTextMLP for layer in full.hf.model.layers))
        self.assertFalse(any("expert" in name or "router" in name for name, _ in full.named_parameters()))
        self.assertIs(full.hf.lm_head.weight, full.hf.model.embed_tokens.weight)
        self.assertEqual(full.hf.config.layer_types,
                         (["linear_attention"] * 3 + ["qwen_sparse_attention"]) * 6)
        for changes in ({"max_seq_len": 0}, {"max_seq_len": 2049}, {"n_layer": 0},
                        {"n_kv_head": 0}, {"n_kv_head": 5}, {"head_dim": 63},
                        {"d_model": 1000}, {"d_ff": -1},
                        {"d_model": 48, "n_head": 16, "head_dim": 3},
                        {"linear_heads": 7}, {"linear_key_heads": 0},
                        {"num_experts": 1}, {"experts_per_token": 1}, {"router_aux_loss_coef": 0.01}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                ModelConfigV5Dense(**changes).hf_config()

    def test_dense_masked_loss_gradients_optimizer_and_generation(self):
        from frlm.optim import build_optimizers
        from run import TrainConfig
        targets = self.x.roll(-1, 1)
        targets[:, 0] = -100
        mask = torch.zeros_like(targets)
        mask[:, :9] = 1
        logits, loss, _ = self.model(self.x, targets, mask, z_loss=0.01)
        valid = mask.bool() & targets.ne(-100)
        expected = torch.nn.functional.cross_entropy(logits[valid].float(), targets[valid])
        expected += 0.01 * logits[valid].float().logsumexp(-1).square().mean()
        torch.testing.assert_close(loss, expected)
        summed = self.model(self.x, targets, mask, z_loss=0.01, loss_reduction="sum")[1]
        torch.testing.assert_close(summed, loss * valid.sum())
        opts, _ = build_optimizers(self.model, TrainConfig(optimizer="muon"))
        before = self.model.hf.model.layers[0].mlp.up_proj.weight.detach().clone()
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for layer in self.model.hf.model.layers for p in layer.mlp.parameters()))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.model.parameters()
                            if p.grad is not None))
        for opt in opts:
            opt.step()
            opt.zero_grad(set_to_none=True)
        self.assertFalse(torch.equal(before, self.model.hf.model.layers[0].mlp.up_proj.weight))
        empty = self.model(self.x, targets, torch.zeros_like(mask), z_loss=0.01)[1]
        self.assertEqual(empty.item(), 0)
        empty.backward()
        self.assertTrue(all(torch.count_nonzero(p.grad) == 0 for p in self.model.parameters()
                            if p.grad is not None))
        with self.assertRaises(ValueError):
            self.model(self.x, targets, loss_reduction="invalid")
        with self.assertRaises(ValueError):
            self.model(torch.zeros(1, 33, dtype=torch.long))
        self.model.eval()
        generated = self.model.generate(self.x, max_new_tokens=3, temperature=0, stop_ids=())
        self.assertEqual(generated.shape, (1, 15))
        torch.testing.assert_close(generated[:, :12], self.x)


@unittest.skipUnless(importlib.util.find_spec("transformers"), "installer requirements-v5.txt")
class Qwen35V5Tests(unittest.TestCase):
    def setUp(self):
        from frlm.model_v5 import ModelConfigV5Qwen35
        torch.set_num_threads(1)
        torch.manual_seed(5502)
        self.cfg = ModelConfigV5Qwen35(vocab_size=300, d_model=128, n_layer=4,
                                       n_head=2, n_kv_head=1, head_dim=64, d_ff=256,
                                       linear_heads=2, linear_key_heads=2,
                                       max_seq_len=32)
        self.preset = "v5-qwen35-230m"
        self.model = model_from_cfg(config_from_dict(self.cfg.to_dict()))
        self.x = torch.randint(5, 300, (1, 12))

    test_causality_cache_and_roundtrip = ModelV5Tests.test_causality_cache_and_roundtrip
    test_trainer_handoff_preserves_global_schedule = ModelV5Tests.test_trainer_handoff_preserves_global_schedule
    test_dense_masked_loss_gradients_optimizer_and_generation = DenseV5Tests.test_dense_masked_loss_gradients_optimizer_and_generation

    def test_native_config_parameter_budget_and_invalid_dimensions(self):
        from frlm.bench_speed import construire
        from frlm.model_v5 import ModelConfigV5Qwen35
        from transformers import Qwen3_5ForCausalLM
        with torch.device("meta"):
            full, cfg = construire(self.preset, 1024, 32768)
        self.assertEqual(full.num_params(), 228_436_896)
        self.assertEqual(cfg.to_dict()["arch"], "v5-qwen35")
        self.assertIs(type(full.hf), Qwen3_5ForCausalLM)
        self.assertEqual(full.hf.config.model_type, "qwen3_5_text")
        self.assertIs(full.hf.lm_head.weight, full.hf.model.embed_tokens.weight)
        self.assertEqual(full.hf.config.layer_types,
                         (["linear_attention"] * 3 + ["full_attention"]) * 6)
        self.assertFalse(any("expert" in name or "router" in name for name, _ in full.named_parameters()))
        for changes in ({"vocab_size": 0}, {"max_seq_len": 0}, {"max_seq_len": 2049},
                        {"n_layer": 3}, {"n_layer": 5}, {"n_kv_head": 0},
                        {"n_kv_head": 3}, {"head_dim": 63}, {"d_model": 1000},
                        {"d_ff": -1}, {"linear_heads": 7}, {"linear_key_heads": 0}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                ModelConfigV5Qwen35(**changes).hf_config()


if __name__ == "__main__":
    unittest.main()
