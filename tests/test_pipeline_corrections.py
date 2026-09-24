"""Régressions CPU de reason45c et frontières du pipeline, données jetables."""
from __future__ import annotations

import copy
import itertools
import json
import random
import re
import signal
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from tokenizers import Tokenizer, models, pre_tokenizers

import run
from frlm import config_from_dict, data as D, model_from_cfg
from frlm import reason_bootstrap_v45 as R
from frlm import synth_programs as S
from frlm.audit_reason_bootstrap_v45 import audit
from frlm.eval_reason_bootstrap_v45 import baseline_report, profile, score_output, stratified_sample
from frlm.modal_preflight import _check_command, _required_files, with_gpu_peak
from frlm.rl_tasks_v45 import make_task
from frlm.verifiers_v45 import verify


def tokenizer(root):
    tok = Tokenizer(models.WordLevel({word: i for i, word in enumerate([*D.SPECIALS, '[UNK]'])}, unk_token='[UNK]'))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_special_tokens(D.SPECIALS)
    tok.save(str(root / 'tokenizer.json'))
    return tok


def retention(root):
    tok = tokenizer(root)
    record = {'messages': [{'role': 'user', 'text': 'Bonjour ?'},
                           {'role': 'assistant', 'text': 'Bonjour, comment allez-vous ?'}]}
    capabilities = {}
    for name in R.BALANCED_REPLAY_WEIGHTS:
        paths = {}
        for split in ('train', 'val'):
            p = root / f'sft_v45_{split}_{name}.bin'
            stat = R._build_bucket(tok, [record], p, 512)
            paths.update({f'{split}_path': p.name, f'{split}_tokens': stat['tokens'],
                          f'{split}_conversations': stat['conversations']})
        capabilities[name] = {**paths, 'actual_supervised': stat['supervised']}
    (root / 'meta.json').write_text(json.dumps({'sft_v45': {'capabilities': capabilities},
                                              'historical_marker': {'unchanged': True}}))
    return tok


def config(vocab=12, arch='v2'):
    return config_from_dict(dict(arch=arch, vocab_size=vocab, n_layer=2, d_model=16,
                                 n_head=2, n_kv_head=1, head_dim=8, d_ff=32, max_seq_len=512))


class GeneratorCorrections(unittest.TestCase):
    def test_regroupements_et_signes_tous_splits(self):
        const = lambda value: {'op': 'const', 'value': value}
        sub = lambda left, right: {'op': 'sub', 'args': [left, right]}
        left, right = sub(sub(const(9), const(4)), const(2)), sub(const(9), sub(const(4), const(2)))
        self.assertEqual(R._natural(left, 'train', random.Random(0)), '((9 moins 4) moins 2)')
        self.assertEqual(R._natural(right, 'structure_holdout', random.Random(0)), '(9 moins (4 moins 2))')
        self.assertEqual((R.evaluate_ast(left), R.evaluate_ast(right)), (3, 7))
        # Choisir le gabarit infixe permet une interprétation indépendante par Python.
        class Infix:
            def choice(self, values):
                return values[-1]
        for values in itertools.product((-3, -1, 0, 2, 9), repeat=3):
            for ops in itertools.product(R.OPS, repeat=2):
                for branch in (0, 1):
                    children = [const(values[0]), const(values[1])]
                    children[branch] = {'op': ops[1], 'args': [children[branch], const(values[2])]}
                    node = {'op': ops[0], 'args': children}
                    text = R._natural(node, 'train', Infix())
                    expression = text.replace('plus', '+').replace('moins', '-').replace('fois', '*')
                    self.assertEqual(eval(expression, {'__builtins__': {}}), R.evaluate_ast(node))
                    for split in (*R.EVAL_SPLITS, R.FINAL_SPLIT):
                        rendered = R._natural(node, split, random.Random(0))
                        self.assertEqual(rendered.count('('), rendered.count(')'))
                        self.assertTrue(rendered.startswith('('))

    def test_ordres_acceptes_sont_exactement_les_tris_topologiques(self):
        lexical_success = total = 0
        for split in ('train', *R.EVAL_SPLITS, R.FINAL_SPLIT):
            for seed in range(150):
                row = R.make_example(seed, split, 'order_steps')
                self.assertNotRegex(row['prompt'], r'Étape|\br\d+\b')
                lines = re.findall(r'([A-Z])\. (\w+) = (\w+|-?\d+) [−+×] (\w+|-?\d+)', row['prompt'])
                self.assertEqual(len(lines), row['operations'])
                valid = []
                for order in itertools.permutations(lines):
                    available = set()
                    for label, name, a, b in order:
                        if any(not part.lstrip('-').isdigit() and part not in available for part in (a, b)):
                            break
                        available.add(name)
                    else:
                        valid.append(','.join(line[0] for line in order))
                self.assertEqual(set(valid), set(row['valid_orders']))
                for answer in valid:
                    self.assertTrue(score_output(row, answer))
                guess = ','.join(line[0] for line in sorted(lines, key=lambda line: line[1]))
                lexical_success += score_output(row, guess)
                total += 1
        self.assertLess(lexical_success / total, 0.55)

    def test_equilibre_erreurs_et_une_seule_erreur_locale(self):
        for split in ('train', *R.EVAL_SPLITS, R.FINAL_SPLIT):
            rows = R._unique_examples(600, split, 45, set())
            for n in (2, 3):
                counts = Counter(row['target'] for row in rows if row['objective'] == 'find_error' and row['operations'] == n)
                self.assertEqual(set(counts), {str(i) for i in range(1, n + 1)})
                self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
            for row in rows:
                self.assertTrue(score_output(row, row['answer']), row)
                if row['objective'] != 'find_error':
                    continue
                values, errors = {}, []
                for i, a, op, b, value in re.findall(r'Étape (\d+) : r\d+ = (r\d+|-?\d+) ([+−×]) (r\d+|-?\d+) = (-?\d+)', row['prompt']):
                    operands = [values[x] if x in values else int(x) for x in (a, b)]
                    result = {'+': lambda: sum(operands), '−': lambda: operands[0] - operands[1], '×': lambda: operands[0] * operands[1]}[op]()
                    if result != int(value):
                        errors.append(i)
                    values[f'r{i}'] = int(value)
                self.assertEqual(errors, [row['target']])

    def test_programmes_et_surfaces_separes(self):
        sets = {}
        for split in ('train', *R.EVAL_SPLITS, R.FINAL_SPLIT):
            rows = R._unique_examples(500, split, 123, set())
            sets[split] = {r['program_key'] for r in rows}
            self.assertTrue(all(r['operations'] >= 2 for r in rows if r['objective'] not in ('execute', 'number_only')))
        for a, b in itertools.combinations(sets, 2):
            self.assertFalse(sets[a] & sets[b])
        for objective in ('find_error', 'masked_step', 'order_steps'):
            prefixes = [R.make_example(5, split, objective)['prompt'].splitlines()[0]
                        for split in ('train', 'surface_holdout', R.FINAL_SPLIT)]
            self.assertEqual(len(set(prefixes)), 3)

    def test_rl_moyenne_sans_copie_et_surfaces_reellement_separees(self):
        for split in ('train', 'dev'):
            for seed in range(1000):
                row = make_task(seed, split, 0.4, 'reasoning_program', 'mean_three')
                values = row.latent_program['args']
                self.assertEqual(sum(values), 3 * row.answer.value)
                self.assertNotIn(row.answer.value, values)
                self.assertTrue(verify(row.answer, str(row.answer.value)).primary_success)
        for capability in ('constraints', 'uncertainty', 'state_tracking'):
            prompts = [{make_task(i, split, 0.4, capability).prompt for i in range(1000)}
                       for split in ('train', 'dev')]
            self.assertFalse(prompts[0] & prompts[1])

    def test_sorties_numeriques_strictes_sans_perdre_une_vraie_reponse(self):
        row = R.make_example(123, objective='number_only')
        self.assertTrue(score_output(row, row['target']))
        for bad in (f"Réponse : {row['target']}", f"{row['target']}.", f"<think>{row['target']}"):
            self.assertFalse(score_output(row, bad))
        positive = {'objective': 'number_only', 'target': '2'}
        self.assertTrue(score_output(positive, '+2'))
        self.assertFalse(score_output(positive, '2.2'))

    def test_baselines_et_profil_stratifie(self):
        rows = R._unique_examples(600, 'iid', 7, set())
        chosen = stratified_sample(rows, 90, 4)
        self.assertEqual(chosen, stratified_sample(rows, 90, 4))
        self.assertEqual({(r['objective'], r['difficulty']) for r in rows}, {(r['objective'], r['difficulty']) for r in chosen})
        baseline = baseline_report(chosen)
        for n in (2, 3):
            metric = baseline[f'objective:find_error/ops_{n}']
            self.assertAlmostEqual(metric['random_expected_rate'], 1 / n)
            self.assertLessEqual(metric['majority_oracle']['hits'], (metric['tasks'] + n - 1) // n)
        for count in (0, 1, 12, 601):
            with self.assertRaises(ValueError):
                stratified_sample(rows, count, 0)
        minimum = stratified_sample(rows, 15, 0)
        self.assertEqual({(r['objective'], r['difficulty']) for r in rows},
                         {(r['objective'], r['difficulty']) for r in minimum})

    def test_remises_exactes_et_consigne_json(self):
        checked = 0
        for seed in range(2000):
            row = S.make_reasoning(random.Random(seed))
            if row['schema_id'] == 'discount':
                price, _, rate = map(int, re.findall(r'\d+', row['program']))
                self.assertEqual(int(row['answer']) * 100, price * (100 - rate))
                checked += 1
            row = S.make_constraints_v45(random.Random(seed))
            if row['schema_id'] == 'constraint_2':
                self.assertIn('double', row['m'][0]['text'])
                self.assertIn('nombre initial', row['m'][0]['text'])
        self.assertGreater(checked, 100)


class PipelineIntegration(unittest.TestCase):
    def test_prepare_audit_corpus_trainer_et_profil_cpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tok = retention(root)
            (root / 'reason_v45_train.bin').write_bytes(b'histoire')
            report = R.prepare(root, examples=1000, eval_per_split=60, max_len=512)
            self.assertEqual((root / 'reason_v45_train.bin').read_bytes(), b'histoire')
            self.assertTrue(json.loads((root / 'meta.json').read_text())['historical_marker']['unchanged'])
            self.assertTrue(audit(root, True, 'reason45c')['ok'])
            with self.assertRaises(FileExistsError):
                R.prepare(root, examples=1000, eval_per_split=30)
            corpus = D.ConversationCorpus(root / report['train_path'], 512)
            x, y, mask = corpus.get_batch(0, 16, device='cpu')
            self.assertEqual(tuple(x.shape), (16, 512))
            self.assertTrue(torch.all(mask[y == 0] == 0))
            params = dict(n_layer=2, d_model=16, n_head=2, n_kv_head=1, head_dim=8, d_ff=32, max_seq_len=512)
            cfg = run.TrainConfig(run_name='tiny', data_dir=str(root), out_dir=str(root / 'runs'),
                                  stage='sft', sft_recipe='reason45c', preset='test', device='cpu',
                                  batch_size=2, grad_accum=2, seq_len=512, optimizer='adamw',
                                  max_steps=1, compile=False, replay_frac=0, warmup=1,
                                  eval_every=100, sample_every=100, eval_iters=1, log_every=1)
            with patch.dict(run.PRESETS, {'test': params}):
                trainer = run.Trainer(cfg)
                reference = copy.deepcopy(trainer.model)
                batches = [trainer.train_data.get_batch(i, cfg.batch_size, cfg.seed, 'cpu')
                           for i in range(cfg.grad_accum)]
                x_all, y_all, mask_all = [torch.cat([batch[j] for batch in batches]) for j in range(3)]
                _, loss, _ = reference(x_all, y_all, mask_all, z_loss=cfg.z_loss)
                loss.backward()
                expected = [p.grad for p in reference.parameters()]
                clip = torch.nn.utils.clip_grad_norm_
                def check_gradients(parameters, max_norm):
                    parameters = list(parameters)
                    for parameter, grad in zip(parameters, expected):
                        if grad is not None:
                            torch.testing.assert_close(parameter.grad, grad, atol=2e-6, rtol=2e-5)
                    return clip(parameters, max_norm)
                old_handler = signal.getsignal(signal.SIGINT)
                try:
                    with patch('torch.nn.utils.clip_grad_norm_', check_gradients):
                        trainer.train()
                finally:
                    signal.signal(signal.SIGINT, old_handler)
                self.assertEqual(trainer.step, 1)
                trainer.metrics_file.close()
                restored = run.Trainer(cfg, resume='latest')
                self.assertEqual(restored.step, 1)
                self.assertTrue(restored.opts[0].state)
                restored.metrics_file.close()
                fresh = run.Trainer(cfg, resume='latest', init_weights_only=True)
                self.assertEqual(fresh.step, 0)
                self.assertFalse(fresh.opts[0].state)
                fresh.metrics_file.close()
                with self.assertRaises(SystemExit):
                    trainer.load_checkpoint(str(root / 'absent.pt'))
                path = trainer.ckpt.resolve('latest')
                payload = torch.load(path, weights_only=False)
                payload.pop('optimizers')
                broken = root / 'sans_adam.pt'
                torch.save(payload, broken)
                with self.assertRaises(ValueError):
                    trainer.load_checkpoint(str(broken))
            # Frontière profil complète, générations simulées (aucun gain du modèle).
            class ConstantSampler:
                description = {'test_double': True}
                def __init__(self, *args):
                    pass
                def generate(self, *args, **kwargs):
                    return '1'
            args = SimpleNamespace(data_dir=str(root), out_dir=str(root / 'runs'), run='tiny', stage='sft',
                                   ckpt='latest', protocol='chat', device='cpu', dtype='fp32',
                                   splits='iid', tasks=30, seed=1, k=2, max_new=8,
                                   report=str(root / 'profile.json'), final_eval=False)
            with patch('frlm.eval_reason_bootstrap_v45.Sampler', ConstantSampler):
                result = profile(args)
            self.assertIn('objective:execute/ops_1', result['metrics'])
            self.assertIn('baselines', result)
            self.assertIn('prompt', result['details'][0])
            with self.assertRaises(FileExistsError):
                profile(args)
            args.report = str(root / 'baselines.json')
            args.baselines_only = True
            with patch('frlm.eval_reason_bootstrap_v45._resolve_checkpoint') as resolve:
                self.assertFalse(profile(args)['model_evaluated'])
                resolve.assert_not_called()
            args.splits = R.FINAL_SPLIT
            with self.assertRaises(ValueError):
                profile(args)
            args.final_eval = True
            with self.assertRaises(ValueError):
                profile(args)
            # Le préflight SFT vérifie les fichiers réels, sans importer Modal.
            _check_command(f'python run.py sft --run tiny --data-dir {root} --out-dir {root / "runs"} '
                           '--sft-recipe reason45c --resume latest --replay-frac 0', root)
            cap = report['capabilities']['ast_execute']
            path = root / cap['train_path']
            path.with_suffix('.mask').write_bytes(b'broken')
            with self.assertRaises((ValueError, RuntimeError)):
                audit(root, True, 'reason45c')

    def test_pas_de_publication_si_longueur_insuffisante(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            retention(root)
            before = (root / 'meta.json').read_bytes()
            with self.assertRaises(ValueError):
                R.prepare(root, examples=1000, eval_per_split=30, max_len=4)
            self.assertEqual(before, (root / 'meta.json').read_bytes())
            self.assertFalse(list(root.glob('reason_v45c*')))
            self.assertFalse(list(root.glob('raw/reason_bootstrap_v45c*')))

    def test_mix_en_tokens_isolation_et_padding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpora = []
            for index, supervised in enumerate((2, 20)):
                path = root / f'{index}.bin'
                tokens = [index + 1] * 24 + [0]
                masks = [0] * (24 - supervised) + [1] * supervised + [0]
                np.array(tokens, dtype=np.uint16).tofile(path)
                np.array(masks, dtype=np.uint8).tofile(path.with_suffix('.mask'))
                corpus = D.ConversationCorpus(path, 32)
                corpora.append((str(index), corpus, D.token_sampling_weight(0.5, 1, supervised)))
            mixture = D.SourceMixtureCorpus(corpora)
            x, y, masks = mixture.get_batch(12, 10000, device='cpu')
            counts = [masks[x[:, 0] == i + 1].sum().item() for i in range(2)]
            self.assertLess(abs(counts[0] / sum(counts) - 0.5), 0.025)
            self.assertTrue(torch.all(masks[:, 24:] == 0))
            for row in x:
                self.assertEqual(len(set(row.tolist()) - {0}), 1)
            path = root / 'unterminated.bin'
            np.array([1, 2, 0, 3, 4], dtype=np.uint16).tofile(path)
            np.array([0, 1, 0, 0, 1], dtype=np.uint8).tofile(path.with_suffix('.mask'))
            with self.assertRaises(ValueError):
                D.ConversationCorpus(path, 32)
            for bad_mask in ([0, 2, 0], [1, 1, 0], [0, 1, 1]):
                np.array([1, 2, 0], dtype=np.uint16).tofile(path)
                np.array(bad_mask, dtype=np.uint8).tofile(path.with_suffix('.mask'))
                with self.assertRaises(ValueError):
                    D.ConversationCorpus(path, 32)

    def test_conversion_poids_exhaustive_1_a_512(self):
        for a in range(1, 513):
            for b in range(1, 513):
                left = D.token_sampling_weight(0.5, 1, a) * a
                right = D.token_sampling_weight(0.5, 1, b) * b
                self.assertLess(abs(left - right), 1e-12)
        for args in ((0, 1, 1), (float('nan'), 1, 1), (0.5, 0, 1), (0.5, 1, 0)):
            with self.assertRaises(ValueError):
                D.token_sampling_weight(*args)

    def test_gradients_accumules_equivalent_batch_global_v2_v3(self):
        torch.manual_seed(11)
        for arch in ('v2', 'v3'):
            model = model_from_cfg(config(arch=arch))
            clone = copy.deepcopy(model)
            x = torch.randint(0, 12, (3, 8))
            y = torch.randint(0, 12, (3, 8))
            mask = torch.tensor([[0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 0]])
            _, loss, _ = model(x, y, mask, z_loss=0.0001, loss_reduction='mean')
            loss.backward()
            for i in range(3):
                _, micro, _ = clone(x[i:i+1], y[i:i+1], mask[i:i+1], z_loss=0.0001, loss_reduction='sum')
                (micro / mask.sum()).backward()
            for a, b in zip(model.parameters(), clone.parameters()):
                if a.grad is not None:
                    torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-5)

    def test_echec_copie_ne_corrompt_pas_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = run.CheckpointManager(Path(tmp))
            manager.save({'step': 1}, 1, is_best=True)
            copyfile = run.shutil.copyfile
            def failure(source, destination):
                if Path(destination).name.startswith('ckpt_best.'):
                    Path(destination).write_bytes(b'partial')
                    raise OSError('interruption simulée')
                return copyfile(source, destination)
            with patch.object(run.shutil, 'copyfile', failure), self.assertRaises(OSError):
                manager.save({'step': 2}, 2, is_best=True)
            self.assertEqual(torch.load(manager.resolve('best'), weights_only=False)['step'], 1)

    def test_preflight_profil_rl_manquant_et_reprise(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tok = tokenizer(root)
            stage = root / 'runs' / 'tiny' / 'sft'
            stage.mkdir(parents=True)
            cfg = config(tok.get_vocab_size())
            payload = {'model_cfg': cfg.to_dict(), 'model': model_from_cfg(cfg).state_dict(), 'step': 1, 'stage': 'sft'}
            torch.save(payload, stage / 'ckpt_best.pt')
            base = f'--run=tiny --data-dir={root} --out-dir={root / "runs"}'
            _check_command(f'python run.py rl-profile-v45 {base}', root)
            _check_command(f'python -m frlm.rl_profile_v45 {base}', root)
            with self.assertRaises(RuntimeError):
                _check_command(f'python run.py rl-v45 {base}', root)
            rl_stage = stage.parent / 'rlvr-v45'
            rl_stage.mkdir()
            torch.save({**payload, 'stage': 'rlvr-v45', 'accepted_updates': 1}, rl_stage / 'ckpt_best.pt')
            _check_command(f'python run.py rl-v45 {base} --resume best --reset-optimizer', root)
            with self.assertRaises(ValueError):
                _check_command(f'python run.py rl-v45 {base} --resume best', root)
            with self.assertRaises(RuntimeError):
                _check_command(f'python run.py rl-v45 {base} --resume best --reset-optimizer --no-refresh-profile', root)
            _, files, _ = _required_files(f'python run.py sft {base} --resume latest', root)
            self.assertIn(root / 'mid_train.bin', files)

    def test_wrapper_n_ajoute_pas_un_argument_train_au_profileur(self):
        cmd = 'python -m frlm.eval_reason_bootstrap_v45 --stage sft --run tiny'
        self.assertEqual(with_gpu_peak(cmd, 989), cmd)
        self.assertTrue(with_gpu_peak('python run.py sft', 989).endswith('--gpu-peak-tflops 989'))
        self.assertEqual(with_gpu_peak('python run.py sft --gpu-peak-tflops=123', 989),
                         'python run.py sft --gpu-peak-tflops=123')


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
