"""Vérification CPU HF -> GGUF -> llama.cpp, avec poids aléatoires (aucun entraînement).

python -m bench.verify_v5_gguf --llama-cpp /chemin/llama.cpp --tokenizer data-v5/tokenizer.json --full-size
"""
import argparse
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np
import torch

from frlm.export_v5 import normalize_config, LLAMA_REVISION
from frlm.model_v5 import ModelConfigV5, Qwen4ExpLM
from frlm.prepare_v5 import QWEN2_PATTERN
from frlm import data as D


def read_logits(path):
    with path.open("rb") as f:
        if f.read(8) != b"_logits_":
            raise ValueError("format llama-perplexity inattendu")
        ctx, vocab, chunks = map(int, np.frombuffer(f.read(12), dtype="<i4"))
        ids = np.frombuffer(f.read(ctx * chunks * 4), dtype="<i4").copy().reshape(chunks, ctx)
        data = np.frombuffer(f.read(), dtype="<u2").reshape(chunks, ctx // 2 - 1, (vocab + 1) // 2 * 2 + 4)
    scale = data[:, :, :2].copy().view("<f4")
    offset = data[:, :, 2:4].copy().view("<f4")
    return ids, data[:, :, 4:4+vocab] * scale + offset


def verify(llama_cpp: Path, tokenizer: Path | None = None, full_size=False):
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    torch.set_num_threads(2)
    torch.manual_seed(55103)
    with tempfile.TemporaryDirectory(prefix="frlm-v5-gguf-") as tmp:
        folder = Path(tmp)
        if tokenizer:
            tok = D.load_tokenizer(tokenizer)
        else:
            tok = Tokenizer(models.BPE())
            tok.pre_tokenizer = pre_tokenizers.Sequence([
                pre_tokenizers.Split(Regex(QWEN2_PATTERN), behavior="isolated"),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
            tok.decoder = decoders.ByteLevel()
            tok.train_from_iterator(["Les nombres 12345 et les fractions : calculons ensemble. " * 50],
                                    trainers.BpeTrainer(vocab_size=320, special_tokens=D.SPECIALS,
                                        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
        cfg = ModelConfigV5(vocab_size=tok.get_vocab_size()) if full_size else ModelConfigV5(
            vocab_size=tok.get_vocab_size(), d_model=128, n_layer=4, n_head=2,
            n_kv_head=1, head_dim=64, d_ff=64, linear_heads=2, linear_key_heads=1,
            ngram_vocab=127, num_experts=2, experts_per_token=1, max_seq_len=256)
        model = Qwen4ExpLM(cfg).eval()
        with torch.no_grad():
            for name, p in model.named_parameters():
                if "norm" in name and "linear_attn.norm" not in name:
                    p.uniform_(-0.1, 0.1)
                if ".ple.conv1d.weight" in name:
                    p.normal_(std=0.02)
        hf = folder / "hf"
        model.hf.save_pretrained(hf)
        fast = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token=D.EOT,
                                       eos_token=D.EOT, pad_token=D.EOT,
                                       additional_special_tokens=D.SPECIALS[1:])
        fast.save_pretrained(hf)
        normalize_config(hf)
        gguf = folder / "model-f32.gguf"
        def run(args, label):
            result = subprocess.run([str(x) for x in args], text=True, capture_output=True, check=False)
            if result.returncode:
                raise RuntimeError(f"{label}: {result.stderr[-4000:]} {result.stdout[-2000:]}")
            if label.startswith("inférence"):
                match = re.search(r"Final estimate: PPL = ([\d.eE+-]+)", result.stderr + result.stdout)
                if match is None or not math.isfinite(float(match[1])):
                    raise RuntimeError(f"{label}: perplexité finale absente ou non finie")
            return result.stdout
        run([sys.executable, "-m", "frlm.export_v5", "--hf-dir", hf, "--llama-cpp", llama_cpp,
             "--output", gguf, "--outtype", "f32"], "conversion")
        texts = ["1234567890 -12,345 1.25e-10", "L’été : ça coûte 12,50 € !\nTrès bien…",
                 "你好 😊 café e\u0301", "\\frac{12}{3} = 4", " a\t b\n\n c  "]
        texts.append("<|im_start|>assistant\n<think>2 + 2</think>4<|im_end|>\n<|endoftext|>")
        for text in texts:
            output = run([llama_cpp / "build/bin/llama-tokenize", "-m", gguf, "-p", text,
                          "--ids", "--no-escape"], "tokenizer")
            ids = json.loads(output[output.index("["):])
            if ids != tok.encode(text).ids:
                raise AssertionError(f"tokenizer divergent : {text!r}")
        prompt = ("Le chat calcule : 123 + 45 = 168. Pour diviser douze objets en trois groupes, "
                  "on place quatre objets dans chaque groupe.\n") * 24
        logits = folder / "logits.bin"
        run([llama_cpp / "build/bin/llama-perplexity", "-m", gguf, "-p", prompt,
             "-c", "64", "-b", "64", "-ub", "64", "-t", "2", "--chunks", "2",
             "-ctk", "f32", "-ctv", "f32", "-fa", "off",
             "--save-all-logits", logits], "inférence f32")
        ids, got = read_logits(logits)
        with torch.no_grad():
            expected = model(torch.from_numpy(ids).long())[0].float().log_softmax(-1)[:, 32:-1].numpy()
        error = float(np.abs(expected - got).max())
        if error > 0.003:
            raise AssertionError(f"log-probabilités HF/GGUF divergentes : {error}")
        quant = folder / "model-q4_k_m.gguf"
        run([llama_cpp / "build/bin/llama-quantize", gguf, quant, "Q4_K_M"], "quantification")
        output = run([llama_cpp / "build/bin/llama-perplexity", "-m", quant, "-p", prompt,
                      "-c", "64", "-b", "64", "-ub", "64", "-t", "2", "--chunks", "1"], "inférence Q4")
        return {"llama_revision": LLAMA_REVISION, "parameters": model.num_params(),
                "tokenizer_cases": len(texts), "f32_max_logprob_error": error,
                "f32_tolerance": 0.003, "q4_k_m_loaded": True,
                "reference_cache": "f32", "reference_flash_attention": False,
                "q4_bytes": quant.stat().st_size, "weights": "random+norms_and_PLE_perturbed"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--llama-cpp", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path)
    p.add_argument("--full-size", action="store_true")
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    result = verify(a.llama_cpp, a.tokenizer, a.full_size)
    print(json.dumps(result, indent=2))
    if a.output:
        a.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
